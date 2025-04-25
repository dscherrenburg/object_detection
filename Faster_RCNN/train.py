import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import cv2
import time
from tqdm import tqdm
import yaml
import shutil

from Faster_RCNN.custom_dataset import CustomDataset
from Faster_RCNN.custom_transforms import CustomTransforms, Resize, ToTensor
from evaluate import validation
from Evaluator import Evaluator, validate

from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.data import build_dataloader, build_fasterrcnn_dataset
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.cfg import get_cfg

class FasterRCNNTrainer:
    def __init__(self, run_dir, patience=10, imgsz=640, resume=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = run_dir
        self.patience, self.imgsz = patience, imgsz
        self.resume = resume
        self.num_workers = min(8, os.cpu_count())  # Use available CPU cores efficiently

        self.args = get_cfg(DEFAULT_CFG_DICT.copy())

        self.run_name = os.path.basename(run_dir)
        model_data = self._extract_data_from_runname(self.run_name)
        self.data_name = model_data['d']
        self.epochs = int(model_data['e'])
        self.batch_size = int(model_data['b'])

        
        self.data_config_dir = "/kaggle/input/object_detection/pytorch/default/2/dataset_configs"

        self._initialize()

    def train(self):
        if not hasattr(self, "best_fitness"):
            self.best_fitness = 0
            self.epochs_no_improve = 0

        for epoch in range(self.last_epoch+1, self.epochs+1):
            start_time = time.time()
            self.model.train()
            total_train_loss = 0
            train_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch:>{len(str(self.epochs))}}/{self.epochs}", dynamic_ncols=True)
            for batch_idx, batch in enumerate(progress_bar):
                images, targets = self.preprocess(batch)

                with autocast(self.device.type):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())  # Avoiding redundant `.to(self.device)`

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # print("\n\n", torch.cuda.memory_allocated() / 1024 ** 2, "MB currently allocated")
                # print(torch.cuda.max_memory_allocated() / 1024 ** 2, "MB peak")

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item() / len(images):.4f}")

            train_loss = total_train_loss / len(self.train_dataloader)

            # === VALIDATION PHASE ===
            self.total_time += time.time() - start_time
            lr = self.lr_scheduler.get_last_lr()[0]

            val_metrics = validate(self.model, self.val_dataloader, conf=self.args.conf, iou=self.args.iou, preprocess=self.preprocess, postprocess=self.postprocess)

            fitness = 0.1 * val_metrics["mAP50"] + 0.9 * val_metrics["mAP50-95"]
            results = [epoch, self.total_time, train_loss] + list(val_metrics.values()) + [fitness, lr]

            val_print = f"{' ' * 6}TRAIN - loss: {train_loss:.3f}, lr: {lr:.1e}  ||    "
            val_print += f"VAL - P: {val_metrics['Precision']:.3f}   R: {val_metrics['Recall']:.3f}   F1: {val_metrics['F1']:.3f}   mAP50: {val_metrics['mAP50']:.4f}   mAP50-95: {val_metrics['mAP50-95']:.4f}   fitness: {fitness:.4f}"

            self.lr_scheduler.step()

            # Save best model
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.epochs_no_improve = 0
                self._save_checkpoint(epoch, results, save_best=True)
                val_print += "   ==> Best model saved ✅"
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    val_print += "   ==> Early stopping triggered ⏳"
                    print(val_print)
                    break  # Stop training
            
            print(val_print)
            self._save_checkpoint(epoch, results)
            self._save_results(results)
            torch.cuda.empty_cache()
        
    def _initialize(self):
        self._load_data()
        self._create_model()
        os.makedirs(self.run_dir, exist_ok=True)

    def _load_data(self):
        print("Loading data...")
        # project_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.run_dir)))
        # data_config_folder = os.path.join(project_folder, "dataset_configs/")
        self.data_yaml_file = os.path.join(self.data_config_dir, self.data_name, "dataset.yaml")

        self.args.data = self.data_yaml_file
        self.args.epochs = self.epochs
        self.args.patience = self.patience
        self.args.batch = self.batch_size
        self.args.imgsz = self.imgsz
        
        self.data = yaml.safe_load(open(self.data_yaml_file, "r"))

        self.train_dataloader = self.get_dataloader(os.path.join(self.data["path"], self.data["train"]), self.batch_size, rank=0, mode="train")
        self.val_dataloader = self.get_dataloader(os.path.join(self.data["path"], self.data["val"]), self.batch_size, rank=0, mode="val")
        self.nc = self.data["nc"]

        self._save_batch_samples(3)
        return
        transform = CustomTransforms([Resize(self.imgsz), ToTensor()])

        def create_loader(split):
            dataset = CustomDataset(self.data_yaml_file, split=split, transform=transform)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == "train"),
                              collate_fn=dataset.collate_fn, num_workers=self.num_workers, pin_memory=True)

        self.train_dataloader = create_loader("train")
        self.val_dataloader = create_loader("val")
        self.nc = self.train_dataloader.dataset.nc

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = build_fasterrcnn_dataset(self.args, dataset_path, batch_size, self.data, mode=mode, rect=mode == "val")
        shuffle = mode == "train"
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def _create_model(self):
        print("Creating model...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        freeze = self.args.get("freeze", 0)
        trainable = max(0, 5 - freeze) if isinstance(freeze, int) else 5
        self.model = fasterrcnn_resnet50_fpn(weights=weights,
                                             trainable_backbone_layers=trainable,
                                             min_size=self.imgsz,
                                             max_size=self.imgsz,
                                             box_score_thresh=0.001, # self.args.get("conf", 0.001),
                                             box_nms_thresh=self.args.get("iou", 0.7),
                                             box_detections_per_img=self.args.get("max_det", 100))
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.nc+1)
        self.model.to(self.device)

        self.scaler = GradScaler(self.device.type, enabled=True)
        self.lr0 = self.args.lr0
        self.momentum = self.args.momentum
        self.weight_decay = self.args.weight_decay
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr0, momentum=self.momentum,
                                         weight_decay=self.weight_decay, nesterov=True)
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10)

        if self.args.compile:
            print("Compiling model...")
            self.model.backbone.body = torch.compile(self.model.backbone.body)


        checkpoint_path = os.path.join(self.run_dir, "weights", "last.pt")
        if self.resume:
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                self.last_epoch = checkpoint['epoch']
                self.total_time = checkpoint['time']
                self.epochs = checkpoint['train_args']['epochs']
                self.best_fitness = checkpoint['best_fitness']
                self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                self.results_file = os.path.join(self.run_dir, "results.csv")
                print(f"Resuming training from epoch {self.last_epoch} with best fitness {self.best_fitness:.4f}.")
                return
            else:
                print("No checkpoint found.")
        if os.path.exists(checkpoint_path):
            inpt = input(f"The folder for {self.run_name} already exists. Press Enter to overwrite it or type 'new' to start a new run.\n")
            if inpt.lower() == "new":
                print("Creating a new run.")
                for i in range(2, 100):
                    if not os.path.exists(self.run_dir + f"_{i}"):
                        self.run_dir += f"_{i}"
                        break
            else:
                shutil.rmtree(self.run_dir)
                print("Overwriting existing model.")
        os.makedirs(self.run_dir, exist_ok=True)
        self.results_file = os.path.join(self.run_dir, "results.csv")
        results = pd.DataFrame(columns=["epoch", "time", "train_loss", "precision", "recall", "f1", "mAP50", "mAP50-95", "fitness", "lr"])
        results.to_csv(self.results_file, index=False)
        self.total_time = 0
        self.last_epoch = 0
        print("Starting training fresh.")


    def _save_checkpoint(self, epoch, results, save_best=False):
        if save_best:
            checkpoint_path = os.path.join(self.run_dir, "weights", "best.pt")
        else:
            checkpoint_path = os.path.join(self.run_dir, "weights", "last.pt")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        checkpoint = {
            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'epoch': epoch,
            'best_fitness': self.best_fitness,
            'epochs_no_improve': self.epochs_no_improve,
            'time': self.total_time,
            'train_args': {
                'model': "FasterRCNN",
                'compile': self.args.compile,
                'data': self.data_yaml_file,
                'epochs': self.epochs,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'imgsz': self.imgsz,
                'save': True,
                'save_period': 1,
                'device': self.device.type,
                'workers': self.num_workers,
                'name': self.run_name,
                'pretrained': True,
                'optimizer': 'SGD',
                'resume': self.resume,
                'lr0': self.lr0,
                'momentum': self.momentum,
                'weight_decay': self.weight_decay,
                'lr_scheduler': 'ReduceLROnPlateau',
                'lr_scheduler_patience': 10,
                'lr_scheduler_factor': 0.1,
            },
            'train_metrics': results[2:],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
    
    def _save_results(self, results):
        with open(self.results_file, "a") as f:
            f.write(",".join(map(str, results)) + "\n")        

    def _save_batch_samples(self, n_batches, data_type=None):
        os.makedirs(self.run_dir, exist_ok=True)
        if data_type is None:
            print("Storing batch samples for train and val datasets...")
            self._save_batch_samples(n_batches, "train")
            self._save_batch_samples(n_batches, "val")
        else:
            if data_type == "train":
                dataloader = self.train_dataloader
            elif data_type == "val":
                dataloader = self.val_dataloader
            else:
                raise ValueError(f"Invalid data type: {data_type}")
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break
                images = batch.get("img")
                targets = batch.get("targets", None)
                # cv2.imwrite(os.path.join(self.run_dir, f"test.png"), images[0].permute(1, 2, 0).cpu().numpy())

                if data_type == "test":
                    images = images.to(self.device)
                    with torch.no_grad():
                        predictions = self.model(images)
                    self._save_batch_sample(images, targets, predictions, name=f"{data_type}_batch{batch_idx}.png")
                else:
                    self._save_batch_sample(images, targets, name=f"{data_type}_batch{batch_idx}.png")
    
    def _save_batch_sample(self, images, targets, predictions=None, name="batch_samples.png"):
        image_file = os.path.join(self.run_dir, name)
        if os.path.exists(image_file):
            return
        image_batch = []
        for i in range(len(images)):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            image = np.clip(image, 0, 1) * 255
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = targets[i]["boxes"].cpu().numpy()
            labels = targets[i]["labels"].cpu().numpy()
            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax = box.astype(int)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # cv2.putText(image, f"{label}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if predictions is not None:
                pred_boxes = predictions[i]["boxes"].cpu().numpy()
                pred_labels = predictions[i]["labels"].cpu().numpy()
                pred_scores = predictions[i]["scores"].cpu().numpy()
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    xmin, ymin, xmax, ymax = box.astype(int)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(image, f"{label} {score:.2f}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            image_batch.append(image)
        image_batch = cv2.vconcat(image_batch)
        cv2.imwrite(image_file, image_batch)
    
    def _extract_data_from_runname(self, run_name):
        run_name_items = [item.split(":") for item in run_name.split("_")]
        run_name_items = [item for item in run_name_items if len(item) == 2]
        return {key: value for key, value in run_name_items}

    def preprocess(self, batch):
        images = batch.get("img")
        targets = batch.get("targets", None)
        device = self.device

        if targets is not None:
            targets = [
                {
                    "boxes": t["boxes"].to(device=device),
                    "labels": (t["labels"]).to(device=device),
                } for t in targets
            ]
        images = images.to(device=device, dtype=torch.half)
        return images, targets

    def postprocess(self, predictions, targets=None):
        predictions = [{
                "boxes": p["boxes"].cpu().numpy(),
                "labels": p["labels"].cpu().numpy(),
                "scores": p["scores"].cpu().numpy(),
            } for p in predictions]
        if targets is not None:
            targets = [{
                "boxes": t["boxes"].cpu().numpy(),
                "labels": t["labels"].cpu().numpy(),
            } for t in targets]
            return predictions, targets
        return predictions
