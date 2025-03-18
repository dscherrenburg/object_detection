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

from Faster_RCNN.custom_dataset import CustomDataset
from Faster_RCNN.custom_transforms import CustomTransforms, Resize, ToTensor
from evaluate import validation


class FasterRCNNTrainer:
    def __init__(self, project_folder, data_name, epochs=50, patience=10, batch_size=1, imgsz=640, resume=True):
        print("\n--- Training Faster RCNN ---\n")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_folder, self.data_name = project_folder, data_name
        self.epochs, self.patience, self.batch_size, self.imgsz = epochs, patience, batch_size, imgsz
        self.resume = resume
        self.num_workers = min(8, os.cpu_count())  # Use available CPU cores efficiently

        self.data_config_folder = os.path.join(project_folder, "dataset_configs/")
        run_name = f"e:{epochs}_b:{batch_size}_d:{data_name}"
        self.run_dir = os.path.join(project_folder, "Faster_RCNN", "runs", run_name)
        self.results_file = os.path.join(self.run_dir, "results.csv")

        self._initialize()

    def train(self):
        best_fitness = 0
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            # epoch += int(self.last_epoch + 1)
            # start_time = time.time()
            # self.model.train()
            # total_train_loss = 0

            # progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch:>{len(str(self.epochs))}}/{self.epochs}", dynamic_ncols=True)
            # for batch_idx, (images, targets, filenames) in enumerate(progress_bar):
            #     images = images.to(self.device)
            #     targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            #     with autocast(self.device.type):
            #         loss_dict = self.model(images, targets)
            #         loss = sum(loss_dict.values())  # Avoiding redundant `.to(self.device)`

            #     self.optimizer.zero_grad(set_to_none=True)
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()

            #     total_train_loss += loss.item()
            #     progress_bar.set_postfix(loss=f"{loss.item() / len(images):.4f}")
            #     torch.cuda.empty_cache()

            # train_loss = total_train_loss / len(self.train_dataloader)


            # === VALIDATION PHASE ===
            start_time = time.time()
            train_loss = 0

            if epoch  == self.epochs:
                val_metrics = validation(self.run_dir, self.model, self.val_dataloader, create_plots=True)
            else:
                val_metrics = validation(self.run_dir, self.model, self.val_dataloader)
            self.total_time += time.time() - start_time
            lr = self.lr_scheduler.get_last_lr()[0]
            results = [epoch+self.last_epoch+1, self.total_time, train_loss] + list(val_metrics.values()) + [lr]

            val_print = f"{' ' * 6}{' ' * (2 * len(str(self.epochs)) + 3)}"
            val_print += f"Train - loss: {train_loss:.4f}, lr: {lr:.2e}  ||    "
            val_print += f"Validation - loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['mAP50-95']:.4f}"

            self.lr_scheduler.step(val_metrics["loss"])

            # Save best model
            fitness = 0.1 * val_metrics["mAP50"] + 0.9 * val_metrics["mAP50-95"]
            if fitness > best_fitness:
                best_fitness = fitness
                epochs_no_improve = 0
                os.makedirs(os.path.join(self.run_dir, "weights"), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.run_dir, "weights", "best.pt"))
                val_print += "   ==> Best model saved ✅"
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    val_print += "   ==> Early stopping triggered ⏳"
                    print(val_print)
                    break  # Stop training
            
            print(val_print)
            self._save_checkpoint()
            self._save_results(results)
        
    def _initialize(self):
        self._load_data()
        self._create_model()
        os.makedirs(self.run_dir, exist_ok=True)

    def _load_data(self):
        print("Loading data...")
        data_yaml_file = os.path.join(self.data_config_folder, self.data_name, "dataset.yaml")
        transform = CustomTransforms([Resize(self.imgsz), ToTensor()])

        def create_loader(split):
            dataset = CustomDataset(data_yaml_file, split=split, transform=transform)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == "train"),
                              collate_fn=dataset.collate_fn, num_workers=self.num_workers, pin_memory=True)

        self.train_dataloader = create_loader("train")
        self.val_dataloader = create_loader("val")
        self.nc = self.train_dataloader.dataset.nc
        self._save_batch_samples(3)

    def _create_model(self):
        print("Creating model...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.nc+1)
        self.model.to(self.device)

        self.scaler = GradScaler(self.device.type)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

        if self.resume:
            checkpoint_path = os.path.join(self.run_dir, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                with open(self.results_file) as f:
                    results = pd.read_csv(f)
                    self.total_time = results["time"].iloc[-1]
                    self.last_epoch = results["epoch"].iloc[-1]
                print(f"Resumed training from checkpoint at epoch {self.last_epoch}.")
                return
            else:
                print("No checkpoint found.")

        results = pd.DataFrame(columns=["epoch", "time", "train_loss", "precision", "recall", "f1", "mAP50", "mAP50-95", "val_loss", "lr"])
        results.to_csv(self.results_file, index=False)
        self.total_time = 0
        self.last_epoch = 0
        print("Starting training fresh.")


    def _save_checkpoint(self):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.run_dir, "checkpoint.pth"))
    
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
            
            for batch_idx, (images, targets, filenames) in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break
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
