import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.amp import GradScaler, autocast
import torch.multiprocessing as mp
import yaml
import numpy as np
import pandas as pd
import cv2
import time
from tqdm import tqdm

from custom_dataset import CustomDataset
from custom_transforms import CustomTransforms, Resize, ToTensor
from evaluate import validation


class FasterRCNNTrainer:
    def __init__(self, project_folder, data_name, epochs=50, batch_size=1, imgsz=640, patience=100, resume=True, confidence_threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.project_folder = project_folder
        self.data_name = data_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.patience = patience
        self.resume = resume
        self.confidence_threshold = confidence_threshold
        self.num_workers = min(8, os.cpu_count())  # Use available CPU cores efficiently

        self.data_config_folder = os.path.join(project_folder, "dataset_configs/")
        self.runs_dir = os.path.join(project_folder, "Faster_RCNN/runs/")
        self.run_name = f"e:{epochs}_b:{batch_size}_data:{data_name}_2"

        self._load_data()
        self._create_model()

    def train(self):
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            epoch += int(self.last_epoch + 1)
            start_time = time.time()
            self.model.train()
            total_train_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch:>{len(str(self.epochs))}}/{self.epochs}", dynamic_ncols=True)
            for batch_idx, (images, targets, filenames) in enumerate(progress_bar):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                with autocast(self.device.type):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())  # Avoiding redundant `.to(self.device)`

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss.item() / len(images):.4f}")
                torch.cuda.empty_cache()

            train_loss = total_train_loss / len(self.train_dataloader)


            # === VALIDATION PHASE ===
            val_metrics = validation(self.model, self.val_dataloader, self.device)
            self.total_time += time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]
            self.results.loc[len(self.results)] = [epoch, self.total_time, train_loss] + list(val_metrics.values()) + [lr]

            val_print = f"{' ' * 6}{' ' * (2 * len(str(self.epochs)) + 3)}"
            val_print += f"Train - loss: {train_loss:.4f}, lr: {lr:.2e}  ||    "
            val_print += f"Validation - loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['mAP50_95']:.4f}"

            self.lr_scheduler.step(val_metrics["loss"])

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
                os.makedirs(os.path.join(self.runs_dir, self.run_name, "weights"), exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.runs_dir, self.run_name, "weights", "best.pt"))
                val_print += "   ==> Best model saved ✅"
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    val_print += "   ==> Early stopping triggered ⏳"
                    print(val_print)
                    break  # Stop training
            
            print(val_print)

            self.results.to_csv(os.path.join(self.runs_dir, self.run_name, "results.csv"), index=False)
            self._save_checkpoint(epoch, self.optimizer, self.lr_scheduler)

    def predict(self, save_images=False, save_labels=True):
        self.model.eval()
        predict_dir = os.path.join(self.runs_dir, self.run_name, "predict")
        labels_dir = os.path.join(predict_dir, "labels")
        os.makedirs(labels_dir, exist_ok=True)
        self._save_batch_samples(3, data_type="test")
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_dataloader, desc="Testing", dynamic_ncols=True)
            for batch_idx, (images, targets, filenames) in enumerate(progress_bar):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                with autocast(self.device.type):
                    predictions = self.model(images)
                
                if save_images:
                    self._save_images(images, targets, filenames, predictions, save_labels=save_labels)
                elif save_labels:
                    self._save_images(predictions=predictions, filenames=filenames, save_labels=save_labels)
        
        print(f"Predictions saved to {predict_dir}")
                    
                    

    def _create_model(self):
        print("Creating model...")
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.nc)
        self.model.to(self.device)

        self.scaler = GradScaler(self.device.type)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, verbose=True)

        os.makedirs(os.path.join(self.runs_dir, self.run_name), exist_ok=True)
        self._save_batch_samples(3)

        if self.resume:
            checkpoint_path = os.path.join(self.runs_dir, self.run_name, "checkpoint.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                self.results = pd.read_csv(os.path.join(self.runs_dir, self.run_name, "results.csv"))
                self.total_time = self.results["time"].iloc[-1]
                self.last_epoch = self.results["epoch"].iloc[-1]
                print(f"Resumed training from checkpoint at epoch {self.last_epoch}.")
                return
            else:
                print("No checkpoint found.")

        self.results = pd.DataFrame(columns=["epoch", "time", "train_loss", "precision", "recall", "f1", "mAP50", "mAP50_95", "val_loss", "lr"])
        self.total_time = 0
        self.last_epoch = 0
        print("Starting training fresh.")


    def _load_data(self):
        print("Loading data...")
        data_yaml = os.path.join(self.data_config_folder, self.data_name, "dataset.yaml")
        dataset_config = yaml.safe_load(open(data_yaml, "r"))
        transform = CustomTransforms([Resize(self.imgsz), ToTensor()])

        def create_loader(split):
            dataset = CustomDataset(dataset_config, split=split, transform=transform)
            return DataLoader(
                dataset, batch_size=self.batch_size, shuffle=(split == "train"),
                collate_fn=dataset.collate_fn, num_workers=self.num_workers, pin_memory=True,
                persistent_workers=True  # Keeps workers alive for performance
            )

        self.train_dataloader = create_loader("train")
        _ = next(iter(self.train_dataloader))  # Load first batch to avoid slow first iteration
        self.val_dataloader = create_loader("val")
        _ = next(iter(self.val_dataloader))  # Load first batch to avoid slow first iteration
        self.test_dataloader = create_loader("test")
        _ = next(iter(self.test_dataloader)) # Load first batch to avoid slow first iteration
        self.nc = dataset_config["nc"] + 1  # Include background class

    def _save_checkpoint(self, epoch, optimizer, lr_scheduler):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.runs_dir, self.run_name, "checkpoint.pth"))

    def _save_batch_samples(self, n_batches, data_type=None):
        if data_type is None:
            print("Storing batch samples for train and val datasets...")
            self._save_batch_samples(n_batches, "train")
            self._save_batch_samples(n_batches, "val")
        else:
            if data_type == "train":
                dataloader = self.train_dataloader
            elif data_type == "val":
                dataloader = self.val_dataloader
            elif data_type == "test":
                dataloader = self.test_dataloader
            else:
                raise ValueError(f"Invalid data type: {data_type}")
            
            for batch_idx, (images, targets, filenames) in enumerate(dataloader):
                if batch_idx >= n_batches:
                    break
                if data_type == "test":
                    images = images.to(self.device)
                    with torch.no_grad():
                        predictions = self.model(images)
                    self._save_batch_sample(images, targets, filenames, predictions, name=f"{data_type}_batch{batch_idx}.png")
                else:
                    self._save_batch_sample(images, targets, filenames, name=f"{data_type}_batch{batch_idx}.png")
        

    def _save_batch_sample(self, images, targets, filenames, predictions=None, name="batch_samples.png"):
        image_file = os.path.join(self.runs_dir, self.run_name, name)
        if os.path.exists(image_file):
            return
        save_dir = os.path.join(self.runs_dir, self.run_name)
        os.makedirs(save_dir, exist_ok=True)
        self._save_images(images, targets, filenames, predictions, save_dir=save_dir, output_name=name)


    def _save_images(self, images=None, targets=None, filenames=None, predictions=None, save_dir=None, save_labels=False, output_name=None):
        if save_dir is None:
            save_dir = os.path.join(self.runs_dir, self.run_name, "predict")
        os.makedirs(save_dir, exist_ok=True)

        n_images = len(images) if images is not None else 0
        n_targets = len(targets) if targets is not None else 0
        n_filenames = len(filenames) if filenames is not None else 0
        n_predictions = len(predictions) if predictions is not None else 0

        if filenames is None:
            filenames = [f"image_{i}.png" for i in range(max(n_images, n_targets, n_filenames, n_predictions))]
        
        image_batch = []
        for i in range(max(n_images, n_targets, n_filenames, n_predictions)):
            if predictions is not None:
                pred_boxes = predictions[i]["boxes"].cpu().numpy()
                pred_labels = predictions[i]["labels"].cpu().numpy()
                pred_scores = predictions[i]["scores"].cpu().numpy()
        
            if images is not None:
                image = images[i].permute(1, 2, 0).cpu().numpy()
                image = np.clip(image, 0, 1) * 255
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if targets is not None:
                    boxes = targets[i]["boxes"].cpu().numpy()
                    labels = targets[i]["labels"].cpu().numpy()
                    for box, label in zip(boxes, labels):
                        xmin, ymin, xmax, ymax = box.astype(int)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(image, f"{label}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if predictions is not None:
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        xmin, ymin, xmax, ymax = box.astype(int)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        cv2.putText(image, f"{label} {score:.2f}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if output_name is not None:
                    image_batch.append(image)
                else:
                    cv2.imwrite(os.path.join(save_dir, os.path.basename(filenames[i])), image)
                    
            if save_labels and predictions is not None:
                with open(os.path.join(save_dir, f"{os.path.basename(filenames[i])}.txt"), "w") as f:
                    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                        xmin, ymin, xmax, ymax = box.astype(int)
                        x, y, w, h = self.test_dataloader.dataset.pascal_to_yolo(xmin, ymin, xmax, ymax)
                        f.write(f"{label-1} {x} {y} {w} {h} {score}\n")
        
        if output_name is not None:
            image_batch = cv2.vconcat(image_batch)
            cv2.imwrite(os.path.join(save_dir, output_name), image_batch)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    trainer = FasterRCNNTrainer(project_folder="/home/daan/object_detection/", data_name="split_2_interval_10", epochs=100, batch_size=2, imgsz=640)
    trainer.train()
    trainer.predict(save_images=True, save_labels=True)
