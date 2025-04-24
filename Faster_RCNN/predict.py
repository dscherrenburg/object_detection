import os
import shutil
import torch
from itertools import zip_longest
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
from torch.amp import autocast
import numpy as np
import cv2
from tqdm import tqdm

from Faster_RCNN.custom_dataset import CustomDataset
from Faster_RCNN.custom_transforms import CustomTransforms, Resize, ToTensor

from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.cfg import get_cfg


class FasterRCNNPredictor:
    def __init__(self, run_dir, conf=0.01, imgsz=640, batch_size=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = run_dir
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.conf_thresh = conf

        self.args = get_cfg(DEFAULT_CFG_DICT.copy())

    def predict(self, model_path=None, save_images=False, save_labels=True):
        if save_images or save_labels:
            self.predict_dir = os.path.join(self.run_dir, "predict")
            self.labels_dir = os.path.join(self.predict_dir, "labels")
            # Check if the dirs are empty
            if len(os.listdir(self.predict_dir)) > 1 or len(os.listdir(self.labels_dir)) > 0:
                i = input("The predict directory is not empty. Do you want to delete the contents? (y/n): ")
                if i.lower() == "y":
                    shutil.rmtree(self.predict_dir)
                    os.makedirs(self.labels_dir, exist_ok=True)
                elif i.lower() == "n":
                    print("Continuing without saving images or labels.")
                    return
                else:
                    print("Invalid input, try again.")
                    return self.predict(model_path, save_images, save_labels)
        
        self._load_data()
        self._load_model(model_path)
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(self.dataloader, desc="Predicting", dynamic_ncols=True)
            for batch_idx, (images, targets, filenames) in enumerate(progress_bar):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                with autocast(self.device.type):
                    predictions = self.model(images)
                self._save_predictions(images, targets, filenames, predictions, save_images, save_labels)
        
    def _save_predictions(self, images, targets, filenames, predictions, save_images=False, save_labels=True):
        for i in range(len(images)):
            image = images[i]
            target = targets[i]
            filename = filenames[i]
            prediction = predictions[i]
            if save_images:
                image = image.permute(1, 2, 0).cpu().numpy()
                image = np.clip(image, 0, 1) * 255
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                t_boxes = target["boxes"].cpu().numpy()
                p_boxes = prediction["boxes"].cpu().numpy()
                p_scores = prediction["scores"].cpu().numpy()
                for t_box, p_box, p_score in zip_longest(t_boxes, p_boxes, p_scores, fillvalue=None):
                    if t_box is not None:
                        xmin, ymin, xmax, ymax = t_box.astype(int)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    if p_box is not None:
                        xmin, ymin, xmax, ymax = p_box.astype(int)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                        cv2.putText(image, f"{p_score:.2f}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imwrite(os.path.join(self.predict_dir, filename), image)
            if save_labels:
                with open(os.path.join(self.labels_dir, filename.replace(".png", ".txt")), "w") as f:
                    for box, label, score in zip(prediction["boxes"].cpu().numpy(),
                                                 prediction["labels"].cpu().numpy(),
                                                 prediction["scores"].cpu().numpy()):
                        xmin, ymin, xmax, ymax = box.astype(int)
                        x, y, w, h = self.dataloader.dataset.pascal_to_yolo(xmin, ymin, xmax, ymax)
                        f.write(f"{label-1} {x} {y} {w} {h} {score}\n")

    def _load_data(self):
        print("Loading data...")
        project_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.run_dir)))
        data_config_folder = os.path.join(project_folder, "dataset_configs")
        data_name = self.run_dir.split(":")[-1].split("_")[0]
        data_yaml_file = os.path.join(data_config_folder, data_name, "dataset.yaml")
        transform = CustomTransforms([Resize(self.imgsz), ToTensor()])

        dataset = CustomDataset(data_yaml_file, split="test", transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size//4, shuffle=False,
            collate_fn=dataset.collate_fn, num_workers=8, pin_memory=True)
        _ = next(iter(self.dataloader))  # Load first batch to avoid slow first iteration
        self.nc = self.dataloader.dataset.nc

    def _load_model(self, model_path=None):
        print("Loading model...")
        if model_path is None:
            model_path = os.path.join(self.run_dir, "weights", "best.pt")
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            exit()

        checkpoint = torch.load(model_path, map_location=self.device)
        train_args = checkpoint['train_args']


        self.model = fasterrcnn_resnet50_fpn(weights=None,
                                             min_size=self.imgsz,
                                             max_size=self.imgsz,
                                             box_score_thresh=self.conf_thresh,
                                             box_nms_thresh=self.args.get("iou", 0.7),
                                             box_detections_per_img=self.args.get("max_det", 100))
        if train_args.get('compile', False):
            self.model.backbone.body = torch.compile(self.model.backbone.body)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.nc+1)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)