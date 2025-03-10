import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_yaml, split="train", transform=None):
        self.dataset_yaml = dataset_yaml
        self.transform = transform
        self.image_paths = self.load_image_paths(dataset_yaml, split)
        self.label_paths = [p.replace("images", "labels").replace(".png", ".txt") for p in self.image_paths]
        self.num_classes = self.get_num_classes()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)  # Normalize & format
        self.img_size = (image.shape[2], image.shape[1])

        # Load labels if they exist
        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            boxes, labels = self.load_labels(label_path)
        else:
            boxes, labels = torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)  # Empty annotations

        # Create target dictionary
        target = {
            "boxes": boxes,      # Shape [N, 4] (empty tensor if no objects)
            "labels": labels,    # Shape [N] (empty tensor if no objects)
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target, os.path.basename(img_path)
    
    def load_image_paths(self, dataset_yaml, split):
        split_file = os.path.join(dataset_yaml["path"], dataset_yaml[split])
        with open(split_file, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def load_labels(self, label_path):
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                data = list(map(float, line.strip().split()))
                class_id, x, y, w, h = data  # YOLO format
                xmin, ymin, xmax, ymax = self.yolo_to_pascal(x, y, w, h)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(class_id) + 1)  # Ensure labels are > 0 (Faster R-CNN requires class IDs to start from 1)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def yolo_to_pascal(self, x, y, w, h):
        """Convert YOLO format (fractions) to Pascal VOC format (absolute pixel values)."""
        xmin = (x - w / 2) * self.img_size[0]
        ymin = (y - h / 2) * self.img_size[1]
        xmax = (x + w / 2) * self.img_size[0]
        ymax = (y + h / 2) * self.img_size[1]
        return xmin, ymin, xmax, ymax
    
    def pascal_to_yolo(self, xmin, ymin, xmax, ymax):
        """Convert Pascal VOC format (absolute pixel values) to YOLO format (fractions)."""
        x = (xmin + xmax) / (2 * self.img_size[0])
        y = (ymin + ymax) / (2 * self.img_size[1])
        w = (xmax - xmin) / self.img_size[0]
        h = (ymax - ymin) / self.img_size[1]
        return x, y, w, h
    
    def get_num_classes(self):
        return len(self.dataset_yaml["names"]) + 1  # Add background class

    @staticmethod
    def collate_fn(batch):
        images, targets, filenames = zip(*batch)
        images = torch.stack(images)
        return images, targets, filenames


def load_dataset_config(config_path):
    """Loads the dataset YAML file."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
