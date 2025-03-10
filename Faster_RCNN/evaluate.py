import torch
from torch.amp import autocast
import os
from torchvision.ops import box_iou
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import numpy as np
import yaml

def validation(model, dataloader, device):
    metrics = EvaluationMetrics(model=model, dataloader=dataloader, device=device)
    results = metrics()
    return {key: np.float64(value) for key, value in results.items()}

def test(pred_dir, data_config_file, iou_threshold=0.5):
    """Evaluates object detection predictions against ground truth labels."""
    data_config = yaml.safe_load(open(data_config_file))
    test_txt = os.path.join(data_config["path"], data_config["test"])
    image_paths = [line.strip() for line in open(test_txt, 'r').readlines() if line.strip()]
    gt_files = [p.replace("images", "labels").replace(".png", ".txt") for p in image_paths]
    metrics = EvaluationMetrics(pred_dir=pred_dir, gt_files=gt_files, iou_threshold=iou_threshold)
    results = metrics()
    return {key: np.float64(value) for key, value in results.items()}
    

class EvaluationMetrics:
    def __init__(self, model=None, dataloader=None, pred_dir=None, gt_files=None, device="cuda", iou_threshold=0.5):
        self.model = model
        self.dataloader = dataloader
        self.pred_dir = pred_dir
        self.gt_files = gt_files
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        self.iou_threshold = iou_threshold
        self.metrics = {key: 0.0 for key in ['precision', 'recall', 'f1', 'mAP50', 'mAP50_95', 'loss']}

    def __call__(self):
        torch.cuda.empty_cache()
        if self.dataloader is not None:
            use_dataloader = True
        elif self.pred_dir is not None and self.gt_files is not None:
            use_dataloader = False
        else:
            raise ValueError("Either dataloader or prediction and ground truth directories must be provided.")
        self.get_metrics(use_dataloader)
        return self.metrics

    def get_metrics(self, use_dataloader=True):
        
        # Calculate final metrics
        if use_dataloader:
            self.compute_metrics_from_dataloader(self.model, self.dataloader, torch.linspace(0.5, 0.95, 10))
        else:
            self.compute_metrics_from_files(self.pred_dir, self.gt_files, torch.linspace(0.5, 0.95, 10))
        
        # Update metrics dictionary
        self.metrics.update({
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mAP50': self.mAP50,
            'mAP50_95': self.mAP50_95,
            'loss': self.loss.item()
        })

    def compute_metrics_from_dataloader(self, model, dataloader, iou_thresholds=[0.5]):
        """
        Compute precision, recall, F1-score, mAP50, and mAP50_95 given a dataloader and model.
        """
        all_predictions = []
        all_ground_truths = []

        model.eval()
        with torch.no_grad():
            for images, targets, filenames in dataloader:
                images = [img.to(self.device) for img in images]
                with autocast(self.device.type):
                    outputs = model(images)

                for i in range(len(images)):
                    gt_boxes = targets[i]['boxes'].to(self.device)
                    gt_labels = targets[i]['labels'].cpu().numpy()
                    
                    pred_boxes = outputs[i]['boxes'].to(self.device)
                    pred_labels = outputs[i]['labels'].cpu().numpy()
                    pred_scores = outputs[i]['scores'].cpu().numpy()
                    
                    # Convert ground truth and predictions to required format
                    ground_truths = [(label, box.tolist()) for label, box in zip(gt_labels, gt_boxes.cpu())]
                    predictions = [(label, score, box.tolist()) for label, score, box in zip(pred_labels, pred_scores, pred_boxes.cpu())]
                    
                    all_ground_truths.extend(ground_truths)
                    all_predictions.extend(predictions)

        num_classes = dataloader.dataset.get_num_classes()
        self.compute_metrics(all_predictions, all_ground_truths, iou_thresholds, num_classes)

    def compute_metrics_from_files(self, pred_dir, gt_files, iou_thresholds=[0.5]):
        """
        Compute precision, recall, F1-score, mAP50, and mAP50_95 given prediction directory and list of gt label files.
        """
        all_predictions = []
        all_ground_truths = []

        for i, gt_file in enumerate(gt_files):
            pred_file = os.path.join(pred_dir, os.path.basename(gt_file))
            with open(pred_file, 'r') as f:
                pred_labels = [list(map(float, line.strip().split())) for line in f.readlines()]
            with open(gt_file, 'r') as f:
                gt_labels = [list(map(float, line.strip().split())) for line in f.readlines()]
            
            pred_labels = [(int(label), score, [x, y, x + w, y + h]) for label, x, y, w, h, score in pred_labels]
            gt_labels = [(int(label), [x, y, x + w, y + h]) for label, x, y, w, h in gt_labels]
            
            all_predictions.extend(pred_labels)
            all_ground_truths.extend(gt_labels)

        num_classes = max([max([p[0] for p in all_predictions]), max([g[0] for g in all_ground_truths])])
        self.compute_metrics(all_predictions, all_ground_truths, iou_thresholds, num_classes)

    def compute_metrics(self, predictions, ground_truths, iou_thresholds=[0.5], num_classes=1):
        """
        Compute precision, recall, F1-score, mAP50, and mAP50_95 given predictions and ground truths.
        """
        ap_per_threshold = []
        self.loss = 0

        for iou_thresh in iou_thresholds:
            ap_per_class = []

            for class_id in range(1, num_classes + 1):
                pred_boxes = [p for p in predictions if p[0] == class_id]
                gt_boxes = [g for g in ground_truths if g[0] == class_id]

                pred_boxes.sort(key=lambda x: x[1], reverse=True)
                
                matched_preds, matched_gts, matched_indices = match_boxes(
                    torch.tensor([p[2] for p in pred_boxes], dtype=torch.float32),
                    torch.tensor([g[1] for g in gt_boxes], dtype=torch.float32),
                    iou_thresh)
                
                if iou_thresh == self.iou_threshold:
                    self.loss = F.l1_loss(matched_preds, matched_gts, reduction='mean')
                
                tp = np.zeros(len(pred_boxes))
                fp = np.zeros(len(pred_boxes))
                num_gt = len(gt_boxes)
                gt_matched = set()
                
                for i, idx in enumerate(matched_indices):
                    if idx is not None and idx not in gt_matched:
                        tp[i] = 1
                        gt_matched.add(idx)
                    else:
                        fp[i] = 1
                
                cum_tp = np.cumsum(tp)
                cum_fp = np.cumsum(fp)
                recalls = np.nan_to_num(cum_tp / max(num_gt, 1))
                precisions = np.nan_to_num(cum_tp / np.maximum(cum_tp + cum_fp, 1e-10))
                
                if iou_thresh == 0.5:
                    self.precision = tp.sum() / max(tp.sum() + fp.sum(), 1e-10)
                    self.recall = tp.sum() / max(num_gt, 1)
                    self.f1 = 2 * (self.precision * self.recall) / max(self.precision + self.recall, 1e-10)
                
                ap = self.compute_ap(precisions, recalls)
                ap_per_class.append(ap)

            ap_per_threshold.append(np.mean(ap_per_class))
        
        self.loss /= len(predictions)
        self.mAP50 = ap_per_threshold[0]
        self.mAP50_95 = np.mean(ap_per_threshold)

    def compute_ap(self, precisions, recalls):
        """Compute AP by integrating the precision-recall curve."""
        recalls = np.concatenate(([0], recalls, [1]))  # Ensure start and end points
        precisions = np.concatenate(([0], precisions, [0]))  # Ensure start and end points

        # Compute precision envelope
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Compute AP as area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]  # Recall change points
        return np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return None, None, []

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    matched_preds, matched_gts = [], []
    assigned_gts = set()
    matched_indices = [None] * len(pred_boxes)

    for pred_idx in range(iou_matrix.shape[0]):
        best_gt_idx = torch.argmax(iou_matrix[pred_idx])
        if iou_matrix[pred_idx, best_gt_idx] >= iou_threshold and best_gt_idx not in assigned_gts:
            matched_preds.append(pred_boxes[pred_idx])
            matched_gts.append(gt_boxes[best_gt_idx])
            assigned_gts.add(best_gt_idx)
            matched_indices[pred_idx] = best_gt_idx

    if len(matched_preds) == 0:
        return None, None, []

    return torch.stack(matched_preds), torch.stack(matched_gts), matched_indices
