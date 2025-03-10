import torch
from torch.amp import autocast
import os
from torchvision.ops import box_iou, generalized_box_iou_loss
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import numpy as np
import yaml
import matplotlib.pyplot as plt

def validation(run_dir, model, dataloader, device):
    metrics = EvaluationMetrics(run_dir=run_dir, model=model, dataloader=dataloader, device=device)
    results = metrics()
    return {key: np.float64(value) for key, value in results.items()}

def test(run_dir, iou_threshold=0.5, create_plots=True):
    """Evaluates object detection predictions against ground truth labels."""
    metrics = EvaluationMetrics(run_dir=run_dir, iou_threshold=iou_threshold, create_plots=create_plots)
    results = metrics()
    return {key: np.float64(value) for key, value in results.items()}
    

class EvaluationMetrics:
    def __init__(self, run_dir, model=None, dataloader=None, create_plots=False, device="cuda", iou_threshold=0.5):
        self.run_dir = run_dir
        self.model = model
        self.dataloader = dataloader
        self.create_plots = create_plots
        self.device = device
        self.iou_threshold = iou_threshold
        if isinstance(self.device, str):
            self.device = torch.device(device)
        self.metrics = {key: 0.0 for key in ['precision', 'recall', 'f1', 'mAP50', 'mAP50_95', 'loss']}

    def __call__(self):
        torch.cuda.empty_cache()
        if self.model is not None and self.dataloader is not None:
            use_dataloader = True
        else:
            use_dataloader = False
            dataset = self.run_dir.split(":")[-1]
            self.label_dir = os.path.join(self.run_dir, "predict", "labels")
            project_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.run_dir)))
            self.data_config_file = os.path.join(project_dir, "dataset_configs", dataset, "dataset.yaml")
        self.get_metrics(use_dataloader)
        return self.metrics

    def get_metrics(self, use_dataloader=True):

        self.precision, self.recall, self.f1 = 0, 0, 0
        self.mAP50, self.mAP50_95 = 0, 0
        self.loss = 0
        self.tp, self.fp, self.fn, self.tn = 0, 0, 0, 0
                
        # Calculate final metrics
        iou_thresholds = torch.linspace(0.5, 0.95, 10)
        if use_dataloader:
            self.compute_metrics_from_dataloader(self.model, self.dataloader, iou_thresholds)
        else:
            self.compute_metrics_from_files(self.label_dir, self.data_config_file, iou_thresholds)
        
        # Update metrics dictionary
        self.metrics.update({
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'mAP50': self.mAP50,
            'mAP50_95': self.mAP50_95,
            'loss': self.loss
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
                    predictions = model(images)
                    
                for i, filename in enumerate(filenames):
                    gt_labels = targets[i]['labels'].cpu().numpy()
                    gt_boxes = targets[i]['boxes'].cpu().numpy()
                    pred_labels = predictions[i]['labels'].cpu().numpy()
                    pred_boxes = predictions[i]['boxes'].cpu().numpy()
                    pred_scores = predictions[i]['scores'].cpu().numpy()

                    all_predictions.append([(int(label), box, score) for label, box, score in zip(pred_labels, pred_boxes, pred_scores)])
                    all_ground_truths.append([(int(label), box) for label, box in zip(gt_labels, gt_boxes)])

        nc = dataloader.dataset.get_num_classes()
        self.compute_metrics(all_predictions, all_ground_truths, iou_thresholds, nc)

    def compute_metrics_from_files(self, pred_dir, data_config_file, iou_thresholds=[0.5]):
        """
        Compute precision, recall, F1-score, mAP50, and mAP50_95 given prediction directory and data_config_file.
        """
        data_config = yaml.safe_load(open(data_config_file))
        nc = data_config["nc"]
        test_txt = os.path.join(data_config["path"], data_config["test"])
        image_paths = [line.strip() for line in open(test_txt, 'r').readlines() if line.strip()]
        gt_files = [p.replace("images", "labels").replace(".png", ".txt") for p in image_paths]

        all_predictions = []
        all_ground_truths = []

        for i, gt_file in enumerate(gt_files):
            pred_file = os.path.join(pred_dir, os.path.basename(gt_file))
            if not os.path.exists(pred_file):
                pred_labels = []
            else:
                with open(pred_file, 'r') as f:
                    pred_labels = [list(map(float, line.strip().split())) for line in f.readlines()]
            with open(gt_file, 'r') as f:
                gt_labels = [list(map(float, line.strip().split())) for line in f.readlines()]

            
            pred_labels = [(int(label), [x, y, w, h], score) for label, x, y, w, h, score in pred_labels]
            gt_labels = [(int(label), [x, y, w, h]) for label, x, y, w, h in gt_labels]
            
            all_predictions.append(pred_labels)
            all_ground_truths.append(gt_labels)
        
        if len(all_predictions) == 0:
            raise ValueError("No predictions found.")
        if len(all_ground_truths) == 0:
            raise ValueError("No ground truths found.")
        if len(all_predictions) != len(all_ground_truths):
            raise ValueError("Number of predictions and ground truths must be equal.")

        self.compute_metrics(all_predictions, all_ground_truths, iou_thresholds, nc)

    def compute_metrics(self, predictions, ground_truths, iou_thresholds=[0.5], num_classes=1):
        """
        Compute precision, recall, F1-score, mAP50, and mAP50_95 given predictions and ground truths.
        """
        ap_per_threshold = []
        
        progress_bar = tqdm(iou_thresholds, desc="    Evaluating:", leave=False)
        for iou_thresh in progress_bar:
            ap_per_class = []

            for class_id in range(num_classes):
                class_preds = [[p for p in preds if p[0] == class_id] for preds in predictions]
                class_gts = [[g for g in gts if g[0] == class_id] for gts in ground_truths]

                ap = self.compute_ap(class_preds, class_gts, iou_threshold=iou_thresh)
                ap_per_class.append(ap)

            ap_per_threshold.append(np.mean(ap_per_class))
        
        self.loss = self.bbox_loss(predictions, ground_truths).item()
        self.mAP50 = ap_per_threshold[0]
        self.mAP50_95 = np.mean(ap_per_threshold)

    def compute_ap(self, predictions, ground_truths, iou_threshold=0.5, create_plots=False):
        """
        Compute average precision (AP) given precision and recall values.
        """
        confidence_thresholds = np.linspace(0, 1, 101)
        
        precisions, recalls = [], []
        for conf_thresh in confidence_thresholds:
            precision, recall = self.compute_precision_recall(
                predictions, ground_truths, confidence_threshold=conf_thresh, iou_threshold=iou_threshold
            )
            precisions.append(precision)
            recalls.append(recall)
            if iou_threshold == self.iou_threshold and conf_thresh == 0.5:
                self.precision = precision
                self.recall = recall
                self.f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        precisions = np.array(precisions)
        recalls = np.array(recalls)

        if iou_threshold==self.iou_threshold and (create_plots or self.create_plots):
            self.create_result_figures(precisions, recalls, confidence_thresholds, iou_threshold=iou_threshold)
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        # Compute precision envelope
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
                
        # Compute AP as the area under the precision-recall curve
        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        
        return ap
    
    def compute_precision_recall(self, predictions, ground_truths, confidence_threshold=0.5, iou_threshold=0.5):
        """
        Compute precision and recall given predictions and ground truths.
        """
        tp, fp, fn, tn = 0, 0, 0, 0
        # filter predictions by confidence score
        predictions = [[p for p in preds if p[2] >= confidence_threshold] for preds in predictions]

        total_gt = sum([len(gts) for gts in ground_truths])
        total_preds = sum([len(preds) for preds in predictions])

        if total_gt == 0 and total_preds == 0:
            return 1, 1
        
        for i in range(len(predictions)):

            # Determine true positives, false positives, and false negatives
            tp_i, fp_i, fn_i, tn_i = 0, 0, 0, 0
            pred_boxes = [yolo_to_pascal(*p[1]) for p in predictions[i]]
            gt_boxes = [yolo_to_pascal(*g[1]) for g in ground_truths[i]]
            if len(gt_boxes) == 0:
                if len(pred_boxes) > 0:
                    fp_i += len(pred_boxes)
                else:
                    tn_i += 1
            elif len(pred_boxes) == 0:
                fn_i += len(gt_boxes)
            else:
                pred_boxes = torch.stack([torch.tensor(p_box) for p_box in pred_boxes])
                gt_boxes = torch.stack([torch.tensor(gt_box) for gt_box in gt_boxes])
                iou_matrix = box_iou(pred_boxes, gt_boxes)
                for j in range(iou_matrix.shape[0]):
                    if torch.max(iou_matrix[j]) >= iou_threshold:
                        tp_i += 1
                fp_i += len(pred_boxes) - tp_i
                fn_i += len(gt_boxes) - tp_i
            tp += tp_i
            fp += fp_i
            fn += fn_i
            tn += tn_i

        assert tp + fp == total_preds, f"TP + FP = {tp + fp} != {total_preds}"
        assert tp + fn == total_gt, f"TP + FN = {tp + fn} != {total_gt}"
        
        if confidence_threshold == 0.5 and iou_threshold == self.iou_threshold:
            self.tp = tp
            self.fp = fp
            self.fn = fn
            self.tn = tn
            
        precision = tp / (tp + fp) if tp + fp > 0 else 1
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        return precision, recall
    
    def bbox_loss(self, preds, gts):
        """
        Computes loss for bounding box predictions across multiple images and classes.

        Args:
            preds (list): List of lists containing tuples (label, bbox, score) for each prediction.
            gts (list): List of lists containing tuples (label, bbox) for each ground truth.

        Returns:
            torch.Tensor: Computed loss value.
        """
        batch_size = len(preds)
        total_loss = 0.0

        for i in range(batch_size):
            if len(preds[i]) == 0 and len(gts[i]) == 0:
                continue  # No loss when both are empty

            if len(gts[i]) == 0:
                total_loss += sum(score for _, _, score in preds[i]) / len(preds[i])  # Penalize false positives
                continue

            if len(preds[i]) == 0:
                total_loss += 1.0  # Penalize missed detections
                continue

            # Convert lists of tuples into tensors
            pl, pb, ps = zip(*preds[i]) if preds[i] else ([], [], [])
            gl, gb = zip(*gts[i]) if gts[i] else ([], [])

            pb = torch.tensor(pb, dtype=torch.float32) if pb else torch.empty((0, 4))
            ps = torch.tensor(ps, dtype=torch.float32) if ps else torch.empty((0,))
            pl = torch.tensor(pl, dtype=torch.int64) if pl else torch.empty((0,), dtype=torch.int64)

            gb = torch.tensor(gb, dtype=torch.float32) if gb else torch.empty((0, 4))
            gl = torch.tensor(gl, dtype=torch.int64) if gl else torch.empty((0,), dtype=torch.int64)

            loss_per_class = 0.0
            unique_classes = torch.cat((pl, gl)).unique()

            for cls in unique_classes:
                pb_cls = pb[pl == cls] if pb.numel() > 0 else torch.empty((0, 4))
                ps_cls = ps[pl == cls] if ps.numel() > 0 else torch.empty((0,))
                gb_cls = gb[gl == cls] if gb.numel() > 0 else torch.empty((0, 4))

                if gb_cls.shape[0] == 0:
                    loss_per_class += ps_cls.mean() if ps_cls.numel() > 0 else 0.0  # Penalize false positives
                    continue

                if pb_cls.shape[0] == 0:
                    loss_per_class += 1.0  # Penalize missed detections
                    continue

                # Compute IoU between each predicted box and each ground truth box
                ious = box_iou(pb_cls, gb_cls)

                # Match predictions to ground truth using IoU
                max_ious, _ = ious.max(dim=1)

                # Localization loss: (1 - IoU) for matched pairs
                loc_loss = 1 - max_ious

                # Confidence loss: penalize false positives based on IoU
                conf_loss = (1 - max_ious) * ps_cls

                loss_per_class += loc_loss.mean() + conf_loss.mean()

            total_loss += loss_per_class

        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)
    
    def create_result_figures(self, precisions, recalls, confidence_thresholds, iou_threshold=0.5):
        """
        Create precision-recall curve given predictions and ground truths.
        """
        # Precision-curve
        p_curve_path = os.path.join(self.run_dir, 'P_curve.png')
        if not os.path.exists(p_curve_path):
            plt.figure()
            plt.plot(confidence_thresholds, precisions, label='Precision')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Precision')
            plt.title(f'Precision-Confidence Curve (IoU={iou_threshold:.2f})')
            # plt.legend()
            plt.savefig(p_curve_path)

        # Recall-curve
        r_curve_path = os.path.join(self.run_dir, 'R_curve.png')
        if not os.path.exists(r_curve_path):
            plt.figure()
            plt.plot(confidence_thresholds, recalls, label='Recall')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('Recall')
            plt.title(f'Recall-Confidence Curve (IoU={iou_threshold:.2f})')
            # plt.legend()
            plt.savefig(r_curve_path)

        # Precision-Recall curve
        pr_curve_path = os.path.join(self.run_dir, 'PR_curve.png')
        if not os.path.exists(pr_curve_path):
            plt.figure()
            plt.plot(recalls, precisions, label='Precision-Recall')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve (IoU={iou_threshold:.2f})')
            # plt.legend()
            plt.savefig(pr_curve_path)

        # F1-Confidence curve
        f1_curve_path = os.path.join(self.run_dir, 'F1_curve.png')
        if not os.path.exists(f1_curve_path):
            f1s = 2 * precisions * recalls / (precisions + recalls)
            plt.figure()
            plt.plot(confidence_thresholds, f1s, label='F1')
            plt.xlabel('Confidence Threshold')
            plt.ylabel('F1')
            plt.title(f'F1-Confidence Curve (IoU={iou_threshold:.2f})')
            # plt.legend()
            plt.savefig(f1_curve_path)
        
        # Confusion matrix
        conf_matrix_path = os.path.join(self.run_dir, 'confusion_matrix.png')
        if self.tp + self.fp + self.fn + self.tn > 0 and not os.path.exists(conf_matrix_path):
            plt.figure()
            plt.imshow([[self.tp, self.fp], [self.fn, self.tn]], cmap='Blues', interpolation='nearest')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.xticks([0, 1], ['Positive', 'Negative'])
            plt.yticks([0, 1], ['Positive', 'Negative'])
            plt.text(0, 0, f'TP: {self.tp}', ha='center', va='center', color='white')
            plt.text(1, 0, f'FP: {self.fp}', ha='center', va='center', color='black')
            plt.text(0, 1, f'FN: {self.fn}', ha='center', va='center', color='black')
            plt.text(1, 1, f'TN: {self.tn}', ha='center', va='center', color='white')
            plt.colorbar()
            plt.savefig(conf_matrix_path)
        
        # Epoch results
        epoch_results = os.path.join(self.run_dir, 'results.csv')
        epoch_results_path = os.path.join(self.run_dir, 'results.png')
        if not os.path.exists(epoch_results_path) and os.path.exists(epoch_results):
            plt.figure()
            header = list(np.genfromtxt(epoch_results, delimiter=',', max_rows=1, dtype=str))
            results = np.genfromtxt(epoch_results, delimiter=',', skip_header=1)
            epochs = results[:, 0]
            f, ax = plt.subplots(2, 3, figsize=(20, 10))
            ax[0, 0].plot(epochs, results[:, header.index('train_loss')])
            ax[0, 0].set_title('Train Loss')
            ax[1, 0].plot(epochs, results[:, header.index('val_loss')])
            ax[1, 0].set_title('Validation Loss')
            ax[0, 1].plot(epochs, results[:, header.index('precision')])
            ax[0, 1].set_title('Precision')
            ax[0, 2].plot(epochs, results[:, header.index('recall')])
            ax[0, 2].set_title('Recall')
            ax[1, 1].plot(epochs, results[:, header.index('mAP50')])
            ax[1, 1].set_title('mAP50')
            ax[1, 2].plot(epochs, results[:, header.index('mAP50_95')])
            ax[1, 2].set_title('mAP50_95')
            plt.savefig(epoch_results_path)
            

    

def pascal_to_yolo(xmin, ymin, xmax, ymax, img_w=1, img_h=1):
    """Convert Pascal VOC format (absolute pixel values) to YOLO format (fractions)."""
    x = (xmin + xmax) / (2 * img_w)
    y = (ymin + ymax) / (2 * img_h)
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return x, y, w, h

def yolo_to_pascal(x, y, w, h, img_w=1, img_h=1):
    """Convert YOLO format (fractions) to Pascal VOC format (absolute pixel values)."""
    xmin = (x - w / 2) * img_w
    ymin = (y - h / 2) * img_h
    xmax = (x + w / 2) * img_w
    ymax = (y + h / 2) * img_h
    return xmin, ymin, xmax, ymax

