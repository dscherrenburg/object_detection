import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from typing import List, Dict, Tuple, Optional
import os

# from podm.metrics import box_iou, precision_recall_curve, average_precision_score, BoundingBox
# from podm.metrics import box_iou, precision_recall_curve, average_precision_score, BoundingBox



# import sys
# import os

# # Adjust the path to where you cloned the repo
# repo_lib_path = os.path.abspath("ODMetrics/lib")
# sys.path.append(repo_lib_path)

from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from Evaluator import Evaluator
from metrics.utils import BBFormat, BBType, CoordinatesType


def image_path_to_label_path(image_path: str) -> str:
    return image_path.replace("/images/", "/labels/").replace(".png", ".txt")


def get_gt_and_pred_paths(gt_file: str, pred_dir: str) -> Tuple[List[str], List[str]]:
    """
    Given a directory of predictions and a ground truth file, returns lists of prediction and ground truth paths.
    Assumes the predictions are in the format: {pred_dir}/{image_name}.txt
    """
    pred_paths = []
    gt_paths = []

    with open(gt_file, 'r') as f:
        gt_lines = f.readlines()

    for line in gt_lines:
        line = line.strip()
        image_name = os.path.basename(line.split('.')[0])
        pred_path = os.path.join(pred_dir, f"{image_name}.txt")
        gt_path = image_path_to_label_path(line)
        if os.path.exists(pred_path) and os.path.exists(gt_path):
            pred_paths.append(pred_path)
            gt_paths.append(gt_path)
        else:
            print(f"Warning: Prediction or ground truth file does not exist for {image_name}")
    return gt_paths, pred_paths


def yolo_to_xyxy(xc, yc, w, h, img_w=1.0, img_h=1.0):
    """Converts YOLO (x_center, y_center, w, h) to (x1, y1, x2, y2)."""
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return x1, y1, x2, y2


def load_yolo_file(path, with_conf=False):
    """Loads a YOLO-format label file (optionally with confidence)."""
    boxes = []
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            boxes.append(parts)
            # if len(parts) < 5:
            #     continue
            # _, xc, yc, w, h = parts[:5]
            # conf = parts[5] if with_conf and len(parts) == 6 else 1.0
            # box = yolo_to_xyxy(xc, yc, w, h)
            # if with_conf:
            #     boxes.append((*box, conf))
            # else:
            #     boxes.append(box)
    return boxes


def box_iou(box1, box2):
    """Compute IoU between two boxes (xyxy)."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_ap(recall, precision):
    """Compute Average Precision (AP) from recall and precision curve."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])


class ObjectDetectionEvaluator:
    def __init__(self, iou_thresholds: List[float] = None, conf_threshold: float = 0.0):
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05).tolist()
        self.conf_threshold = conf_threshold
        self.all_boxes = BoundingBoxes()
        self.evaluator = Evaluator(show_plots=True)


    def get_metrics(self, iou_thresh: Optional[float] = 0.5, conf_thresh: Optional[float] = 0.25) -> Dict[str, float]:
        self.evaluator.load(self.all_boxes)
        metrics = self.evaluator.evaluate(self.all_boxes, iou_thresh, conf_thresh)
        return metrics
    
    def plot_pr_curve(self, class_id=0, iou_thresh: float = 0.5):
        print(iou_thresh)
        self.evaluator.PlotPrecisionRecallCurve(self.all_boxes, # Object containing all bounding boxes (ground truths and detections)
                                   IOUThreshold=iou_thresh, # IOU threshold
                                   showAP=True, # Show Average Precision in the title of the plot
                                   showInterpolatedPrecision=True) # Don't plot the interpolated precision curve

    
    def _precision_recall_score(self, y_true, y_scores, conf_thresh):
        """Compute precision and recall scores."""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_scores = np.where(y_scores >= conf_thresh, 1, 0)

        tp = np.sum((y_true == 1) & (y_scores == 1))
        fp = np.sum((y_true == 0) & (y_scores == 1))
        fn = np.sum((y_true == 1) & (y_scores == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return precision, recall

    def compute(self, iou_thresh=0.5, conf_thresh=0.25) -> Dict[str, float]:
        """Returns dictionary with mAP and AP50."""
        aps = []
        for iou in self.iou_thresholds:
            y_true, y_scores = self._evaluate_at_iou(iou)
            ap = average_precision_score(y_true, y_scores)
            aps.append(ap)
        
        # Compute precision and recall at the specified IoU threshold
        y_true, y_scores = self._evaluate_at_iou(iou_thresh)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        

        ap, prec, rec, total_gt = self._evaluate_at_iou(iou_thresh)
        ap50, prec50, rec50, _ = self._evaluate_at_iou(0.5)
        return {
            "Precision": np.mean(prec),
            "Recall": np.mean(rec),
            "mAP50": ap50,
            "mAP50-95": float(np.mean(aps)),
            "num_images": len(self.matches),
            "num_gt_total": total_gt
        }


from tqdm import tqdm

def evaluate_yolo_files(gt_files: List[str], pred_files: List[str], conf_thresh=0.25, iou_thresh=0.5):
    assert len(gt_files) == len(pred_files), "GT and prediction lists must be equal length"
    
    evaluator = ObjectDetectionEvaluator(conf_threshold=conf_thresh)
    tot_preds = 0

    for gt_path, pred_path in zip(gt_files, pred_files):
        image_name = os.path.basename(gt_path).split('.')[0]
        gt_boxes = load_yolo_file(gt_path, with_conf=False)
        pred_boxes = load_yolo_file(pred_path, with_conf=True)
        tot_preds += len(pred_boxes)
        evaluator.add_batch([pred_boxes], [gt_boxes], [image_name])  # wrap in list for single image
    metrics = evaluator.get_metrics(iou_thresh=iou_thresh, conf_thresh=conf_thresh)
    # evaluator.plot_pr_curve(iou_thresh=iou_thresh)
    return metrics



def main(split_nr: int = 1):
    data_name = f"split-{split_nr}(3)+interval-5+distance-(0-200)"
    model_name = "FRCNN"
    # model_name = "YOLO11n"
    project_folder = "/home/daan/object_detection/"
    run_name = f"m:{model_name}_e:100_b:8_d:{data_name}"
    confidence = 0.7
    iou = 0.5


    model_dir = "YOLO" if model_name.startswith("YOLO") else "Faster_RCNN"
    gt_file = os.path.join(project_folder, "dataset_configs", data_name, "test.txt")
    pred_dir = os.path.join(project_folder, model_dir, "runs", run_name, "predict", "labels")

    gt_files, pred_files = get_gt_and_pred_paths(gt_file, pred_dir)

    results = evaluate_yolo_files(gt_files, pred_files, conf_thresh=confidence, iou_thresh=iou)
    print("Results:")
    print(results)
    return


if __name__ == "__main__":
    main(1)