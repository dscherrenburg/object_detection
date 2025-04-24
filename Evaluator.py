import os
import sys
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from metrics.BoundingBox import BoundingBox
from metrics.BoundingBoxes import BoundingBoxes
from metrics.utils import BBFormat, BBType, CoordinatesType

from Faster_RCNN.predict import FasterRCNNPredictor
from YOLO.predict import YOLOPredictor


def validate(model, dataloader, conf, iou, preprocess, postprocess):
    model.eval()
    evaluator = Evaluator()
    progress_bar = tqdm(dataloader, desc="   Validation", dynamic_ncols=True, leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        images, targets = preprocess(batch)
        with torch.no_grad(), torch.amp.autocast(images.device.type):
            predictions = model(images)
            predictions, targets = postprocess(predictions, targets)
        evaluator.add(predictions, targets)
    metrics = evaluator.validate(conf=conf, iou=iou)
    return metrics

def evaluate(run_dir, conf=0.5, iou=0.5, save_path=None, show_plots=False):
    evaluator = Evaluator(conf_threshold=conf, save_path=save_path, show_plots=show_plots)

    preds_dir = os.path.join(run_dir, "predict", "labels")
    if not os.path.exists(preds_dir):
        raise FileNotFoundError(f"Predictions directory not found: {preds_dir}")
    
    data_name = run_dir.split(":")[-1].split("_")[0]
    project_folder = os.path.dirname(os.path.dirname(os.path.dirname(run_dir)))
    data_config_folder = os.path.join(project_folder, "dataset_configs")
    test_txt = os.path.join(data_config_folder, data_name, "test.txt")
    if not os.path.exists(test_txt):
        raise FileNotFoundError(f"Test file not found: {test_txt}")
    
    with open(test_txt) as f:
        gt_images = [line.strip() for line in f if line.strip()]
        gt_label_files = [p.replace("images", "labels").replace(".png", ".txt") for p in gt_images]
        image_names = [os.path.basename(p).split('.')[0] for p in gt_label_files]

    for image_name, gt_label_file in zip(image_names, gt_label_files):
        if not os.path.exists(gt_label_file):
            raise FileNotFoundError(f"Ground truth label file not found: {gt_label_file}")
        with open(gt_label_file) as f:
            gts = []
            for line in f:
                if line.strip():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    gts.append((int(class_id), x, y, w, h))
        preds = []
        pred_label_file = os.path.join(preds_dir, image_name + ".txt")
        if os.path.exists(pred_label_file):
            with open(pred_label_file) as f:
                for line in f:
                    if line.strip():
                        class_id, x, y, w, h, conf = map(float, line.strip().split())
                        preds.append((int(class_id), x, y, w, h, conf))
        else:
            print(f"Prediction file not found: {pred_label_file}")
            continue

        evaluator.add_batch([preds], [gts], [image_name])

    metrics = evaluator.evaluate(iou=iou, conf=conf, plot=save_path is not None)
    return metrics




class Evaluator:
    def __init__(self, 
                 boundingboxes: BoundingBoxes = None,
                 iou_thresholds: List[float] = None, 
                 conf_threshold: float = 0.0,
                 imgsz: tuple = (640, 360),
                 save_path: str = None,
                 show_plots: bool = False):
        self.boundingboxes = boundingboxes or BoundingBoxes()
        self.iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05).tolist()
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.save_path = save_path
        self.show_plots = show_plots
        self.i = 0

    def load(self, boundingboxes):
        """
        Loads the bounding boxes into the evaluator.
        Args:
            boundingboxes: List of bounding boxes.
        """
        self.boundingboxes = boundingboxes
    
    def add(self, predictions, targets):
        """
        Adds a list of predictions and targets to the evaluator.
        Args:
            predictions: List of prediction dictionaries with keys 'boxes', 'labels', and 'scores'.
            targe
            targets: List of targets dictionaries with keys 'boxes' and 'labels'.
        """
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            target_boxes = target['boxes']
            target_labels = target['labels']

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                bb = BoundingBox(self.i, label, *box, CoordinatesType.Absolute, self.imgsz, BBType.Detected, score, format=BBFormat.XYX2Y2)
                self.boundingboxes.addBoundingBox(bb)

            for box, label in zip(target_boxes, target_labels):
                bb = BoundingBox(self.i, label, *box, CoordinatesType.Absolute, self.imgsz, BBType.GroundTruth, format=BBFormat.XYX2Y2)
                self.boundingboxes.addBoundingBox(bb)
            self.i += 1

    def add_batch(self, predictions: List[List[Tuple[int, float, float, float, float, float]]],
                  ground_truths: List[List[Tuple[int, float, float, float, float]]], 
                  image_names: List):
        """
        Adds a batch of predictions and ground truths.

        Each prediction is (id, x1, y1, x2, y2, confidence).
        Each ground truth is (id, x1, y1, x2, y2).
        """
        for preds, gts, image_name in zip(predictions, ground_truths, image_names):
            for pred in preds:
                class_id, x, y, w, h, conf = pred
                bb = BoundingBox(image_name,class_id,x,y,w,h,CoordinatesType.Relative, self.imgsz, BBType.Detected, conf, format=BBFormat.XYWH)
                self.boundingboxes.addBoundingBox(bb)
            for gt in gts:
                class_id, x, y, w, h = gt
                bb = BoundingBox(image_name,class_id,x,y,w,h,CoordinatesType.Relative, self.imgsz, BBType.GroundTruth, format=BBFormat.XYWH)
                self.boundingboxes.addBoundingBox(bb)

    def reset(self):
        """
        Resets the evaluator.
        """
        self.boundingboxes = BoundingBoxes()
    
    def validate(self, iou=0.5, conf=0.25):
        """
        Validates the model.
        """
        return self.get_metrics(iou, conf)
    
    def evaluate(self, iou=0.5, conf=0.25, plot=False):
        """
        Evaluates the model.
        Args:
            iou: IoU threshold.
            conf: Confidence threshold.
            plot: If True, plots the metrics.
        Returns:
            A dictionary containing the metrics.
        """
        metrics = self.get_metrics(iou, conf)

        if plot:
            self.create_plots(iou, conf)

        return metrics
    
    def get_metrics(self, iou=0.5, conf=0.25):
        """
        Computes the metrics for the given bounding boxes, IoU and confidence values.
        """
        if not self.boundingboxes or len(self.boundingboxes.getBoundingBoxes()) == 0:
            raise ValueError("Bounding boxes are not loaded.")
        results = {}
        # Precision, Recall, F1 and mAP
        r = self.evaluate_at_iou(iou, conf)
        results.update(r)
        del results['mAP']

        # mAP@50
        r = self.evaluate_at_iou(0.5, conf)
        results["mAP50"] = r['mAP']

        # mAP@50-95
        maps = []
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        for iou in iou_thresholds:
            r = self.evaluate_at_iou(iou, conf)
            maps.append(r['mAP'])
        results['mAP50-95'] = np.mean(maps)
        return results
        

    def get_tp_fp(self, iou_thresh):
        """
        Computes the True Positives (TP) and False Positives (FP) for each class.
        """
        tps_fps = {}
        groundTruths = []
        detections = []
        classes = []
        matched_ious = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in self.boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(), 1,
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
                ])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c, use filename as key
            gts = {}
            npos = 0
            for g in groundTruths:
                if g[1] == c:
                    npos += 1
                    gts[g[0]] = gts.get(g[0], []) + [g]

            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            confidences = [d[2] for d in dects]
            # create dictionary with amount of gts for each image
            det = {key: np.zeros(len(gts[key])) for key in gts}

            # Loop through detections
            for d in range(len(dects)):
                # Find ground truth image
                gt = gts[dects[d][0]] if dects[d][0] in gts else []
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou > iouMax:
                        iouMax = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iouMax >= iou_thresh:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        matched_ious.append(iouMax)
                    else:
                        FP[d] = 1  # count as false positive
                # - A detected "cat" is overlaped with a GT "cat" with IoU >= IoUThreshold.
                else:
                    FP[d] = 1  # count as false positive
            
            tps_fps[c] = (TP, FP, confidences, npos, matched_ious)
            return tps_fps
    
    def evaluate_at_iou(self, iou=0.5, conf=0.25):
        """
        Computes the metrics for the given bounding boxes, IoU and confidence values.
        Args:
            iou: IoU threshold.
            conf: Confidence threshold.
        Returns:
            A list of dictionaries containing the metrics for each class.
        """
        aps = []
        recalls = []
        precisions = []
        f1s = []
        ious = []

        tps_fps = self.get_tp_fp(iou)
        for c, (TP, FP, confidences, npos, matched_ious) in tps_fps.items():
            # average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            aps.append(ap)

            # Filter TP and FP based on confidence
            confidences = np.array(confidences)
            TP = TP[confidences >= conf]
            FP = FP[confidences >= conf]
            recall = np.sum(TP) / (npos + 1e-20)
            precision = np.sum(TP) / (np.sum(TP) + np.sum(FP) + 1e-20)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            ious.append(np.mean(matched_ious)) if matched_ious else ious.append(0.0)
        
        # Calculate mean average precision
        mAP = np.mean(aps) if aps else 0.0
        mRecall = np.mean(recalls) if recalls else 0.0
        mPrecision = np.mean(precisions) if precisions else 0.0
        mF1 = np.mean(f1s) if f1s else 0.0
        mIoU = np.mean(ious) if ious else 0.0
        results = {
            'Recall': mRecall,
            'Precision': mPrecision,
            'F1': mF1,
            'IoU': mIoU,
            'mAP': mAP,
        }
        return results
    
    def create_plots(self, iou=0.5, conf=0.25):
        """
        Creates the plots for the given bounding boxes, IoU and confidence values.
        Args:
            iou: IoU threshold.
            conf: Confidence threshold.
        """
        data = {}
        tps_fps = self.get_tp_fp(iou)
        assert len(tps_fps) > 0, "No detections found."
        assert len(tps_fps) == 1, "Multiple classes found, this requires the code to be modified."
        for c, (TP, FP, confidences, npos, ious) in tps_fps.items():
            # average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            f1 = 2 * (prec * rec) / (prec + rec + 1e-20)

            data[c] = {
                'recall': rec,
                'precision': prec,
                'f1': f1,
                'confidences': confidences
            }
        
        # Plot Precision-Recall curve
        for c, d in data.items():
            self.PlotMetrics(d['recall'], d['precision'], x_label='Recall', y_label='Precision', iou=iou)
            self.PlotMetrics(d['confidences'], d['precision'], x_label='Confidence', y_label='Precision', iou=iou)
            self.PlotMetrics(d['confidences'], d['recall'], x_label='Confidence', y_label='Recall', iou=iou)
            self.PlotMetrics(d['confidences'], d['f1'], x_label='Confidence', y_label='F1', iou=iou)


    def PlotMetrics(self, x_metric, y_metric, x_label='X', y_label='Y', iou=None,
                    title=None, savePath='default', showGraphic='default'):
        """
        PlotMetrics
        Plot a graph for two given metrics.
        Args:
            x_metric: Values for the X-axis.
            y_metric: Values for the Y-axis.
            x_label (optional): Label for the X-axis (default = 'X').
            y_label (optional): Label for the Y-axis (default = 'Y').
            title (optional): Title of the plot (default = 'Metrics Plot').
            savePath (optional): If informed, the plot will be saved as an image in this path.
            showGraphic (optional): If True, the plot will be shown (default = True).
        """
        if savePath == 'default':
            savePath = self.save_path
        if showGraphic == 'default':
            showGraphic = self.show_plots

        name = f'{y_label}-{x_label}' if x_label.lower() != 'confidence' else f'{y_label}'
        if title is None:
            title = name + ' Curve'
        if iou is not None:
            title += f'  (IoU={iou:.2f})'

        plt.plot(x_metric, y_metric)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid()
        if savePath is not None:
            if not os.path.exists(os.path.dirname(savePath)):
                os.makedirs(os.path.dirname(savePath))
            name += f'_curve({iou:.2f})' if iou is not None else '_curve'
            plt.savefig(os.path.join(savePath, name + '.png'), dpi=300)

        if showGraphic is True:
            plt.show()
            plt.waitforbuttonpress()
        plt.close()

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1+i] != mrec[i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    # For each detections, calculate IoU with reference
    @staticmethod
    def _getAllIoUs(reference, detections):
        ret = []
        bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference, bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou, reference, d))  # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)  # sort by iou (from highest to lowest)

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)