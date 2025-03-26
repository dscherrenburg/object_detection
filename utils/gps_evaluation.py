import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from sklearn.metrics.pairwise import haversine_distances

import pandas as pd

class GPS_Evaluation:
    def __init__(self, data_folder, prediction_folder, rosbags, dock_coordinates):
        """
        Initializes the GPS_Evaluation class.
        
        Parameters:
        - data_folder: Folder containing images, labels and gps_coords folders.
        - prediction_folder: Folder containing the predictions.
        - rosbags: List of rosbag names to evaluate.
        - dock_coordinates: Tuple of (latitude, longitude) representing the dock's location.
        """
        self.data_folder = data_folder
        self.prediction_folder = prediction_folder
        self.rosbags = rosbags if isinstance(rosbags, list) else [rosbags]
        self.dock_coordinates = dock_coordinates
    
    def extract_data(self):
        self.predictions = []
        self.labels = []
        self.gps_coords = []

        # Extract labels
        for rosbag in self.rosbags:
            labels_folder = os.path.join(self.data_folder, 'labels', rosbag)
            for file in os.listdir(labels_folder):
                name_split = file.split('.')
                if name_split[-1] == 'txt' and name_split[0].isdigit():
                    timestamp = int(name_split[0])
                    with open(os.path.join(labels_folder, file), 'r') as f:
                        labels = []
                        for line in f.readlines():
                            line = line.strip().split()
                            labels.append((int(line[0]), list(map(float, line[1:5]))))
                        self.labels.append((timestamp, labels))
        self.labels = sorted(self.labels, key=lambda x: x[0])
        print(f"Extracted {len(self.labels)} labels.")

        # Extract predictions
        timestamps = [label[0] for label in self.labels]
        for file in os.listdir(self.prediction_folder):
            name_split = file.split('.')
            if name_split[-1] == 'txt' and name_split[0].isdigit():
                timestamp = int(name_split[0])
                # print(timestamp)
                if timestamp not in timestamps:
                    continue
                with open(os.path.join(self.prediction_folder, file), 'r') as f:
                    preds = []
                    for line in f.readlines():
                        line = line.strip().split()
                        preds.append((int(line[0]), list(map(float, line[1:5])), float(line[5])))
                    self.predictions.append((timestamp, preds))
        # Add empty predictions for missing timestamps
        pred_timestamps = [p[0] for p in self.predictions] 
        for timestamp in timestamps:
            if timestamp not in pred_timestamps:
                self.predictions.append((timestamp, []))
        self.predictions = sorted(self.predictions, key=lambda x: x[0])
        print(f"Extracted {len(self.predictions)} predictions.")
        
        # Extract GPS coordinates
        for rosbag in self.rosbags:
            with open(os.path.join(self.data_folder, 'gps_coords', f"{rosbag}.txt"), 'r') as f:
                for line in f.readlines():
                    line = line.strip().split()
                    self.gps_coords.append((int(line[0]), float(line[1]), float(line[2])))
        self.gps_coords = sorted(self.gps_coords, key=lambda x: x[0])
        print(f"Extracted {len(self.gps_coords)} GPS coordinates.")
        
        if len(self.predictions) == 0:
            raise ValueError("No predictions found in the prediction folder.")
        if len(self.labels) == 0:
            raise ValueError("No labels found in the labels folder.")
        if len(self.gps_coords) == 0:
            raise ValueError("No GPS coordinates found in the gps_coords folder.")
        if len(self.predictions) != len(self.labels) or len(self.predictions) != len(self.gps_coords):
            raise ValueError("Number of predictions, labels and GPS coordinates do not match.")
        if self.predictions[0][0] != self.labels[0][0] or self.predictions[0][0] != self.gps_coords[0][0]:
            print(self.predictions[0][0], self.labels[0][0], self.gps_coords[0][0])
            raise ValueError("Timestamps of predictions, labels and GPS coordinates do not match.")
        print(f"Extracted {len(self.predictions)} predictions, labels and GPS coordinates.")

    def calculate_distances(self):
        dock_lat, dock_lon = self.dock_coordinates
        self.distances = []
        for stamp, lat, lon in self.gps_coords:
            # Simple Haversine distance calculation (in meters)
            lat1, lon1 = np.radians(dock_lat), np.radians(dock_lon)
            lat2, lon2 = np.radians(lat), np.radians(lon)
            distance = haversine_distances([[lat1, lon1], [lat2, lon2]])[0][1] * 6371000
            self.distances.append((stamp, distance))
        self.distances = np.array(self.distances)
        print(f"Calculated {len(self.distances)} distances from dock location.")

    def iou(self, box1, box2):
        """Calculate the Intersection over Union (IoU) of two bounding boxes.
            box: (x_center, y_center, width, height)
        """
        if len(box1) != 4 or len(box2) != 4:
            return 0
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1, y1, x1b, y1b = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        x2, y2, x2b, y2b = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

        # Get the coordinates of the intersection rectangle
        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1b, x2b)
        yB = min(y1b, y2b)

        # Calculate intersection area
        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Calculate union area
        box1_area = (w1 + 1) * (h1 + 1)
        box2_area = (w2 + 1) * (h2 + 1)
        union = box1_area + box2_area - intersection

        # Calculate the Intersection over Union (IoU)
        iou = intersection / union if union > 0 else 0
        return iou
    
    def compute_ap(self, precisions, recalls):
        """Compute the average precision, given the precision and recall values."""
        recalls = np.concatenate([[0], np.array(recalls), [1]])
        precisions = np.concatenate([[0], np.array(precisions), [0]])
        # Ensure precision is non-decreasing
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i-1] = max(precisions[i-1], precisions[i])
        # Compute ap as the area under PR curve
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap

    def evaluate_image(self, predictions, ground_truths, iou_threshold=0.5, conf_threshold=None):
        """Compute AP for a single image, ensuring class matches before IoU comparison."""
        if conf_threshold is not None:
            predictions = [pred for pred in predictions if pred[2] >= conf_threshold]
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)  # Sort by confidence
        tps, fps = [], []
        assigned = set()  # Track assigned ground-truth boxes

        for pred_class, pred_box, conf in predictions:
            matched_gt = [
                (i, self.iou(pred_box, gt_box)) for i, (gt_class, gt_box) in enumerate(ground_truths) 
                if gt_class == pred_class]
            if not matched_gt:
                fps.append(1)  # No matching ground truth found
                tps.append(0)
                continue

            best_idx, best_iou = max(matched_gt, key=lambda x: x[1])  # Get best match based on IoU

            if best_iou >= iou_threshold and best_idx not in assigned:
                tps.append(1)
                fps.append(0)
                assigned.add(best_idx)
            else:
                tps.append(0)
                fps.append(1)

        if conf_threshold is not None:
            recall = sum(tps) / len(ground_truths) if ground_truths else 0
            precision = sum(tps) / len(predictions) if predictions else 0
            return precision, recall
        else:
            tps = np.cumsum(tps)
            fps = np.cumsum(fps)
            recalls = tps / len(ground_truths) if ground_truths else np.array([])
            precisions = tps / (tps + fps) if (tps + fps).any() else np.array([])
            return precisions, recalls

    def compute_aps(self, iou_threshold=0.5):
        self.aps = []
        for i in range(len(self.predictions)):
            p_stamp, predictions = self.predictions[i]
            l_stamp, labels = self.labels[i]
            if p_stamp != l_stamp:
                raise ValueError("Timestamps of predictions and labels do not match.")
            precisions, recalls = self.evaluate_image(predictions, labels, iou_threshold)
            ap = self.compute_ap(precisions, recalls)
            self.aps.append((p_stamp, ap))
        print(f"Computed {len(self.aps)} AP values.")

    def plot_metrics_vs_distance(self, num_bins=20, iou_threshold=0.5, confidence_thresholds=[0.5], metrics=["precision", "recall", "f1"]):
        """Create plots of several metrics vs distances from the dock."""

        plots = {}
        for metric in metrics:
            plots[metric] = plt.figure(f"{metric.capitalize()} vs Distance from Dock (IoU: {iou_threshold})", figsize=(10, 6))
            plt.xlabel("Distance from Dock (m)")
            plt.ylabel(metric.capitalize())
            plt.title(f"{metric.capitalize()} vs Distance from Dock (IoU Threshold: {iou_threshold})")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        distances = self.distances[:, 1]


        for confidence_threshold in confidence_thresholds:
            scores = {"precision": [], "recall": [], "f1": []}
            for i in range(len(self.predictions)):
                p_stamp, predictions = self.predictions[i]
                l_stamp, labels = self.labels[i]
                if p_stamp != l_stamp:
                    raise ValueError("Timestamps of predictions and labels do not match.")
                precision, recall = self.evaluate_image(predictions, labels, iou_threshold, confidence_threshold)
                if 'precision' in metrics:
                    scores['precision'].append(precision)
                if 'recall' in metrics:
                    scores["recall"].append(recall)
                if 'f1' in metrics:
                    scores["f1"].append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0)

            for metric in metrics:
                # Compute binned statistics
                values = np.array(scores[metric])
                bin_means, bin_edges, _ = binned_statistic(distances, values, statistic='mean', bins=num_bins)
                bin_stds, _, _ = binned_statistic(distances, values, statistic='std', bins=num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                plot = plots[metric]
                plt.figure(plot.number)
                plt.plot(bin_centers, bin_means, label=f'Confidence {confidence_threshold}', linewidth=2)
                plt.fill_between(bin_centers, bin_means - bin_stds, bin_means + bin_stds, alpha=0.2)
                plt.legend()
                
        for metric in metrics:
            plot = plots[metric]
            plt.show()


    def plot_ap_vs_distance(self, num_bins=20, iou_threshold=0.5):
        """Plot the a score vs distance from the dock."""
        distances = pd.DataFrame(self.distances, columns=["timestamp", "distance"])
        aps = pd.DataFrame(self.aps, columns=["timestamp", "ap"])

        # Merge the two DataFrames based on timestamp
        merged_df = pd.merge(distances, aps, on="timestamp", how="inner")
        distances = merged_df['distance'].to_numpy()
        scores = merged_df['ap'].to_numpy()

        # Compute binned statistics for means and standard deviations
        bin_means, bin_edges, _ = binned_statistic(distances, scores, statistic='mean', bins=num_bins)
        bin_stds, _, _ = binned_statistic(distances, scores, statistic='std', bins=num_bins)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize confidence values for color mapping
        norm_scores = (bin_means - np.min(bin_means)) / (np.max(bin_means) - np.min(bin_means) + 1e-6)
        
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bin_centers, bin_means, width=(bin_edges[1] - bin_edges[0]) * 0.9, 
                    color=plt.cm.viridis(norm_scores), edgecolor="black", yerr=bin_stds, capsize=5)

        # Labels and title
        plt.xlabel("Distance from Dock (m)")
        plt.ylabel("Average Precision")
        plt.title(f"Average Precision vs Distance from Dock (IoU Threshold: {iou_threshold})")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def plot_gps_vs_dock(self):
        gps_df = pd.DataFrame(self.gps_coords, columns=["timestamp", "latitude", "longitude"])
        gps_df['ap'] = [a[1] for a in self.aps]  # Assuming AP list is ordered by timestamp

        # Convert latitude and longitude to meters relative to the dock location
        dock_lat, dock_lon = self.dock_coordinates
        earth_radius = 6371000  # Earth radius in meters
        def lat_lon_to_meters(lat, lon):
            """Convert latitude and longitude to meters relative to the dock location."""
            dlat = np.radians(lat - dock_lat)
            dlon = np.radians(lon - dock_lon)
            x = dlon * earth_radius * np.cos(np.radians(dock_lat))
            y = dlat * earth_radius
            return x, y
        gps_df['x'], gps_df['y'] = zip(*gps_df.apply(lambda row: lat_lon_to_meters(row['latitude'], row['longitude']), axis=1))
        dock_x, dock_y = lat_lon_to_meters(dock_lat, dock_lon)

        # Adjust plot limits to start slightly before the dock location
        x_min = min(gps_df['x'].min(), dock_x) - 10
        y_min = min(gps_df['y'].min(), dock_y) - 10
        x_max = max(gps_df['x'].max(), dock_x) + 10
        y_max = max(gps_df['y'].max(), dock_y) + 10

        # Create a scatter plot of GPS locations, colored by confidence
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(gps_df['x'], gps_df['y'], c=gps_df['ap'], cmap='viridis')
        plt.colorbar(scatter, label='Average Precision')
        plt.scatter(dock_x, dock_y, color='red', marker='x', label="Dock Location")
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('GPS Locations vs Dock Location (in meters)')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.grid()
        plt.show()

    def evaluate(self, bins=40, iou_threshold=0.5):
        """Run the complete evaluation by extracting data and plotting results."""
        self.extract_data()
        self.calculate_distances()
        # self.compute_aps(iou_threshold)
        # self.plot_ap_vs_distance(bins, iou_threshold)
        self.plot_metrics_vs_distance(bins, iou_threshold, confidence_thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], metrics=["precision", "recall", "f1"])
        self.plot_gps_vs_dock()


if __name__ == "__main__":
    data_folder = '/home/daan/Data/dock_data'  # Folder containing images, labels and gps_coords folders
    project_folder = "/home/daan/object_detection/"

    dock_lat, dock_lon = 51.82106762532677, 4.888864958997777  # Dock location coordinates
    rosbag_name = ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]

    run_name = "m:YOLO11s_e:600_b:8_d:split-1(2)"
    iou_threshold = 0.1
    bins = 40



    model = "YOLO" if run_name.startswith("m:YOLO") else run_name.split("_")[0][2:]
    prediction_folder = os.path.join(project_folder, model, "runs", run_name, "predict", "labels")
    gps_eval = GPS_Evaluation(data_folder, prediction_folder, rosbag_name, (dock_lat, dock_lon))
    gps_eval.evaluate(bins=bins, iou_threshold=iou_threshold)
