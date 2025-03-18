import os
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from scipy.stats import binned_statistic

from cv_bridge import CvBridge
import pandas as pd
from datetime import datetime

class GPS_Evaluation:
    def __init__(self, rosbag, label_folder, dock_coordinates):
        """
        Initializes the GPS_Evaluation class.
        
        Parameters:
        - rosbag_path: Path to the rosbag file containing GPS data.
        - label_folder: Folder path containing prediction confidence txt files (timestamp-based).
        - dock_coordinates: Tuple of (latitude, longitude) representing the dock's location.
        """
        self.rosbag = rosbag
        self.label_folder = label_folder
        self.dock_coordinates = dock_coordinates
        self.gps_data = []
        self.confidences = []
        self.timestamps = []
        self.distances = []

    def extract_gps_data(self, topic):
        """Extract GPS data from the given rosbag topic."""
        typestore = get_typestore(Stores.ROS2_FOXY)
        self.gps_data = []
        self.first_timestamp, self.last_timestamp = None, None

        with Reader(self.rosbag) as reader:
            for connection, timestamp, rawdata in reader.messages():
                if self.first_timestamp is None:
                    self.first_timestamp = timestamp
                self.last_timestamp = timestamp

                if connection.topic == topic:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    self.gps_data.append((timestamp, msg.latitude, msg.longitude))

        print(f"Extracted {len(self.gps_data)} GPS data points from {len(self.rosbag)} rosbags.")

    def extract_confidences(self):
        """Extract object detection confidence from txt files in the label folder."""
        label_files = sorted(os.listdir(self.label_folder))
        for label_file in label_files:
            if label_file.endswith('.txt'):
                timestamp = int(label_file.split('.')[0])
                if timestamp < self.first_timestamp or timestamp > self.last_timestamp:
                    continue
                with open(os.path.join(self.label_folder, label_file), 'r') as f:
                    confidence = max([float(line.strip().split()[-1]) for line in f.readlines()])  # Assuming highest confidence
                    self.confidences.append((timestamp, confidence))
                    self.timestamps.append(timestamp)

        print(f"Extracted {len(self.confidences)} confidence values from {len(label_files)} label files.")

    def interpolate_gps_data(self):
        """Interpolate missing GPS data points."""
        gps_data = sorted(self.gps_data, key=lambda x: x[0])
        self.timestamps = list(set(self.timestamps))
        self.gps_data = []
        for timestamp in self.timestamps:
            previous_gps_data = None
            next_gps_data = None
            for gps_point in gps_data:
                if gps_point[0] == timestamp:
                    self.gps_data.append(gps_point)
                    break
                elif gps_point[0] < timestamp:
                    previous_gps_data = gps_point
                elif gps_point[0] > timestamp:
                    next_gps_data = gps_point
                    break
            
            if previous_gps_data and next_gps_data:
                previous_time, previous_lat, previous_lon = previous_gps_data
                next_time, next_lat, next_lon = next_gps_data
                time_diff = next_time - previous_time
                lat_diff = next_lat - previous_lat
                lon_diff = next_lon - previous_lon
                time_ratio = (timestamp - previous_time) / time_diff
                lat = previous_lat + lat_diff * time_ratio
                lon = previous_lon + lon_diff * time_ratio
                self.gps_data.append((timestamp, lat, lon))
            elif previous_gps_data and not next_gps_data:
                self.gps_data.append(previous_gps_data)
            elif not previous_gps_data and next_gps_data:
                self.gps_data.append(next_gps_data)
            
        print(f"Interpolated missing GPS data points. Total: {len(self.gps_data)}")

    def calculate_distances(self):
        """Calculate the Euclidean distance between GPS points and the dock location."""
        dock_lat, dock_lon = self.dock_coordinates
        for timestamp, lat, lon in self.gps_data:
            # Simple Haversine distance calculation (in meters)
            lat1, lon1 = np.radians(dock_lat), np.radians(dock_lon)
            lat2, lon2 = np.radians(lat), np.radians(lon)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            radius = 6371000  # Earth radius in meters
            distance = radius * c
            self.distances.append((timestamp, distance))

        print(f"Calculated {len(self.distances)} distances from dock location.")

    def plot_gps_vs_dock(self):
        """Plot GPS locations and color them based on confidence."""
        gps_df = pd.DataFrame(self.gps_data, columns=["timestamp", "latitude", "longitude"])
        gps_df['confidence'] = [c[1] for c in self.confidences]  # Assuming confidence list is ordered by timestamp

        # Create a scatter plot of GPS locations, colored by confidence
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(gps_df['longitude'], gps_df['latitude'], c=gps_df['confidence'], cmap='viridis')
        plt.colorbar(scatter, label='Confidence')
        plt.scatter(self.dock_coordinates[1], self.dock_coordinates[0], color='red', marker='x', label="Dock Location")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('GPS Locations vs Dock Location')
        plt.legend()
        plt.show()

    # def plot_confidence_vs_distance(self):
    #     """Plot confidence vs distance with trend line and standard deviation shading."""
    #     distance_df = pd.DataFrame(self.distances, columns=["timestamp", "distance"])
    #     confidence_df = pd.DataFrame(self.confidences, columns=["timestamp", "confidence"])

    #     # Merge the two DataFrames based on timestamp
    #     merged_df = pd.merge(distance_df, confidence_df, on="timestamp", how="inner")
    #     distances = merged_df['distance'].to_numpy()
    #     confidences = merged_df['confidence'].to_numpy()

    #     # Scatter plot with color gradient
    #     plt.figure(figsize=(10, 6))
    #     scatter = plt.scatter(distances, confidences, c=confidences, cmap='viridis', alpha=0.7, edgecolors='k', s=confidences * 50)

    #     # Compute trend line (Quadratic fit)
    #     if len(distances) > 5:
    #         z = np.polyfit(distances, confidences, 2)  # Quadratic fit
    #         p = np.poly1d(z)
    #         sorted_distances = np.sort(distances)
    #         trend_values = p(sorted_distances)

    #         # Compute standard deviation in distance bins
    #         num_bins = 10  # Adjust as needed
    #         bin_means, bin_edges, _ = binned_statistic(distances, confidences, statistic='mean', bins=num_bins)
    #         bin_stds, _, _ = binned_statistic(distances, confidences, statistic='std', bins=num_bins)

    #         # Compute bin centers
    #         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    #         # Interpolate standard deviation to match smooth trend line
    #         std_interpolated = np.interp(sorted_distances, bin_centers, bin_stds, left=bin_stds[0], right=bin_stds[-1])

    #         # Plot trend line
    #         plt.plot(sorted_distances, trend_values, "r-", label="Trend Line")

    #         # Fill area with standard deviation band
    #         plt.fill_between(sorted_distances, trend_values - std_interpolated, trend_values + std_interpolated, 
    #                         color='r', alpha=0.3, label="Std Deviation")

    #     # Add colorbar
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_label('Confidence')

    #     plt.xlabel('Distance (meters)')
    #     plt.ylabel('Confidence')
    #     plt.title('Confidence vs Distance from Dock')
    #     plt.grid(True, linestyle="--", alpha=0.6)
    #     plt.legend()
    #     plt.show()

    def plot_confidence_vs_distance(self, num_bins=20):
        """Plot confidence vs distance as a vertical bar plot with standard deviation."""
        distance_df = pd.DataFrame(self.distances, columns=["timestamp", "distance"])
        confidence_df = pd.DataFrame(self.confidences, columns=["timestamp", "confidence"])

        # Merge the two DataFrames based on timestamp
        merged_df = pd.merge(distance_df, confidence_df, on="timestamp", how="inner")
        distances = merged_df['distance'].to_numpy()
        confidences = merged_df['confidence'].to_numpy()

        # Compute binned statistics for means and standard deviations
        bin_means, bin_edges, _ = binned_statistic(distances, confidences, statistic='mean', bins=num_bins)
        bin_stds, _, _ = binned_statistic(distances, confidences, statistic='std', bins=num_bins)
        
        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize confidence values for color mapping
        norm_confidences = (bin_means - np.min(bin_means)) / (np.max(bin_means) - np.min(bin_means) + 1e-6)
        
        # Create the bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(bin_centers, bin_means, width=(bin_edges[1] - bin_edges[0]) * 0.9, 
                    color=plt.cm.viridis(norm_confidences), edgecolor="black", yerr=bin_stds, capsize=5)

        # # Color bar legend
        # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=np.min(bin_means), vmax=np.max(bin_means)))
        # cbar = plt.colorbar(sm)
        # cbar.set_label("Average Confidence")

        # Labels and title
        plt.xlabel("Distance from Dock (meters)")
        plt.ylabel("Mean Confidence")
        plt.title("Average Confidence vs Distance from Dock")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        plt.show()

    def evaluate(self, topic):
        """Run the complete evaluation by extracting data and plotting results."""
        self.extract_gps_data(topic)
        self.extract_confidences()
        self.interpolate_gps_data()
        self.calculate_distances()
        self.plot_gps_vs_dock()
        self.plot_confidence_vs_distance()


if __name__ == "__main__":
    dock_lat, dock_lon = 51.81477243831143, 4.766330011467225  # Dock location coordinates
    rosbag = "/home/daan/Data/dock_data/rosbags/rosbag2_2024_09_17-16_02_29"
    label_folder = '/home/daan/object_detection/YOLO/runs/m:YOLO11n_e:300_b:8_d:split-1/predict/labels'
    gps_eval = GPS_Evaluation(rosbag, label_folder, (dock_lat, dock_lon))
    gps_eval.evaluate('/sensors/gnss/fix')
