import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from geopy.distance import geodesic
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores


def extract_gps_coords(rosbags_folder, gps_topic, image_topics, rosbags=None, gps_data_dir=None):
    if gps_data_dir is None:
        gps_data_dir = os.path.join(os.path.dirname(rosbags_folder), "gps_data")
    os.makedirs(gps_data_dir, exist_ok=True)

    # Get all rosbags in the folder
    if rosbags is None:
        rosbags = os.listdir(rosbags_folder)

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    for bag_name in rosbags:
        data_file = os.path.join(gps_data_dir, f"{bag_name}.csv")
        if os.path.exists(data_file):
            continue
        bag_path = f"{rosbags_folder}/{bag_name}"
        gps_coords = []
        image_timestamps = []

        with Reader(bag_path) as reader:
            connections = {conn.topic: conn for conn in reader.connections}
            # Read messages
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == gps_topic:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    timestamp = int(f"{msg.header.stamp.sec:010d}{msg.header.stamp.nanosec:09d}")
                    gps_coords.append((timestamp, msg.latitude, msg.longitude))
                elif connection.topic in image_topics:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    timestamp = int(f"{msg.header.stamp.sec:010d}{msg.header.stamp.nanosec:09d}")
                    image_timestamps.append(timestamp)
            
            if len(gps_coords) == 0:
                print(f"No GPS data found in {bag_name}, so trying AIS data")
                
                key = "244820507"
                json_dir = "/home/daan/Downloads/plot_coordinates/json_files/ais"
                ais_file = os.path.join(json_dir, f"{bag_name[8:]}.json")
                if not os.path.exists(ais_file):
                    print(f"AIS data file {ais_file} does not exist.")
                    continue
                ais_data = json.load(open(ais_file))
                if key not in ais_data:
                    print(f"No AIS data found in {ais_file}")
                    continue
                data = ais_data[key]
                for item in data:
                    stamp = item['time'] * 1e9
                    lat = item['latitude']
                    lon = item['longitude']
                    if lat == 0 or lon == 0 or lat == math.nan or lon == math.nan:
                        continue
                    gps_coords.append((stamp, lat, lon))
                
                if len(gps_coords) == 0:
                    print(f"No AIS data found in {ais_file}")
                    continue

            print(f"Extracted {len(gps_coords)} GPS data points and {len(image_timestamps)} image timestamps from {bag_name}")
            
        # Interpolate GPS coordinates for image timestamps
        gps_timestamps = np.array([data[0] for data in gps_coords])
        latitudes = np.array([data[1] for data in gps_coords])
        longitudes = np.array([data[2] for data in gps_coords])
        image_timestamps = np.array(image_timestamps)
        if len(image_timestamps) == 0:
            print(f"No image timestamps found in {bag_name}")
            continue
        if len(gps_timestamps) == 0:
            print(f"No GPS data found in {bag_name}")
            continue
        # Interpolate GPS coordinates for image timestamps
        latitudes_interp = np.interp(image_timestamps, gps_timestamps, latitudes)
        longitudes_interp = np.interp(image_timestamps, gps_timestamps, longitudes)
        # Create a DataFrame for the interpolated data
        gps_data = pd.DataFrame({
            "timestamp": image_timestamps,
            "latitude": latitudes_interp,
            "longitude": longitudes_interp,
        })
        gps_data.to_csv(data_file, index=False)

def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    """Converts latitude and longitude to meters relative to a reference point."""
    earth_radius = 6378137  # Radius of Earth in meters
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    lat_m = dlat * earth_radius
    lon_m = dlon * earth_radius * np.cos(np.radians(ref_lat))
    return lat_m, lon_m

def haversine(coord1, coord2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Earth radius in meters

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def compute_distances_and_angles(gps_data_dir, dock_coords):
    """Computes distances (m) and angles (deg) from GPS data to dock coordinates."""
    for rosbag in dock_coords.keys():
        gps_data_file = os.path.join(gps_data_dir, f"{rosbag}.csv")
        gps_data = pd.read_csv(gps_data_file)

        # Check if data already contains distances and angles
        if "distance" in gps_data.columns and "angle" in gps_data.columns:
            continue
        
        dock_coordinates = dock_coords[rosbag]
        dock_center = np.mean(dock_coordinates, axis=0)

        # Distance
        gps_data["distance"] = gps_data.apply(
            lambda row: geodesic((row["latitude"], row["longitude"]), dock_center).meters, axis=1
        )
        
        # Angle
        dock_start = np.array(dock_coordinates[0][::-1])  # (lon, lat)
        dock_end = np.array(dock_coordinates[1][::-1])    # (lon, lat)
        dock_center = (dock_start + dock_end) / 2
        dock_vec = dock_end - dock_start
        dock_vec = dock_vec / np.linalg.norm(dock_vec)
        dock_normal = np.array([-dock_vec[1], dock_vec[0]])

        def compute_angle(lat, lon):
            pos = np.array([lon, lat])
            vec_to_cam = pos - dock_center
            vec_to_cam = vec_to_cam / np.linalg.norm(vec_to_cam)
            angle_rad = np.arctan2(
                np.cross(dock_normal, vec_to_cam),
                np.dot(dock_normal, vec_to_cam)
            )
            return np.degrees(angle_rad)

        gps_data["angle"] = gps_data.apply(lambda row: compute_angle(row["latitude"], row["longitude"]), axis=1)
        # gps_data["angle"] = gps_data["angle"].apply(lambda x: (x + 180) % 360 - 180)  # Normalize to [-180, 180]
        # Save the updated data
        gps_data.to_csv(gps_data_file, index=False)


def plot_polar_heatmap(gps_data_folder, rosbags):
    """Visualizes the locations of frame captures relative to a dock (at origin) using a polar scatter plot."""
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(10, 9))
    ax = plt.subplot(111, polar=True)

    all_angles = []
    all_distances = []

    for rosbag in rosbags:
        gps_data_file = os.path.join(gps_data_folder, f"{rosbag}.csv")
        if not os.path.exists(gps_data_file):
            print(f"GPS data file {gps_data_file} does not exist.")
            continue

        gps_data = pd.read_csv(gps_data_file)
        angles = gps_data["angle"]
        distances = gps_data["distance"]

        # Project angles to [-90, 90] range
        # angles = np.where(angles < 0, angles + 360, angles)
        # angles = np.where(angles > 180, angles - 360, angles)
        # angles = np.where(angles < -90, angles + 180, angles)
        # angles = np.where(angles > 90, angles - 180, angles)
        angles = -np.radians(angles)

        all_angles.extend(angles)
        all_distances.extend(distances)

    # Scatter plot with colormap
    scatter = ax.scatter(all_angles, all_distances, 
                         c=all_distances, cmap="viridis", 
                         alpha=0.5, s=40, edgecolors='none')

    # Polar settings
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.grid(True)

    # Set tick label font sizes
    ax.tick_params(axis='both', labelsize=12)

    # Add x-axis label manually below the plot
    plt.figtext(0.5, 0.215, "Distance to dock (m)", ha='center', fontsize=13)

    # # Title
    # ax.set_title("Spatial Distribution of Frame Captures\nRelative to Dock Location (Located at Origin)",
    #              va="bottom", fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig("/home/daan/Documents/Thesis/distribution.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    rosbags_folder = "/home/daan/Data/dock_data/rosbags"
    gps_data_dir = "/home/daan/Data/dock_data/gps_data"
    image_topics = ["/sensors/rgb_camera/image_raw/sampled", "/sensors/rgb_camera/image_raw"]
    gps_topic = "/sensors/gnss/fix"

    dock_data = {
            "rosbag2_2024_09_17-16_02_29": [(51.81478814973421, 4.766302229483678), (51.81475913198714, 4.76643566938163)],
            "rosbag2_2024_09_17-19_47_16": [(51.81478814973421, 4.766302229483678), (51.81475913198714, 4.76643566938163)],

            "rosbag2_2024_09_17-16_34_11": [(51.82398941105339, 4.729554088975325), (51.82399604232434, 4.729184614674523)],
            "rosbag2_2024_09_17-19_32_52": [(51.82398941105339, 4.729554088975325), (51.82399604232434, 4.729184614674523)],
            "rosbag2_2024_11_18-12_55_24": [(51.82398941105339, 4.729554088975325), (51.82399604232434, 4.729184614674523)],

            "rosbag2_2024_09_17-15_33_18": [(51.81997966741068, 4.846921773627031), (51.82000660934062, 4.846496643487814)],
            "rosbag2_2024_09_17-15_37_39": [(51.81997966741068, 4.846921773627031), (51.82000660934062, 4.846496643487814)],
            "rosbag2_2024_09_17-15_40_51": [(51.81997966741068, 4.846921773627031), (51.82000660934062, 4.846496643487814)],

            "rosbag2_2024_09_17-17_04_31": [(51.833911861591304, 4.673126410998313), (51.834212391859495, 4.673363907670344)],

            "rosbag2_2024_09_17-17_13_13": [(51.85023606430054, 4.657714278249804), (51.8500165309032, 4.657859117540136)],

            "rosbag2_2024_09_17-18_54_49": [(51.814849954559314, 4.6581413684089), (51.815021158652726, 4.658418977048704)],

            "rosbag2_2024_09_17-14_40_19": [(51.82109416005252, 4.8889499241694025), (51.82103406025281, 4.888777592235998)],
            "rosbag2_2024_09_17-14_48_48": [(51.82109416005252, 4.8889499241694025), (51.82103406025281, 4.888777592235998)],
            "rosbag2_2024_09_17-14_58_53": [(51.82109416005252, 4.8889499241694025), (51.82103406025281, 4.888777592235998)],
            "rosbag2_2024_09_17-20_27_46": [(51.82109416005252, 4.8889499241694025), (51.82103406025281, 4.888777592235998)],

            "rosbag2_2024_09_17-15_59_28": [(51.81687526993932, 4.779191387516475), (51.81691506383338, 4.7790639825969363)],
    }

    # for key, coords in dock_data.items():
    #     width = haversine(coords[0], coords[1])
    #     print(f"{key}: {width:.2f} meters")

    # exit()

    # Extract GPS coordinates
    gps_data = extract_gps_coords(rosbags_folder, gps_topic, image_topics, rosbags=dock_data.keys(), gps_data_dir=gps_data_dir)
    # Compute distances and angles
    gps_data = compute_distances_and_angles(gps_data_dir, dock_coords=dock_data)
    # Plot polar heatmap
    plot_polar_heatmap(gps_data_dir, rosbags=dock_data.keys())


























def plot_label_size_vs_distance(self, image_size=(640, 360)):
        """Plot label size vs distance from dock."""
        label_widths = []
        label_heights = []
        label_areas = []
        for _, labels in self.labels:
            if len(labels) == 0:
                label_widths.append(0)
                label_heights.append(0)
                label_areas.append(0)
            else:
                for c, (x, y, w, h) in labels:
                    w = w * image_size[0]
                    h = h * image_size[1]
                    label_widths.append(w)
                    label_heights.append(h)
                    label_areas.append(w * h)
        label_widths = np.array(label_widths[:-1])
        label_heights = np.array(label_heights[:-1])
        label_areas = np.array(label_areas[:-1])
        distances = self.distances[:, 1]

        print(f"Number of labels: {len(label_widths)}")
        print(f"Number of distances: {len(distances)}")
        # Create bar plots for width, height, area vs distance
        fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        fig.suptitle("Label Size vs Distance from Dock", fontsize=16)
        
        axs[0].bar(distances, label_widths, width=0.5, color='blue', alpha=0.7)
        axs[0].set_ylabel("Width (pixels)")
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        axs[1].bar(distances, label_heights, width=0.5, color='green', alpha=0.7)
        axs[1].set_ylabel("Height (pixels)")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        axs[2].bar(distances, label_areas, width=0.5, color='red', alpha=0.7)
        axs[2].set_xlabel("Distance from Dock (m)")
        axs[2].set_ylabel("Area (pixels^2)")
        axs[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()