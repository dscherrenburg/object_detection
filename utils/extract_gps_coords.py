import os
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from sensor_msgs.msg import NavSatFix, Image

def extract_gps_coords_from_rosbags(folder_path, image_topics, gps_topic, output_folder=None):
    # Get all rosbags in the folder
    rosbag_dirs = os.listdir(folder_path)

    # Initialize the type store
    typestore = get_typestore(Stores.ROS2_FOXY)

    for rosbag in rosbag_dirs:
        bag_path = os.path.join(folder_path, rosbag)
        if output_folder:
            output_file = os.path.join(output_folder, f"{rosbag}.txt")
        else:
            output_file = os.path.join(folder_path, f"{rosbag}.txt")

        # Store GPS and image data
        gps_data = []
        image_timestamps = []

        # Open the rosbag
        with Reader(bag_path) as reader:
            # Get connections for the topics
            connections = {conn.topic: conn for conn in reader.connections}
            image_topic = None
            for topic in image_topics:
                if topic in connections:
                    image_topic = topic
                    break
            if image_topic is None:
                print(f"None of the image topics {image_topics} found in {rosbag}")
                continue
            gps_conn = connections.get(gps_topic)
            if not gps_conn:
                print(f"GPS topic {gps_topic} not found in {rosbag}")
                continue

            # Read messages
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == gps_topic:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    timestamp = int(f"{msg.header.stamp.sec:010d}{msg.header.stamp.nanosec:09d}")
                    gps_data.append((timestamp, msg.latitude, msg.longitude))
                elif connection.topic == image_topic:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                    timestamp = int(f"{msg.header.stamp.sec:010d}{msg.header.stamp.nanosec:09d}")
                    image_timestamps.append(timestamp)

            print(f"Extracted {len(gps_data)} GPS data points and {len(image_timestamps)} image timestamps from {rosbag}")

        # Interpolate GPS coordinates for image timestamps
        gps_timestamps = np.array([data[0] for data in gps_data])
        latitudes = np.array([data[1] for data in gps_data])
        longitudes = np.array([data[2] for data in gps_data])

        with open(output_file, 'w') as f:
            if len(image_timestamps) == 0:
                print(f"No image timestamps found in {rosbag}")
                continue
            if len(gps_timestamps) == 0:
                print(f"No GPS data found in {rosbag}")
                continue

            for img_time in image_timestamps:
                # Check if exact match exists
                if img_time in gps_timestamps:
                    idx = np.where(gps_timestamps == img_time)[0][0]
                    f.write(f"{img_time} {latitudes[idx]} {longitudes[idx]}\n")
                else:
                    # Interpolate GPS coordinates
                    lat = np.interp(img_time, gps_timestamps, latitudes)
                    lon = np.interp(img_time, gps_timestamps, longitudes)
                    f.write(f"{img_time} {lat} {lon}\n")



if __name__ == "__main__":
    folder_path = "/home/daan/Data/dock_data/rosbags"  # Replace with the folder containing rosbags
    image_topics = ["/sensors/rgb_camera/image_raw/sampled", "/sensors/rgb_camera/image_raw"]  # Replace with your image topics
    gps_topic = "/sensors/gnss/fix"  # Replace with your GPS topic
    output_folder = "/home/daan/Data/dock_data/gps_coords"  # Replace with the output folder

    extract_gps_coords_from_rosbags(folder_path, image_topics, gps_topic, output_folder)