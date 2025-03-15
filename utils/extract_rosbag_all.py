import os
import cv2
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import get_typestore, Stores
from cv_bridge import CvBridge
import tqdm


def extract_images_from_rosbag(bag_name, data_folder, image_topics):
    bag_folder = os.path.join(data_folder, "rosbags", bag_name)
    output_folder = os.path.join(data_folder, "images", bag_name)
    bridge = CvBridge()
    count = 0
    typestore = get_typestore(Stores.ROS2_FOXY)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    with Reader(bag_folder) as reader:
        # Check which image topic is present in the rosbag
        image_topic = None
        for topic in image_topics:
            if topic in reader.topics:
                image_topic = topic
                break
        
        if image_topic is None:
            print(f"None of the image topics {image_topics} found in {bag_name}")
            return  
        
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == image_topic:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                img_timestamp = f"{msg.header.stamp.sec:010d}{msg.header.stamp.nanosec:09d}"
                img_filename = os.path.join(output_folder, f"{img_timestamp}.png")
                if os.path.isfile(img_filename):
                    continue
                cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_filename, cv_img)
                count += 1

    print(f"Extracted {count} images from {bag_name}")

    return count


def extract_images_from_rosbags(data_folder, image_topics):
    failed_bags = []
    total_count = 0
    bags = os.listdir(os.path.join(data_folder, "rosbags"))
    output_folder = os.path.join(data_folder, "images")

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    else:
        key = input("Images already extracted, press Y to continue or any other key to exit: ")
        if key != "Y":
            return

    for bag_name in tqdm.tqdm(bags, desc="Extracting images from rosbags", unit=" bags"):
        try:
            if not os.path.isdir(os.path.join(output_folder, bag_name)):
                os.makedirs(os.path.join(output_folder, bag_name))
            else:
                print(f"Images from {bag_name} already extracted")
                continue
            count = extract_images_from_rosbag(bag_name, data_folder, image_topics)
            total_count += count
        except Exception as e:
            print(f"Error extracting images from {bag_name}: {e}")
            failed_bags.append(bag_name)
            os.rmdir(os.path.join(output_folder, bag_name))
    print(f"Failed to extract images from {len(failed_bags)} bags: {failed_bags}")
    print(f"Extracted {total_count} images from {len(bags)-len(failed_bags)}/{len(bags)} bags")


if __name__ == "__main__":

    image_topics = ["/sensors/rgb_camera/image_raw/sampled", "/sensors/rgb_camera/image_raw"]
    data_folder = "/home/daan/Data/dock_data/"

    # bag_name = "rosbag2_2024_09_17-16_34_11"
    # extract_images_from_rosbag(bag_name, data_folder, image_topics)

    extract_images_from_rosbags(data_folder, image_topics)
