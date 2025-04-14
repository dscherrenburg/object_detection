import os
from pathlib import Path
import pandas as pd

# Define dataset root
DATASET_ROOT = "/home/daan/Data/dock_data"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
GPS_DIR = os.path.join(DATASET_ROOT, "gps_data")
CONFIG_DIR = "dataset_configs"

def get_valid_folders():
    """Finds folders that have both an image and an annotation."""
    image_folders = {f.name for f in Path(IMAGES_DIR).iterdir() if f.is_dir()}
    annotation_folders = {f.name for f in Path(LABELS_DIR).iterdir() if f.is_dir()}
    return sorted(image_folders & annotation_folders)

def create_dataset_config(train_folders, val_folders, test_folders, config_name, interval=1, distance_range=None, min_label_size=None, image_size=(640,360)):
    """Generates dataset configuration files."""
    config_path = os.path.join(CONFIG_DIR, config_name)
    os.makedirs(config_path, exist_ok=True)

    global skip_count
    skip_count = 0

    # Helper function to write file paths
    def write_split_file(split_name, folders, interval=interval, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size):
        global skip_count
        split_file = os.path.join(config_path, f"{split_name}.txt")
        if min_label_size > 0:
            print("Warning: min_label_size > 0, skipping images with labels smaller than this size, thus all images with no labels will be skipped.")
        with open(split_file, "w") as f:
            for folder in folders:
                image_folder = os.path.join(IMAGES_DIR, folder)
                gps_data = pd.read_csv(os.path.join(GPS_DIR, folder + ".csv"))
                for img in sorted(os.listdir(image_folder))[::interval]:
                    if img.endswith((".jpg", ".jpeg", ".png")):
                        skip = False
                        if min_label_size and min_label_size > 0:
                            label_file = os.path.join(LABELS_DIR, folder, img.split(".")[0] + ".txt")
                            if os.path.exists(label_file):
                                with open(label_file, "r") as label_f:
                                    lines = label_f.readlines()
                                    skip = False
                                    if min_label_size > 0:
                                        if len(lines) == 0:
                                            skip = True
                                            skip_count += 1
                                        else:
                                            for class_id, x, y, w, h in (line.split() for line in lines):
                                                if float(w) < min_label_size/image_size[0] or float(h) < min_label_size/image_size[1]:
                                                    skip = True
                                                    skip_count += 1
                                                    break
                            else:
                                print(f"Warning: Label file {label_file} does not exist.")
                                continue

                        if distance_range:
                            timestamp = int(img.split(".")[0])
                            distance= gps_data.loc[gps_data['timestamp'] == timestamp, 'distance'].values[0]
                            if not distance_range[0] < distance < distance_range[1]:
                                skip = True
                                skip_count += 1
                        if not skip:
                            f.write(f"{os.path.join(image_folder, img)}\n")

    # Generate train, val, and test txt files
    write_split_file("train", train_folders)
    write_split_file("val", val_folders)
    write_split_file("test", test_folders, interval=1, min_label_size=0)

    # Create dataset.yaml file
    yaml_content = f"""path: {os.path.abspath(config_path)}
train: {'train.txt'}
val: {'val.txt'}
test: {'test.txt'}
nc: 1
names: ["dock"]
"""
    with open(os.path.join(config_path, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"✅ Dataset configuration '{config_name}' created successfully in '{config_path}'.")
    return skip_count

if __name__ == "__main__":
    valid_folders = get_valid_folders()

    # Go through labels in the valid_folders and print the number of non-empty labels
    for folder in valid_folders:
        label_folder = os.path.join(LABELS_DIR, folder)
        label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt") and f.startswith("1")]
        total_rows = sum(len(open(os.path.join(label_folder, label_file)).readlines()) for label_file in label_files)
        print(f"{folder}: {total_rows} total docks in label files")

    exit()


























    del valid_folders[valid_folders.index("rosbag2_2024_09_17-15_59_28")]

    ### Define dataset splits ###
    val_sets_1 = [["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
                ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
                ["rosbag2_2024_09_17-17_04_31"],
                ["rosbag2_2024_09_17-17_13_13"],
                ["rosbag2_2024_09_17-18_54_49"],
                ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]]
    test_folders_1 = ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"]
    val_sets_2 = [["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
                ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
                ["rosbag2_2024_09_17-17_04_31"],
                ["rosbag2_2024_09_17-17_13_13"],
                ["rosbag2_2024_09_17-18_54_49"],
                ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"]]
    test_folders_2 = ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]

    data_version = 1
    interval = 5
    min_label_size = 1
    distance_range = (0, 200)
    image_size = (640, 360)
    #############################

    if data_version == 1:
        val_sets = val_sets_1
        test_folders = test_folders_1
    elif data_version == 2:
        val_sets = val_sets_2
        test_folders = test_folders_2

    for i, val_folders in enumerate(val_sets):
        name = f"split-{i+1}"
        name += f"({data_version})" if data_version > 1 else ""
        name += f"+interval-{interval}" if (interval and interval > 1) else ""
        name += f"+label-{min_label_size}" if (min_label_size and min_label_size > 2) else ""
        name += f"+distance-({distance_range[0]}-{distance_range[1]})" if (distance_range and distance_range[1] > 0) else ""
        val_folders = set(val_folders) & set(valid_folders)
        test_folders = set(test_folders) & set(valid_folders)
        train_folders = set(valid_folders) - val_folders - test_folders

        skip_count = create_dataset_config(list(train_folders), list(val_folders), list(test_folders), name, interval=interval, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size)
    
    print(f"Total images left out due to min label size or distance range: {skip_count}")
