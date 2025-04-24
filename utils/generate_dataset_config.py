import os
from pathlib import Path
import pandas as pd

# Define dataset root
DATASET_ROOT = "/home/daan/Data/dock_data"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
GPS_DIR = os.path.join(DATASET_ROOT, "gps_data")
CONFIG_DIR = "/home/daan/object_detection/dataset_configs"

def get_valid_folders():
    """Finds folders that have both an image and an annotation."""
    image_folders = {f.name for f in Path(IMAGES_DIR).iterdir() if f.is_dir()}
    annotation_folders = {f.name for f in Path(LABELS_DIR).iterdir() if f.is_dir()}
    return sorted(image_folders & annotation_folders)

def load_gps_data(gps_file_path):
    try:
        gps_df = pd.read_csv(gps_file_path)
        return gps_df.set_index('timestamp')['distance'].to_dict()
    except Exception as e:
        print(f"Failed to load GPS data from {gps_file_path}: {e}")
        return {}

def label_too_small(label_path, min_label_size, image_size):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) == 0:
                return True
            for line in lines:
                _, _, _, w, h = map(float, line.strip().split())
                if w * image_size[0] < min_label_size or h * image_size[1] < min_label_size:
                    return True
    except Exception as e:
        print(f"Error reading label file {label_path}: {e}")
        return True
    return False

def distance_out_of_range(img_name, gps_map, distance_range):
    try:
        timestamp = int(Path(img_name).stem)
        distance = gps_map.get(timestamp, None)
        if distance is None:
            return True
        return not (distance_range[0] < distance < distance_range[1])
    except Exception as e:
        print(f"Error parsing timestamp or distance for {img_name}: {e}")
        return True

def write_split_file(split_name, folders, images_dir, labels_dir, gps_dir, config_path,
                     interval=1, distance_range=None, min_label_size=0, image_size=(640, 480)):

    split_file_path = os.path.join(config_path, f"{split_name}.txt")
    skipped_info = {
        "missing_label": 0,
        "small_label": 0,
        "out_of_range": 0,
        "written": 0
    }

    with open(split_file_path, "w") as output_file:
        for folder in folders:
            image_folder = os.path.join(images_dir, folder)
            gps_file = os.path.join(gps_dir, f"{folder}.csv")
            gps_map = load_gps_data(gps_file)

            image_list = sorted(os.listdir(image_folder))[::interval]

            for img_name in image_list:
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(image_folder, img_name)
                label_path = os.path.join(labels_dir, folder, Path(img_name).stem + ".txt")

                # Label size check
                if min_label_size > 0:
                    if not os.path.exists(label_path):
                        skipped_info["missing_label"] += 1
                        continue
                    if label_too_small(label_path, min_label_size, image_size):
                        skipped_info["small_label"] += 1
                        continue

                # Distance range check
                if distance_range:
                    if distance_out_of_range(img_name, gps_map, distance_range):
                        skipped_info["out_of_range"] += 1
                        continue

                output_file.write(f"{img_path}\n")
                skipped_info["written"] += 1

    print(f"   [{split_name:<5}] Total kept: {skipped_info['written']:>5}  |  Skipped - Small label: {skipped_info['small_label']:>5}, "
          f"Missing label: {skipped_info['missing_label']:>5}, "
          f"Distance out of range: {skipped_info['out_of_range']:>5}")

    return skipped_info["written"], skipped_info


def create_dataset_config(train_folders, val_folders, test_folders, config_name, interval=1, distance_range=None, min_label_size=None, image_size=(640,360)):
    """Generates dataset configuration files."""
    config_path = os.path.join(CONFIG_DIR, config_name)
    os.makedirs(config_path, exist_ok=True)
    print(f"Creating dataset configuration '{config_name}'")

    # Generate train, val, and test txt files
    train_size, _ = write_split_file("train", train_folders, IMAGES_DIR, LABELS_DIR, GPS_DIR, config_path,
                                     interval=interval, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size)
    val_size, _ = write_split_file("val", val_folders, IMAGES_DIR, LABELS_DIR, GPS_DIR, config_path,
                                   interval=interval, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size)
    test_size, _ = write_split_file("test", test_folders, IMAGES_DIR, LABELS_DIR, GPS_DIR, config_path,
                                    interval=1, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size)

    if any(size == 0 for size in [train_size, val_size, test_size]):
        print(f"Warning: One of the splits in '{config_name}' is empty. Please check your folder structure and filters.")
        return 0, 0, 0

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
    return train_size, val_size, test_size

if __name__ == "__main__":
    valid_folders = get_valid_folders()

    # # Go through labels in the valid_folders and print the number of non-empty labels
    # for folder in valid_folders:
    #     label_folder = os.path.join(LABELS_DIR, folder)
    #     label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt") and f.startswith("1")]
    #     total_rows = sum(len(open(os.path.join(label_folder, label_file)).readlines()) for label_file in label_files)
    #     print(f"{folder}: {total_rows} total docks in label files")

    # exit()

    # del valid_folders[valid_folders.index("rosbag2_2024_09_17-15_59_28")]

    data_sets = {
        "Waterbus dock Rosmolenweg":            ["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
        "Bunkerstation Dekker & Stam":          ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
        "Waterbus dock Noordhoek":              ["rosbag2_2024_09_17-17_04_31"],
        "Waterbus dock Noordeinde":             ["rosbag2_2024_09_17-17_13_13"],
        "Waterbus dock Hooikade":               ["rosbag2_2024_09_17-18_54_49"],
        "Waterbus dock Hollandse Biesbosch":    ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"],
        "Veersteiger Boven Hardinxveld":        ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"],
        "Private dock Middeldiep":              ["rosbag2_2024_09_17-15_59_28"],
        }

    ### Define dataset splits ###
    val_sets_1 = [["Waterbus dock Rosmolenweg"],
                  ["Bunkerstation Dekker & Stam"],
                  ["Waterbus dock Noordhoek"],
                  ["Waterbus dock Noordeinde"],
                  ["Waterbus dock Hooikade"],
                  ["Veersteiger Boven Hardinxveld"]]
    test_set_1 = ["Waterbus dock Hollandse Biesbosch"]
    val_sets_2 = [["Waterbus dock Rosmolenweg"],
                  ["Bunkerstation Dekker & Stam"],
                  ["Waterbus dock Noordhoek"],
                  ["Waterbus dock Noordeinde"],
                  ["Waterbus dock Hooikade"],
                  ["Waterbus dock Hollandse Biesbosch"]]
    test_set_2 = ["Veersteiger Boven Hardinxveld"]
    val_sets_3 = [["Waterbus dock Rosmolenweg"],
                  ["Bunkerstation Dekker & Stam"],
                  ["Waterbus dock Noordhoek", "Private dock Middeldiep"],
                  ["Waterbus dock Noordeinde", "Waterbus dock Hooikade"]]
    test_set_3 = ["Waterbus dock Hollandse Biesbosch"]
    # val_sets_1 = [["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
    #             ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
    #             ["rosbag2_2024_09_17-17_04_31"],
    #             ["rosbag2_2024_09_17-17_13_13"],
    #             ["rosbag2_2024_09_17-18_54_49"],
    #             ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]]
    # test_folders_1 = ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"]
    # val_sets_2 = [["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
    #             ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
    #             ["rosbag2_2024_09_17-17_04_31"],
    #             ["rosbag2_2024_09_17-17_13_13"],
    #             ["rosbag2_2024_09_17-18_54_49"],
    #             ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"]]
    # test_folders_2 = ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]





    data_version = 3
    interval = 1
    min_label_size = 5
    distance_range = (0, 200)  # (0, 200) for all, (0, 100) for only the first two
    distance_range = None
    image_size = (640, 360)




    #############################

    all_folders = set()
    for folders in data_sets.values():
        all_folders.update(folders)
    all_folders = sorted(all_folders)

    if data_version == 1:
        val_sets = val_sets_1
        test_set = test_set_1
    elif data_version == 2:
        val_sets = val_sets_2
        test_set = test_set_2
    elif data_version == 3:
        val_sets = val_sets_3
        test_set = test_set_3


    version_name = ""
    version_name += f"({data_version})" if data_version > 1 else ""
    version_name += f"+interval-{interval}" if (interval and interval > 1) else ""
    version_name += f"+label-{min_label_size}" if (min_label_size and min_label_size > 2) else ""
    version_name += f"+distance-({distance_range[0]}-{distance_range[1]})" if (distance_range and distance_range[1] > 0) else ""


    data_config_csv = os.path.join(CONFIG_DIR, "version" + version_name + ".csv")
    with open(data_config_csv, "w") as f:
        f.write("split,train, , val, , test, \n")
        f.write(" , docks, frames, docks, frames, docks, frames\n")

        test_folders = set()
        for test_dock in test_set:
            if test_dock in data_sets.keys():
                test_folders.update(data_sets[test_dock])
            else:
                raise ValueError(f"Test dock {test_dock} not found in data_sets.")
        
        for i, val_docks in enumerate(val_sets):
            val_folders = set()
            for val_dock in val_docks:
                if val_dock in data_sets.keys():
                    val_folders.update(data_sets[val_dock])
                else:
                    raise ValueError(f"Validation dock {val_dock} not found in data_sets.")

            train_folders = set(all_folders) - val_folders - test_folders
            train_docks_names = set(data_sets.keys()) - set(val_docks) - set(test_set)

            if len(train_folders) < 1:
                print(f"Warning: No training folders left after removing validation and test folders.")
                continue
            if len(val_folders) < 1:
                print(f"Warning: No validation folders left after removing test folders.")
                continue
            if len(test_folders) < 1:
                print(f"Warning: No test folders left after removing validation folders.")
                continue

            split_name = f"split-{i+1}"
            config_name = split_name + version_name

            train_size, val_size, test_size = create_dataset_config(list(train_folders), list(val_folders), list(test_folders), config_name, interval=interval, distance_range=distance_range, min_label_size=min_label_size, image_size=image_size)
            f.write(f"{split_name}, \"{', '.join(train_docks_names)}\", {train_size}, \"{', '.join(val_docks)}\", {val_size}, \"{', '.join(test_set)}\", {test_size}\n")

    print(f"Dataset configuration CSV file created at {data_config_csv}.")



