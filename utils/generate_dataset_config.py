import os
from pathlib import Path

# Define dataset root
DATASET_ROOT = "/home/daan/Data/dock_data"
IMAGES_DIR = os.path.join(DATASET_ROOT, "images")
LABELS_DIR = os.path.join(DATASET_ROOT, "labels")
CONFIG_DIR = "dataset_configs"

def get_valid_folders():
    """Finds folders that have both an image and an annotation."""
    image_folders = {f.name for f in Path(IMAGES_DIR).iterdir() if f.is_dir()}
    annotation_folders = {f.name for f in Path(LABELS_DIR).iterdir() if f.is_dir()}
    return sorted(image_folders & annotation_folders)

def create_dataset_config(train_folders, val_folders, test_folders, config_name, interval=1):
    """Generates dataset configuration files."""
    config_path = os.path.join(CONFIG_DIR, config_name)
    os.makedirs(config_path, exist_ok=True)

    # Helper function to write file paths
    def write_split_file(split_name, folders, interval=interval):
        split_file = os.path.join(config_path, f"{split_name}.txt")
        with open(split_file, "w") as f:
            for folder in folders:
                image_folder = os.path.join(IMAGES_DIR, folder)
                for img in sorted(os.listdir(image_folder))[::interval]:
                    if img.endswith(".png" or ".jpg" or ".jpeg"):
                        f.write(os.path.join(image_folder, img) + "\n")

    # Generate train, val, and test txt files
    write_split_file("train", train_folders, interval)
    write_split_file("val", val_folders, interval)
    write_split_file("test", test_folders, interval=1)

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

if __name__ == "__main__":
    valid_folders = get_valid_folders()

    ### Define dataset splits ###
    val_sets = [["rosbag2_2024_09_17-16_34_11", "rosbag2_2024_09_17-19_32_52", "rosbag2_2024_11_18-12_55_24"],
                ["rosbag2_2024_09_17-15_33_18", "rosbag2_2024_09_17-15_37_39", "rosbag2_2024_09_17-15_40_51"],
                ["rosbag2_2024_09_17-17_04_31"],
                ["rosbag2_2024_09_17-17_13_13"],
                ["rosbag2_2024_09_17-18_54_49"],
                ["rosbag2_2024_09_17-16_02_29", "rosbag2_2024_09_17-19_47_16"]]
    test_folders = ["rosbag2_2024_09_17-14_40_19", "rosbag2_2024_09_17-14_48_48", "rosbag2_2024_09_17-14_58_53", "rosbag2_2024_09_17-20_27_46"]
    interval = 1
    #############################

    for i, val_folders in enumerate(val_sets):
        name = f"split-{i+1}+interval-{interval}(2)" if interval > 1 else f"split-{i+1}(2)"
        val_folders = set(val_folders) & set(valid_folders)
        test_folders = set(test_folders) & set(valid_folders)
        train_folders = set(valid_folders) - val_folders - test_folders

        create_dataset_config(list(train_folders), list(val_folders), list(test_folders), name, interval=interval)
        exit()
