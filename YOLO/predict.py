import os
os.environ['YOLO_VERBOSE'] = 'False'
import shutil
from itertools import islice
from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, project_folder: str, run_name: str, conf_thres=0.01, imgsz=640, batch_size=8):
        print("\n--- Predicting YOLO ---\n")
        self.conf_thres = conf_thres
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.run_dir = os.path.join(project_folder, "YOLO", "runs", run_name)
        self.data_config_folder = os.path.join(project_folder, "dataset_configs")
        self.data_name = run_name.split(":")[-1]

    def predict(self, model_path=None):
        if model_path is not None:
            model = YOLO(model_path)
        elif os.path.exists(os.path.join(self.run_dir, "weights", "best.pt")):
            model = YOLO(os.path.join(self.run_dir, "weights", "best.pt"))
        else:
            raise FileNotFoundError("Model not found. Please provide a valid model path.")
        
        test_txt = os.path.join(self.data_config_folder, self.data_name, "test.txt")
        results_path = os.path.join(self.run_dir, "predict")
        os.makedirs(results_path, exist_ok=True)

        with open(test_txt) as f:
            test_images = [line.strip() for line in f if line.strip()]

        def batch(iterable, size):
            it = iter(iterable)
            while chunk := list(islice(it, size)):
                yield chunk

        for i, batch_images in enumerate(batch(test_images, self.batch_size * 10)):
            model.predict(batch_images, batch=self.batch_size, save=True, save_txt=True, conf=self.conf_thres, save_conf=True, imgsz=self.imgsz, verbose=False)

        self._move_results(results_path)
        return results_path

    def _move_results(self, results_path):
        if os.path.exists("./runs/detect"):
            dirs = os.listdir("./runs/detect")
            dirs = [d for d in dirs if d.startswith("predict")]
            if not dirs:
                print("No predictions found.")
                return
            # Get the dir with the latest timestamp
            latest_dir = max(dirs, key=lambda x: os.path.getmtime(os.path.join("./runs/detect", x)))
            default_path = os.path.join("./runs", "detect", latest_dir)
            for root, _, files in os.walk(default_path):
                for file in files:
                    dst_path = os.path.join(results_path, os.path.relpath(os.path.join(root, file), default_path))
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(os.path.join(root, file), dst_path)
            shutil.rmtree("runs")
        else:
            print("No predictions found.")
        print("Predictions saved in", results_path)