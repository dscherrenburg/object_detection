import os
os.environ['YOLO_VERBOSE'] = 'False'
import shutil
from itertools import islice
from ultralytics import YOLO

class YOLOPredictor:
    def __init__(self, run_dir: str, conf=0.001, imgsz=640, batch_size=8):
        self.run_dir = run_dir
        self.conf_thres = conf
        self.imgsz = imgsz
        self.batch_size = batch_size

    def predict(self, model_path=None, save_images=False, save_labels=True):
        if model_path is not None:
            model = YOLO(model_path)
        elif os.path.exists(os.path.join(self.run_dir, "weights", "best.pt")):
            model = YOLO(os.path.join(self.run_dir, "weights", "best.pt"))
        else:
            raise FileNotFoundError("Model not found. Please provide a valid model path.")
        
        results_path = os.path.join(self.run_dir, "predict")
        os.makedirs(results_path, exist_ok=True)

        def batch(iterable, size):
            it = iter(iterable)
            while chunk := list(islice(it, size)):
                yield chunk

        test_images = self._load_images()
        for i, batch_images in enumerate(batch(test_images, self.batch_size * 25)):
            model.predict(batch_images, batch=self.batch_size, save=save_images, save_txt=save_labels, conf=self.conf_thres, save_conf=True, imgsz=self.imgsz, verbose=False)

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

    def _load_images(self):
        project_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.run_dir)))
        data_config_folder = os.path.join(project_folder, "dataset_configs")
        data_name = self.run_dir.split(":")[-1].split("_")[0]
        test_txt = os.path.join(data_config_folder, data_name, "test.txt")
        with open(test_txt) as f:
            test_images = [line.strip() for line in f if line.strip()]
        return test_images
