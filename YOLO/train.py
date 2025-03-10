import os
import shutil
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, project_folder, data_name, epochs=50, batch_size=8, imgsz=640, resume=True, conf_thres=0.5):
        self.project_folder, self.data_name = project_folder, data_name
        self.epochs, self.batch_size, self.imgsz = epochs, batch_size, imgsz
        self.resume, self.conf_thres = resume, conf_thres
        self.data_config_folder = os.path.join(project_folder, "dataset_configs")
        self.runs_dir = os.path.join(project_folder, "YOLO/runs")
        self.run_name = f"e:{epochs}_b:{batch_size}_data:{data_name}"
        self.model_path = os.path.join(project_folder, "YOLO", "yolo11s.pt")
        self.best_model = None      

    def train(self):
        old_run_name, model_path, last_epoch = self._get_resume_info()
        YOLO(model_path).train(
            data=os.path.join(self.data_config_folder, self.data_name, "dataset.yaml"),
            epochs=self.epochs - last_epoch,
            batch=self.batch_size, imgsz=self.imgsz, device=[0],
            project=self.runs_dir, name=self.run_name, resume=self.resume
        )
        if old_run_name:
            self._merge_results(old_run_name)
        self.best_model_path = os.path.join(self.runs_dir, self.run_name, "weights", "best.pt")
        self.best_model = YOLO(self.best_model_path)
        return self.best_model_path

    def predict(self, img_path, model_path=None):
        if model_path is not None:
            model = YOLO(model_path)
        elif self.best_model is not None:
            model = self.best_model
        elif self.best_model_path is not None:
            model = YOLO(self.best_model_path)
        elif os.path.exists(os.path.join(self.runs_dir, self.run_name, "weights", "best.pt")):
            model = YOLO(os.path.join(self.runs_dir, self.run_name, "weights", "best.pt"))
        else:
            raise ValueError("No model found.")
        
        prediction = model.predict(img_path, save=False, save_txt=False, conf=self.conf_thres, imgsz=self.imgsz)
        return []

    def test(self, model_path=None):
        model = YOLO(model_path or os.path.join(self.runs_dir, self.run_name, "weights", "best.pt"))
        test_txt = os.path.join(self.data_config_folder, self.data_name, "test.txt")
        results_path = os.path.join(self.runs_dir, self.run_name, "predict")
        os.makedirs(results_path, exist_ok=True)
        
        with open(test_txt) as f:
            test_images = [line.strip() for line in f if line.strip()]
        
        for img in test_images:
            model.predict(img, save=True, save_txt=True, conf=0.01, save_conf=True, imgsz=self.imgsz)
        self._move_results(results_path)

    def _get_resume_info(self):
        last_weights = os.path.join(self.runs_dir, self.run_name, "weights", "last.pt")
        if not self.resume or not os.path.exists(self.runs_dir):
            return None, self.model_path, 0
        
        if os.path.exists(last_weights):
            print("Resuming training from last.pt")
            return None, last_weights, 0
        
        existing_run = os.path.join(self.runs_dir, self.run_name)
        if os.path.exists(existing_run):
            print("Resuming training from existing run.")
            shutil.rmtree(existing_run)
            return None, self.model_path, 0
        
        best_old_run, largest_epoch = None, 0
        for dir in os.listdir(self.runs_dir):
            if dir.split("_")[1:] == self.run_name.split("_")[1:]:
                try:
                    epoch_num = int(dir.split("_")[0][2:])
                    if epoch_num > largest_epoch:
                        largest_epoch, best_old_run = epoch_num, dir
                except ValueError:
                    continue
        
        if best_old_run:
            results_file = os.path.join(self.runs_dir, best_old_run, "results.csv")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    last_epoch = int(f.readlines()[-1].split(",")[0])
                print(f"Resuming from {best_old_run} at epoch {last_epoch}.")
                return best_old_run, os.path.join(self.runs_dir, best_old_run, "weights", "last.pt"), last_epoch
        
        print("No previous run found. Starting fresh.")
        return None, self.model_path, 0
    
    def _merge_results(self, old_run_name):
        old_file, new_file = [os.path.join(self.runs_dir, run, "results.csv") for run in [old_run_name, self.run_name]]
        if not all(map(os.path.exists, [old_file, new_file])):
            return
        
        with open(old_file) as f_old, open(new_file) as f_new:
            old_results, new_results = f_old.readlines(), f_new.readlines()
        
        last_epoch = int(old_results[-1].split(",")[0])
        adjusted_new = [f"{int(line.split(',')[0]) + last_epoch + 1},{','.join(line.split(',')[1:])}" for line in new_results[1:]]
        
        with open(new_file, "w") as f_new:
            f_new.writelines(old_results + adjusted_new)

    def _move_results(self, results_path):
        default_predict = os.path.join("runs", "detect", "predict")
        if os.path.exists(default_predict):
            for root, _, files in os.walk(default_predict):
                for file in files:
                    dst_path = os.path.join(results_path, os.path.relpath(os.path.join(root, file), default_predict))
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.move(os.path.join(root, file), dst_path)
            shutil.rmtree("runs")
        else:
            print("No predictions found.")
        print("Predictions saved in", results_path)

if __name__ == "__main__":
    project_folder = "/home/daan/object_detection/"
    data_name = "split_2"
    epochs = 100
    batch_size = 8
    imgsz = 640
    train = False
    test = True
    resume = True
    confidence_threshold = 0.2
    
    trainer = YOLOTrainer(project_folder, data_name, epochs, batch_size, imgsz, resume, confidence_threshold)
    
    trainer.train() if train else None
    trainer.test() if test else None
