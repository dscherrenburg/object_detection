import os
import shutil
from ultralytics import YOLO
from itertools import islice

class YOLOTrainer:
    def __init__(self, project_folder, model_name, data_name, epochs=50, patience=10, batch_size=8, imgsz=640, resume=True, single_cls=True):
        print("\n--- Training YOLO ---\n")
        self.project_folder, self.model_name, self.data_name = project_folder, model_name, data_name
        self.epochs, self.patience, self.batch_size, self.imgsz = epochs, patience, batch_size, imgsz
        self.resume, self.single_cls = resume, single_cls
        self.data_config_folder = os.path.join(project_folder, "dataset_configs")
        self.runs_dir = os.path.join(project_folder, "YOLO", "runs")
        self.run_name = f"m:{model_name}_e:{epochs}_b:{batch_size}_d:{data_name}"

    def train(self):
        old_run_name, model_path, last_epoch = self._get_resume_info()
        try:
            YOLO(model_path).train(
                data=os.path.join(self.data_config_folder, self.data_name, "dataset.yaml"),
                epochs=self.epochs - last_epoch,
                batch=self.batch_size, imgsz=self.imgsz, device=[0],
                project=self.runs_dir, name=self.run_name, resume=self.resume, patience=self.patience,
                verbose=True, single_cls=self.single_cls, half=True
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving progress...")
            last_weights = os.path.join(self.runs_dir, self.run_name, "weights", "last.pt")
            if os.path.exists(last_weights):
                print(f"Progress saved in {last_weights}. You can resume training later.")
            else:
                print("No checkpoint found. Consider resuming from the latest available model.")
            if old_run_name:
                self._merge_results(old_run_name)
    
        if old_run_name:
            self._merge_results(old_run_name)
        best_model_path = os.path.join(self.runs_dir, self.run_name, "weights", "best.pt")
        return best_model_path
    
    def _get_resume_info(self):
        # Get the default model path
        self.model_path = os.path.join(self.project_folder, "YOLO", "models", self.model_name.lower() + ".pt")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model {self.model_name} not found.")

        # Check if we should resume training
        if not self.resume:
            print("Starting fresh.")
            return None, self.model_path, 0
        
        # Check if the run already exists
        self.run_dir = os.path.join(self.runs_dir, self.run_name)
        if os.path.exists(self.run_dir):
            last_weights = os.path.join(self.run_dir, "weights", "last.pt")
            if os.path.exists(last_weights):
                print(f"Resuming training from {last_weights}.")
                return None, last_weights, 0
            else:
                print("No checkpoint found. Starting fresh.")
                self.resume = False
                return None, self.model_path, 0
            
        # Check if we should resume from an existing run
        best_old_run, largest_epoch = None, 0
        for dir in os.listdir(self.runs_dir):
            run_data = self._extract_data_from_runname(dir)
            if run_data['m'] == self.model_name and run_data['d'] == self.data_name and int(run_data['b']) == self.batch_size and int(run_data['e']) < self.epochs:
                if int(run_data['e']) > largest_epoch:
                    largest_epoch, best_old_run = int(run_data['e']), dir
        if best_old_run:
            results_file = os.path.join(self.runs_dir, best_old_run, "results.csv")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    last_epoch = int(f.readlines()[-1].split(",")[0])
                print(f"Resuming from {best_old_run} at epoch {last_epoch}.")
                self.resume = False
                return best_old_run, os.path.join(self.runs_dir, best_old_run, "weights", "last.pt"), last_epoch
        
        print("No checkpoint found. Starting fresh.")
        self.resume = False
        return None, self.model_path, 0
    
    def _merge_results(self, old_run_name):
        old_file, new_file = [os.path.join(self.runs_dir, run, "results.csv") for run in [old_run_name, self.run_name]]
        if not all(map(os.path.exists, [old_file, new_file])):
            return
        
        with open(old_file) as f_old, open(new_file) as f_new:
            old_results, new_results = f_old.readlines(), f_new.readlines()
        
        last_epoch = int(old_results[-1].split(",")[0])
        last_time = old_results[-1].split(",")[1]
        adjusted_new = [f"{int(line.split(',')[0]) + last_epoch}, {line.split(',')[1] + last_time},{','.join(line.split(',')[2:])}" for line in new_results[1:]]
        
        with open(new_file, "w") as f_new:
            f_new.writelines(old_results + adjusted_new)
    
    def _extract_data_from_runname(self, run_name):
        run_name_items = [item.split(":") for item in run_name.split("_")]
        run_name_items = [item for item in run_name_items if len(item) == 2]
        return {key: value for key, value in run_name_items}
    


if __name__ == "__main__":
    project_folder = "/home/daan/object_detection/"
    data_name = "split_1_interval_10"
    epochs = 1
    batch_size = 8
    imgsz = 640
    train = True
    test = True
    resume = True
    confidence_threshold = 0.2
    
    trainer = YOLOTrainer(project_folder, data_name, epochs, batch_size, imgsz, resume, confidence_threshold)
    
    trainer.train() if train else None
