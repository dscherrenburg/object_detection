import os
import shutil
from ultralytics import YOLO
from itertools import islice
from torch import nn
import torch
import math

class YOLOTrainer:
    def __init__(self, run_dir, patience=10, imgsz=640, resume=True):
        self.run_dir = run_dir
        self.patience, self.imgsz = patience, imgsz
        self.resume = resume

        self.runs_dir = os.path.dirname(run_dir)
        self.project_folder = os.path.dirname(os.path.dirname(self.runs_dir))
        self.run_name = os.path.basename(run_dir)
        self.data_config_folder = os.path.join(self.project_folder, "dataset_configs")
        model_data = self._extract_data_from_runname(self.run_name)
        self.model_name = model_data['m']
        self.data_name = model_data['d']
        self.epochs = int(model_data['e'])
        self.batch_size = int(model_data['b'])

    def train(self):
        old_run_name, model_path, last_epoch = self._get_resume_info()
        try:
            # ckpt = torch.load(model_path)
            # ckpt['epoch'] = 257
            # torch.save(ckpt, model_path)
            yolo = YOLO(model_path)
            yolo.train(
                data=os.path.join(self.data_config_folder, self.data_name, "dataset.yaml"),
                epochs=self.epochs - last_epoch, resume=self.resume,
                batch=self.batch_size, imgsz=self.imgsz, device=[0, 1],
                project=self.runs_dir, name=self.run_name, patience=self.patience,
                verbose=False, half=True, max_det=100,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving progress...")
            last_weights = os.path.join(self.run_dir, "weights", "last.pt")
            if os.path.exists(last_weights):
                print(f"Progress saved in {last_weights}. You can resume training later.")
            else:
                print("No checkpoint found. Consider resuming from the latest available model.")
            if old_run_name:
                self._merge_results(old_run_name)
    
        if old_run_name:
            self._merge_results(old_run_name)
        best_model_path = os.path.join(self.run_dir, "weights", "best.pt")
        return best_model_path
    
    def _get_resume_info(self):
        # Get the default model path
        self.model_path = os.path.join(self.project_folder, "YOLO", "models", self.model_name.lower() + ".pt")
        if not os.path.exists(self.model_path):
            self.model_path = self.model_name.lower() + ".pt"

        # Check if we should resume training
        if not self.resume:
            if os.path.exists(self.run_dir):
                inpt = input(f"Run {self.run_name} already exists. Press Enter to overwrite it or type 'new' to start a new run.\n")
                if inpt.lower() == "new":
                    print("Creating a new run.")
                else:
                    shutil.rmtree(self.run_dir)
                    print("Overwriting existing run.")
            else:
                print("Starting fresh.")
            return None, self.model_path, 0
        
        # Check if the run already exists
        if os.path.exists(self.run_dir):
            last_weights = os.path.join(self.run_dir, "weights", "last.pt")
            if os.path.exists(last_weights):
                print(f"Resuming training from {last_weights}.")
                return None, last_weights, 0
            else:
                print("No checkpoint found. Starting fresh.")
                shutil.rmtree(self.run_dir)
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
    
