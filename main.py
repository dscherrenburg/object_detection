import os
import csv

from YOLO.train import YOLOTrainer
from YOLO.predict import YOLOPredictor
from Faster_RCNN.train import FasterRCNNTrainer
from Faster_RCNN.predict import FasterRCNNPredictor
from Evaluator import Evaluator, evaluate


def save_metrics_to_csv(model, run_name, metrics, csv_path):
    """Saves the evaluation metrics to a CSV file."""
    fieldnames = ['model', 'data', 'epochs', 'batch size', 'best epoch'] + list(metrics.keys())
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    model_folder = "YOLO" if model.startswith("YOLO") else model
    run_csv = os.path.join(os.path.dirname(csv_path), model_folder, "runs", run_name, "results.csv")
    if not os.path.exists(run_csv):
        epochs = run_name.split("_")[0][2:]
    else:
        with open(run_csv, 'r') as f:
            epochs = int(f.readlines()[-1].split(",")[0])
    
    batch_size = run_name.split("_")[2][2:]
    data = run_name.split(":")[-1]

    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
            
        writer.writerow({
            'model': model,
            'data': data,
            'epochs': epochs,
            'batch size': batch_size,
            'best epoch': '',
            **metrics
        })


def main(project_folder, run_name, patience, imgsz, train=False, resume=False, test=False):
    model = run_name.split("_")[0].split(":")[1]
    batch_size = int(run_name.split("_")[2].split(":")[1])

    if model.upper().startswith("YOLO"):
        trainer = YOLOTrainer
        predictor = YOLOPredictor
        run_dir = os.path.join(project_folder, "YOLO", "runs", run_name)

    elif model.upper() == "FRCNN":
        trainer = FasterRCNNTrainer
        predictor = FasterRCNNPredictor
        run_dir = os.path.join(project_folder, "Faster_RCNN", "runs", run_name)

    else:
        raise ValueError("Model must be either 'YOLO[version]' or 'FRCNN'.")
    
    # Train the model
    if train:
        print(f"\n--- Training {model} ---\n")
        trainer = trainer(run_dir, patience, imgsz, resume)
        trainer.train()
    
    if test:
        print(f"\n--- Testing {model} ---\n")
        predictor = predictor(run_dir, conf=0.01, imgsz=imgsz, batch_size=batch_size)
        predictor.predict(model_path=None, save_images=False, save_labels=True)
        metrics = evaluate(run_dir, conf=0.001, iou=0.5, save_path=run_dir)

        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  - {key}: {value}")

        # Save metrics to CSV
        csv_path = os.path.join(project_folder, "evaluation_metrics.csv")
        save_metrics_to_csv(model, run_name, metrics, csv_path)



if __name__ == "__main__":
    train = False
    resume = False
    test = False


    project_folder = "/home/daan/object_detection/"
    # model = "YOLO11n"                                          # 'YOLO[version]' or 'Faster_RCNN'
    model = "FRCNN"
    data_name = "split-1(3)+interval-5+distance-(0-200)"                       # Must be a a dataset in the project_folder/dataset_configs folder
    epochs = 100
    patience = epochs // 2
    batch_size = 8
    imgsz = 640
    train = True
    resume = True
    test = True



    if model.upper() == "YOLO":
        print("Using default YOLO11n model.")
        model = "YOLO11n"

    
    # run_name = f"m:{model}_e:{epochs}_b:{batch_size}_d:{data_name}"
    # main(project_folder, run_name, patience, imgsz, train, resume, test)
    # exit()

    # for i in range(1, 5):
    #     data_name = f"split-{i}(3)+label-5"
    #     print(f"\n--- Running {model} on {data_name} ---\n")
    #     run_name = f"m:{model}_e:{epochs}_b:{batch_size}_d:{data_name}"
    #     main(project_folder, run_name, patience, imgsz, train, resume, test)
    
    model = "YOLO11n"                                          # 'YOLO[version]' or 'Faster_RCNN'
    for i in range(2, 5):
        data_name = f"split-{i}(3)+label-5"
        print(f"\n--- Running {model} on {data_name} ---\n")
        run_name = f"m:{model}_e:{epochs}_b:{batch_size}_d:{data_name}"
        main(project_folder, run_name, patience, imgsz, train, resume, test)