import os
import csv

from YOLO.train import YOLOTrainer
from YOLO.predict import YOLOPredictor
from Faster_RCNN.train import FasterRCNNTrainer
from Faster_RCNN.predict import FasterRCNNPredictor
from evaluate import evaluation


def save_metrics_to_csv(model, run_name, metrics, csv_path):
    """Saves the evaluation metrics to a CSV file."""
    fieldnames = ['model', 'data', 'epochs', 'batch_size'] + list(metrics.keys())
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
    
    batch_size = run_name.split("_")[1][2:]
    data = run_name.split(":")[-1]

    with open(csv_path, 'a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'model': model,
            'data': data,
            'epochs': epochs,
            'batch_size': batch_size,
            **metrics
        })


def main(project_folder, model, data_name, epochs, patience, imgsz, train, resume, predict, test):
    if model.startswith("YOLO"):
        if model =="YOLO":
            model = "YOLO11n"
        batch_size = 8
        run_name = f"m:{model}_e:{epochs}_b:{batch_size}_d:{data_name}"
        if train:
            trainer = YOLOTrainer(project_folder=project_folder,
                                  model_name=model,
                                  data_name=data_name,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  imgsz=imgsz,
                                  patience=patience,
                                  resume=resume)
            trainer.train()
        if predict:
            predictor = YOLOPredictor(project_folder=project_folder,
                                      run_name=run_name,
                                      conf_thres=0.01,
                                      imgsz=imgsz,
                                      batch_size=batch_size)
            predictor.predict()

    elif model == "Faster_RCNN":
        batch_size = 2
        run_name = f"e:{epochs}_b:{batch_size}_d:{data_name}"
        if train:
            trainer = FasterRCNNTrainer(project_folder=project_folder,
                                        data_name=data_name,
                                        epochs=epochs,
                                        patience=patience,
                                        imgsz=imgsz,
                                        batch_size=batch_size,
                                        resume=resume)
            trainer.train()
        if predict:
            predictor = FasterRCNNPredictor(project_folder=project_folder,
                                            run_name=run_name,
                                            conf_thres=0.01,
                                            imgsz=imgsz,
                                            batch_size=batch_size)
            predictor.predict(save_images=True, save_labels=True)
    else:
        raise ValueError("Model must be either 'YOLO[version]' or 'Faster_RCNN'.")
    
    if test:
        print(f"\n--- Evaluating {model} ---\n")
        model_folder = "YOLO" if model.startswith("YOLO") else model
        run_dir = os.path.join(project_folder, model_folder, "runs", run_name)
        metrics = evaluation(run_dir=run_dir, iou_threshold=0.5, create_plots=True)
        
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"   - {key}: {value:.4f}")
        
        # Save metrics to CSV
        csv_path = os.path.join(project_folder, "evaluation_metrics.csv")
        save_metrics_to_csv(model, run_name, metrics, csv_path)



if __name__ == "__main__":
    project_folder = "/home/daan/object_detection/"
    model = "YOLO11n"                                          # 'YOLO[version]' or 'Faster_RCNN'
    # model = "Faster_RCNN"
    data_name = "split-1"                       # Must be a a dataset in the project_folder/dataset_configs folder
    epochs = 300
    patience = epochs // 2
    # patience = epochs
    imgsz = 640
    train = False
    resume = False
    predict = True
    test = True


    main(project_folder, model, data_name, epochs, patience, imgsz, train, resume, predict, test)