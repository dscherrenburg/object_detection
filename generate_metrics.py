import os
import glob
import argparse
import numpy as np
import csv
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from evaluate import test

    
def plot_precision_recall_curve(all_labels, all_scores):
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.show()

def save_metrics_to_csv(model, run_name, metrics, csv_path):
    """Saves the evaluation metrics to a CSV file."""
    if not os.path.exists(csv_path):
        fieldnames = ['model', 'data', 'epochs', 'batch_size'] + list(metrics.keys())
        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    epochs = run_name.split("_")[0][2:]
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

if __name__ == "__main__":
    project_dir = "/home/daan/object_detection/"

    dataset_name = "split_2_interval_10"
    run_name = f"e:100_b:2_data:{dataset_name}"
    model = "Faster-RCNN"

    dataset_name = "split_2"
    run_name = f"e:100_b:8_data:{dataset_name}"
    model = "YOLO"
    
    run_dir = os.path.join(project_dir, model, "runs", run_name)
    metrics = test(run_dir=run_dir, create_plots=False)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics to CSV
    csv_path = os.path.join(project_dir, "evaluation_metrics.csv")
    save_metrics_to_csv(model, run_name, metrics, csv_path)
    
    # plot_precision_recall_curve(all_labels, all_scores)
