import argparse
import os
from YOLO.train import YOLOTrainer
from Faster_RCNN.train import FasterRCNNTrainer

def main():
    parser = argparse.ArgumentParser(description="Train and test object detection models.")
    parser.add_argument("--model", type=str, choices=["yolo", "faster_rcnn"], required=True, help="Choose the model to train: 'yolo' or 'faster_rcnn'")
    parser.add_argument("--project_folder", type=str, default="/home/daan/object_detection/", help="Path to the project folder")
    parser.add_argument("--data_name", type=str, default="split_3", help="Dataset name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--test", action="store_true", help="Run testing after training")
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for testing")
    
    args = parser.parse_args()
    
    if args.model == "yolo":
        trainer = YOLOTrainer(
            project_folder=args.project_folder,
            data_name=args.data_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            resume=args.resume,
            confidence_threshold=args.confidence_threshold,
        )
    elif args.model == "faster_rcnn":
        trainer = FasterRCNNTrainer(
            project_folder=args.project_folder,
            data_name=args.data_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            resume=args.resume,
            confidence_threshold=args.confidence_threshold,
        )
    
    if args.train:
        trainer.train()
    
    if args.test:
        trainer.test()

if __name__ == "__main__":
    main()
