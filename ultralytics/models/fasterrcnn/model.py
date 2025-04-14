# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.utils import DEFAULT_CFG_DICT, RANK

from .predict import FasterRCNNPredictor
from .train import FasterRCNNTrainer
from .val import FasterRCNNValidator
from ultralytics.nn.tasks import FasterRCNNDetectionModel



class FasterRCNN(Model):
    def __init__(self, model="fasterrcnn.pt", task="detect"):
        super().__init__(task=task)
        print("### INITIALIZING FasterRCNN ###")

    def _load(self, weights: str = None, task="detect") -> None:
        """
        Load a Faster R-CNN model from torchvision or a .pt checkpoint.
        """
        print("### LOADING FasterRCNN MODEL ###")

        if weights and weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            super()._load(weights, task=task)
            return

        if weights and weights.endswith(".pt") and Path(weights).exists():
            super()._load(weights, task=task)
            return

        args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model = FasterRCNNDetectionModel(nc=args.get("nc", 2), imgsz=args.get("imgsz"), verbose=(RANK == -1))

        self.ckpt = None
        self.ckpt_path = weights
        self.overrides["model"] = weights or "FasterRCNN"
        self.overrides["task"] = task

    @property
    def task_map(self) -> dict:
        """
        Map tasks to their corresponding Ultralytics pipeline components.
        """
        return {
            "detect": {
                "predictor": FasterRCNNPredictor,
                "trainer": FasterRCNNTrainer,
                "validator": FasterRCNNValidator
            }
        }
