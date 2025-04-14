# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
from pathlib import Path

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes


class FasterRCNNPredictor(BasePredictor):
    """
    Predictor class for the FasterRCNN model, extending the BasePredictor class.
    This class is responsible for making predictions using the FasterRCNN model and post-processing the results.
    It handles the conversion of model outputs into a format suitable for further analysis and visualization.
    """

    def postprocess(self, preds) -> list[Results]:
        """
        Postprocesses predictions from the FasterRCNN model.
        Only converts torchvision model outputs into Ultralytics Results.

        Args:
            preds (list[dict]): List of prediction dictionaries from the model.

        Returns:
            list[Results]: List of processed results.
        """
        results = []
        for i, pred in enumerate(preds):
            pred = {k: v.cpu() for k, v in pred.items()}

            path = self.batch[i].get('im_file', None) if self.batch else None
            name = Path(path).name if path else f'image{i}'
            orig_img = self.batch[i]['im0'] if self.batch else None

            result = Results(orig_img=orig_img, path=path, names=self.model.names)
            result.boxes = Boxes(pred)
            results.append(result)

        return results