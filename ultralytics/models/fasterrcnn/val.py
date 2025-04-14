# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import numpy as np

from ultralytics.data import YOLODataset
from ultralytics.data.augment import Compose, Format, v8_transforms
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr, ops
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.ops import Profile
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode

__all__ = ("FasterRCNNValidator")  # tuple or list


from ultralytics.models.yolo.detect import DetectionValidator

class FasterRCNNValidator(DetectionValidator):
    """
    Validator for Faster R-CNN models that avoids unnecessary prediction computation
    by calling `forward_with_losses` instead of `model(images)` during validation.
    """

    def __call__(self, trainer=None, model=None):
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        self.nc = 1

        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = True
            model = trainer.model
            model = model.half()
            self.loss = np.zeros(4).astype(np.float32)  # [box, obj, cls, dfl]
            # self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            # model = AutoBackend(
            #     weights=model or self.args.model,
            #     device=select_device(self.args.device, self.args.batch),
            #     dnn=self.args.dnn,
            #     data=self.args.data,
            #     fp16=self.args.half,
            # )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        # model.validating = True
        # model.set_threshold_mode("val")
        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            print(f"\n\n\n\n\n\nValidation batch {batch}")

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            
            # Inference & Loss
            with torch.no_grad(), dt[1], dt[2]:
                images = batch.get("img")
                targets = batch.get("targets", None)
                device = self.device

                if targets is not None:
                    targets = [
                        {
                            "boxes": t["boxes"].to(device=device),
                            "labels": (t["labels"]).to(device=device),
                        } for t in targets
                    ]

                # Normalize images to [0, 1]
                images = images.to(device=device, dtype=torch.half)
                preds = model(images)
                model.train()
                losses = model(images, targets).values()
                print(f"Predictions: {preds}")
                print(f"Losses: {losses}")
                model.eval()
                if self.training:
                    self.loss += losses

            # # Postprocess
            # with torch.no_grad(), dt[3]:
            #     preds = self.postprocess(preds)

            print(f"Postprocessed predictions: {preds}")

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        model.validating = False
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            return stats
        
    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and annotations.

        Returns:
            (dict): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        print(f"cls: {cls}, bbox: {bbox}")
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        # if len(cls):
        #     bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        #     print(f"bbox: {bbox}")
        #     ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}