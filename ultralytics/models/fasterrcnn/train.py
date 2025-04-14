# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import math
import torch.distributed as dist
import torch.nn as nn
from datetime import datetime
from copy import deepcopy, copy

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import FasterRCNNDetectionModel
from ultralytics.utils import RANK, LOGGER, LOCAL_RANK, __version__, callbacks, ops
from ultralytics.utils.checks import check_imgsz, check_amp
from ultralytics.utils.torch_utils import torch_distributed_zero_first, EarlyStopping, convert_optimizer_state_dict_to_fp16
from ultralytics.data import build_dataloader, build_fasterrcnn_dataset
from ultralytics.utils.plotting import plot_images

from .val import FasterRCNNValidator


class FasterRCNNTrainer(DetectionTrainer):
    """
    Trainer class for the FasterRCNN model, extending the DetectionTrainer class.
    This class is responsible for training the FasterRCNN model on a given dataset.
    """

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """
        Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = build_fasterrcnn_dataset(self.args, dataset_path, batch_size, self.data, mode=mode, rect=mode == "val")
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return a FasterRCNN model for object detection tasks.
        """
        print("###  GETTING FasterRCNN MODEL FROM TRAINER  ###")
        model = FasterRCNNDetectionModel(weights=weights, args=self.args, ch=3, verbose=verbose and RANK == -1)
        return model
    
    def _setup_train(self, world_size):
        """Build dataloaders and optimizer on correct rank process."""
        # Model
        self.run_callbacks("on_pretrain_routine_start")
        print("###  SETTING UP TRAINER  ###")
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        # # Freeze layers
        # freeze_list = (
        #     self.args.freeze
        #     if isinstance(self.args.freeze, list)
        #     else range(self.args.freeze)
        #     if isinstance(self.args.freeze, int)
        #     else []
        # )
        # always_freeze_names = [".dfl"]  # always freeze these layers
        # freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        # self.freeze_layer_names = freeze_layer_names
        # for k, v in self.model.named_parameters():
        #     # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        #     if any(x in k for x in freeze_layer_names):
        #         LOGGER.info(f"Freezing layer '{k}'")
        #         v.requires_grad = False
        #     elif not v.requires_grad and v.dtype.is_floating_point:  # only floating point Tensor can require gradients
        #         LOGGER.info(
        #             f"WARNING ⚠️ setting 'requires_grad=True' for frozen layer '{k}'. "
        #             "See ultralytics.engine.trainer for customization of frozen layers."
        #         )
        #         v.requires_grad = True

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()

        # Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size, rank=-1, mode="val"
            )
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            # self.ema = ModelEMA(self.model)
            if self.args.plots:
                self.plot_training_labels()

        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        # Scheduler
        self._setup_scheduler()
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self.run_callbacks("on_pretrain_routine_end")

    def _model_train(self):
        """Set model in training mode and freeze backbone if needed."""
        self.model.train()  # Set entire model to training mode

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        checkpoint = {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": deepcopy(self.model).half() if not self.ema else None,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            }
        
        if self.ema:
            checkpoint["ema"] = deepcopy(self.ema.ema).half()  # save ema model
            checkpoint["updates"] = self.ema.updates  # save ema updates
        
        torch.save(checkpoint, buffer)
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        # if self.args.close_mosaic and self.epoch == (self.epochs - self.args.close_mosaic - 1):
        #    (self.wdir / "last_mosaic.pt").write_bytes(serialized_ckpt)  # save mosaic checkpoint

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "obj_loss", "rpn_loss"
        return FasterRCNNValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
    
    def plot_training_samples(self, batch, ni):
        """
        Plot training samples with their annotations.

        Args:
            batch (dict): Dictionary containing batch data.
            ni (int): Number of iterations.
        """
        bboxes = ops.xyxy2xywh(batch["bboxes"])  # convert xyxy to xywh
        plot_images(
            images=batch["img"].cpu(),
            batch_idx=batch["batch_idx"].cpu(),
            cls=batch["cls"].squeeze(-1).cpu(),
            bboxes=bboxes.cpu(),
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )