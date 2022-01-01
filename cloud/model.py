from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.tensor import Tensor

from cloud.batch_augs import CutMix
from cloud.ema import ModelEMA
from cloud.utils import build_object, load_metrics


class Cloud(pl.LightningModule):
    """Class for training segmentation models"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg

        self.model = build_object(self.cfg["model"])
        self.criterion = build_object(self.cfg["criterion"])

        self.metrics = load_metrics(self.cfg["metrics"])

        # self.cutmix = build_object(cfg["augmentation"]["cutmix"])

        # if self.cfg["experiment"]["ema"]:
        #     self.ema = ModelEMA(self.model)
        #     self.ema.to(self.cfg["experiment"]["device"])
        # else:
        #     self.ema = None

    def on_fit_start(self) -> None:
        if self.cfg["experiment"]["ema"]:
            self.ema = ModelEMA(self.model)
        else:
            self.ema = None

    def configure_optimizers(self):
        optimizer = build_object(self.cfg["optimizer"], params=self.model.parameters())

        if "scheduler" in self.cfg.keys():
            scheduler = build_object(self.cfg["scheduler"], optimizer=optimizer)

            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            return {"optimizer": optimizer}

        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

        # mixed_images, mixed_targets = self.cutmix(batch["chip"], batch["label"])
        # x = mixed_images
        # y = mixed_targets.long()

        x = batch["chip"]
        y = batch["label"].long()

        out = self(x)

        train_loss = self.criterion(out, y)

        with torch.no_grad():
            for name, metric in self.metrics.items():
                metric_value = metric(out, y)

                self.log(
                    f"train_{name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return train_loss

    def training_step_end(self, batch_parts):
        if self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        x = batch["chip"]
        y = batch["label"].long()

        if self.ema is not None:
            out = self.ema(x)
        else:
            out = self(x)
        val_loss = self.criterion(out, y)

        with torch.no_grad():
            for name, metric in self.metrics.items():
                metric_value = metric(out, y)

                self.log(
                    f"val_{name}",
                    metric_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
