from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.tensor import Tensor

from cloud.utils import build_object, load_metrics


class Cloud(pl.LightningModule):
    """Class for training segmentation models"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg

        self.model = build_object(self.cfg["model"])
        self.criterion = build_object(self.cfg["criterion"])

        self.metrics = load_metrics(self.cfg["metrics"])

    def configure_optimizers(self):
        optimizer = build_object(self.cfg["optimizer"], params=self.model.parameters())

        scheduler = build_object(self.cfg["scheduler"], optimizer=optimizer)

        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:

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

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:

        x = batch["chip"]
        y = batch["label"].long()

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
