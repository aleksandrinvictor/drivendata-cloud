import os
import random
import warnings
from typing import Any, Dict, List, Optional

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from cloud.utils import load_augs

warnings.filterwarnings("ignore")


def minmax_scale(x):
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def minmax_scale_percentile(x):
    return (x - np.nanpercentile(x, 2)) / (np.nanpercentile(x, 98) - np.nanpercentile(x, 2))


def scale_std(x):
    return (x - (np.nanmean(x) - np.nanstd(x) * 2)) / (
        (np.nanmean(x) + np.nanstd(x) * 2) - (np.nanmean(x) - np.nanstd(x) * 2)
    )


class CloudDataset(Dataset):
    """Reads in images, transforms pixel values, and serves a
    dictionary containing chip ids, image tensors, and
    label masks (where available).
    """

    def __init__(
        self,
        x_paths: pd.DataFrame,
        y_paths: Optional[pd.DataFrame] = None,
        transforms: Optional[A.Compose] = None,
        bands: Optional[List[str]] = None,
    ):
        """
        Instantiate the CloudDataset class.
        """
        self.data = x_paths
        self.label = y_paths
        self.transforms = transforms
        self.bands: List[str] = bands or ["B02", "B03", "B04", "B08"]

    def __len__(self):
        return len(self.data)

    def _load_image(self, idx: int) -> np.ndarray:
        img_paths = self.data.loc[idx]

        with rasterio.open(img_paths["B02_path"]) as b:
            blue = b.read(1).astype("float32")

        with rasterio.open(img_paths["B03_path"]) as b:
            green = b.read(1).astype("float32")

        with rasterio.open(img_paths["B04_path"]) as b:
            red = b.read(1).astype("float32")

        blue = minmax_scale_percentile(blue)
        green = minmax_scale_percentile(green)
        red = minmax_scale_percentile(red)

        return np.stack([red, green, blue], axis=2).astype("float32")

    def _load_nir(self, idx: int) -> np.ndarray:
        img_paths = self.data.loc[idx]
        with rasterio.open(img_paths["B08_path"]) as b:
            nir = b.read(1).astype("float32")

        nir = minmax_scale_percentile(nir)

        return nir

    def _load_mask(self, idx: int) -> np.ndarray:
        assert isinstance(self.label, pd.DataFrame)

        label_path = self.label.loc[idx].label_path
        with rasterio.open(label_path) as lp:
            y_arr = lp.read(1).astype("float32")

        y_arr = np.expand_dims(y_arr, -1)
        return y_arr

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        # Loads an n-channel image from a chip-level dataframe
        image = self._load_image(idx)

        if self.label is None:
            mask = None
        else:
            mask = self._load_mask(idx)

        if self.transforms is not None:

            if self.label is None:
                augmented = self.transforms(image=image)
            else:
                augmented = self.transforms(image=image, mask=mask)
                mask = augmented["mask"]

            image = augmented["image"]

        item = {"chip_id": self.data.loc[idx].chip_id, "chip": image}

        if self.label is not None:
            item["label"] = mask

        return item


class GridDataset(CloudDataset):
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:

        indices = [idx] + random.choices(range(len(self)), k=3)

        img_grid = torch.zeros(3, 256, 256)
        mask_grid = torch.zeros(1, 256, 256)

        yc = random.randint(16, 240)
        xc = random.randint(16, 240)

        start_row_indexes = np.array([0, 0, yc, yc])
        end_row_indexes = np.array([yc, yc, 256, 256])

        start_col_indexes = np.array([0, xc, 0, xc])
        end_col_indexes = np.array([xc, 256, xc, 256])

        len_x = end_row_indexes - start_row_indexes
        len_y = end_col_indexes - start_col_indexes

        for i in range(4):
            item = super().__getitem__(indices[i])

            if i == 0:
                chip_id = item["chip_id"]

            y1, y2 = start_row_indexes[i], end_row_indexes[i]
            x1, x2 = start_col_indexes[i], end_col_indexes[i]

            i, j, h, w = transforms.RandomCrop.get_params(item["chip"], output_size=(len_x[i], len_y[i]))

            img_grid[:, y1:y2, x1:x2] = TF.crop(item["chip"], i, j, h, w)
            mask_grid[:, y1:y2, x1:x2] = TF.crop(item["label"], i, j, h, w)

        return {"chip": img_grid, "label": mask_grid, "chip_id": chip_id}


class MosaicDataset(CloudDataset):
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:

        indices = [idx] + random.choices(range(len(self)), k=3)

        img_grid = torch.zeros(3, 256, 256)
        mask_grid = torch.zeros(1, 256, 256)

        yc = random.randint(32, 224)
        xc = random.randint(32, 224)

        start_row_indexes = np.array([0, 0, yc, yc])
        end_row_indexes = np.array([yc, yc, 256, 256])

        start_col_indexes = np.array([0, xc, 0, xc])
        end_col_indexes = np.array([xc, 256, xc, 256])

        len_x = end_row_indexes - start_row_indexes
        len_y = end_col_indexes - start_col_indexes

        for cell_idx in range(4):
            item = super().__getitem__(indices[cell_idx])

            if cell_idx == 0:
                chip_id = item["chip_id"]

            y1, y2 = start_row_indexes[cell_idx], end_row_indexes[cell_idx]
            x1, x2 = start_col_indexes[cell_idx], end_col_indexes[cell_idx]

            i, j, h, w = transforms.RandomResizedCrop.get_params(item["chip"], scale=(0.9, 1.0), ratio=(1.0, 1.0))

            img_grid[:, y1:y2, x1:x2] = TF.resized_crop(
                item["chip"], i, j, h, w, size=(len_x[cell_idx], len_y[cell_idx])
            )
            mask_grid[:, y1:y2, x1:x2] = TF.resized_crop(
                item["label"], i, j, h, w, size=(len_x[cell_idx], len_y[cell_idx])
            )

        return {"chip": img_grid, "label": mask_grid, "chip_id": chip_id}


class ChessMixDataset(CloudDataset):
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:

        indices = [idx] + random.choices(range(len(self)), k=7)

        img_grid = torch.zeros(3, 256, 256)
        mask_grid = torch.zeros(1, 256, 256)

        start_row_indexes = np.array([0, 0, 64, 64, 128, 128, 192, 192])
        end_row_indexes = np.array([64, 64, 128, 128, 192, 192, 256, 256])

        start_col_indexes = np.array([0, 128, 64, 192, 0, 128, 64, 192])
        end_col_indexes = np.array([64, 192, 128, 256, 64, 192, 128, 256])

        len_x = end_row_indexes - start_row_indexes
        len_y = end_col_indexes - start_col_indexes

        for cell_idx in range(8):
            item = super().__getitem__(indices[cell_idx])

            if cell_idx == 0:
                chip_id = item["chip_id"]

            y1, y2 = start_row_indexes[cell_idx], end_row_indexes[cell_idx]
            x1, x2 = start_col_indexes[cell_idx], end_col_indexes[cell_idx]

            i, j, h, w = transforms.RandomCrop.get_params(item["chip"], output_size=(len_x[cell_idx], len_y[cell_idx]))

            img_grid[:, y1:y2, x1:x2] = TF.crop(item["chip"], i, j, h, w)
            mask_grid[:, y1:y2, x1:x2] = TF.crop(item["label"], i, j, h, w)

            if cell_idx in [0, 1, 4, 5]:
                img_grid[:, y1:y2, x1 + 64 : x2 + 64] = TF.hflip(img_grid[:, y1:y2, x1:x2])
                mask_grid[:, y1:y2, x1 + 64 : x2 + 64] = -1
            else:
                img_grid[:, y1:y2, x1 - 64 : x2 - 64] = TF.hflip(img_grid[:, y1:y2, x1:x2])
                mask_grid[:, y1:y2, x1 - 64 : x2 - 64] = -1

        return {"chip": img_grid, "label": mask_grid, "chip_id": chip_id}


class CloudDataModule(pl.LightningDataModule):
    """Class for stroing dataloaders"""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datamodule in train of test mode

        Parameters
        ----------
        stage: Optional[str]
            setup stage

        Returns
        -------
        None
        """

        bands = ["B02", "B03", "B04", "B08"]
        basepath = self.cfg["experiment"]["datapath"]
        feature_cols = ["chip_id", "coverage"] + [f"{band}_path" for band in bands]

        if stage == "fit" or stage is None:

            train = pd.read_csv(os.path.join(basepath, f"{self.cfg['experiment']['fold']}/train.csv"))
            train_x = train[feature_cols].copy()
            train_y = train[["chip_id", "label_path"]].copy()

            val = pd.read_csv(os.path.join(basepath, f"{self.cfg['experiment']['fold']}/val.csv"))
            val_x = val[feature_cols].copy()
            val_y = val[["chip_id", "label_path"]].copy()

            train_augs = load_augs(self.cfg["augmentation"]["train"])
            val_augs = load_augs(self.cfg["augmentation"]["val"])

            self.train_dataset = CloudDataset(x_paths=train_x, y_paths=train_y, transforms=train_augs, bands=bands)

            self.valid_dataset = CloudDataset(x_paths=val_x, y_paths=val_y, transforms=val_augs, bands=bands)

    def train_dataloader(self) -> DataLoader:
        """Returns train dataloader

        Returns
        -------
        DataLoader
        """
        num_workers = self.cfg["experiment"]["num_workers"]

        if num_workers is None:
            num_workers = 0

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg["experiment"]["train_batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Returns val dataloader

        Returns
        -------
        DataLoader
        """

        num_workers = self.cfg["experiment"]["num_workers"]

        if num_workers is None:
            num_workers = 0

        val_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg["experiment"]["val_batch_size"],
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )
        return val_dataloader
