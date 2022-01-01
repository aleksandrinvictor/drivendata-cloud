import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

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

        return np.stack([red, green, blue], axis=2)

        # band_arrs = []
        # for band in self.bands:

        #     # WARNING. SKIPPING BAND!!!
        #     if band == "B08":
        #         continue

        #     with rasterio.open(img_paths[f"{band}_path"]) as b:
        #         band_arr = b.read(1).astype("float32")
        #     band_arrs.append(band_arr)

        # return np.stack(band_arrs, axis=-1)

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
        # basepath = os.path.join(self.cfg["experiment"]["datapath"], "folds")
        basepath = self.cfg["experiment"]["datapath"]
        feature_cols = ["chip_id"] + [f"{band}_path" for band in bands]

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
