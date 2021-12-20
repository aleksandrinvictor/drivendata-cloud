import argparse
import logging
import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from cloud.dataset import CloudDataset
from cloud.model import Cloud
from cloud.tta import TTA
from cloud.utils import build_object, load_augs


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="./assets/exp0")
    parser.add_argument("--data_path", default="./data")

    return parser.parse_args()


class Predictor:
    def __init__(self, model_path: str, x_paths: pd.DataFrame, predictions_dir: os.PathLike, device: str) -> None:
        """Initializes Predictor class

        Parameters
        ----------
        model_path: str
            path to model

        Returns
        -------
        None
        """

        self.model_path = model_path
        self.device = torch.device(device)
        self.x_paths = x_paths
        self.predictions_dir = predictions_dir

        self.models = []
        self.configs = []
        self.num_checkpoints = 4

        for i in range(self.num_checkpoints):
            path = os.path.join(model_path, f"fold_{i}")
            checkpoint_path = os.path.join(path, "checkpoints/best.ckpt")
            config_path = os.path.join(path, "cfg.yml")

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            self.configs.append(cfg)

            model = build_object(cfg["model"])
            model = Cloud.load_from_checkpoint(checkpoint_path, cfg=cfg)
            model.to(self.device)
            model.eval()

            self.models.append(model)

        self.tta = None
        if "tta" in self.configs[0]["augmentation"].keys():
            self.tta = TTA(self.configs[0]["augmentation"]["tta"])


class TestPredictor(Predictor):
    def __init__(self, model_path: str, x_paths: pd.DataFrame, device: str, predictions_dir: os.PathLike) -> None:
        super().__init__(model_path, x_paths, predictions_dir, device)

        temp_cfg = self.configs[0]

        test_augs = load_augs(temp_cfg["augmentation"]["test"])

        test_dataset = CloudDataset(x_paths=x_paths, bands=["B02", "B03", "B04", "B08"], transforms=test_augs)
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=temp_cfg["experiment"]["val_batch_size"],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )

    @torch.no_grad()
    def predict(self):

        for batch in tqdm(self.test_dataloader):

            x = batch["chip"].to(self.device)

            preds = []

            for i in range(self.num_checkpoints):
                model = self.models[i]

                if self.tta:
                    batch_pred = self.tta(model, x)
                else:
                    batch_pred = model(x)

                batch_pred = F.interpolate(batch_pred, size=(512, 512), mode="bilinear")
                batch_pred = torch.softmax(batch_pred, dim=1)[:, 1].detach().cpu().numpy()

                preds.append(batch_pred)

            preds = np.stack(preds, axis=0).mean(axis=0)
            preds = (preds > 0.5).astype("uint8")

            for chip_id, pred in zip(batch["chip_id"], preds):
                chip_pred_path = self.predictions_dir / f"{chip_id}.tif"
                chip_pred_im = Image.fromarray(pred)
                chip_pred_im.save(chip_pred_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parse_args()

    path = pathlib.PurePath(args.model_path)

    x_paths = pd.read_csv("./data/folds/0/val.csv")

    logger.info(f"Model: {path.name}")

    pred_dir = Path("./predictions")
    pred_dir.mkdir(exist_ok=True, parents=True)

    test_predictor = TestPredictor(args.model_path, x_paths, device="cuda", predictions_dir=pred_dir)
    test_predictor.predict()
