import argparse
import logging
import os
import pathlib
from copy import deepcopy
from glob import glob
from pathlib import Path

from typing import Any, Dict, List, Union
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from cloud.dataset import CloudDataset
from cloud.model import Cloud
from cloud.postprocess import PostProcess
from cloud.tta import TTA
from cloud.utils import build_object, load_augs, load_metrics
from torch.utils.data import DataLoader


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", default="./assets/exp0")
    parser.add_argument("--data_path", default="./data")

    return parser.parse_args()


class Predictor:
    def __init__(self, model_path: str, x_paths: pd.DataFrame, device: str, num_folds: int) -> None:
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

        self.device = device
        self.x_paths = x_paths

        self.models = []
        self.configs = []
        self.num_folds = num_folds

        for i in range(self.num_folds):
            path = os.path.join(model_path, f"fold_{i}")

            checkpoint_paths = glob(os.path.join(path, "checkpoints/*.ckpt"))

            config_path = os.path.join(path, "cfg.yml")

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            self.configs.append(cfg)

            for checkpoint_path in checkpoint_paths:
                model = build_object(cfg["model"])
                model = Cloud.load_from_checkpoint(checkpoint_path, cfg=cfg)
                model.to(self.device)
                model.eval()

                self.models.append(model)

        self.tta = None
        if "tta" in self.configs[0]["augmentation"].keys():
            self.tta = TTA(self.configs[0]["augmentation"]["tta"])

        self.postprocess = None
        if "postprocess" in self.configs[0].keys():
            self.postprocess = PostProcess(self.configs[0]["postprocess"])


class TestPredictor(Predictor):
    def __init__(
        self, model_path: str, x_paths: pd.DataFrame, device: str, predictions_dir: os.PathLike, num_folds: int = 5
    ) -> None:
        super().__init__(model_path, x_paths, device, num_folds)

        self.predictions_dir = predictions_dir

        temp_cfg = self.configs[0]

        test_augs = load_augs(temp_cfg["augmentation"]["test"])

        test_dataset = CloudDataset(x_paths=x_paths, bands=["B02", "B03", "B04", "B08"], transforms=test_augs)
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=temp_cfg["experiment"]["val_batch_size"],
            num_workers=os.cpu_count(),
            shuffle=False,
            pin_memory=True,
        )

    @torch.no_grad()
    def predict(self):

        for batch in tqdm(self.test_dataloader):

            x = batch["chip"].to(self.device)

            preds = []

            for i in range(len(self.models)):
                model = self.models[i]

                if self.tta:
                    batch_pred = self.tta(model, x)
                else:
                    batch_pred = model(x)

                # batch_pred = F.interpolate(batch_pred, size=(512, 512), mode="bilinear")
                batch_pred = torch.softmax(batch_pred, dim=1)[:, 1].detach().cpu().numpy()

                if self.postprocess:
                    batch_pred = self.postprocess(batch_pred)

                preds.append(batch_pred)

            preds = np.stack(preds, axis=0).mean(axis=0)
            preds = (preds > 0.5).astype("uint8")

            for chip_id, pred in zip(batch["chip_id"], preds):
                chip_pred_path = self.predictions_dir / f"{chip_id}.tif"
                chip_pred_im = Image.fromarray(pred)
                chip_pred_im.save(chip_pred_path)


class PseudoLabelsPredictor(Predictor):
    def __init__(
        self,
        model_path: str,
        x_paths: pd.DataFrame,
        device: str,
        output_label_path: str,
        conf_thres: float,
        num_folds: int = 5,
    ) -> None:
        super().__init__(model_path, x_paths, device, num_folds)

        temp_cfg = self.configs[0]

        val_augs = load_augs(temp_cfg["augmentation"]["val"])

        dataset = CloudDataset(x_paths=x_paths, bands=["B02", "B03", "B04", "B08"], transforms=val_augs)
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=temp_cfg["experiment"]["val_batch_size"],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )

        self.conf_thres = conf_thres

        self.output_label_path = Path(output_label_path)
        self.output_label_path.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def predict(self):

        pseudo_label_chips = []
        pseudo_label_mask_paths = []

        for batch in tqdm(self.dataloader):

            x = batch["chip"].to(self.device)

            preds = []

            for i in range(len(self.models)):
                model = self.models[i]

                if self.tta:
                    batch_pred = self.tta(model, x)
                else:
                    batch_pred = model(x)

                batch_pred = F.interpolate(batch_pred, size=(512, 512), mode="bilinear")
                batch_pred = torch.softmax(batch_pred, dim=1)[:, 1].detach().cpu().numpy()

                if self.postprocess:
                    batch_pred = self.postprocess(batch_pred)

                preds.append(batch_pred)

            preds = np.stack(preds, axis=0).mean(axis=0).squeeze()

            zero_pixels = (preds < 0.1).sum(axis=(1, 2))
            one_pixels = (preds > 0.9).sum(axis=(1, 2))

            area = preds[0].shape[0] * preds[0].shape[1]

            num_confident = zero_pixels + one_pixels
            confidence = num_confident / area

            confidence_mask = confidence > self.conf_thres

            confident_chip_ids = np.array(batch["chip_id"])[confidence_mask]
            confident_preds = preds[confidence_mask]
            confident_preds = (confident_preds > 0.5).astype("uint8")

            pseudo_label_chips.append(confident_chip_ids)

            for chip_id, pred in zip(confident_chip_ids, confident_preds):
                chip_pred_path = self.output_label_path / f"{chip_id}.tif"

                pseudo_label_mask_paths.append(chip_pred_path)

                chip_pred_im = Image.fromarray(pred)
                chip_pred_im.save(chip_pred_path)

        pseudo_label_chips = np.concatenate(pseudo_label_chips)
        pseudo_label_mask_paths = np.array(pseudo_label_mask_paths).astype(str)

        pseudo_labels = self.x_paths[self.x_paths["chip_id"].isin(pseudo_label_chips)]
        pseudo_labels["label_path"] = pseudo_label_mask_paths

        return pseudo_labels


class ValPredictor(Predictor):
    def __init__(self, model_path: str, device: str, threshold: float = 0.5, num_folds: int = 5) -> None:
        super().__init__(model_path, None, device, num_folds)

        metrcis_config = deepcopy(self.configs[0]["metrics"])
        metrcis_config[0]["params"]["threshold"] = threshold

        # self.post_process = post_process

        if self.postprocess:
            metrcis_config[0]["params"]["activation"] = None
            metrcis_config[0]["params"]["threshold"] = None

        self.metrics = load_metrics(metrcis_config)

    def _make_dataloader(self, fold: int) -> torch.utils.data.DataLoader:
        cfg = self.configs[fold]

        val_augs = load_augs(cfg["augmentation"]["val"])

        x_paths = pd.read_csv(os.path.join(cfg["experiment"]["datapath"], f"{fold}/val.csv"))

        dataset = CloudDataset(
            x_paths=x_paths, y_paths=x_paths, bands=["B02", "B03", "B04", "B08"], transforms=val_augs
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["experiment"]["val_batch_size"],
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )

        return dataloader

    @torch.no_grad()
    def evaluate(self):

        metrics: Dict[Any, Any] = {i: {} for i in range(self.num_folds)}

        for i in range(self.num_folds):

            fold_metrics = {}

            model = self.models[i]

            dataloader = self._make_dataloader(i)

            for batch in tqdm(dataloader):

                x = batch["chip"].to(self.device)
                y = batch["label"].to(self.device).long()

                if self.tta:
                    batch_pred = self.tta(model, x)
                else:
                    batch_pred = model(x)

                if self.postprocess:
                    batch_pred = torch.softmax(batch_pred, dim=1)[:, 1].detach().cpu().numpy()

                    batch_pred = self.postprocess(batch_pred)

                    batch_pred = (batch_pred > 0.5).astype("uint8")
                    batch_pred = torch.tensor(batch_pred).unsqueeze(dim=1)
                    y = y.to("cpu")

                for name, metric in self.metrics.items():
                    fold_metrics[name] = fold_metrics.get(name, 0) + metric(batch_pred, y).item() / len(dataloader)

            metrics[i] = fold_metrics

            # if i == 0 and fold_metrics["IoU"] < 0.8828757375584806:
            #     metrics["cv_score_IoU"] = -100
            #     return metrics

            for name, metric_val in fold_metrics.items():
                print(f"fold: {i}, metric: {name}, val: {metric_val}")
                metrics[f"cv_score_{name}"] = metrics.get(f"cv_score_{name}", 0) + metric_val / self.num_folds

        return metrics


def make_dataloader(cfg: Dict[str, Any], fold: int) -> torch.utils.data.DataLoader:
    val_augs = load_augs(cfg["augmentation"]["val"])

    x_paths = pd.read_csv(os.path.join(cfg["experiment"]["datapath"], f"{fold}/val.csv"))

    dataset = CloudDataset(x_paths=x_paths, y_paths=x_paths, bands=["B02", "B03", "B04", "B08"], transforms=val_augs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["experiment"]["val_batch_size"],
        num_workers=8,
        shuffle=False,
        pin_memory=True,
    )


def eval_ensemble(model_paths: List[str], device: str = "cuda") -> None:
    predictors: List[Predictor] = []

    for path in model_paths:
        predictors.append(Predictor(path, x_paths=None, device="cuda", num_folds=5))

    metrics_config = deepcopy(predictors[0].configs[0]["metrics"])
    metrics = load_metrics(metrics_config)

    num_folds = predictors[0].num_folds

    metrics_dict: Dict[Any, Any] = {i: {} for i in range(num_folds)}

    for i in range(num_folds):

        fold_metrics = {}

        cfg = predictors[0].configs[0]
        dataloader = make_dataloader(cfg, i)
        models = [p.models[i] for p in predictors]

        for batch in tqdm(dataloader):

            x = batch["chip"].to(device)
            y = batch["label"].to(device).long()

            final_batch_pred = 0

            for j, model in enumerate(models):
                if predictors[j].tta:
                    final_batch_pred += predictors[j].tta(model, x)
                else:
                    final_batch_pred += model(x)

            final_batch_pred /= len(models)

            for name, metric in metrics.items():
                fold_metrics[name] = fold_metrics.get(name, 0) + metric(final_batch_pred, y).item() / len(dataloader)

        metrics_dict[i] = fold_metrics

        for name, metric_val in fold_metrics.items():
            print(f"fold: {i}, metric: {name}, val: {metric_val}")
            metrics_dict[f"cv_score_{name}"] = metrics_dict.get(f"cv_score_{name}", 0) + metric_val / num_folds

    return metrics_dict


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parse_args()

    path = pathlib.PurePath(args.model_path)

    # x_paths = pd.read_csv("./data/init_folds/0/val.csv")

    logger.info(f"Model: {path.name}")

    # pred_dir = Path("./predictions")
    # pred_dir.mkdir(exist_ok=True, parents=True)

    # test_predictor = TestPredictor(args.model_path, x_paths, device="cuda", predictions_dir=pred_dir, num_folds=5)
    # test_predictor.predict()

    val_predictor = ValPredictor(args.model_path, device="cuda", num_folds=5)
    metrics = val_predictor.evaluate()

    print(metrics)

