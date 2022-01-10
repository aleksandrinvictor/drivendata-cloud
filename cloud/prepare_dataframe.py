import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


def load_mask(path: str) -> np.ndarray:

    with rasterio.open(path) as lp:
        mask = lp.read(1).astype("float32")

    return mask


def get_coverage(data_path: str, chip_ids: np.ndarray) -> np.ndarray:

    coverage = []
    for chip_id in tqdm(chip_ids, desc="counting coverage"):

        path = os.path.join(data_path, f"train_labels/{chip_id}.tif")
        mask = load_mask(path)

        coverage.append(mask.sum() / (mask.shape[0] * mask.shape[1]))

    return np.array(coverage)


def add_paths(df: pd.DataFrame, feature_dir: Path, label_dir: Path) -> pd.DataFrame:
    """
    Given dataframe with a column for chip_id, returns a dataframe with a column
    added indicating the path to each band's TIF image as "{band}_path", eg "B02_path".
    A column is also added to the dataframe with paths to the label TIF, if the
    path to the labels directory is provided.
    """

    bands: List[str] = ["B02", "B03", "B04", "B08"]

    for band in bands:
        df[f"{band}_path"] = feature_dir / df["chip_id"] / f"{band}.tif"
        assert df[f"{band}_path"][0].exists()
    if label_dir is not None:
        df["label_path"] = label_dir / (df["chip_id"] + ".tif")
        assert df["label_path"][0].exists()

    return df


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data")

if __name__ == "__main__":
    args = parser.parse_args()

    path = Path(args.data_path)

    feature_dir = path / "train_features"
    label_dir = path / "train_labels"

    df = pd.read_csv(os.path.join(args.data_path, "train_metadata.csv"))
    df = add_paths(df, feature_dir, label_dir)
    df["coverage"] = get_coverage(args.data_path, df["chip_id"].values)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df.datetime.dt.year
    df["month"] = df.datetime.dt.month
    df["hour"] = df.datetime.dt.hour

    df.to_csv(os.path.join(args.data_path, "train.csv"), index=False)

    with open(os.path.join(args.data_path, "bad_chips.yml")) as f:
        bad_chips = yaml.load(f, yaml.Loader)

    init_samples = df[~df["chip_id"].isin(bad_chips["overall"])]

    init_samples.to_csv(os.path.join(args.data_path, "init_samples.csv"), index=False)
