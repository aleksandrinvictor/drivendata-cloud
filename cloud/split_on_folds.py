import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rasterio
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
parser.add_argument("--n_splits", type=int, default=4)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--output_path", default="./data/folds")

if __name__ == "__main__":
    args = parser.parse_args()

    path = Path(args.data_path)

    feature_dir = path / "train_features"
    label_dir = path / "train_labels"

    df = pd.read_csv(os.path.join(args.data_path, "train_metadata.csv"))
    df = add_paths(df, feature_dir, label_dir)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df.datetime.dt.year
    df["month"] = df.datetime.dt.month

    df["coverage"] = get_coverage(args.data_path, df["chip_id"].values)

    mskf = MultilabelStratifiedKFold(
        n_splits=args.n_splits, **{"shuffle": args.shuffle, "random_state": args.random_state}
    )

    stratify_cols = ["location", "year", "month", "coverage"]

    for i, (train_index, val_index) in enumerate(
        mskf.split(
            df["chip_id"],
            df[stratify_cols],
        )
    ):
        fold_path = os.path.join(args.output_path, str(i))
        os.makedirs(fold_path, exist_ok=True)

        df.iloc[train_index].to_csv(os.path.join(fold_path, "train.csv"), index=False)
        df.iloc[val_index].to_csv(os.path.join(fold_path, "val.csv"), index=False)
