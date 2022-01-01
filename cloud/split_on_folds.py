import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/train.csv")
parser.add_argument("--n_splits", type=int, default=5)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--random_state", type=int, default=42)
parser.add_argument("--output_path", default="./data/folds")

if __name__ == "__main__":
    args = parser.parse_args()

    path = Path(args.data_path)

    df = pd.read_csv(args.data_path)

    stratify_cols = ["location", "year", "month", "hour", "coverage"]

    if args.n_splits > 1:
        mskf = MultilabelStratifiedKFold(
            n_splits=args.n_splits, **{"shuffle": args.shuffle, "random_state": args.random_state}
        )

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
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            df, df[stratify_cols], test_size=0.2, random_state=args.random_state
        )

        fold_path = os.path.join(args.output_path, "0")
        os.makedirs(fold_path, exist_ok=True)

        X_train.to_csv(os.path.join(fold_path, "train.csv"), index=False)
        X_test.to_csv(os.path.join(fold_path, "val.csv"), index=False)
