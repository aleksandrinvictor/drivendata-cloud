import argparse
from pathlib import Path

import pandas as pd

from cloud.inference import PseudoLabelsPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./assets/exp16")
parser.add_argument("--train_data_path", default="./data/init_samples.csv")
parser.add_argument("--test_data_path", default="./data/train.csv")
parser.add_argument("--output_label_path", default="./data/pseudo_labels_1")
parser.add_argument("--conf_thres", default=0.95)

if __name__ == "__main__":
    args = parser.parse_args()

    model_path = Path(args.model_path)

    train_data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    test_data = test_data[~test_data["chip_id"].isin(train_data["chip_id"])]
    test_data = test_data.reset_index(drop=True)

    predictor = PseudoLabelsPredictor(
        model_path,
        x_paths=test_data,
        device="cuda",
        output_label_path=args.output_label_path,
        conf_thres=args.conf_thres,
    )

    pseudo_labels = predictor.predict()

    pseudo_labels = pd.concat([pseudo_labels, train_data], axis=0, ignore_index=True)

    pseudo_labels.to_csv(f"{args.output_label_path}.csv", index=False)
