import argparse
from copy import deepcopy
from typing import Any, Dict

import cv2
import numpy as np
import optuna
from torch.tensor import Tensor

from cloud.inference import ValPredictor
from cloud.postprocess import PostProcess


def objective(trial: optuna.trial.Trial, model_path: str) -> float:

    # kernel_size = trial.suggest_int("kernel_size", 2, 5, step=1)
    sigma = trial.suggest_float("sigma", 1, 3, step=1)
    mode = trial.suggest_categorical("mode", ["reflect", "constant", "nearest", "mirror", "wrap"])

    # kernel_size = trial.suggest_int("kernel_size", 3, 5, step=1)

    # min_area = trial.suggest_int("min_area", 2, (kernel_size ** 2) // 2 + 1, step=1)

    val_predictor = ValPredictor(model_path, device="cuda")

    temp_cfg = deepcopy(val_predictor.configs[0]["postprocess"])
    temp_cfg[0]["params"]["sigma"] = sigma
    temp_cfg[0]["params"]["mode"] = mode
    # temp_cfg[0]["params"]["patches_threshold"] = patches_threshold
    # temp_cfg[0]["params"]["pixel_threshold"] = pixel_threshold

    val_predictor.postprocess = PostProcess(temp_cfg)

    metrics = val_predictor.evaluate()

    return metrics["cv_score_IoU"]


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./assets/exp0")


if __name__ == "__main__":
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda x: objective(x, args.model_path), n_trials=50)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
