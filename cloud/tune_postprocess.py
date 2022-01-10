import argparse
from typing import Any, Dict

import cv2
import numpy as np
import optuna
from torch.tensor import Tensor

from cloud.inference import ValPredictor


class PostProcess:
    def __init__(self, func: str, kernel_size: int) -> None:
        self.func = func
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def __call__(self, x: Tensor) -> Any:

        for i in range(x.shape[0]):
            if self.func == "opening":
                x[i] = cv2.morphologyEx(x[i], cv2.MORPH_OPEN, self.kernel)
            elif self.func == "closing":
                x[i] = cv2.morphologyEx(x[i], cv2.MORPH_CLOSE, self.kernel)
            elif self.func == "erosion":
                x[i] = cv2.erode(x[i], self.kernel, iterations=1)
            elif self.func == "dilation":
                x[i] = cv2.dilate(x[i], self.kernel, iterations=1)

        return x


def objective(trial: optuna.trial.Trial, model_path: str) -> float:

    func = trial.suggest_categorical("func", ["opening", "closing", "erosion", "dilation"])
    kernel_size = trial.suggest_int("kernel_size", 3, 10, step=1)

    post_process = PostProcess(func, kernel_size)

    val_predictor = ValPredictor(model_path, device="cuda", post_process=post_process)
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
