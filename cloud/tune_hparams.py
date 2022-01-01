from copy import deepcopy
from typing import Any, Dict

import optuna
import yaml
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything

from cloud.dataset import CloudDataModule
from cloud.model import Cloud


def objective(trial: optuna.trial.Trial, cfg: Dict[str, Any]) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    beta = trial.suggest_float("beta", 0.5, 5.0, step=0.5)
    lam_min = trial.suggest_float("lam_min", 0.8, 0.95, step=0.01)
    lam_max = trial.suggest_float("lam_max", 0.8, 0.99, step=0.01)
    num_boxes = trial.suggest_int("num_boxes", 4, 8, step=1)

    if lam_min > lam_max:
        return -1e6

    trial_cfg = deepcopy(cfg)
    trial_cfg["augmentation"]["cutmix"]["params"]["beta"] = beta
    trial_cfg["augmentation"]["cutmix"]["params"]["lam_min"] = lam_min
    trial_cfg["augmentation"]["cutmix"]["params"]["lam_max"] = lam_max
    trial_cfg["augmentation"]["cutmix"]["params"]["num_boxes"] = num_boxes
    trial_cfg["augmentation"]["cutmix"]["params"]["p"] = 1.0

    seed_everything(trial_cfg["experiment"]["seed"])
    model = Cloud(trial_cfg)
    datamodule = CloudDataModule(trial_cfg)

    trainer = Trainer(
        max_epochs=3,
        gpus=1,
        profiler=cfg["experiment"]["profiler"],
        logger=True,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_IoU")],
        deterministic=True,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics["val_IoU"].item()


if __name__ == "__main__":
    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(lambda x: objective(x, cfg), n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
