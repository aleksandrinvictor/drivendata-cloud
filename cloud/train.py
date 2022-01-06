import argparse
import logging
import os
from typing import Any, Dict

import torch
import wandb
import yaml
from pytorch_lightning import Trainer, seed_everything

from cloud.dataset import CloudDataModule
from cloud.model import Cloud
from cloud.utils import build_object

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=0)


def train(cfg: Dict[str, Any], fold: int) -> None:

    cfg["experiment"]["fold"] = fold

    exp_path = f"./assets/{cfg['experiment']['id']}/fold_{fold}"
    cfg["callbacks"][1]["params"]["dirpath"] = os.path.join(exp_path, "checkpoints")

    os.makedirs(exp_path)

    with open(os.path.join(exp_path, "cfg.yml"), "w") as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    seed_everything(cfg["experiment"]["seed"])

    logger.info("Starting run:")
    logger.info(f"Model: {cfg['model']['class_name']}")

    # Setup device
    if "device" in cfg["experiment"]:
        device = cfg["experiment"]["device"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    callbacks = []
    for callback_cfg in cfg["callbacks"]:
        callbacks.append(build_object(callback_cfg))

    exp_logger = build_object(
        cfg["logger"],
        save_dir=exp_path,
        name=f"{cfg['experiment']['id']}_fold{fold}",
        group=cfg["experiment"]["id"],
        reinit=True,
    )

    if "pretrained" in cfg["experiment"].keys():
        model = Cloud.load_from_checkpoint(checkpoint_path=cfg["experiment"]["pretrained"], cfg=cfg)
    else:
        model = Cloud(cfg)

    datamodule = CloudDataModule(cfg)

    trainer = Trainer(
        max_epochs=cfg["experiment"]["max_epochs"],
        gpus=1,
        profiler=cfg["experiment"]["profiler"],
        logger=exp_logger,
        callbacks=callbacks,
        deterministic=True,
        gradient_clip_val=1.0,
        # accumulate_grad_batches=4,
    )

    # # Run learning rate finder
    # lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule, num_training=1000)

    # # Plot with
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("lr.png")

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()

    # print(f"new_lr: {new_lr}")

    trainer.fit(model, datamodule=datamodule)

    wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    with open("config.yml", "r") as f:
        cfg = yaml.safe_load(f)

    args = parser.parse_args()

    train(cfg, args.fold)
