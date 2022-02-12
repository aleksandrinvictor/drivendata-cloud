from typing import List, Dict
import torch.nn as nn
import os
import yaml
from cloud.utils import build_object
from cloud.model import Cloud
import torch
import segmentation_models_pytorch as smp


class Ensembler(nn.Module):
    def __init__(self, model_paths: List[str], device: str):
        super(Ensembler, self).__init__()

        self.device = device
        self.num_models = len(model_paths)

        self.models = []

        for model_path in model_paths:
            config_path = os.path.join(model_path, "cfg.yml")
            checkpoint_path = os.path.join(model_path, "checkpoints/best.ckpt")

            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)

            model = build_object(cfg["model"])
            model = Cloud.load_from_checkpoint(checkpoint_path, cfg=cfg)
            model.to(self.device)
            model.eval()
            model.freeze()

            self.models.append(model)


class LinearEnsembler(Ensembler):
    def __init__(self, model_paths: List[str], device: str):
        super().__init__(model_paths=model_paths, device=device)

        self.one_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(1 / self.num_models)) for _ in range(self.num_models)]
        )

        self.zero_params = nn.ParameterList(
            [nn.Parameter(torch.tensor(1 / self.num_models)) for _ in range(self.num_models)]
        )

    def forward(self, x):

        model_output = self.models[0](x)

        out_one = model_output[:, 1].unsqueeze(dim=1)
        out_zero = model_output[:, 0].unsqueeze(dim=1)

        for i in range(1, self.num_models):
            model_output = self.models[i](x)

            one_probs = model_output[:, 1].unsqueeze(dim=1)
            zero_probs = model_output[:, 0].unsqueeze(dim=1)

            out_one += self.one_params[i] * one_probs
            out_zero += self.zero_params[i] * zero_probs

        out = torch.cat([out_zero, out_one], dim=1) / self.num_models

        return out


class UnetEnsembler(Ensembler):
    def __init__(self, model_paths: List[str], device: str, backbone: str = "resnet34"):
        super().__init__(model_paths=model_paths, device=device)

        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=7,
            classes=2,
        )

    def forward(self, x):
        outs = [x]
        for m in self.models:
            outs.append(m(x))

        outs = torch.cat(outs, dim=1)

        return self.model(outs)


class ConvEnsembler(Ensembler):
    def __init__(self, model_paths: List[str], device: str):
        super().__init__(model_paths=model_paths, device=device)

        self.model = nn.Conv2d(4, 2, kernel_size=1)

    def forward(self, x):
        outs = []
        for m in self.models:
            outs.append(m(x))

        outs = torch.cat(outs, dim=1)

        return self.model(outs)
