from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor

from .crf import GaussCRF, cloud_conf
from .fpa import FeaturePyramidAttention
from .refinement import RefimentLayer


class Unet(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class UnetFPA(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

        self.model.decoder.center = nn.Sequential(
            nn.Conv2d(2048, 64),
            nn.ReLU(inplace=True),
            FeaturePyramidAttention(64),
            nn.Conv2d(64, 2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class UnetPlusPlus(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class UnetCRF(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            if "tu" in backbone:
                checkpoint = torch.load(checkpoint_path)
                self.model.encoder.model.load_state_dict(checkpoint)
            else:
                checkpoint = torch.load(checkpoint_path)
                self.model.encoder.load_state_dict(checkpoint)

        self.crf = RefimentLayer()

    def forward(self, inputs: Tensor) -> Tensor:
        model_out = self.model(inputs)
        return self.crf(model_out, inputs)
