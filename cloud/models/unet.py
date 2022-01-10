from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor

from .fpa import FeaturePyramidAttention

# def convert_relu_to_silu(model):
#     for child_name, child in model.named_children():
#         if isinstance(child, nn.ReLU):
#             setattr(model, child_name, nn.SiLU(inplace=True))
#         else:
#             convert_relu_to_silu(child)


class Unet(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

        # self.model.segmentation_head[0].bias = nn.parameter.Parameter(
        #     torch.tensor([np.log(0.58), np.log(0.42)], dtype=torch.float)
        # )
        # convert_relu_to_silu(self.model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class UnetFPA(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

        # self.model.segmentation_head[0].bias = nn.parameter.Parameter(
        #     torch.tensor([np.log(0.58), np.log(0.42)], dtype=torch.float)
        # )
        self.model.decoder.center = nn.Sequential(
            nn.Conv2d(2048, 64),
            nn.ReLU(inplace=True),
            FeaturePyramidAttention(64),
            nn.Conv2d(64, 2048),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
        # out = self.model.encoder(inputs)
        # out[-1] = self.fpa(out[-1])
        # out = self.model.decoder(out)

        # return self.model.segmentation_head(out)


class UnetPlusPlus(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

        # self.model.segmentation_head[0].bias = nn.parameter.Parameter(
        #     torch.tensor([np.log(0.58), np.log(0.42)], dtype=torch.float)
        # )
        # convert_relu_to_silu(self.model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
