from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor


class Unet(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
