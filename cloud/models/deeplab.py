from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor


class DeepLab(nn.Module):
    def __init__(self, backbone: str = "resnet34", checkpoint_path: Optional[str] = None) -> None:
        super().__init__()

        self.model = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2)

        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.model.encoder.load_state_dict(checkpoint)

        # self.model.segmentation_head[0].bias = nn.parameter.Parameter(
        #     torch.tensor([np.log(0.58), np.log(0.42)], dtype=torch.float)
        # )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)
