from turtle import forward
from kornia import torch
import torch.nn as nn
from torch import Tensor


class RefimentLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, pred_mask: Tensor, original_image: Tensor) -> Tensor:
        return self.alpha * pred_mask + self.beta * self.conv(original_image)
        # return (pred_mask + self.conv(original_image)) / 2
        # return torch.maximum(pred_mask, self.conv(original_image))
