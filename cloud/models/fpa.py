import torch.nn as nn
from torch import Tensor


class ConvBatchNormBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        upsample: bool = False,
    ):
        super(ConvBatchNormBlock, self).__init__()

        self.block = nn.ModuleList()

        if upsample:
            self.block.append(
                nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=1,
                )
            )
        else:
            self.block.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        self.block.append(nn.BatchNorm2d(in_channels))
        self.block.append(nn.ReLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.block:
            x = layer(x)
        return x


class FeaturePyramidAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(FeaturePyramidAttention, self).__init__()

        self.in_channels = in_channels

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU(inplace=True)

        # Pyramid block
        self.conv_main = ConvBatchNormBlock(in_channels, kernel_size=1, stride=1)

        # self.conv7x7_1 = ConvBatchNormBlock(in_channels, kernel_size=7, stride=1, padding=3)

        # self.conv7x7_2 = ConvBatchNormBlock(in_channels, kernel_size=7, stride=1, padding=3)

        self.conv5x5_1 = ConvBatchNormBlock(in_channels, kernel_size=5, stride=1, padding=2)

        self.conv5x5_2 = ConvBatchNormBlock(in_channels, kernel_size=5, stride=1, padding=2)

        self.conv3x3_1 = ConvBatchNormBlock(in_channels, kernel_size=3, stride=1, padding=1)

        self.conv3x3_2 = ConvBatchNormBlock(in_channels, kernel_size=3, stride=1, padding=1)

        # Upsample
        self.upsample_1 = ConvBatchNormBlock(in_channels, kernel_size=3, stride=2, padding=1, upsample=True)

        self.upsample_2 = ConvBatchNormBlock(in_channels, kernel_size=3, stride=2, padding=1, upsample=True)

        # self.upsample_3 = ConvBatchNormBlock(in_channels, kernel_size=3, stride=2, padding=1, upsample=True)

        # Global pooling branch
        self.gb_conv = ConvBatchNormBlock(in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:

        # Main path
        out_conv_main = self.conv_main(x)

        # Downsample path
        # out_conv7x7 = self.conv7x7_1(self.pooling(x))

        # out_conv5x5 = self.conv5x5_1(self.pooling(out_conv7x7))
        out_conv5x5 = self.conv5x5_1(self.pooling(x))

        out_conv3x3 = self.conv3x3_1(self.pooling(out_conv5x5))

        # Middle path
        # out_conv7x7 = self.conv7x7_2(out_conv7x7)

        out_conv5x5 = self.conv5x5_2(out_conv5x5)

        out_conv3x3 = self.conv3x3_2(out_conv3x3)

        # Upsample path
        out = self.upsample_1(out_conv3x3)

        out = self.upsample_2(out + out_conv5x5)

        # out = self.upsample_3(out + out_conv7x7)

        out = out * out_conv_main

        # Global pooling branch
        out_gb = nn.AvgPool2d(x.shape[2:])(x)
        out_gb = self.gb_conv(out_gb)

        return self.relu(out + out_gb)
