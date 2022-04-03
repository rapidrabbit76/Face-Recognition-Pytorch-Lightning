import torch
import torch.nn as nn
import numpy as np
from typing import *

__all__ = [
    "ResidualBlock",
    "AttentionStage1",
    "AttentionStage2",
    "AttentionStage3",
]

Normalization = nn.BatchNorm2d
Activation = nn.ReLU


class ResidualBlock(nn.Module):
    def __init__(self, inp: int, outp: int, s: int = 1):
        super().__init__()
        self.shortcut = nn.Identity()
        self.bottleneck = nn.Sequential(
            Normalization(inp),
            Activation(inplace=True),
            nn.Conv2d(inp, outp // 4, 1, 1, bias=False),
            Normalization(outp // 4),
            Activation(inplace=True),
            nn.Conv2d(outp // 4, outp // 4, 3, s, 1, bias=False),
            Normalization(outp // 4),
            Activation(inplace=True),
            nn.Conv2d(outp // 4, outp, 1, 1, bias=False),
        )

        if inp != outp or s != 1:
            self.shortcut = nn.Conv2d(inp, outp, 1, s, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        x = self.bottleneck(x)
        return x + res


class AttentionStage1(nn.Module):

    # input size is 56*56
    def __init__(
        self,
        inp: int,
        outp: int,
        size1: Tuple[int, int] = (56, 56),
        size2: Tuple[int, int] = (28, 28),
        size3: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.share_residual_block = ResidualBlock(inp, outp)
        self.trunk_branches = nn.Sequential(
            ResidualBlock(inp, outp),
            ResidualBlock(inp, outp),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block1 = ResidualBlock(inp, outp)
        self.skip_connect1 = ResidualBlock(inp, outp)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block2 = ResidualBlock(inp, outp)
        self.skip_connect2 = ResidualBlock(inp, outp)

        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mask_block3 = nn.Sequential(
            ResidualBlock(inp, outp),
            ResidualBlock(inp, outp),
        )

        self.interpolation3 = nn.UpsamplingBilinear2d(size=size3)
        self.mask_block4 = ResidualBlock(inp, outp)

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.mask_block5 = ResidualBlock(inp, outp)

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.mask_block6 = nn.Sequential(
            Normalization(outp),
            Activation(inplace=True),
            nn.Conv2d(outp, outp, 1, 1, bias=False),
            Normalization(outp),
            Activation(inplace=True),
            nn.Conv2d(outp, outp, 1, 1, bias=False),
            nn.Sigmoid(),
        )

        self.last_block = ResidualBlock(inp, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.share_residual_block(x)
        out_trunk = self.trunk_branches(x)

        out_pool1 = self.mpool1(x)
        out_block1 = self.mask_block1(out_pool1)
        out_skip_connect1 = self.skip_connect1(out_block1)

        out_pool2 = self.mpool2(out_block1)
        out_block2 = self.mask_block2(out_pool2)
        out_skip_connect2 = self.skip_connect2(out_block2)

        out_pool3 = self.mpool3(out_block2)
        out_block3 = self.mask_block3(out_pool3)
        #
        out_inter3 = self.interpolation3(out_block3) + out_block2
        out = out_inter3 + out_skip_connect2
        out_block4 = self.mask_block4(out)

        out_inter2 = self.interpolation2(out_block4) + out_block1
        out = out_inter2 + out_skip_connect1
        out_block5 = self.mask_block5(out)

        out_inter1 = self.interpolation1(out_block5) + out_trunk
        out_block6 = self.mask_block6(out_inter1)

        out = (1 + out_block6) + out_trunk
        out_last = self.last_block(out)

        return out_last


class AttentionStage2(nn.Module):

    # input image size is 28*28
    def __init__(
        self,
        inp: int,
        outp: int,
        size1: Tuple[int, int] = (28, 28),
        size2: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(inp, outp)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(inp, outp),
            ResidualBlock(inp, outp),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = ResidualBlock(inp, outp)
        self.skip1_connection_residual_block = ResidualBlock(inp, outp)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2_blocks = nn.Sequential(
            ResidualBlock(inp, outp),
            ResidualBlock(inp, outp),
        )

        self.interpolation2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3_blocks = ResidualBlock(inp, outp)
        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)
        self.softmax4_blocks = nn.Sequential(
            Normalization(outp),
            Activation(inplace=True),
            nn.Conv2d(outp, outp, kernel_size=1, stride=1, bias=False),
            Normalization(outp),
            Activation(inplace=True),
            nn.Conv2d(outp, outp, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )
        self.last_blocks = ResidualBlock(inp, outp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)

        out_interp2 = self.interpolation2(out_softmax2) + out_softmax1
        out = out_interp2 + out_skip1_connection

        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = self.interpolation1(out_softmax3) + out_trunk
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = (1 + out_softmax4) * out_trunk
        out_last = self.last_blocks(out)

        return out_last


class AttentionStage3(nn.Module):

    # input image size is 14*14
    def __init__(self, in_channels, out_channels, size1=(14, 14)):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1_blocks = nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(in_channels, out_channels),
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2_blocks = nn.Sequential(
            Normalization(out_channels),
            Activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            Normalization(out_channels),
            Activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

        self.last_blocks = ResidualBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)

        out_interp1 = self.interpolation1(out_softmax1) + out_trunk
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = (1 + out_softmax2) * out_trunk
        out_last = self.last_blocks(out)

        return out_last
