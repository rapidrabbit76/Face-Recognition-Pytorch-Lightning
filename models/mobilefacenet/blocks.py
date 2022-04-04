import torch
from torch import nn

from typing import *


__all__ = [
    "ConvBlock",
    "DepthWiseConv",
    "BottleNeck",
]


Normalization = nn.BatchNorm2d
Activation = nn.PReLU


class ConvBlock(nn.Module):
    def __init__(self, inp: int, outp: int, k: int, s: int, p: int, dw=False, act=True):
        super().__init__()

        if dw:
            layer = [nn.Conv2d(inp, outp, k, s, p, groups=inp, bias=False)]
        else:
            layer = [nn.Conv2d(inp, outp, k, s, p, bias=False)]

        layer += [Normalization(outp)]

        if act:
            layer += [Activation(outp)]

        self.net = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthWiseConv(nn.Module):
    def __init__(
        self, inp: int, outp: int, k: int, s: int, p: int, norm=True, act=True
    ):
        super().__init__()
        layer = [nn.Conv2d(inp, outp, k, s, p, groups=inp, bias=False)]
        if norm:
            layer += [Normalization(outp)]
        if act:
            layer += [Activation(inp)]
        self.net = nn.Sequential(*layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BottleNeck(nn.Module):
    def __init__(self, inp: int, outp: int, s: int, expansion: int):
        super().__init__()
        self.connect = s == 1 and inp == outp
        self.conv = nn.Sequential(
            ConvBlock(inp, inp * expansion, 1, 1, 0),
            DepthWiseConv(inp * expansion, inp * expansion, 3, s, 1),
            ConvBlock(inp * expansion, outp, 1, 1, 0, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = x
        x = self.conv(x)
        return x + sc if self.connect else x
