import torch
from torch import nn

from typing import *

__all__ = [
    "SE",
    "ChannelAttention",
    "SpatialAttention",
    "IRBottleNeck",
    "IRBottleNeck_SE",
    "IRBottleNeck_CA",
    "IRBottleNeck_SA",
    "IRBottleNeck_CBA",
]


class SE(nn.Module):
    """Squeeze and Excitation"""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return input * x


class ChannelAttention(nn.Module):
    """Channel Attention Module"""

    def __init__(self, dim: int, r: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        x = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        x = self.sigmoid(x)
        # ATTENTION!
        return input * x


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        avg_c = torch.mean(x, 1, True)
        max_c, _ = torch.max(x, 1, True)
        x = torch.cat((avg_c, max_c), 1)
        x = self.conv(x)
        x = self.sigmoid(x)
        # ATTENTION!
        return input * x


class IRBottleNeck(nn.Module):
    """Improved Residual Bottlenecks as IRB"""

    def __init__(self, inp: int, outp: int, s: int):
        super().__init__()
        assert s in [1, 2], "stride shuld be 1 or 2"
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
            nn.Conv2d(outp, outp, 3, s, 1, bias=False),
            nn.BatchNorm2d(outp),
        )
        self.shortcut_layer = nn.Identity()

        if inp != outp or s != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inp, outp, 1, s, bias=False),
                nn.BatchNorm2d(outp),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return shortcut + res


class IRBottleNeck_SE(IRBottleNeck):
    """IRB + SE"""

    def __init__(self, inp: int, outp: int, s: int):
        nn.Module.__init__(self)
        assert s in [1, 2], "stride shuld be 1 or 2"

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
            nn.Conv2d(outp, outp, 3, s, 1, bias=False),
            nn.BatchNorm2d(outp),
            SE(outp, 16),
        )

        self.shortcut_layer = nn.Identity()

        if inp != outp or s != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inp, outp, 1, s, bias=False),
                nn.BatchNorm2d(outp),
            )


class IRBottleNeck_CA(IRBottleNeck):
    """IRB + Channel Attention"""

    def __init__(self, inp: int, outp: int, s: int):
        nn.Module.__init__(self)
        assert s in [1, 2], "stride shuld be 1 or 2"

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
            nn.Conv2d(outp, outp, 3, s, 1, bias=False),
            nn.BatchNorm2d(outp),
            ChannelAttention(outp, 16),
        )

        self.shortcut_layer = nn.Identity()

        if inp != outp or s != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inp, outp, 1, s, bias=False),
                nn.BatchNorm2d(outp),
            )


class IRBottleNeck_SA(IRBottleNeck):
    """IRB + Spatial Attention"""

    def __init__(self, inp: int, outp: int, s: int):
        nn.Module.__init__(self)
        assert s in [1, 2], "stride shuld be 1 or 2"

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
            nn.Conv2d(outp, outp, 3, s, 1, bias=False),
            nn.BatchNorm2d(outp),
            SpatialAttention(),
        )
        self.shortcut_layer = nn.Identity()

        if inp != outp or s != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inp, outp, 1, s, bias=False),
                nn.BatchNorm2d(outp),
            )


class IRBottleNeck_CBA(IRBottleNeck):
    """IRB + Channel Attention + Spatial Attention (CBAM)"""

    def __init__(self, inp: int, outp: int, s: int):
        nn.Module.__init__(self)
        assert s in [1, 2], "stride shuld be 1 or 2"

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
            nn.Conv2d(outp, outp, 3, s, 1, bias=False),
            nn.BatchNorm2d(outp),
            ChannelAttention(outp, 16),
            SpatialAttention(),
        )
        self.shortcut_layer = nn.Identity()

        if inp != outp or s != 1:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(inp, outp, 1, s, bias=False),
                nn.BatchNorm2d(outp),
            )
