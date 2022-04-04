import torch
from torch import nn
import math
from typing import *
from models.mobilefacenet.blocks import *


class MobileFaceNetBase(nn.Module):
    def __init__(
        self,
        image_channels: int,
        block: nn.Module,
        embedding_size: int,
        config: List[int],
        **kwargs,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            ConvBlock(image_channels, 64, 3, 2, 1),
            ConvBlock(64, 64, 3, 1, 1, dw=True),
        )

        inp = 64
        inner_layers = []
        for t, c, n, s in config:
            inner_layers += [self._make_layer(block, inp, c, n, s, t)]
            inp = c

        self.inner_layers = nn.Sequential(*inner_layers)

        self.output_layers = nn.Sequential(
            ConvBlock(128, 512, 1, 1, 0),
            ConvBlock(512, 512, 7, 1, 0, dw=True, act=False),
            ConvBlock(512, embedding_size, 1, 1, 0, act=False),
            nn.Flatten(),
        )

    @classmethod
    def _make_layer(
        cls, block: nn.Module, inp: int, c: int, n: int, s: int, t: int
    ) -> nn.Module:
        layers = [block(inp, c, s, t)]
        layers += [block(c, c, 1, t) for _ in range(1, n)]
        return nn.Sequential(*layers)

    def initialization_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.inner_layers(x)
        x = self.output_layers(x)
        return x


def MobileFaceNet(image_channels: int, embedding_size: int, **kwargs):
    block = BottleNeck
    config: List[int] = [
        [2, 64, 5, 2],
        [4, 128, 1, 2],
        [2, 128, 6, 1],
        [4, 128, 1, 2],
        [2, 128, 2, 1],
    ]  # t,   c, n, s

    return MobileFaceNetBase(
        image_channels,
        block,
        embedding_size,
        config,
        **kwargs,
    )
