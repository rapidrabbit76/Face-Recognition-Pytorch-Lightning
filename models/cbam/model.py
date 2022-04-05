import torch
from torch import nn

from typing import *
from models.cbam.blocks import *


__all__ = [
    # BASE
    "CBAMResNet",
    # IResNet
    "IResnet50",
    "IResnet100",
    "IResnet152",
    # Squeeze and Excitation
    "IResnetSE50",
    "IResnetSE100",
    "IResnetSE152",
    # Channel Attention
    "IResnetCA50",
    "IResnetCA100",
    "IResnetCA152",
    # Spatial Attention
    "IResnetSA50",
    "IResnetSA100",
    "IResnetSA152",
    # Convolutional Block Attention Module
    "IResnetCBAM50",
    "IResnetCBAM100",
    "IResnetCBAM152",
]


class CBAMResNet(nn.Module):
    def __init__(
        self,
        image_channels: int,
        block: nn.Module,
        num_layers: List[int],
        channels_list: List[int],
        embedding_size: int,
        dropout_rate: float,
    ):
        super().__init__()

        inp = outp = channels_list[0]
        self.input_layer = nn.Sequential(
            nn.Conv2d(image_channels, outp, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outp),
            nn.PReLU(outp),
        )

        inner_layers = []
        for idx, layer in enumerate(num_layers):
            outp = channels_list[idx + 1]
            inner_layers += [self._make_layer(block, inp, outp, layer, 2)]
            inp = outp

        self.inner_layers = nn.Sequential(*inner_layers)

        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(outp),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(outp * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    @classmethod
    def _make_layer(
        cls, block: nn.Module, inp: int, outp: int, blocks: int, s: int
    ) -> nn.Sequential:
        layers = [block(inp, outp, s)]
        layers += [block(outp, outp, 1) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.inner_layers(x)
        x = self.output_layer(x)
        return x

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


FILTER_LIST: List[int] = [64, 64, 128, 256, 512]
BLOCK_TABLE: Dict[str, Any] = {
    "ir": IRBottleNeck,
    "se": IRBottleNeck_SE,
    "ca": IRBottleNeck_CA,
    "sa": IRBottleNeck_SA,
    "cba": IRBottleNeck_CBA,
}
LAYER_TABLE: Dict[str, List[int]] = {
    "50": [3, 4, 14, 3],
    "100": [3, 13, 30, 3],
    "152": [3, 8, 36, 3],
}


def IResnet50(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ir"]
    num_layers = LAYER_TABLE["50"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnet100(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ir"]
    num_layers = LAYER_TABLE["100"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnet152(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ir"]
    num_layers = LAYER_TABLE["152"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSE50(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["se"]
    num_layers = LAYER_TABLE["50"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSE100(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["se"]
    num_layers = LAYER_TABLE["100"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSE152(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["se"]
    num_layers = LAYER_TABLE["152"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCA50(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ca"]
    num_layers = LAYER_TABLE["50"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCA100(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ca"]
    num_layers = LAYER_TABLE["100"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCA152(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["ca"]
    num_layers = LAYER_TABLE["152"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSA50(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["sa"]
    num_layers = LAYER_TABLE["50"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSA100(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["sa"]
    num_layers = LAYER_TABLE["100"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetSA152(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["sa"]
    num_layers = LAYER_TABLE["152"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCBAM50(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["cba"]
    num_layers = LAYER_TABLE["50"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCBAM100(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["cba"]
    num_layers = LAYER_TABLE["100"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )


def IResnetCBAM152(
    image_channels: int, embedding_size: int, dropout_rate: float, **kwargs
):
    block = BLOCK_TABLE["cba"]
    num_layers = LAYER_TABLE["152"]
    return CBAMResNet(
        image_channels,
        block,
        num_layers,
        FILTER_LIST,
        embedding_size,
        dropout_rate,
    )
