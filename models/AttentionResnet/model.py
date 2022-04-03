import torch
import torch.nn as nn
import numpy as np
from models.AttentionResnet.block import *


class AttentionResnet56(nn.Module):
    def __init__(
        self,
        image_channels: int,
        embedding_size=512,
        dropout_rate=0.4,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionStage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionStage2(512, 512)
        self.residual_block3 = ResidualBlock(512, 512, 2)
        self.attention_module3 = AttentionStage3(512, 512)
        self.residual_block4 = ResidualBlock(512, 512, 2)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.output_layer(x)

        return x


class AttentionResnet92(nn.Module):

    # for input size 112
    def __init__(
        self,
        image_channels: int,
        embedding_size: int = 512,
        dropout_rate: float = 0.4,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                image_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_block1 = ResidualBlock(64, 256)
        self.attention_module1 = AttentionStage1(256, 256)
        self.residual_block2 = ResidualBlock(256, 512, 2)
        self.attention_module2 = AttentionStage2(512, 512)
        self.attention_module2_2 = AttentionStage2(512, 512)  # tbq add
        self.residual_block3 = ResidualBlock(512, 1024, 2)
        self.attention_module3 = AttentionStage3(1024, 1024)
        self.attention_module3_2 = AttentionStage3(1024, 1024)  # tbq add
        self.attention_module3_3 = AttentionStage3(1024, 1024)  # tbq add
        self.residual_block4 = ResidualBlock(1024, 2048, 2)
        self.residual_block5 = ResidualBlock(2048, 2048)
        self.residual_block6 = ResidualBlock(2048, 2048)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(2048 * 7 * 7, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        x = self.attention_module3_2(x)
        x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.output_layer(x)
        return x
