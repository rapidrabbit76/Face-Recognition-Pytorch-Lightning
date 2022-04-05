from models.mobilefacenet import *
from models.cbam import *
from models.AttentionResnet import *


__all__ = [
    # Attention Resnet
    "AttentionResnet56",
    "AttentionResnet92",
    # MobileFaceNet
    "MobileFaceNet",
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
    "Backbone",
]


MODEL_TABLE = {
    # MobileFaceNet
    "MobileFaceNet": MobileFaceNet,
    # Attention Resnet
    "AttentionResnet56": AttentionResnet56,
    "AttentionResnet92": AttentionResnet92,
    # IResNet
    "IResnet50": IResnet50,
    "IResnet100": IResnet100,
    "IResnet152": IResnet152,
    # Squeeze and Excitation
    "IResnetSE50": IResnetSE50,
    "IResnetSE100": IResnetSE100,
    "IResnetSE152": IResnetSE152,
    # Channel Attention
    "IResnetCA50": IResnetCA50,
    "IResnetCA100": IResnetCA100,
    "IResnetCA152": IResnetCA152,
    # Spatial Attention
    "IResnetSA50": IResnetSA50,
    "IResnetSA100": IResnetSA100,
    "IResnetSA152": IResnetSA152,
    # Convolutional Block Attention Module
    "IResnetCBAM50": IResnetCBAM50,
    "IResnetCBAM100": IResnetCBAM100,
    "IResnetCBAM152": IResnetCBAM152,
}


def Backbone(**kwargs):
    backbone = kwargs.get("backbone")
    Model = MODEL_TABLE[backbone]
    return Model(**kwargs)
