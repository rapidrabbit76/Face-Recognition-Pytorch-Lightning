from unittest import mock
import numpy as np

import pytest
import torch
from experiment.arc_face_net import ArcFaceTrainer
from models import Backbone
from easydict import EasyDict


torch.set_grad_enabled(False)

model_list = [
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
]

transfoms = []


def build_args(**kwargs):
    return EasyDict(kwargs)


@pytest.fixture(scope="session")
def args():
    return build_args(
        num_classes=10,
        transforms="base",
        batch_size=2,
        input_channels=3,
        num_cached_parquet=10,
        refresh_frequncy=4,
        local_cache=4,
        image_size=112,
        embedding_size=512,
        dropout_rate=0.5,
        pretrained=True,
        max_epochs=1,
        optimizer="SGD",
        nesterov=True,
        lr=0.1,
        weight_decay=0.9,
        momentum=0.1,
        milestones=[0, 1, 2],
        lr_step_gamma=0.1,
        interval="epoch",
        frequency=1,
        m=0.5,
        s=32.0,
    )


@pytest.fixture(scope="session", params=model_list)
def backbone(request, args):
    return Backbone(
        backbone=request.param,
        image_channels=args.input_channels,
        embedding_size=args.embedding_size,
        dropout_rate=args.dropout_rate,
    )


@pytest.fixture(scope="session")
def batch(args):
    shape = [
        args.batch_size,
        args.input_channels,
        args.image_size,
        args.image_size,
    ]
    return torch.zeros(shape), torch.zeros(args.batch_size, dtype=torch.int64)


@pytest.fixture(scope="session")
def test_batch(args):
    shape = [
        args.batch_size,
        args.input_channels,
        args.image_size,
        args.image_size,
    ]
    return (
        torch.zeros(shape),
        torch.zeros(shape),
        torch.zeros(args.batch_size, dtype=torch.int64),
    )


@pytest.fixture(scope="session")
def transform_batch(args):
    shape = [
        args.image_size,
        args.image_size,
        args.input_channels,
    ]
    return np.zeros(shape, np.uint8), 0


@pytest.fixture(scope="session")
def trainer(args):
    return ArcFaceTrainer(
        args,
        Backbone(
            backbone="MobileFaceNet",
            image_channels=args.input_channels,
            embedding_size=args.embedding_size,
            dropout_rate=args.dropout_rate,
        ),
    )
