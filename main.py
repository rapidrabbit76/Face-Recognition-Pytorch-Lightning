import os
from argparse import ArgumentParser
from typing import Union, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from time import time

from datamodule import DATAMODULE_TABLE
from transfoms import TRANSFORMS_TABLE
from models import MODEL_TABLE, Backbone
from experiment import ArcFaceTrainer


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def hyperparameters():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    model_candidate = list(MODEL_TABLE.keys())

    # experiment
    parser.add_argument("--seed", type=int, default=9423)
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["wandb", "tb"],
        default="tb",
    )

    parser.add_argument("--experiment_name", type=str, default="Arc FACE")
    parser.add_argument("--train_data_dir", type=str)
    parser.add_argument("--validation_data_dir", type=str)
    parser.add_argument("--save_dir", type=str, default="log")
    parser.add_argument("--log_save_dir", type=str, default="ckpt")
    parser.add_argument("--num_workers", type=int, default=16)

    # data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["casia", "lfw"],
        default="lfw",
    )
    parser.add_argument(
        "--transforms", type=str, default="base", choices=["base"]
    )
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--batch_size", type=int)

    # model
    parser.add_argument(
        "--backbone",
        type=str,
        choices=model_candidate,
    )
    parser.add_argument("--pretrained", type=Union[bool, int], default=False)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    # metric
    parser.add_argument("--m", type=float, default=0.5)
    parser.add_argument("--s", type=float, default=30.0)

    # callbacks
    parser.add_argument("--callbacks_verbose", action="store_true")
    parser.add_argument("--callbacks_refresh_rate", type=int, default=1)
    parser.add_argument("--callbacks_monitor", type=str, default="val/acc")
    parser.add_argument("--callbacks_mode", type=str, default="max")
    parser.add_argument("--checkpoint_top_k", type=int, default=5)
    parser.add_argument("--earlystooping_min_delta", type=float, default=0)
    parser.add_argument("--earlystooping_patience", type=float, default=10e5)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--nesterov", type=str2bool, default=True)

    # optimizer scheduler
    parser.add_argument(
        "--milestones",
        type=List[int],
        default=[200, 400, 500],
    )
    parser.add_argument("--interval", type=str, default="epoch")
    parser.add_argument("--frequency", type=int, default=1)
    parser.add_argument("--lr_step_gamma", type=float)

    # artifacts
    parser.add_argument("--opset_version", type=int, default=11)
    args = pl.Trainer.parse_argparser(parser.parse_args())
    return args


def objective(args: dict) -> float:
    # Seed setting
    seed_everything(args.seed)

    # select table
    datamodule = DATAMODULE_TABLE[args.dataset]
    transforms = TRANSFORMS_TABLE[args.transforms]

    # build DataModule
    image_shape = [
        args.input_channels,
        args.image_size,
        args.image_size,
    ]

    train_transforms = transforms(image_shape=image_shape, train=True)
    test_transforms = transforms(image_shape=image_shape, train=False)
    validation_transforms = transforms(image_shape=image_shape, train=False)

    dm = datamodule(
        args,
        train_transforms=train_transforms,
        validation_transforms=validation_transforms,
        test_transforms=test_transforms,
    )

    # Model
    backbone = Backbone(
        backbone=args.backbone,
        image_channels=args.input_channels,
        embedding_size=args.embedding_size,
        dropout_rate=args.dropout_rate,
    )
    model = ArcFaceTrainer(args, backbone=backbone)

    # Logger
    logger = TensorBoardLogger(
        "tensorboard",
        name=f"{args.backbone}-{args.dataset}-{args.transforms}",
    )

    # Callbacks
    callbacks = [
        # tqdm refresh rate setting
        TQDMProgressBar(
            refresh_rate=args.callbacks_refresh_rate,
        ),
        # model checkpoint
        ModelCheckpoint(
            dirpath=args.save_dir,
            filename=(
                "[{epoch:04d}]-[{step:06d}]-"
                "[{train/acc:.4f}]-[{val/acc:.4f}]"
            ),
            auto_insert_metric_name=False,
            monitor=args.callbacks_monitor,
            mode=args.callbacks_mode,
            save_top_k=args.checkpoint_top_k,
            save_last=True,
            verbose=args.callbacks_verbose,
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        EarlyStopping(
            monitor=args.callbacks_monitor,
            mode=args.callbacks_mode,
            min_delta=args.earlystooping_min_delta,
            patience=args.earlystooping_patience,
            verbose=args.callbacks_verbose,
        ),
    ]

    # Trainer setting
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
    )

    # Training
    trainer.fit(model, datamodule=dm)
    test_info = trainer.test(model, datamodule=dm)[0]

    # model save
    example_inputs = torch.rand([1] + image_shape)
    ts_path = os.path.join(args.save_dir, f"{args.backbone}.jit")
    model.to_torchscript(ts_path, example_inputs=example_inputs)

    onnx_path = os.path.join(args.save_dir, f"{args.backbone}.onnx")
    input_names = ["inputs"]
    output_names = ["output"]

    model.to_onnx(
        file_path=onnx_path,
        input_sample=example_inputs,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset_version,
    )

    return test_info


if __name__ == "__main__":
    args = hyperparameters()
    info = objective(args=args)
