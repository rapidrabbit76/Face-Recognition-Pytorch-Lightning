from typing import Optional, Tuple, Dict, Union, List, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.functional as tmf
from metrics import ArcMarginProduct


_batch_types = Tuple[torch.Tensor, torch.Tensor]
_step_return_types = Tuple[torch.Tensor, torch.Tensor]


class ArcFaceTrainer(pl.LightningModule):
    def __init__(self, args, backbone: nn.Module) -> None:
        super().__init__()
        self.save_hyperparameters(args)

        self.backbone = backbone
        self.initialize_weights(self.backbone)

        self.metric_fn = ArcMarginProduct(
            in_feature=self.hparams.embedding_size,
            out_feature=self.hparams.num_classes,
            m=self.hparams.m,
            s=self.hparams.s,
        )
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        backbone_params = self.backbone.parameters()
        metric_params = self.metric_fn.parameters()
        params = [{"params": backbone_params}, {"params": metric_params}]

        # Select optimizer
        optimizer = optim.SGD(
            params=params,
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=self.hparams.nesterov,
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.hparams.milestones,
            gamma=self.hparams.lr_step_gamma,
        )

        # Scheduler dict setting
        lr_scheduler = {
            "scheduler": scheduler,
            "interval": self.hparams.interval,
            "frequency": self.hparams.frequency,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    @staticmethod
    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Pytorch feed forward path"
        feature = self.backbone(x)
        return feature

    def step(self, batch: _batch_types) -> _step_return_types:
        "feed forward step"
        image, label = batch
        label = label.long()
        feature = self(image)
        output = self.metric_fn(feature, label)
        loss = self.criterion(output, label)
        acc = tmf.accuracy(output, label)
        return loss, acc

    def training_step(
        self, batch: _batch_types, batch_idx: int
    ) -> torch.Tensor:
        loss, acc = self.step(batch)
        self.log_dict(
            dictionary={
                "train/loss": loss.item(),
                "train/acc": acc,
            },
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        return self._test_common_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._test_common_step(batch, batch_idx)

    def _test_common_step(
        self, batch, batch_idx
    ) -> Dict[str, Union[torch.Tensor, float, np.ndarray]]:
        image_a, image_b, label = batch
        feature_a = self(image_a)
        feature_b = self(image_b)

        return (feature_a, feature_b, label)

    def validation_epoch_end(self, outputs):
        return self._logging(outputs, mode="val")

    def test_epoch_end(self, outputs):
        return self._logging(outputs, mode="test")

    def _logging(self, outputs, mode: str) -> Dict[str, Any]:
        info = self._calculate_accuracy(outputs)
        self.log_dict(
            {
                f"{mode}/acc": info["acc"],
                f"{mode}/threshold": info["threshold"],
            },
            prog_bar=True,
        )

        return {f"{mode}/acc": info["acc"]}

    @classmethod
    def _calculate_accuracy(
        cls,
        outputs: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> Dict[str, float]:
        f_a = torch.cat([a for a, _, _ in outputs])
        f_b = torch.cat([b for _, b, _ in outputs])
        label = torch.cat([l for _, _, l in outputs])

        sim = F.cosine_similarity(f_a, f_b).cpu()
        label = label.cpu()

        th = cls._get_best_threshold(sim, label)
        acc = tmf.accuracy(sim, label, threshold=th)
        return {
            "acc": acc,
            "threshold": th,
        }

    @classmethod
    def _get_best_threshold(
        cls, sim: torch.Tensor, label: torch.Tensor, th_num: int = 10000
    ):
        accuracy = np.zeros([2 * th_num + 1, 1])
        thresholds = np.arange(-th_num, th_num + 1) * 1.0 / th_num

        # search best thresholds
        for i, th in enumerate(thresholds):
            accuracy[i] = tmf.accuracy(sim, label, threshold=th)

        max_index = np.squeeze(accuracy == np.max(accuracy))
        threshold = np.mean(thresholds[max_index])
        return threshold

    def to_torchscript(
        self,
        file_path: str = None,
        example_inputs: Optional[Any] = None,
        **kwargs,
    ):
        mode = self.training
        torchscript_module = torch.jit.trace(
            func=self.backbone.eval(), example_inputs=example_inputs, **kwargs
        )
        self.train(mode)
        torch.jit.save(torchscript_module, file_path)
        return torchscript_module

    def to_onnx(
        self,
        file_path: str,
        input_sample: Optional[Any] = None,
        **kwargs,
    ):
        mode = self.training

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Could not export to ONNX since neither `input_sample` nor"
                    " `model.example_input_array` attribute is set."
                )
            input_sample = self.example_input_array

        input_sample = self._apply_batch_transfer_handler(input_sample)

        if "example_outputs" not in kwargs:
            self.eval()
            if isinstance(input_sample, Tuple):
                kwargs["example_outputs"] = self(*input_sample)
            else:
                kwargs["example_outputs"] = self(input_sample)

        torch.onnx.export(
            self.backbone.eval(), input_sample, file_path, **kwargs
        )
        self.train(mode)
