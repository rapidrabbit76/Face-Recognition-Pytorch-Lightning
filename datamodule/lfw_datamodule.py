from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import LFWPairs, LFWPeople

__all__ = ["LFWDataModule"]


class LFWDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        train_transforms: Callable,
        validation_transforms: Callable,
        test_transforms: Callable,
    ):
        super().__init__()
        self.save_hyperparameters(args)
        self.train_transforms = train_transforms
        self.validation_transforms = validation_transforms
        self.test_transforms = test_transforms

    def prepare_data(self) -> None:
        LFWPairs(
            root=self.hparams.validation_data_dir,
            download=True,
        )
        LFWPeople(
            root=self.hparams.train_data_dir,
            download=True,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_ds = LFWPeople(
                root=self.hparams.train_data_dir,
                download=True,
                transform=self.train_transforms,
            )

            self.val_ds = LFWPairs(
                root=self.hparams.validation_data_dir,
                split="train",
                transform=self.validation_transforms,
                download=True,
            )

        if stage == "test" or stage is None:
            self.test_ds = LFWPairs(
                root=self.hparams.validation_data_dir,
                split="10fold",
                transform=self.test_transforms,
                download=False,
            )
        return super().setup(stage=stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        print("test")
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
