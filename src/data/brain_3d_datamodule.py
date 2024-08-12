from typing import Any, Dict, Optional, Tuple, Sequence
import os
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from monai import transforms
from .components.disease_dataset import SingleDataset


class Brain3DDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "data/",
            data_types: Sequence[str] = ('AD',),
            info_keys: Sequence[str] = ('age', 'gender',),
            remove_covar: bool = True,
            preprocess: [bool, str] = None,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            cache: bool = True,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        # data transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                transforms.EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),  # to tensor
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),  # to tensor
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = SingleDataset(
                self.hparams.data_dir,
                self.hparams.data_types,
                baseline=True,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.train_transforms,
                cache=self.hparams.cache,
            )
            self.data_test = SingleDataset(
                self.hparams.data_dir,
                self.hparams.data_types,
                baseline=False,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.test_transforms,
                cache=self.hparams.cache,
            )

            os.environ["DATASET_TRAIN_COUNT"] = str(len(self.data_train))
            os.environ["DATASET_VAL_COUNT"] = str(len(self.data_test))

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=ConcatDataset(
                [
                    self.data_train,
                    self.data_test
                ]
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
