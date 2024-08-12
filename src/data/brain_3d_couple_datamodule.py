from typing import Any, Dict, Optional, Tuple, Sequence

import os
import numpy as np
import json
import pickle
import joblib
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, SubsetRandomSampler
from monai import transforms
from nilearn.image import load_img
from nilearn.maskers import NiftiMasker
from .utiles import apply_covariance_correction, uniform
from tqdm import tqdm
from lightning.pytorch.utilities import CombinedLoader
from .components.disease_dataset import SingleDataset, CombinedDataset


def load_patient_info(json_file):
    with open(json_file, "r") as file:
        return json.load(file)


class Brain3DCoupleDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str = "data/",
            data_types: Sequence[str] = ('AD',),
            infinite_data_types: Sequence[str] = ('CN',),
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
        self.transforms = transforms.Compose(
            [
                transforms.EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),  # to tensor
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_train_infinite: Optional[Dataset] = None
        self.data_test_infinite: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test and not self.data_train_infinite and not self.data_test_infinite:
            self.data_train = SingleDataset(
                self.hparams.data_dir,
                self.hparams.data_types,
                baseline=True,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.transforms,
                cache=self.hparams.cache,
            )

            self.data_test = SingleDataset(
                self.hparams.data_dir,
                self.hparams.data_types,
                baseline=False,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.transforms,
                cache=self.hparams.cache,
            )

            os.environ["DATASET_TRAIN_COUNT"] = str(len(self.data_train))
            os.environ["DATASET_VAL_COUNT"] = str(len(self.data_test))

            self.data_train_infinite = SingleDataset(
                self.hparams.data_dir,
                self.hparams.infinite_data_types,
                baseline=True,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.transforms,
                cache=self.hparams.cache,
            )

            self.data_test_infinite = SingleDataset(
                self.hparams.data_dir,
                self.hparams.infinite_data_types,
                baseline=False,
                info_keys=self.hparams.info_keys,
                remove_covar=self.hparams.remove_covar,
                preprocess=self.hparams.preprocess,
                transform=self.transforms,
                cache=self.hparams.cache,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=CombinedDataset(self.data_train, self.data_train_infinite),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=CombinedDataset(self.data_test, self.data_test_infinite),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=ConcatDataset(
                [
                    CombinedDataset(self.data_train, self.data_train_infinite),
                    CombinedDataset(self.data_test, self.data_test_infinite)
                ]
            ),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
