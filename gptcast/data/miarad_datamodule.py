from typing import Any, Dict, Optional, Tuple, Union

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from pandas import DataFrame
from pathlib import Path
 
from gptcast.data.miarad import Miarad, MiaradN
from gptcast.utils.downloads import download_dataset


class MiaradDataModule(LightningDataModule):
    """LightningDataModule for Miarad dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    default_data_path = Path(__file__).parent.parent.parent.resolve() / "data"
    default_train_tarfile_path = str(default_data_path / "miarad_training.tar"),
    default_train_metadata_path = str(default_data_path / "miarad_training.csv"),
    default_test_tarfile_path = str(default_data_path / "miarad_test.tar"),
    default_test_metadata_path = str(default_data_path / "miarad_test.csv"),

    @classmethod
    def  load_from_zenodo(cls, path: Optional[str] = None, **kwargs) -> 'MiaradDataModule':
        """Load the dataset from Zenodo. 

        Args:
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            MiaradDataModule: The dataset.
        """
        data_path = cls.default_data_path if path is None else Path(path)
        assert download_dataset("miarad", data_path), "Failed to download miarad dataset"
        return cls(
            train_tarfile_path = str(data_path / "miarad_training.tar"),
            train_metadata_path_or_df = str(data_path / "miarad_training.csv"),
            test_tarfile_path = str(data_path / "miarad_test.tar"),
            test_metadata_path_or_df = str(data_path / "miarad_test.csv"),
            **kwargs
        )


    def __init__(
        self,
        *,
        train_tarfile_path: str = default_train_tarfile_path,
        train_metadata_path_or_df: Union[str, DataFrame] = default_train_metadata_path,
        test_tarfile_path: str = default_test_tarfile_path,
        test_metadata_path_or_df: Union[str, DataFrame] = default_test_metadata_path,
        clip_and_normalize: Tuple[float] = (0,60,-1,1),
        crop: int = 256,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        seq_len: int = 1,
        stack_seq: str = None,
        smart_crop: bool = False,
    ):
        super().__init__()

        

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        if self.data_train is None and stage == "fit":
            if self.hparams.seq_len > 1:
                trainset = MiaradN(
                    crop = self.hparams.crop, 
                    clip_and_normalize = self.hparams.clip_and_normalize,
                    tarfile_path = self.hparams.train_tarfile_path,
                    metadata_path_or_df = self.hparams.train_metadata_path_or_df,
                    seq_len = self.hparams.seq_len,
                    stack_seq = self.hparams.stack_seq,
                    smart_crop = self.hparams.smart_crop
                )
            else:
                trainset = Miarad(
                    crop = self.hparams.crop, 
                    clip_and_normalize = self.hparams.clip_and_normalize,
                    tarfile_path = self.hparams.train_tarfile_path,
                    metadata_path_or_df = self.hparams.train_metadata_path_or_df
                )
            self.data_train, self.data_val, = random_split(
                dataset=trainset,
                lengths=[0.95, 0.05],
                generator=torch.Generator().manual_seed(42),
            )

        if self.data_test is None and stage == "test":
            if self.hparams.seq_len > 1:
                testset = MiaradN(
                    crop = self.hparams.crop, 
                    clip_and_normalize = self.hparams.clip_and_normalize,
                    tarfile_path = self.hparams.test_tarfile_path,
                    metadata_path_or_df = self.hparams.test_metadata_path_or_df,
                    seq_len = self.hparams.seq_len,
                    stack_seq = self.hparams.stack_seq,
                    smart_crop=self.hparams.smart_crop
                )
            else:
                testset = Miarad(
                    crop = self.hparams.crop, 
                    clip_and_normalize = self.hparams.clip_and_normalize,
                    tarfile_path = self.hparams.test_tarfile_path,
                    metadata_path_or_df = self.hparams.test_metadata_path_or_df
                )
            self.data_test = testset

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )


    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
