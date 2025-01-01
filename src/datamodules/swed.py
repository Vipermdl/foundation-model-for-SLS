import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
os.environ["GDAL_NUM_THREADS"] = "4"

import torch
import lightning.pytorch as pl

from src.datasets import SWED

S2_MEAN = torch.tensor(
    [ 560.0944, 669.804, 938.80133, 1104.3877, 1374.6317, 1826.4297, 2012.0166, 2095.8945, 2159.6338, 2191.1506, 2105.7383, 1568.9834]
)

S2_STD = torch.tensor(
    [175.07619, 236.3873, 268.17673, 328.9421, 326.24823, 404.8634, 447.36502, 486.22122, 464.84232, 450.85526, 413.28418, 369.56287]
)


class SWEDDataModule(pl.LightningDataModule):
    """Pytorch Lightning data module class for mados."""

    mean = S2_MEAN
    std = S2_STD

    def __init__(
        self,
        root="data/",
        batch_size=32,
        num_workers=0,
        modality=None,
        train_transforms=None,
        val_transforms=None,
    ):
        """MADOSDataModule constructor."""
        super(SWEDDataModule).__init__()
        self.root = root
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers
        self.modality = modality
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def prepare_data(self):
        """Method to prepare data."""
        pass

    def setup(
        self,
        stage="fit",
        drop_last=False,
    ):
        """Method to setup dataset and corresponding splits."""
        for split in ["train", "val", "test"]:            
            assert self.modality in ['s1', 's2']
            if split == 'train':
                setattr(
                    self,
                    f"{split}_dataset",
                    SWED(
                        self.root,
                        split=split,
                        transforms=self.train_transforms,
                    ),
                )
            else:
                setattr(
                    self,
                    f"{split}_dataset",
                    SWED(
                        self.root,
                        split='test',
                        transforms=self.val_transforms,
                    ),
                )

        self.drop_last = drop_last

    def train_dataloader(self):
        """Return training dataset loader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        """Return validation dataset loader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        """Return test dataset loader."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )
