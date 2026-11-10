"""ISLES-2022 Dataset and Preprocessing Module.

Exports:
- parse_isles_dataset: BIDS folder scanner and case dictionary builder.
- get_kfold_splits: K-Fold cross-validation splitter.
- get_isles_transforms: MONAI preprocessing and patch-sampling pipelines.
- build_isles_dataloaders: PyTorch/MONAI DataLoader constructor.
- ISLESCase: Dataclass representing a multi-modal case.
"""

from src.dataset.dataloader import (
    build_isles_dataloaders,
    build_kfold_dataloaders,
)
from src.dataset.parser import ISLESCase, get_kfold_splits, parse_isles_dataset
from src.dataset.transforms import (
    get_isles_transforms,
    get_train_transforms,
    get_val_transforms,
)

__all__ = [
    "ISLESCase",
    "parse_isles_dataset",
    "get_kfold_splits",
    "get_isles_transforms",
    "get_train_transforms",
    "get_val_transforms",
    "build_isles_dataloaders",
    "build_kfold_dataloaders",
]
