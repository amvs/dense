from torch.utils.data import random_split, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from dense.helpers import LoggerManager
import torch

def stratify_split(dataset, train_size, seed=123):
    """
    Stratified random split.
    To ensure that each subset has approximately the same class distribution

    Args:
        dataset: torch Dataset returning (x, y)
        train_size: int or float, specifies the size or ratio of the first subset
        seed: random seed for reproducibility

    Returns:
        A tuple of Subset objects with the given lengths.

    Usage:
        subset1, subset2 = stratify_split(dataset, train_size, seed=42)
    """
    total_len = len(dataset)
    # extract targets, assuming dataset[i] (__getitem__(i)) returns (x, y)
    targets = [dataset[i][1] for i in range(total_len)]

    # Single stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed
    )

    idx1, idx2 = next(sss.split(range(total_len), targets))

    subset1 = Subset(dataset, idx1)
    subset2 = Subset(dataset, idx2)

    return subset1, subset2

def split_train_val(train_dataset, train_ratio=0.1, batch_size=64, seed=123, train_val_ratio=4, drop_last=False):
    """
    Split an existing train_dataset further into train and validation subsets
    with a fixed seed for reproducibility.
    When train subset is small, the validation subset will also be small.
    The ratio between train and val subsets is determined by train_val_ratio.
    By default, train : val = 4 : 1

    Args:
        train_dataset: torch.utils.data.Dataset
            The original training dataset to be split.
        train_ratio: float
            The ratio of the training subset to the total dataset.
    """
    total_len = len(train_dataset)
    train_len = int(total_len * train_ratio)
    val_len = train_len // train_val_ratio
    used_len = train_len + val_len
    if used_len > total_len:
        val_len = total_len - train_len
        used_len = total_len # adjust used_len accordingly
        train_subset, val_subset = random_split(
            train_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(seed)
        )
    else: # we discard some data to maintain the train:val ratio
          # this mimics the situation when dataset is small in practice
        used_dataset, _ = stratify_split(
            train_dataset, train_size=used_len,
            seed=seed
        )
        train_subset, val_subset = stratify_split(
            used_dataset, train_size=train_len,
            seed=seed
        )
    discard_len = total_len - len(train_subset) - len(val_subset)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    logger = LoggerManager.get_logger()
    logger.info(f"Split train dataset into train and val subsets...")
    logger.info(f"[Train Ratio: {train_ratio}] Train size: {train_len}, "
                 f"Val size: {val_len}, Used Size: {used_len}, Discard Size: {discard_len}, "
                 f"Train:Val={train_val_ratio}:1")

    return train_loader, val_loader