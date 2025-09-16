from torch.utils.data import random_split, DataLoader
from dense.helpers import LoggerManager
import torch

def split_train_val(train_dataset, val_ratio=0.1, batch_size=64, seed=123):
    """
    Split an existing train_dataset into train and validation subsets
    with a fixed seed for reproducibility.
    """
    total_len = len(train_dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len

    train_subset, val_subset = random_split(
        train_dataset, [train_len, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    logger = LoggerManager.get_logger()
    logger.info(f"Split train dataset into train and val subsets...")
    logger.info(f"[Ratio: {val_ratio}] Train size: {train_len}, Val size: {val_len}")

    return train_loader, val_loader