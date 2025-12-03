from torch.utils.data import random_split, DataLoader
from dense.helpers import LoggerManager
import torch

def split_train_val(train_dataset, val_ratio=0.1, batch_size=64, seed=123, drop_last=False, train_size=None):
    """
    Split an existing train_dataset into train and validation subsets
    with a fixed seed for reproducibility.

    Args:
        train_dataset: The dataset to split
        val_ratio: Ratio of data to reserve for validation (default: 0.1)
        batch_size: Batch size for both loaders (default: 64)
        seed: Random seed for reproducibility (default: 123)
        drop_last: Whether to drop the last incomplete batch (default: False)
        train_size: Optional fixed size for training set. If provided, limits 
                    training data to this size, with remaining data used for validation.
                    Cannot exceed total_len - int(total_len * val_ratio).

    Returns:
        Tuple of (train_loader, val_loader)
    """
    total_len = len(train_dataset)
    if train_size is not None:
        train_size = min(train_size, total_len - int(total_len * val_ratio))
    else:
        train_size = total_len - int(total_len * val_ratio)

    val_len = total_len - train_size

    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_len],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=drop_last)
    logger = LoggerManager.get_logger()
    logger.info(f"Split train dataset into train and val subsets...")
    logger.info(f"[Train size: {train_size}, Val size: {val_len}]")

    return train_loader, val_loader