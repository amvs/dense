"""
Shared utilities for data loading and preprocessing.
Used by train_wph.py and train_wph_svm.py.
"""
from functools import partial
from training.datasets import get_loaders, split_train_val


def load_and_split_data(config, worker_init_fn, batch_size=None):
    """
    Load dataset and create train/val/test splits.
    
    Args:
        config: Configuration dictionary
        worker_init_fn: Function for DataLoader worker initialization
        batch_size: Batch size (uses config["batch_size"] if None)
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, nb_class, image_shape)
    """
    dataset = config["dataset"]
    batch_size = batch_size or config["batch_size"]
    test_ratio = config["test_ratio"]
    train_val_ratio = config.get("train_val_ratio", 4)
    train_ratio = config["train_ratio"]
    
    # Load dataset
    if dataset == "mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=1-test_ratio,
            worker_init_fn=worker_init_fn
        )
    else:
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            resize=resize,
            deeper_path=deeper_path,
            batch_size=batch_size,
            train_ratio=1-test_ratio,
            worker_init_fn=worker_init_fn
        )
    
    # Split train into train/val
    train_loader, val_loader = split_train_val(
        train_loader.dataset,
        train_ratio=train_ratio,
        train_val_ratio=train_val_ratio,
        batch_size=batch_size,
        drop_last=True
    )
    
    return train_loader, val_loader, test_loader, nb_class, image_shape
