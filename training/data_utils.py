"""
Shared utilities for data loading and preprocessing.
Used by train_wph.py and train_wph_svm.py.
"""
from training.datasets import get_loaders, split_train_val
from training.datasets.outex import get_available_problems


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
    drop_last = config.get("drop_last", True)
    if dataset in ["mnist", "cifar"]:
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            batch_size=batch_size,
            train_ratio=1-test_ratio,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last
        )
    elif dataset == 'kthtips2b':
        root_dir = config["root_dir"]
        train_loader, val_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            root_dir=root_dir,
            deeper_path=config.get("deeper_path", ''),
            resize=config["resize"],
            batch_size=batch_size,
            train_ratio=config["train_ratio"],
            worker_init_fn=worker_init_fn,
            fold=config.get("fold", None),
            drop_last=drop_last
        )
        return train_loader, val_loader, test_loader, nb_class, image_shape
    elif dataset == 'akash2sharma/tiny-imagenet':
        from pathlib import Path
        root_dir = str(Path(config["root_dir"]) / config.get("deeper_path", ''))
        train_loader, val_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            root_dir=root_dir,
            resize=config["resize"],
            batch_size=batch_size,
            train_ratio=config["train_ratio"],
            train_val_ratio=train_val_ratio,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last
        )
        return train_loader, val_loader, test_loader, nb_class, image_shape
    elif dataset.startswith('outex'):
        root_dir = config["root_dir"]
        available_problems = get_available_problems(root_dir)
        problem_id = available_problems[config.get("fold", 0)]
        train_loader, val_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            root_dir=root_dir,
            resize=config["resize"],
            batch_size=batch_size,
            train_ratio=config["train_ratio"],
            train_val_ratio=train_val_ratio,
            worker_init_fn=worker_init_fn,
            problem_id=problem_id,
            drop_last=drop_last
        )
        # outex loader already returns train/val/test split, no need to split further
        return train_loader, val_loader, test_loader, nb_class, image_shape
    else:
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            resize=resize,
            deeper_path=deeper_path,
            batch_size=batch_size,
            train_ratio=1-test_ratio,
            worker_init_fn=worker_init_fn,
            drop_last=drop_last
        )
    
    # Split train into train/val (only for datasets that don't return val_loader)
    if dataset not in ['kthtips2b', 'akash2sharma/tiny-imagenet'] and not dataset.startswith('outex'):
        train_loader, val_loader = split_train_val(
            train_loader.dataset,
            train_ratio=train_ratio,
            train_val_ratio=train_val_ratio,
            batch_size=batch_size,
            drop_last=drop_last
        )
    
    return train_loader, val_loader, test_loader, nb_class, image_shape
