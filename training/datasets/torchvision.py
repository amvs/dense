from pathlib import Path
import hashlib
import inspect
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from dense.helpers import LoggerManager
import torch




def _stats_cache_path(identifier: str, resize: int | None, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    suffix = f"r{resize}" if resize is not None else "rNone"
    return p / f"meanstd_{h}_{suffix}.npz"


def _make_dataset_identifier(dataset, dataset_name: str | None) -> str:
    base_ds = dataset
    indices = getattr(dataset, "indices", None)
    for _ in range(5):
        parent = getattr(base_ds, "dataset", None)
        if parent is None:
            break
        parent_indices = getattr(parent, "indices", None)
        if indices is not None and parent_indices is not None:
            indices = [parent_indices[i] for i in indices]
        elif indices is None and parent_indices is not None:
            indices = parent_indices
        base_ds = parent

    base_root = getattr(base_ds, "root", None) or getattr(base_ds, "root_dir", None) or ""
    classes = getattr(base_ds, "classes", None)
    class_str = ",".join(classes) if classes is not None else "none"
    name = dataset_name or type(base_ds).__name__
    return f"{name}|root={base_root}|n={len(dataset)}|classes={class_str}"


def _get_targets(dataset):
    if isinstance(dataset, Subset):
        base_targets = _get_targets(dataset.dataset)
        if base_targets is None:
            return None
        return [base_targets[i] for i in dataset.indices]
    if isinstance(dataset, ConcatDataset):
        targets = []
        for ds in dataset.datasets:
            ds_targets = _get_targets(ds)
            if ds_targets is None:
                return None
            targets.extend(ds_targets)
        return targets
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "labels"):
        return list(dataset.labels)
    if hasattr(dataset, "y"):
        return list(dataset.y)
    if hasattr(dataset, "samples"):
        return [s[1] for s in dataset.samples]
    return None


def _stratified_split(dataset, train_ratio: float, seed: int = 42):
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    train_len = min(train_len, max(1, total_len - 1))
    targets = _get_targets(dataset)
    if targets is None:
        raise ValueError("Unable to extract targets for stratified split.")
    indices = np.arange(total_len)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_len, random_state=seed)
    train_idx, test_idx = next(sss.split(indices, targets))
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def _set_transform(dataset, transform):
    if isinstance(dataset, Subset):
        _set_transform(dataset.dataset, transform)
    elif isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            _set_transform(ds, transform)
    elif hasattr(dataset, "transform"):
        dataset.transform = transform


def _init_torchvision_dataset(dataset_cls, root: str, split: str, transform, download: bool, dataset_kwargs):
    params = inspect.signature(dataset_cls).parameters
    kwargs = dict(dataset_kwargs or {})
    if "root" in params and "root" not in kwargs:
        kwargs["root"] = root
    if "transform" in params and "transform" not in kwargs:
        kwargs["transform"] = transform
    if "download" in params and "download" not in kwargs:
        kwargs["download"] = download
    if "train" in params and "train" not in kwargs and "split" not in kwargs:
        kwargs["train"] = split == "train"
    if "split" in params and "split" not in kwargs:
        kwargs["split"] = split
    return dataset_cls(**kwargs)


def compute_mean_std(dataset, batch_size: int = 64, num_workers: int = 4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channel_sum = None
    channel_sq_sum = None
    num_pixels = 0
    num_batches = 0

    for imgs, _ in loader:
        if imgs.dim() != 4:
            raise ValueError("Expected images in NCHW format.")
        imgs = imgs.to(device)
        if channel_sum is None:
            channel_sum = imgs.sum(dim=(0, 2, 3))
            channel_sq_sum = (imgs ** 2).sum(dim=(0, 2, 3))
        else:
            channel_sum += imgs.sum(dim=(0, 2, 3))
            channel_sq_sum += (imgs ** 2).sum(dim=(0, 2, 3))
        num_pixels += imgs.size(0) * imgs.size(2) * imgs.size(3)
        num_batches += 1

    mean = channel_sum / num_pixels
    std = (channel_sq_sum / num_pixels - mean ** 2).sqrt()
    logger = LoggerManager.get_logger()
    logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total) on {device}")
    return mean.cpu().tolist(), std.cpu().tolist()


def load_or_compute_mean_std(dataset, resize: int | None, dataset_name: str | None, force_recompute: bool = False):
    logger = LoggerManager.get_logger()
    identifier = _make_dataset_identifier(dataset, dataset_name)
    cache_path = _stats_cache_path(identifier, resize)

    if cache_path.exists() and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_resize = data.get("resize", resize)
            if (cached_resize is None and resize is None) or int(cached_resize) == int(resize):
                mean = data["mean"].tolist()
                std = data["std"].tolist()
                logger.info(f"Loaded mean/std from cache {cache_path}")
                return mean, std
            logger.info("Cache mismatch (resize). Recomputing stats.")
        except Exception:
            logger.info(f"Failed to load cache {cache_path}; recomputing stats")

    mean, std = compute_mean_std(dataset)
    try:
        np.savez_compressed(cache_path, mean=mean, std=std, resize=resize)
        logger.info(f"Saved mean/std to cache {cache_path}")
    except Exception as e:
        logger.info(f"Failed to save mean/std cache: {e}")
    return mean, std


def get_torchvision_loaders(
    dataset_cls,
    dataset_name: str | None = None,
    root: str = "data",
    resize: int | None = None,
    batch_size: int = 64,
    train_ratio: float = 0.8,
    worker_init_fn=None,
    drop_last: bool = True,
    download: bool = True,
    num_workers: int = 4,
    dataset_kwargs=None,
    dataset_kwargs_train=None,
    dataset_kwargs_test=None,
    pre_transforms=None,
):
    logger = LoggerManager.get_logger()

    base_transforms = []
    if pre_transforms is not None:
        base_transforms.extend(pre_transforms)
    if resize is not None:
        base_transforms.append(transforms.Resize((resize, resize)))
    base_transforms.append(transforms.ToTensor())
    base_transform = transforms.Compose(base_transforms)

    if dataset_kwargs_train is None and dataset_kwargs_test is None:
        dataset_kwargs_train = dict(dataset_kwargs or {})
        dataset_kwargs_test = dict(dataset_kwargs or {})

    train_base = _init_torchvision_dataset(
        dataset_cls,
        root=root,
        split="train",
        transform=base_transform,
        download=download,
        dataset_kwargs=dataset_kwargs_train,
    )
    test_base = _init_torchvision_dataset(
        dataset_cls,
        root=root,
        split="test",
        transform=base_transform,
        download=download,
        dataset_kwargs=dataset_kwargs_test,
    )

    if train_ratio == 1.0:
        train_dataset = train_base
        test_dataset = test_base
    else:
        full_dataset = ConcatDataset([train_base, test_base])
        train_dataset, test_dataset = _stratified_split(full_dataset, train_ratio=train_ratio)

    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize, dataset_name=dataset_name)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")

    norm_transform = transforms.Compose(base_transforms + [transforms.Normalize(mean, std)])
    _set_transform(train_dataset, norm_transform)
    _set_transform(test_dataset, norm_transform)
    logger.info("Updated dataset transforms with normalization")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        num_workers=num_workers,
    )

    nb_class = len(getattr(train_base, "classes", []))
    sample_img, _ = train_dataset[0]
    logger.info(f"[Ratio:{train_ratio}] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    return train_loader, test_loader, nb_class, sample_img.shape


def get_cifar_loaders(batch_size=64, train_ratio=0.8, worker_init_fn=None, drop_last=True, **kwargs):
    return get_torchvision_loaders(
        datasets.CIFAR10,
        dataset_name="cifar10",
        root="data",
        batch_size=batch_size,
        train_ratio=train_ratio,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        download=True,
        **kwargs
    )