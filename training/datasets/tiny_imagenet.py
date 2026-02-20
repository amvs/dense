from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from dense.helpers import LoggerManager
import os
from training.datasets.base import stratify_split
import hashlib
import numpy as np


class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        TinyImageNet dataset with train/val/test splits.

        The dataset structure is:
        - train/: Contains class folders (e.g., n02124075/) with images
        - val/: Contains images/ folder and val_annotations.txt
        - test/: Contains images/ folder

        Parameters
        ----------
        root_dir : str
            Root directory containing 'train', 'val', 'test' subfolders
        split : str
            One of 'train', 'val', 'test'
        transform : callable
            Transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}

        if split == 'train':
            self._load_train_split()
        elif split == 'val':
            self._load_val_split()
        elif split == 'test':
            self._load_test_split()
        else:
            raise ValueError(f"Invalid split: {split}")

    def _load_class_mapping(self):
        """Load class mapping from wnids.txt."""
        wnids_file = self.root_dir / 'wnids.txt'
        if wnids_file.exists():
            with open(wnids_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            raise FileNotFoundError(f"wnids.txt not found at {self.root_dir}")

    def _load_train_split(self):
        """Load training split with class folders."""
        self._load_class_mapping()
        train_dir = self.root_dir / 'train'

        for cls in self.classes:
            cls_folder = train_dir / cls
            if cls_folder.exists():
                # TinyImageNet train may have nested structure: train/class_id/images/
                # or directly train/class_id/
                for img_file in sorted(cls_folder.rglob("*.JPEG")) + sorted(cls_folder.rglob("*.jpeg")):
                    self.image_paths.append(img_file)
                    self.labels.append(self.class_to_idx[cls])

    def _load_val_split(self):
        """Load validation split with annotations file."""
        self._load_class_mapping()
        val_dir = self.root_dir / 'val'
        images_dir = val_dir / 'images'
        annotations_file = val_dir / 'val_annotations.txt'

        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        filename = parts[0]
                        class_id = parts[1]
                        img_path = images_dir / filename
                        if img_path.exists():
                            self.image_paths.append(img_path)
                            self.labels.append(self.class_to_idx[class_id])

    def _load_test_split(self):
        """Load test split (no labels)."""
        self._load_class_mapping()
        test_dir = self.root_dir / 'test' / 'images'
        if test_dir.exists():
            self.image_paths = sorted(test_dir.glob("*.JPEG")) + sorted(test_dir.glob("*.jpeg"))
            self.labels = [-1] * len(self.image_paths)  # dummy labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_mean_std(dataset):
    """Compute mean and std for a dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = torch.zeros(3, device=device)
    sq_mean = torch.zeros(3, device=device)
    num_pixels = 0
    num_batches = 0

    for idx, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device)
        # Sum over batch, height, width (N, H, W)
        mean += imgs.sum(dim=(0, 2, 3))
        sq_mean += (imgs ** 2).sum(dim=(0, 2, 3))
        num_pixels += imgs.size(0) * imgs.size(2) * imgs.size(3)
        num_batches += 1

    mean /= num_pixels
    std = (sq_mean / num_pixels - mean ** 2).sqrt()
    logger = LoggerManager.get_logger()
    logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total) on {device}")
    return mean.cpu().tolist(), std.cpu().tolist()


def _stats_cache_path(identifier: str, resize: int, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"meanstd_{h}_r{resize}.npz"


def _make_dataset_identifier(dataset) -> str:
    # Unwrap Subset if needed
    base_ds = getattr(dataset, 'dataset', dataset)
    base_dir = getattr(base_ds, 'root_dir', None) or getattr(base_ds, 'root', None) or str(getattr(base_ds, 'image_paths', '')[:5])
    classes = getattr(base_ds, 'classes', None)
    # include a short fingerprint: base_dir + number of samples + class list
    return f"{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}"


def load_or_compute_mean_std(dataset, resize: int, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache."""
    logger = LoggerManager.get_logger()
    identifier = _make_dataset_identifier(dataset)
    cache_path = _stats_cache_path(identifier, resize)

    if cache_path.exists() and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_resize = int(data.get('resize', resize))
            if cached_resize == resize:
                mean = data['mean'].tolist()
                std = data['std'].tolist()
                logger.info(f"Loaded mean/std from cache {cache_path}")
                return mean, std
            else:
                logger.info(f"Cache mismatch (resize). Recomputing stats.")
        except Exception:
            logger.info(f"Failed to load cache {cache_path}; recomputing stats")

    # compute mean/std and save
    mean, std = compute_mean_std(dataset)
    try:
        np.savez_compressed(cache_path, mean=mean, std=std, resize=resize)
        logger.info(f"Saved mean/std to cache {cache_path}")
    except Exception as e:
        logger.info(f"Failed to save mean/std cache: {e}")
    return mean, std


def get_tinyimagenet_loaders(root_dir, resize=64, batch_size=64, train_ratio=0.8, train_val_ratio=4, worker_init_fn=None, drop_last=True):
    """
    Load TinyImageNet dataset with train/val/test splits.
    
    The original 'train' split is split into train and val.
    The original 'val' split becomes the test set.
    The original 'test' split is not used (no labels).

    Parameters
    ----------
    root_dir : str
        Root directory containing 'train', 'val', 'test' subfolders
    resize : int
        Image size for resizing
    batch_size : int
        Batch size for DataLoader
    train_ratio : float
        Ratio of data used for training+validation (from original train split)
    train_val_ratio : int
        Ratio between train and val sets (default 4:1)
    worker_init_fn : callable
        Worker initialization function for DataLoader
    drop_last : bool
        Whether to drop last batch if incomplete

    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    test_loader : DataLoader
        Test data loader (original val split)
    nb_class : int
        Number of classes
    sample_shape : tuple
        Shape of sample image (C, H, W)
    """
    logger = LoggerManager.get_logger()

    logger.info(f"Loading TinyImageNet dataset from {root_dir}...")

    # Initial transform without normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
    ])

    # Load original train and val datasets
    full_train_dataset = TinyImageNetDataset(root_dir, split='train', transform=transform)
    test_dataset = TinyImageNetDataset(root_dir, split='val', transform=transform)

    logger.info(f"Dataset loaded. Full train: {len(full_train_dataset)}, Test (val): {len(test_dataset)}")

    # Split the original train into train and val using stratified split
    train_len = int(len(full_train_dataset) * train_ratio)
    val_len = train_len // train_val_ratio
    used_len = train_len + val_len
    
    if used_len > len(full_train_dataset):
        val_len = len(full_train_dataset) - train_len
        used_len = len(full_train_dataset)
    
    # First split: get the used portion
    used_dataset, _ = stratify_split(
        full_train_dataset, train_size=used_len, seed=42
    )
    
    # Second split: split used portion into train and val
    train_dataset, val_dataset = stratify_split(
        used_dataset, train_size=train_len, seed=43
    )

    logger.info(f"Split train dataset - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # compute (or load cached) mean and std from TRAIN set only (no data leakage)
    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")

    # Apply normalization using train statistics to all splits
    normalized_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std)
    ])

    # Update datasets' transforms
    train_dataset.dataset.transform = normalized_transform
    val_dataset.dataset.transform = normalized_transform
    test_dataset.transform = normalized_transform

    logger.info(f"Updated dataset transforms with normalization")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             worker_init_fn=worker_init_fn, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)

    nb_class = len(full_train_dataset.classes)
    sample_img, _ = train_dataset[0]

    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")

    return train_loader, val_loader, test_loader, nb_class, sample_img.shape
