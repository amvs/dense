"""
KTH-TIPS (original) dataset loader.

The original KTH-TIPS dataset has:
- 10 classes
- 81 images per class
- Grayscale images
"""
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
from dense.helpers import LoggerManager
from training.datasets.base import stratify_split, CenterCropToSquare
from training.datasets.balanced_sampler import BalancedBatchSampler
from training.datasets.scale_augmentation import ScaleAugmentedDataset
import hashlib
import numpy as np
from collections import defaultdict
import random


class KTHTipsDataset(Dataset):
    """
    Original KTH-TIPS texture dataset.
    
    Directory structure (typical):
        root/
            class_0/
                img_*.png
            class_1/
                img_*.png
            ...
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to dataset root directory
            transform: Transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Classes are subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Collect all images from each class
        for cls_name in self.classes:
            cls_path = self.root_dir / cls_name
            # Collect image files (png, jpg, bmp, etc.)
            img_files = (
                sorted(cls_path.glob("*.png")) +
                sorted(cls_path.glob("*.jpg")) +
                sorted(cls_path.glob("*.JPG")) +
                sorted(cls_path.glob("*.bmp")) +
                sorted(cls_path.glob("*.BMP"))
            )
            for img_file in img_files:
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_dataset_statistics(dataset):
    """
    Compute dataset statistics: number of classes and examples per class.
    
    Returns:
        dict with keys:
            - num_classes: int
            - examples_per_class: dict mapping class_idx -> count
            - min_examples_per_class: int (minimum across all classes)
            - max_examples_per_class: int (maximum across all classes)
            - total_examples: int
            - is_balanced: bool (True if all classes have same count)
    """
    class_to_count = defaultdict(int)
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_count[label] += 1
    
    num_classes = len(class_to_count)
    examples_per_class = dict(class_to_count)
    counts = list(examples_per_class.values())
    min_examples = min(counts)
    max_examples = max(counts)
    total_examples = sum(counts)
    is_balanced = min_examples == max_examples
    
    return {
        "num_classes": num_classes,
        "examples_per_class": examples_per_class,
        "min_examples_per_class": min_examples,
        "max_examples_per_class": max_examples,
        "total_examples": total_examples,
        "is_balanced": is_balanced
    }


def create_balanced_train_subset(dataset, example_per_class, seed=42):
    """
    Create a balanced training subset with exactly example_per_class examples from each class.
    
    Args:
        dataset: Dataset to sample from
        example_per_class: Number of examples to take from each class
        seed: Random seed for reproducibility
    
    Returns:
        Subset with balanced examples
    """
    rng = random.Random(seed)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_to_indices[label].append(idx)
    
    # Check that we have enough examples per class
    for cls, indices in class_to_indices.items():
        if len(indices) < example_per_class:
            raise ValueError(
                f"Class {cls} has only {len(indices)} examples, "
                f"cannot sample {example_per_class} examples per class."
            )
    
    # Sample example_per_class from each class
    train_indices = []
    for cls in sorted(class_to_indices.keys()):
        indices = class_to_indices[cls]
        rng.shuffle(indices)
        train_indices.extend(indices[:example_per_class])
    
    return Subset(dataset, train_indices)


def compute_mean_std(dataset):
    """Compute mean and std for grayscale dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.tensor(0.0)
    sq_mean = torch.tensor(0.0)
    num_pixels = 0
    num_batches = 0

    for idx, (imgs, _) in enumerate(loader):
        # imgs shape: (batch, 1, H, W) for grayscale
        batch_size = imgs.size(0)
        imgs_flat = imgs.view(batch_size, -1)  # (batch, H*W)
        mean += imgs_flat.sum()
        sq_mean += (imgs_flat ** 2).sum()
        num_pixels += batch_size * imgs_flat.size(1)
        num_batches += 1

    mean = (mean / num_pixels).item()
    var = (sq_mean / num_pixels) - mean ** 2
    std = var.sqrt().item()
    
    logger = LoggerManager.get_logger()
    logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total)")
    return (mean,), (std,)


def _stats_cache_path(identifier: str, resize: int, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"meanstd_{h}_r{resize}.npz"


def _make_kthtips_identifier(dataset, root_dir) -> str:
    base_dir = root_dir
    classes = getattr(dataset, 'classes', None)
    return f"kthtips|{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}"


def load_or_compute_mean_std(dataset, resize: int, root_dir: str = None, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.
    
    For grayscale images, returns tuple of 1 value: (mean,) and (std,).
    """
    logger = LoggerManager.get_logger()
    identifier = _make_kthtips_identifier(dataset, root_dir)
    cache_path = _stats_cache_path(identifier, resize)

    if cache_path.exists() and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_resize = int(data.get('resize', resize))
            if cached_resize == resize:
                mean = tuple(data['mean'].tolist())
                std = tuple(data['std'].tolist())
                logger.info(f"Loaded mean/std from cache {cache_path}")
                return mean, std
            else:
                logger.info(f"Cache mismatch (resize). Recomputing stats.")
        except Exception as e:
            logger.info(f"Failed to load cache {cache_path}; recomputing stats: {e}")

    # Compute mean/std and save
    mean, std = compute_mean_std(dataset)
    try:
        np.savez_compressed(cache_path, mean=np.array(mean), std=np.array(std), resize=resize)
        logger.info(f"Saved mean/std to cache {cache_path}")
    except Exception as e:
        logger.info(f"Failed to save mean/std cache: {e}")
    
    return mean, std


def get_kthtips_loaders(root_dir, resize, batch_size=64, worker_init_fn=None,
                        example_per_class=None, drop_last=False, seed=42,
                        use_balanced_batches=True, use_scale_augmentation=False,
                        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Load original KTH-TIPS dataset.
    
    Args:
        root_dir: Path to KTH-TIPS dataset root directory
        resize: Target size for resizing (e.g., 200)
        batch_size: Batch size for data loaders (must be multiple of num_classes if use_balanced_batches=True)
        worker_init_fn: Worker initialization function
        example_per_class: Number of examples per class for training (creates balanced dataset)
        drop_last: Whether to drop last incomplete batch
        seed: Random seed for reproducibility
        use_balanced_batches: If True, use BalancedBatchSampler to ensure equal examples per class in each batch
        use_scale_augmentation: If True, augment training images with scale factors [1, sqrt(2), 2, 2*sqrt(2)]
        train_ratio: Ratio of training data (default: 0.7)
        val_ratio: Ratio of validation data (default: 0.15)
        test_ratio: Ratio of test data (default: 0.15)
        
    Returns:
        train_loader, val_loader, test_loader, nb_class, sample_img.shape, stats_dict
    """
    logger = LoggerManager.get_logger()
    
    # Validate ratios sum to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Initial transform: center crop to square, then to tensor and resize
    transform = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
    ])
    
    # Create full dataset
    full_dataset = KTHTipsDataset(root_dir, transform=transform)
    
    # Compute dataset statistics
    stats = compute_dataset_statistics(full_dataset)
    nb_class = stats["num_classes"]
    logger.info(f"Dataset statistics: {stats}")
    
    # Split dataset into train/val/test
    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    test_len = total_len - train_len - val_len
    
    # Stratified split to maintain class distribution
    train_dataset_raw, val_test_dataset = stratify_split(full_dataset, train_size=train_len, seed=seed)
    val_dataset, test_dataset = stratify_split(val_test_dataset, train_size=val_len, seed=seed)
    
    logger.info(f"Split dataset: Train={len(train_dataset_raw)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Apply example_per_class to create balanced training subset
    if example_per_class is not None:
        # Check that we have enough examples per class in training set
        train_stats = compute_dataset_statistics(train_dataset_raw)
        if example_per_class > train_stats["min_examples_per_class"]:
            raise ValueError(
                f"example_per_class ({example_per_class}) exceeds minimum examples per class in training set "
                f"({train_stats['min_examples_per_class']})"
            )
        train_dataset = create_balanced_train_subset(train_dataset_raw, example_per_class, seed=seed)
        logger.info(f"Created balanced training subset: {example_per_class} examples per class = {len(train_dataset)} total")
        # Update stats to reflect actual training set
        stats["train_examples_per_class"] = example_per_class
        stats["train_total_examples"] = len(train_dataset)
    else:
        # Use all available training data
        train_dataset = train_dataset_raw
        train_stats = compute_dataset_statistics(train_dataset)
        stats["train_examples_per_class"] = train_stats["min_examples_per_class"]
        stats["train_total_examples"] = len(train_dataset)
        logger.info(f"Using all training data: {len(train_dataset)} examples")
    
    # Compute (or load cached) mean and std from ORIGINAL TRAIN set (before augmentation)
    # Normalization stats are computed from original training data, not augmented data
    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples (before augmentation)...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize, root_dir=root_dir)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")
    
    # Apply normalization using train statistics to all splits
    # For grayscale: mean and std are tuples of single values
    normalized_transform_full = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std)
    ])
    
    # Apply normalization to all datasets
    train_dataset.transform = normalized_transform_full
    val_dataset.transform = normalized_transform_full
    test_dataset.transform = normalized_transform_full
    logger.info(f"Applied normalization transforms to all datasets")
    
    # Apply scale augmentation to training dataset AFTER normalization
    # Each training image is augmented to have 4 different scales
    if use_scale_augmentation:
        logger.info("Applying scale augmentation to training dataset (after normalization)...")
        original_train_size = len(train_dataset)
        # Wrap training dataset with scale augmentation
        # ScaleAugmentedDataset will create 4 versions of each image
        train_dataset = ScaleAugmentedDataset(train_dataset, target_size=resize)
        logger.info(f"Scale augmentation: {original_train_size} -> {len(train_dataset)} samples (4x increase)")
        # Update stats to reflect augmented dataset
        stats["train_total_examples"] = len(train_dataset)
        stats["train_examples_per_class"] = stats["train_examples_per_class"] * 4  # 4 scales per example
    
    # Create data loaders with balanced batches if requested
    if use_balanced_batches:
        # Validate batch_size is multiple of num_classes
        if batch_size % nb_class != 0:
            # Adjust batch_size to nearest multiple
            adjusted_batch_size = (batch_size // nb_class) * nb_class
            if adjusted_batch_size == 0:
                adjusted_batch_size = nb_class
            logger.warning(
                f"batch_size ({batch_size}) is not a multiple of num_classes ({nb_class}). "
                f"Adjusting to {adjusted_batch_size}."
            )
            batch_size = adjusted_batch_size
        
        # Use balanced batch sampler for training
        train_sampler = BalancedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            num_classes=nb_class,
            shuffle=True,
            seed=seed
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            worker_init_fn=worker_init_fn,
            drop_last=False
        )
        logger.info(f"Using BalancedBatchSampler: {batch_size // nb_class} examples per class per batch")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            drop_last=False
        )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False)
    
    sample_img, _ = train_dataset[0]
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    
    return train_loader, val_loader, test_loader, nb_class, sample_img.shape, stats


if __name__ == "__main__":
    # Test script
    root_dir = "data/datasets/KTH-TIPS"
    train_loader, val_loader, test_loader, nb_class, img_shape, stats = get_kthtips_loaders(
        root_dir=root_dir,
        resize=200,
        batch_size=30,  # Must be multiple of 10 (num_classes)
        example_per_class=5,
        use_balanced_batches=True,
        use_scale_augmentation=True
    )
    print(f"Dataset loaded: {nb_class} classes, image shape: {img_shape}")
