from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from dense.helpers import LoggerManager
from training.datasets.base import stratify_split
import hashlib
import numpy as np


class CenterCropToSquare:
    """Custom transform to center crop rectangular images to square."""
    def __call__(self, img):
        w, h = img.size
        size = min(w, h)
        left = (w - size) // 2
        top = (h - size) // 2
        right = left + size
        bottom = top + size
        return img.crop((left, top, right, bottom))


class KHTTips2bDataset(Dataset):
    """
    KTH-TIPS2-b texture dataset.
    
    Directory structure:
        root/
            material_1/
                sample_a/
                    img_*.png
                sample_b/
                    img_*.png
                ...
            material_2/
                ...
    """
    def __init__(self, root_dir, transform=None, sample_filter=None):
        """
        Args:
            root_dir: Path to dataset root
            transform: Transform to apply to images
            sample_filter: List of sample names to include (e.g., ['sample_a', 'sample_b'])
                          If None, includes all samples
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.sample_filter = sample_filter

        # Materials are the top-level subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Traverse materials -> samples -> images
        for material in self.classes:
            material_path = self.root_dir / material
            for sample_dir in sorted(material_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                # Filter samples if needed
                if self.sample_filter is not None and sample_dir.name not in self.sample_filter:
                    continue
                # Collect all image files (png, jpg, etc.)
                for img_file in sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.JPG")):
                    self.image_paths.append(img_file)
                    self.labels.append(self.class_to_idx[material])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale (model expects 1 channel)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_kthtips2b_loaders(root_dir, resize, batch_size=64, worker_init_fn=None, fold=None, train_ratio=1.0, drop_last=True):
    """
    Load KTH-TIPS2-b dataset with leave-one-sample-out cross-validation.
    
    Uses 4-fold cross-validation where each fold holds out a different sample:
    - Fold 0: test=sample_a, val=sample_b, train=sample_c+sample_d
    - Fold 1: test=sample_b, val=sample_c, train=sample_a+sample_d
    - Fold 2: test=sample_c, val=sample_d, train=sample_a+sample_b
    - Fold 3: test=sample_d, val=sample_a, train=sample_b+sample_c
    
    Args:
        root_dir: Path to KTH-TIPS2-b dataset root
        resize: Target size for resizing (e.g., 200)
        batch_size: Batch size for data loaders
        worker_init_fn: Worker initialization function
        fold: Fold index (0-3). If None, uses all samples with standard train/val/test split.
        train_ratio: Fraction of training data to use (default 1.0). Set to < 1.0 to reduce training set.
        
    Returns:
        train_loader, val_loader, test_loader, nb_class, sample_img.shape
        
    Note:
        For proper leave-one-sample-out cross-validation, train the model 4 times 
        (once for each fold) and average the results.
    """
    logger = LoggerManager.get_logger()
    
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    
    samples = ['sample_a', 'sample_b', 'sample_c', 'sample_d']
    
    if fold is not None:
        if not 0 <= fold <= 3:
            raise ValueError(f"fold must be 0-3, got {fold}")
        
        # Determine which sample goes to which split
        test_sample = samples[fold]
        val_sample = samples[(fold + 1) % 4]
        train_samples = [samples[(fold + 2) % 4], samples[(fold + 3) % 4]]
        
        logger.info(f"Using fold {fold}: test={test_sample}, val={val_sample}, train={train_samples}")
    else:
        logger.info("No fold specified, using all samples with standard train/val/test split")
        test_sample = None
        val_sample = None
        train_samples = None
    
    # Initial transform: center crop to square, then to tensor and resize
    transform = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
    ])
    
    # Create datasets for each split
    if fold is not None:
        # Each dataset uses a separate sample - test and val are held out, not derived from train
        train_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=train_samples)
        val_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=[val_sample])
        test_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=[test_sample])
        
        # Use the train dataset for computing statistics
        full_dataset = train_dataset
    else:
        # Standard split: use all samples, compute statistics, then split
        full_dataset = KHTTips2bDataset(root_dir, transform=transform)
        total_len = len(full_dataset)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)
        
        train_dataset, val_test_dataset = stratify_split(full_dataset, train_size=train_len, seed=42)
        val_dataset, test_dataset = stratify_split(val_test_dataset, train_size=val_len, seed=42)
    
    # Apply train_ratio to reduce training data if needed (test_dataset is held out separately)
    if train_ratio < 1.0:
        original_train_len = len(train_dataset)
        reduced_train_len = int(original_train_len * train_ratio)
        train_dataset, _ = stratify_split(train_dataset, train_size=reduced_train_len, seed=42)
        logger.info(f"Reduced training set from {original_train_len} to {len(train_dataset)} samples (train_ratio={train_ratio})")
    
    # Compute (or load cached) mean and std from TRAIN set only (no data leakage)
    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize, root_dir=root_dir, fold=fold)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")
    
    # Apply normalization using train statistics to all splits
    # For grayscale: mean and std are tuples of single values
    normalized_transform = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std)  # mean/std are tuples: (mean,) and (std,)
    ])
    
    # Update datasets' transforms
    train_dataset.transform = normalized_transform
    val_dataset.transform = normalized_transform
    test_dataset.transform = normalized_transform
    logger.info(f"Updated dataset transforms with normalization")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
    
    nb_class = len(full_dataset.classes)
    sample_img, _ = train_dataset[0]
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    
    return train_loader, val_loader, test_loader, nb_class, sample_img.shape


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


def _make_kthtips2b_identifier(dataset, root_dir, fold=None) -> str:
    base_dir = root_dir
    classes = getattr(dataset, 'classes', None)
    fold_str = f"|fold={fold}" if fold is not None else ""
    return f"kthtips2b|{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}{fold_str}"


def load_or_compute_mean_std(dataset, resize: int, root_dir: str = None, fold: int = None, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.
    
    For grayscale images, returns tuple of 1 value: (mean,) and (std,).
    """
    logger = LoggerManager.get_logger()
    identifier = _make_kthtips2b_identifier(dataset, root_dir, fold=fold)
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


if __name__ == "__main__":
    # Script to run and compute statistics for KTH-TIPS2-b
    # Example: Use fold 0 for leave-one-sample-out cross-validation
    root_dir = "data/datasets/KTH-TIPS2-b"
    train_loader, val_loader, test_loader, nb_class, img_shape = get_kthtips2b_loaders(root_dir, resize=200, batch_size=64, fold=0)
