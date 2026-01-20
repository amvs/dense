from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from dense.helpers import LoggerManager
from training.datasets.base import stratify_split, CenterCropToSquare
import hashlib
import numpy as np


class OutexDataset(Dataset):
    """
    Outex texture dataset loader.
    
    Directory structure:
        root/
            images/
                000000.bmp
                000001.bmp
                ...
            <problem_id>/
                classes.txt
                train.txt
                test.txt
            ...
    """
    def __init__(self, root_dir, problem_id='000', split='train', transform=None):
        """
        Args:
            root_dir: Path to Outex dataset root
            problem_id: Problem configuration ID (e.g., '000', '001', '002')
            split: 'train' or 'test'
            transform: Transform to apply to images
        """
        self.root_dir = Path(root_dir)
        self.problem_id = problem_id
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []

        problem_dir = self.root_dir / problem_id
        
        # Load classes
        classes_file = problem_dir / 'classes.txt'
        self.classes = []
        self.class_to_idx = {}
        
        with open(classes_file, 'r') as f:
            num_classes = int(f.readline().strip())
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_name = parts[0]
                    class_id = int(parts[1])
                    self.classes.append(class_name)
                    self.class_to_idx[class_id] = class_name
        
        # Load train or test split
        split_file = problem_dir / f'{split}.txt'
        with open(split_file, 'r') as f:
            num_samples = int(f.readline().strip())
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    filename = parts[0]
                    class_id = int(parts[1])
                    image_path = self.root_dir / 'images' / filename
                    self.image_paths.append(image_path)
                    self.labels.append(class_id)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale (model expects 1 channel)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_outex_loaders(root_dir, resize, batch_size=64, worker_init_fn=None, problem_id='000', train_ratio=1.0, train_val_ratio=4, drop_last=False):
    """
    Load Outex dataset with train/val/test splits.
    
    For multi-problem datasets (e.g., TC-00012), can run multiple experiments
    using different problem configurations for cross-validation.
    
    Args:
        root_dir: Path to Outex dataset root
        resize: Target size for resizing (e.g., 200)
        batch_size: Batch size for data loaders
        worker_init_fn: Worker initialization function
        problem_id: Problem configuration ID (e.g., '000', '001', '002'). 
                   If None, lists available problems.
        train_ratio: Fraction of training data to use (default 1.0). Set to < 1.0 to reduce training set.
        train_val_ratio: Ratio of test to validation data (default 4, meaning test:val = 4:1).
        drop_last: Whether to drop the last incomplete batch in training loader (default False).
        
    Returns:
        train_loader, val_loader, test_loader, nb_class, sample_img.shape
    """
    logger = LoggerManager.get_logger()
    
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    
    logger.info(f"Loading Outex dataset from {root_dir} with problem_id={problem_id}")
    
    # Initial transform: center crop to square, then to tensor and resize (grayscale)
    transform = transforms.Compose([
        CenterCropToSquare(),  # Center crop to square before resize (preserves aspect ratio)
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((resize, resize)),
    ])
    
    # Load train and test datasets
    train_dataset = OutexDataset(root_dir, problem_id=problem_id, split='train', transform=transform)
    test_dataset = OutexDataset(root_dir, problem_id=problem_id, split='test', transform=transform)
    
    # Get number of classes before any splitting (Subset objects don't have .classes attribute)
    nb_class = len(test_dataset.classes)
    
    # Apply train_ratio to reduce training data if needed
    if train_ratio < 1.0:
        original_train_len = len(train_dataset)
        reduced_train_len = int(original_train_len * train_ratio)
        train_dataset, _ = stratify_split(train_dataset, train_size=reduced_train_len, seed=42)
        logger.info(f"Reduced training set from {original_train_len} to {len(train_dataset)} samples (train_ratio={train_ratio})")
    
    # Create validation split from TEST data instead of train data
    # This preserves all training data and makes sure val data includes rotations/illumination changes
    # train_val_ratio determines the ratio test:val (e.g., 4 means test:val = 4:1)
    total_test_len = len(test_dataset)
    val_len = total_test_len // (train_val_ratio + 1)
    test_len = total_test_len - val_len
    test_dataset, val_dataset = stratify_split(test_dataset, train_size=test_len, seed=42)
    
    # Compute (or load cached) mean and std from TRAIN set only (no data leakage)
    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize, root_dir=root_dir, problem_id=problem_id)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")
    
    # Apply normalization using train statistics to all splits (grayscale)
    normalized_transform = transforms.Compose([
        CenterCropToSquare(),  # Center crop to square before resize (preserves aspect ratio)
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std)  # mean/std are tuples: (mean,) and (std,)
    ])
    
    # Update datasets' transforms
    train_dataset.transform = normalized_transform
    val_dataset.transform = normalized_transform
    test_dataset.transform = normalized_transform
    logger.info(f"Updated dataset transforms with normalization")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False)
    
    sample_img, _ = train_dataset[0]
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    
    return train_loader, val_loader, test_loader, nb_class, sample_img.shape


def get_available_problems(root_dir):
    """Get list of available problem configurations in dataset.
    
    Args:
        root_dir: Path to Outex dataset root
        
    Returns:
        List of problem IDs (e.g., ['000', '001', '002'])
    """
    root_path = Path(root_dir)
    problems_file = root_path / 'problems.txt'
    
    problems = []
    if problems_file.exists():
        with open(problems_file, 'r') as f:
            num_problems = int(f.readline().strip())
            for line in f:
                problem_id = line.strip()
                if problem_id and (root_path / problem_id).is_dir():
                    problems.append(problem_id)
    
    return sorted(problems)


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
    return (mean,), (std,)  # Return as tuples for compatibility with Normalize


def _stats_cache_path(identifier: str, resize: int, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"meanstd_{h}_r{resize}.npz"


def _make_outex_identifier(dataset, root_dir, problem_id=None) -> str:
    base_dir = root_dir
    classes = getattr(dataset, 'classes', None)
    problem_str = f"|problem={problem_id}" if problem_id is not None else ""
    return f"outex|{base_dir}|n={len(dataset)}|classes={len(classes) if classes is not None else 'none'}{problem_str}"


def load_or_compute_mean_std(dataset, resize: int, root_dir: str = None, problem_id: str = None, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.
    
    For RGB images, returns tuple of 3 values (one per channel).
    """
    logger = LoggerManager.get_logger()
    identifier = _make_outex_identifier(dataset, root_dir, problem_id=problem_id)
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
    # Script to run and compute statistics for Outex
    # Example: Load Outex-TC-00012-bmp dataset with problem 000
    root_dir = "data/datasets/Outex-TC-00012-bmp"
    
    # List available problems
    problems = get_available_problems(root_dir)
    print(f"Available problems: {problems}")
    
    # Load dataset with first problem
    if problems:
        train_loader, val_loader, test_loader, nb_class, img_shape = get_outex_loaders(
            root_dir, resize=200, batch_size=64, problem_id=problems[0]
        )
