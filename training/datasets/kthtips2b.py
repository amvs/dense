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
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Materials are the top-level subdirectories
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Traverse materials -> samples -> images
        for material in self.classes:
            material_path = self.root_dir / material
            for sample_dir in sorted(material_path.iterdir()):
                if not sample_dir.is_dir():
                    continue
                # Collect all image files (png, jpg, etc.)
                for img_file in sorted(sample_dir.glob("*.png")) + sorted(sample_dir.glob("*.jpg")) + sorted(sample_dir.glob("*.JPG")):
                    self.image_paths.append(img_file)
                    self.labels.append(self.class_to_idx[material])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Keep color images
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_kthtips2b_loaders(root_dir, resize, batch_size=64, train_ratio=0.8, worker_init_fn=None):
    """
    Load KTH-TIPS2-b dataset with appropriate transforms.
    
    Args:
        root_dir: Path to KTH-TIPS2-b dataset root
        resize: Target size for resizing (e.g., 200)
        batch_size: Batch size for data loaders
        train_ratio: Ratio of training to total data
        worker_init_fn: Worker initialization function
        
    Returns:
        train_loader, test_loader, nb_class, sample_img.shape
    """
    logger = LoggerManager.get_logger()
    
    # Initial transform: center crop to square, then to tensor and resize
    transform = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
    ])
    
    dataset = KHTTips2bDataset(root_dir, transform=transform)
    
    # Compute split sizes
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    
    # Split dataset FIRST, before computing statistics
    train_dataset, test_dataset = stratify_split(
        dataset, train_size=train_len,
        seed=42
    )
    
    # Compute (or load cached) mean and std from TRAIN set only (no data leakage)
    logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
    mean, std = load_or_compute_mean_std(train_dataset, resize=resize, root_dir=root_dir)
    logger.info(f"Normalization stats - mean: {mean}, std: {std}")
    
    # Apply normalization using train statistics to both train and test
    normalized_transform = transforms.Compose([
        CenterCropToSquare(),
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std)
    ])
    
    # Update the base dataset's transform so both subsets use it
    dataset.transform = normalized_transform
    logger.info(f"Updated dataset transform with normalization")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
    
    nb_class = len(dataset.classes)
    sample_img, _ = train_dataset[0]
    logger.info(f"[Ratio:{train_ratio}] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    
    return train_loader, test_loader, nb_class, sample_img.shape


def compute_mean_std(dataset):
    """Compute mean and std for RGB dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    mean = torch.tensor([0.0, 0.0, 0.0])
    sq_mean = torch.tensor([0.0, 0.0, 0.0])
    num_pixels = 0
    num_batches = 0

    for idx, (imgs, _) in enumerate(loader):
        # imgs shape: (batch, 3, H, W)
        batch_size = imgs.size(0)
        imgs_flat = imgs.view(batch_size, 3, -1)  # (batch, 3, H*W)
        mean += imgs_flat.sum(dim=(0, 2))
        sq_mean += (imgs_flat ** 2).sum(dim=(0, 2))
        num_pixels += batch_size * imgs_flat.size(2)
        num_batches += 1

    mean = (mean / num_pixels).tolist()
    var = (sq_mean / num_pixels) - torch.tensor(mean) ** 2
    std = var.sqrt().tolist()
    
    logger = LoggerManager.get_logger()
    logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total)")
    return tuple(mean), tuple(std)


def _stats_cache_path(identifier: str, resize: int, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"meanstd_{h}_r{resize}.npz"


def _make_kthtips2b_identifier(dataset, root_dir) -> str:
    base_dir = root_dir
    classes = getattr(dataset, 'classes', None)
    return f"kthtips2b|{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}"


def load_or_compute_mean_std(dataset, resize: int, root_dir: str = None, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.
    
    For RGB images, returns tuple of 3 values (one per channel).
    """
    logger = LoggerManager.get_logger()
    identifier = _make_kthtips2b_identifier(dataset, root_dir)
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
    root_dir = "data/datasets/KTH-TIPS2-b"
    train_loader, test_loader, nb_class, img_shape = get_kthtips2b_loaders(root_dir, resize=200, batch_size=64)
