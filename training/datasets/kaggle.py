from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
from dense.helpers import LoggerManager
import kagglehub
import os
from training.datasets.base import stratify_split
import hashlib
import numpy as np
from pathlib import Path
def get_kaggle_dataset(dataset: str) -> str:
    """
    Download (or reuse cached) Kaggle dataset via kagglehub.

    Parameters
    ----------
    dataset : str
        The Kaggle dataset identifier, e.g. "roustoumabdelmoula/textures-dataset".

    Returns
    -------
    str
        Local path to the dataset files.
    """
    os.environ['KAGGLEHUB_CACHE'] = './data'
    # kagglehub automatically caches datasets
    path = kagglehub.dataset_download(dataset)
    if not os.path.exists(path):
        raise RuntimeError(f"Dataset path {path} not found after download.")
    return path

class KaggleDataset(Dataset):
    def __init__(self, root_dir, deeper_path='', transform=None, color=False):
        self.root_dir = Path(root_dir) / deeper_path
        self.transform = transform
        self.color = color  # If True, keep color images; if False, convert to grayscale
        self.image_paths = []
        self.labels = []

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_folder = self.root_dir / cls
            for img_file in sorted(cls_folder.glob("*.png")) + sorted(cls_folder.glob("*.JPG")) +  sorted(cls_folder.glob("*.jpg")) + sorted(cls_folder.glob("*.bmp")):
                self.image_paths.append(img_file)
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.color:
            image = Image.open(img_path).convert("RGB")  # color image
        else:
            image = Image.open(img_path).convert("L")  # grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_kaggle_loaders(dataset_name, resize, deeper_path, batch_size=64, train_ratio=0.8, worker_init_fn=None, drop_last=True, color=False, pre_transforms=None):
    logger = LoggerManager.get_logger()
    # Early check: try to load cached mean/std using a deterministic identifier
    try:
        kh_version = getattr(kagglehub, '__version__', 'unknown')
    except Exception:
        kh_version = 'unknown'
    early_identifier = f"{dataset_name}|deeper={deeper_path}|kh={kh_version}|color={color}"
    early_cache = _stats_cache_path(early_identifier, resize)
    stats_loaded = False
    if early_cache.exists():
        try:
            data = np.load(early_cache, allow_pickle=True)
            if color:
                mean = data['mean'].tolist()
                std = data['std'].tolist()
            else:
                mean = float(data['mean'].tolist())
                std = float(data['std'].tolist())
            logger.info(f"Loaded mean/std from early cache {early_cache}")
            stats_loaded = True
        except Exception:
            logger.info(f"Failed to load early cache {early_cache}; will compute/load later.")

    path = get_kaggle_dataset(dataset_name) # download if not exists
    logger.info(f"Load dataset {dataset_name} from Kaggle...")
    logger.info(f"Dataset path: {path} + {deeper_path}")
    # Initial transform without normalization
    transform_list = []
    if pre_transforms is not None:
        transform_list.extend(pre_transforms)
    transform_list.append(transforms.ToTensor())
    if not color:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.Resize((resize, resize)))
    transform = transforms.Compose(transform_list)
    
    dataset = KaggleDataset(path, deeper_path, transform=transform, color=color)
    if len(dataset) == 0:
        logger.error(
            "No images found for Kaggle dataset. "
            f"dataset={dataset_name}, path={path}, deeper_path={deeper_path}"
        )
        raise ValueError(
            "Kaggle dataset is empty. Check that the dataset downloaded correctly and "
            "that deeper_path points to the directory with class subfolders."
        )

    # compute split sizes
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len

    # split dataset FIRST, before computing statistics
    train_dataset, test_dataset = stratify_split(
        dataset, train_size=train_len,
        seed=42
    )
    
    # compute (or load cached) mean and std from TRAIN set only (no data leakage)
    if not stats_loaded:
        logger.info(f"Computing/loading normalization statistics from {len(train_dataset)} training samples...")
        mean, std = load_or_compute_mean_std(train_dataset, resize=resize, color=color)
        logger.info(f"Normalization stats - mean: {mean}, std: {std}")
    else:
        logger.info(f"Using early-loaded normalization stats - mean: {mean}, std: {std}")
    
    # Apply normalization using train statistics to both train and test
    norm_transform_list = []
    if pre_transforms is not None:
        norm_transform_list.extend(pre_transforms)
    norm_transform_list.append(transforms.ToTensor())
    if not color:
        norm_transform_list.append(transforms.Grayscale(num_output_channels=1))
    norm_transform_list.append(transforms.Resize((resize, resize)))
    if color:
        norm_transform_list.append(transforms.Normalize(mean, std))
    else:
        norm_transform_list.append(transforms.Normalize((mean,), (std,)))
    normalized_transform = transforms.Compose(norm_transform_list)
    
    # Update the base dataset's transform so both subsets use it
    dataset.transform = normalized_transform
    logger.info(f"Updated dataset transform with normalization")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=worker_init_fn, drop_last=drop_last, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn, drop_last=False, num_workers=4)
    nb_class = len(dataset.classes)
    sample_img, _ = train_dataset[0]
    logger.info(f"[Ratio:{train_ratio}] Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    logger.info(f"# class: {nb_class}, shape {sample_img.shape}")
    return train_loader, test_loader, nb_class, sample_img.shape

def compute_mean_std(dataset, color=False):
    """Compute mean and std for a dataset (grayscale or color)."""
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    if color:
        # For color (RGB), compute per-channel statistics
        mean = torch.zeros(3)
        sq_mean = torch.zeros(3)
        num_pixels = 0
        num_batches = 0
        for idx, (imgs, _) in enumerate(loader):
            # imgs shape: (B, 3, H, W)
            for c in range(3):
                mean[c] += imgs[:, c, :, :].sum()
                sq_mean[c] += (imgs[:, c, :, :] ** 2).sum()
            num_pixels += imgs.size(2) * imgs.size(3) * imgs.size(0)
            num_batches += 1
        mean /= num_pixels
        std = (sq_mean / num_pixels - mean ** 2).sqrt()
        logger = LoggerManager.get_logger()
        logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total)")
        return mean.tolist(), std.tolist()
    else:
        # For grayscale, compute single value statistics
        mean = 0.0
        sq_mean = 0.0
        num_pixels = 0
        num_batches = 0
        for idx, (imgs, _) in enumerate(loader):
            imgs = imgs.view(imgs.size(0), -1)  # flatten H*W
            mean += imgs.sum()
            sq_mean += (imgs ** 2).sum()
            num_pixels += imgs.numel()
            num_batches += 1
        mean /= num_pixels
        std = (sq_mean / num_pixels - mean ** 2).sqrt()
        logger = LoggerManager.get_logger()
        logger.info(f"Computed mean/std from {num_batches} batches ({num_pixels} pixels total)")
        return mean.item(), std.item()


def _stats_cache_path(identifier: str, resize: int, cache_dir: str = "data/stats") -> Path:
    h = hashlib.sha1(identifier.encode("utf-8")).hexdigest()[:10]
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"meanstd_{h}_r{resize}.npz"


def _make_dataset_identifier(dataset, color=False) -> str:
    # Unwrap Subset if needed
    base_ds = getattr(dataset, 'dataset', dataset)
    base_dir = getattr(base_ds, 'root_dir', None) or getattr(base_ds, 'root', None) or str(getattr(base_ds, 'image_paths', '')[:5])
    classes = getattr(base_ds, 'classes', None)
    # include a short fingerprint: base_dir + number of samples + class list + color mode
    return f"{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}|color={color}"


def load_or_compute_mean_std(dataset, resize: int, color=False, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.

    Cache key includes a fingerprint of the dataset and the `kagglehub` version.
    """
    logger = LoggerManager.get_logger()
    identifier = _make_dataset_identifier(dataset, color=color)
    cache_path = _stats_cache_path(identifier, resize)

    # get kagglehub version if available
    try:
        kh_version = getattr(kagglehub, '__version__', 'unknown')
    except Exception:
        kh_version = 'unknown'

    if cache_path.exists() and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=True)
            cached_resize = int(data.get('resize', resize))
            cached_kh = data.get('kagglehub_version', kh_version)
            if cached_resize == resize and (str(cached_kh) == str(kh_version)):
                if color:
                    mean = data['mean'].tolist()
                    std = data['std'].tolist()
                else:
                    mean = float(data['mean'].tolist())
                    std = float(data['std'].tolist())
                logger.info(f"Loaded mean/std from cache {cache_path}")
                return mean, std
            else:
                logger.info(f"Cache mismatch (resize or kagglehub version). Recomputing stats.")
        except Exception:
            logger.info(f"Failed to load cache {cache_path}; recomputing stats")

    # compute mean/std and save
    mean, std = compute_mean_std(dataset, color=color)
    try:
        np.savez_compressed(cache_path, mean=mean, std=std, resize=resize, kagglehub_version=kh_version)
        logger.info(f"Saved mean/std to cache {cache_path}")
    except Exception as e:
        logger.info(f"Failed to save mean/std cache: {e}")
    return mean, std


if __name__ == "__main__":
    # script to run and compute the statistics
    dataset_name = "smohsensadeghi/curet-dataset"  # replace with your path
    train_loader, test_loader, nb_class, img_shape = get_kaggle_loaders(dataset_name, resize=64, batch_size=64)
    #CURET dataset mean: 0.317774, std: 0.213433