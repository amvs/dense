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
    def __init__(self, root_dir, deeper_path='', transform=None):
        self.root_dir = Path(root_dir) / deeper_path
        self.transform = transform
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
        image = Image.open(img_path).convert("L")  # grayscale only
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



def get_kaggle_loaders(dataset_name, resize, deeper_path, batch_size=64, train_ratio=0.8, worker_init_fn=None, drop_last=True):
    logger = LoggerManager.get_logger()
    # Early check: try to load cached mean/std using a deterministic identifier
    try:
        kh_version = getattr(kagglehub, '__version__', 'unknown')
    except Exception:
        kh_version = 'unknown'
    early_identifier = f"{dataset_name}|deeper={deeper_path}|kh={kh_version}"
    early_cache = _stats_cache_path(early_identifier, resize)
    stats_loaded = False
    if early_cache.exists():
        try:
            data = np.load(early_cache, allow_pickle=True)
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize, resize)),
    ])
    dataset = KaggleDataset(path, deeper_path, transform=transform)

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
        mean, std = load_or_compute_mean_std(train_dataset, resize=resize)
        logger.info(f"Normalization stats - mean: {mean:.6f}, std: {std:.6f}")
    else:
        logger.info(f"Using early-loaded normalization stats - mean: {mean:.6f}, std: {std:.6f}")
    
    # Apply normalization using train statistics to both train and test
    normalized_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((resize, resize)),
        transforms.Normalize((mean,), (std,))
    ])
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

def compute_mean_std(dataset):
    """Compute mean and std for a grayscale dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
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


def _make_dataset_identifier(dataset) -> str:
    # Unwrap Subset if needed
    base_ds = getattr(dataset, 'dataset', dataset)
    base_dir = getattr(base_ds, 'root_dir', None) or getattr(base_ds, 'root', None) or str(getattr(base_ds, 'image_paths', '')[:5])
    classes = getattr(base_ds, 'classes', None)
    # include a short fingerprint: base_dir + number of samples + class list
    return f"{base_dir}|n={len(dataset)}|classes={','.join(classes) if classes is not None else 'none'}"


def load_or_compute_mean_std(dataset, resize: int, force_recompute: bool = False):
    """Load cached mean/std if present and compatible, otherwise compute and cache.

    Cache key includes a fingerprint of the dataset and the `kagglehub` version.
    """
    logger = LoggerManager.get_logger()
    identifier = _make_dataset_identifier(dataset)
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
                mean = float(data['mean'].tolist())
                std = float(data['std'].tolist())
                logger.info(f"Loaded mean/std from cache {cache_path}")
                return mean, std
            else:
                logger.info(f"Cache mismatch (resize or kagglehub version). Recomputing stats.")
        except Exception:
            logger.info(f"Failed to load cache {cache_path}; recomputing stats")

    # compute mean/std and save
    mean, std = compute_mean_std(dataset)
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