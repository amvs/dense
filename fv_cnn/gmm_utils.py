from __future__ import annotations

from typing import Dict, Any, List, Optional, Iterator, Tuple
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from fv_cnn.descriptor_pca import LocalPCATransform
from fv_cnn.region_utils import get_region_crops


def compute_matconvnet_mean(train_loader: DataLoader, logger=None) -> List[float]:
    sum_rgb = np.zeros(3, dtype=np.float64)
    count = 0
    for batch in train_loader:
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        if not isinstance(images, torch.Tensor):
            raise TypeError("Expected tensor images in train_loader for mean computation")
        if images.ndim == 3:
            images = images.unsqueeze(0)
        sum_rgb += images.sum(dim=(0, 2, 3)).double().cpu().numpy()
        count += images.shape[0] * images.shape[2] * images.shape[3]
    if count == 0:
        raise RuntimeError("No samples found when computing MatConvNet mean")
    mean_rgb = sum_rgb / count
    mean_bgr = mean_rgb[::-1] * 255.0
    mean_bgr_list = mean_bgr.tolist()
    if logger is not None:
        logger.log(f"Computed matconvnet_mean (BGR, 0..255): {mean_bgr_list}")
    return mean_bgr_list


def apply_local_pca(descriptors: np.ndarray, local_pca: Optional[LocalPCATransform]) -> np.ndarray:
    if local_pca is None:
        return descriptors
    return local_pca.transform(descriptors)


def _iter_image_descriptors(
    train_loader: DataLoader,
    extractor,
    region_cfg: Dict[str, Any],
    max_per_image: Optional[int],
    seed: int,
) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    extractor.eval()
    if hasattr(extractor, "device"):
        device = extractor.device
    else:
        try:
            device = next(extractor.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for batch in train_loader:
            if len(batch) == 2:
                images, _ = batch
                regions_batch = None
            else:
                images, _, regions_batch = batch[0], batch[1], batch[2]
            images = images.to(device)
            for i in range(images.size(0)):
                img = images[i]
                regions = None
                if regions_batch is not None:
                    regions = regions_batch[i]
                crops = get_region_crops(img, regions, region_cfg)
                desc_parts = []
                for crop in crops:
                    feats = extractor.extract(crop)
                    desc_parts.extend([d.detach().cpu().numpy() for d in feats["descriptors"]])
                if not desc_parts:
                    continue
                desc = np.concatenate(desc_parts, axis=0)
                if max_per_image is not None and desc.shape[0] > max_per_image:
                    idx = rng.choice(desc.shape[0], size=max_per_image, replace=False)
                    desc = desc[idx]
                yield desc


def collect_descriptors(
    train_loader: DataLoader,
    extractor,
    reservoir_size: Optional[int] = None,
    seed: int = 42,
    region_cfg: Optional[Dict[str, Any]] = None,
    max_per_image: Optional[int] = None,
    memmap_path: Optional[str] = None,
    memmap_size: Optional[int] = None,
    local_pca: Optional[LocalPCATransform] = None,
) -> Tuple[np.ndarray, int]:
    region_cfg = region_cfg or {}
    rng = np.random.default_rng(seed)
    total_seen = 0
    seen = 0

    if memmap_path is not None and memmap_size is None and reservoir_size is None:
        raise ValueError("memmap_size or reservoir_size must be set when using memmap_path")

    target_size = reservoir_size
    if memmap_size is not None:
        target_size = memmap_size if target_size is None else min(target_size, memmap_size)

    if target_size is None:
        collected: List[np.ndarray] = []
        for desc in _iter_image_descriptors(train_loader, extractor, region_cfg, max_per_image, seed):
            total_seen += desc.shape[0]
            desc = apply_local_pca(desc, local_pca)
            collected.append(desc)
        if not collected:
            raise RuntimeError("No descriptors collected; check dataset/extractor.")
        return np.concatenate(collected, axis=0), total_seen

    storage = None
    filled = 0
    for desc in _iter_image_descriptors(train_loader, extractor, region_cfg, max_per_image, seed):
        total_seen += desc.shape[0]
        desc = apply_local_pca(desc, local_pca)
        if storage is None:
            if memmap_path is not None:
                os.makedirs(os.path.dirname(memmap_path) or ".", exist_ok=True)
                storage = np.memmap(memmap_path, mode="w+", dtype=np.float32, shape=(target_size, desc.shape[1]))
            else:
                storage = np.empty((target_size, desc.shape[1]), dtype=desc.dtype)
        for row in desc:
            seen += 1
            if filled < target_size:
                storage[filled] = row
                filled += 1
            else:
                j = rng.integers(0, seen)
                if j < target_size:
                    storage[j] = row

    if storage is None or filled == 0:
        raise RuntimeError("No descriptors collected; check dataset/extractor.")
    return storage[:filled], total_seen
