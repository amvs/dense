from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _to_pil(image: torch.Tensor | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.ndim == 2:
        image = image.unsqueeze(0)
    return transforms.ToPILImage()(image.cpu())


def _mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return x1, y1, x2, y2


def _normalize_box(box: Sequence[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w, h = image_size
    if max(x1, y1, x2, y2) <= 1.0:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def _expand_box(
    box: Tuple[int, int, int, int],
    border: float,
    image_size: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    if border <= 0:
        return box
    x1, y1, x2, y2 = box
    w, h = image_size
    bw = x2 - x1
    bh = y2 - y1
    scale = 1.0 + border
    new_w = bw * scale
    new_h = bh * scale
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    nx1 = max(0, int(round(cx - new_w / 2.0)))
    ny1 = max(0, int(round(cy - new_h / 2.0)))
    nx2 = min(w, int(round(cx + new_w / 2.0)))
    ny2 = min(h, int(round(cy + new_h / 2.0)))
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _regions_to_boxes(regions, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
    if regions is None:
        return []
    if isinstance(regions, torch.Tensor):
        regions = regions.cpu().numpy()
    if isinstance(regions, np.ndarray):
        if regions.ndim == 1 and regions.size == 4:
            return [_normalize_box(regions.tolist(), image_size)]
        if regions.ndim == 2 and regions.shape[1] == 4:
            return [_normalize_box(b, image_size) for b in regions]
        if regions.ndim == 2:
            box = _mask_to_box(regions != 0)
            return [_normalize_box(box, image_size)] if box is not None else []
        if regions.ndim == 3:
            boxes = []
            for i in range(regions.shape[0]):
                box = _mask_to_box(regions[i] != 0)
                if box is not None:
                    boxes.append(_normalize_box(box, image_size))
            return boxes
    if isinstance(regions, (list, tuple)):
        boxes = []
        for item in regions:
            if item is None:
                continue
            if isinstance(item, (list, tuple, np.ndarray, torch.Tensor)):
                item_arr = np.array(item)
                if item_arr.ndim == 1 and item_arr.size == 4:
                    boxes.append(_normalize_box(item_arr.tolist(), image_size))
                else:
                    boxes.extend(_regions_to_boxes(item, image_size))
        return boxes
    return []


def _grid_boxes(image_size: Tuple[int, int], region_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    w, h = image_size
    if region_size <= 0:
        return [(0, 0, w, h)]
    if region_size > w or region_size > h:
        return [(0, 0, w, h)]
    stride = max(1, stride)
    boxes = []
    for y in range(0, h - region_size + 1, stride):
        for x in range(0, w - region_size + 1, stride):
            boxes.append((x, y, x + region_size, y + region_size))
    if not boxes:
        boxes.append((0, 0, w, h))
    return boxes


def get_region_crops(
    image: torch.Tensor | Image.Image,
    regions,
    region_cfg: Dict[str, Any],
) -> List[Image.Image]:
    pil_img = _to_pil(image)
    w, h = pil_img.size
    mode = (region_cfg.get("mode") or "none").lower()
    border = float(region_cfg.get("border", 0.0))
    boxes = _regions_to_boxes(regions, (w, h)) if regions is not None else []
    if not boxes:
        if mode == "grid":
            region_size = int(region_cfg.get("size", min(w, h)))
            stride = int(region_cfg.get("stride", region_size))
            boxes = _grid_boxes((w, h), region_size, stride)
        else:
            boxes = [(0, 0, w, h)]
    crops = []
    for box in boxes:
        box = _expand_box(box, border, (w, h))
        crops.append(pil_img.crop(box))
    return crops


def pool_region_codes(codes: List[np.ndarray], pooling: str) -> np.ndarray:
    if len(codes) == 1:
        return codes[0]
    pooling = pooling.lower()
    if pooling == "mean":
        return np.mean(codes, axis=0)
    if pooling == "max":
        return np.max(codes, axis=0)
    raise ValueError(f"Unsupported region_pooling: {pooling}")
