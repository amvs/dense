from typing import Tuple, Optional
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_preprocess(resize: Optional[int] = None) -> transforms.Compose:
    t = []
    if resize is not None:
        t.append(transforms.Resize((resize, resize)))
    t.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(t)


def flatten_feature_map(feat: torch.Tensor) -> torch.Tensor:
    """feat: (C,H,W) → (N,D) with N=H*W, D=C."""
    if feat.dim() == 4:
        # assume (B,C,H,W) and B==1
        feat = feat.squeeze(0)
    C, H, W = feat.shape
    return feat.permute(1, 2, 0).reshape(-1, C).contiguous()
