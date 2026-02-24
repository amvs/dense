from typing import Tuple, Optional, Sequence
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MATCONVNET_MEAN = [123.68, 116.779, 103.939]


def _matconvnet_preprocess(mean: Sequence[float]) -> transforms.Compose:
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)

    def _fn(img):
        x = transforms.ToTensor()(img) * 255.0
        x = x[[2, 1, 0], :, :]  # RGB -> BGR
        return x - mean_tensor

    return transforms.Compose([transforms.Lambda(_fn)])


def build_preprocess(
    resize: Optional[int] = None,
    mode: str = "imagenet",
    matconvnet_mean: Optional[Sequence[float]] = None,
) -> transforms.Compose:
    t = []
    if resize is not None:
        t.append(transforms.Resize((resize, resize)))
    mode = mode.lower()
    if mode == "imagenet":
        t.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    elif mode == "matconvnet":
        mean = matconvnet_mean or MATCONVNET_MEAN
        t.append(_matconvnet_preprocess(mean))
    else:
        raise ValueError(f"Unsupported preprocess mode: {mode}")
    return transforms.Compose(t)


def flatten_feature_map(feat: torch.Tensor) -> torch.Tensor:
    """feat: (C,H,W) → (N,D) with N=H*W, D=C."""
    if feat.dim() == 4:
        # assume (B,C,H,W) and B==1
        feat = feat.squeeze(0)
    C, H, W = feat.shape
    return feat.permute(1, 2, 0).reshape(-1, C).contiguous()
