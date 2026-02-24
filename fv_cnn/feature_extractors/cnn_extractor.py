from typing import List, Dict, Any, Optional, Sequence
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .base_extractor import BaseFeatureExtractor
from .utils import build_preprocess, flatten_feature_map


def _load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "alexnet":
        return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "vgg11":
        return models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "vgg19":
        return models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    raise ValueError(f"Unsupported backbone: {name}")


def _feature_module(model: nn.Module, backbone: str, layer: str) -> nn.Module:
    b = backbone.lower()
    # map named layers to actual submodules
    if b == "alexnet":
        if layer == "conv5_3": # match with vgg layer name for sweeps
            return nn.Sequential(*list(model.features.children())[:12])  # up to conv5 activation
        raise ValueError("Unsupported layer for AlexNet: {layer}")
    if b == "vgg16":
        if layer == "conv5_3":
            return nn.Sequential(*list(model.features.children())[:30])  # up to conv5_3 relu
        raise ValueError("Unsupported layer for VGG16: {layer}")
    if b == "vgg11":
        if layer == "conv5_3":
            return nn.Sequential(*list(model.features.children())[:20])  # up to conv5_3 relu
        raise ValueError("Unsupported layer for VGG11: {layer}")
    if b == "vgg19":
        if layer == "conv5_3":
            return nn.Sequential(*list(model.features.children())[:36])  # up to conv5_3 relu
    if b == "resnet50":
        if layer == "layer4":
            # return everything up to layer4
            return nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                 model.layer1, model.layer2, model.layer3, model.layer4)
        raise ValueError("Unsupported layer for ResNet50: {layer}")
    raise ValueError(f"Unsupported backbone: {backbone}")


class MultiScaleCNNExtractor(BaseFeatureExtractor):
    def __init__(self,
                 backbone: str = "vgg16",
                 feature_layer: str = "conv5_3",
                 scales: Optional[List[float]] = None,
                 min_edge: int = 30,
                 max_sqrt_hw: int = 1024,
                 preprocess_mode: str = "imagenet",
                 matconvnet_mean: Optional[Sequence[float]] = None,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.backbone = backbone
        self.feature_layer = feature_layer
        self.scales = scales or [
            2.82842712,
            2.0,
            1.41421356,
            1.0,
            0.70710678,
            0.5,
            0.35355339,
            0.25,
            0.17677670,
            0.125,
        ]
        self.min_edge = min_edge
        self.max_sqrt_hw = max_sqrt_hw
        self.preprocess_mode = preprocess_mode
        self.matconvnet_mean = matconvnet_mean
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # backbone and feature extractor as buffers (not trainable)
        model = _load_backbone(backbone)
        feature_model = _feature_module(model, backbone, feature_layer)
        for p in model.parameters():
            p.requires_grad = False
        for p in feature_model.parameters():
            p.requires_grad = False
        self.model = model.to(self.device)
        self.feature_model = feature_model.to(self.device)
        self.model.eval()
        self.feature_model.eval()

        # store preprocessing as a buffer-like attribute (not a Parameter)
        self.preprocess = build_preprocess(
            resize=None,
            mode=self.preprocess_mode,
            matconvnet_mean=self.matconvnet_mean,
        )

    @torch.no_grad()
    def extract(self, image: torch.Tensor | Image.Image, resize: Optional[int] = None) -> Dict[str, Any]:
        # accept PIL or CHW tensor; convert to PIL for resizing and scaling conveniences
        if isinstance(image, torch.Tensor):
            # assume CHW in [0,1]
            image = transforms.ToPILImage()(image)
        assert isinstance(image, Image.Image)
        
        # convert grayscale to RGB if needed (for pretrained models expecting RGB)
        if image.mode == "L":
            image = image.convert("RGB")

        base_w, base_h = image.size
        descriptors: List[torch.Tensor] = []
        used_scales: List[float] = []

        for s in self.scales:
            # enforce constraints
            scaled_w = int(base_w * s)
            scaled_h = int(base_h * s)
            if min(scaled_w, scaled_h) < self.min_edge:
                continue
            if (scaled_w * scaled_h) ** 0.5 > self.max_sqrt_hw:
                continue
            img_s = image.resize((scaled_w, scaled_h), resample=Image.BICUBIC)
            x = self.preprocess(img_s).unsqueeze(0).to(self.device)
            feat = self.feature_model(x)  # (1, C, H, W)
            desc = flatten_feature_map(feat)  # (N, D)
            descriptors.append(desc.cpu())
            used_scales.append(s)

        return {"descriptors": descriptors, "scales": used_scales}
