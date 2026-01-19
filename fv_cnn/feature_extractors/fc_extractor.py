from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from .base_extractor import BaseFeatureExtractor


def _load_backbone(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "alexnet":
        return models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "vgg16":
        return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    if name == "resnet50":
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    raise ValueError(f"Unsupported backbone: {name}")


def _fc_module(model: nn.Module, backbone: str, layer: str) -> nn.Module:
    b = backbone.lower()
    if b == "alexnet":
        if layer == "fc7":
            return nn.Sequential(model.features, nn.Flatten(), model.classifier[:6])  # up to fc7
    if b == "vgg16":
        if layer == "fc7":
            return nn.Sequential(model.features, nn.Flatten(), model.classifier[:6])
    if b == "resnet50":
        if layer in ("avgpool", "penultimate"):
            # outputs pooled features before final fc
            return nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                 model.layer1, model.layer2, model.layer3, model.layer4,
                                 model.avgpool, nn.Flatten())
    raise ValueError(f"Unsupported backbone/layer: {backbone}:{layer}")


class FCFeatureExtractor(BaseFeatureExtractor):
    def __init__(self,
                 backbone: str = "vgg16",
                 fc_layer: str = "fc7",
                 device: Optional[torch.device] = None):
        super().__init__()
        self.backbone = backbone
        self.fc_layer = fc_layer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_backbone(backbone)
        fc_model = _fc_module(model, backbone, fc_layer)
        for p in model.parameters():
            p.requires_grad = False
        for p in fc_model.parameters():
            p.requires_grad = False
        self.model = model.to(self.device)
        self.fc_model = fc_model.to(self.device)
        self.model.eval()
        self.fc_model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract(self, image: torch.Tensor | Image.Image, resize: Optional[int] = None) -> Dict[str, Any]:
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        feat = self.fc_model(x)  # (1, D)
        return {"descriptor": feat.squeeze(0).cpu()}  # single global feature
