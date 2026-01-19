from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch import nn


class BaseFeatureExtractor(nn.Module, ABC):
    """Abstract multi-scale feature extractor interface."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract(self, image: torch.Tensor, resize: Optional[int] = None) -> Dict[str, Any]:
        """Extract features for one image.
        Args:
            image: CHW tensor in [0,1]; may accept PIL and convert internally.
            resize: optional side length for resizing.
        Returns:
            Dict with keys: 'descriptors' (List[Tensor N_i x D]), 'scales' (List[float]).
        """
        raise NotImplementedError
