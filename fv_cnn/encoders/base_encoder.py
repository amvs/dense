from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseEncoder(torch.nn.Module, ABC):
    """Abstract encoder for local descriptor sets → global codes."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, descriptors: np.ndarray) -> None:
        """Train encoder parameters on a large set of local descriptors.
        Args:
            descriptors: (N, D) array of sampled local descriptors.
        """
        raise NotImplementedError

    @abstractmethod
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        """Encode a set of local descriptors into a single global code.
        Args:
            descriptors: (N, D) array of local descriptors for one image.
        Returns:
            (M,) array representing the global code (e.g., FV of size 2*D*K).
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save(self, path)

    @staticmethod
    def load(path: str, map_location: str | torch.device | None = None) -> "BaseEncoder":
        return torch.load(path, map_location=map_location)
