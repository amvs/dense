"""SVM classifier wrapper for WPH features."""
import torch
from torch import nn
from typing import Optional


class SVMClassifier(nn.Module):
    """
    Placeholder classifier for SVM training.
    
    This module doesn't perform classification directly but provides an interface
    compatible with the WPHClassifier framework. The actual SVM training is done
    externally using sklearn, and features are extracted via extract_features().
    """
    
    def __init__(self, input_dim: int, num_classes: Optional[int] = None):
        """
        Initialize SVM classifier placeholder.
        
        Args:
            input_dim (int): Dimension of input features.
            num_classes (int, optional): Number of classes (not used for SVM, kept for API consistency).
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        For SVM, forward pass just returns features since classification happens externally.
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Same as input features.
        """
        return features
    
    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract features for external SVM training.
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Features ready for SVM training.
        """
        return features
