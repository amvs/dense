"""Linear classifier with optional batch normalization."""
import torch
from torch import nn


class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        """
        Linear classifier with optional batch normalization.
        
        Args:
            input_dim (int): Dimension of input features.
            num_classes (int): Number of classes for classification.
            use_batch_norm (bool): Whether to include a batch normalization layer before the classifier.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
    

        # Define the classifier layer
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes).
        """
        
        # Compute logits
        logits = self.classifier(features)
        return logits
