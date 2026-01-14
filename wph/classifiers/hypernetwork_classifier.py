"""HyperNetwork-based classifier for WPH features."""
import torch
from torch import nn
import math


class HyperNetworkClassifier(nn.Module):
    def __init__(self, num_classes, metadata_dim = 10, hidden_dim=64):
        """
        Initialize the HyperNetwork.
        Instead of learning a linear classifier, which might be large (num_features x num_classes),
        we generate the weights from metadata (positional encodings) associated with each feature.
        
        Args:
            num_classes (int): Number of classes in the classification task.
            metadata_dim (int): Dimensionality of the metadata for each feature.
            hidden_dim (int): Hidden dimension for the weight generator network.
        """
        super().__init__()

        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.feature_metadata = None
        
        self.net = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features, return_weights=False):
        # features shape: [Batch, Num_Features]
        # metadata shape: [Num_Features, Metadata_Dim]
        
        # 1. Predict all weights at once
        # thetas shape: [Num_Features, Num_Classes]
        thetas = self.net(self.feature_metadata) 
        
        # 2. Perform the classification via Einstein Summation
        # b: batch, f: features, c: classes
        logits = torch.einsum('bf, fc -> bc', features, thetas)
        
        if return_weights:
            return logits, thetas
        else:
            return logits
    
    def _get_positional_encoding_single(self, values, dims=10):
        pe = torch.zeros(len(values), dims)
        div_term = torch.exp(torch.arange(0, dims, 2) * -(math.log(10000.0) / dims))
    
        # Fill sine and cosine
        pe[:, 0::2] = torch.sin(values.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(values.unsqueeze(1) * div_term)
        return pe
    
    def get_positional_encoding(self, feature_metadata, dims):
        if feature_metadata.dim() == 1:
            return self._get_positional_encoding_single(feature_metadata, dims)
        else:
            return torch.cat([self._get_positional_encoding_single(feature_metadata[i], dims) for i in range(feature_metadata.shape[0])], dim=0)

    def set_feature_metadata(self, metadata: torch.Tensor):
        """
        Set the metadata for features. This should be called after the feature extractor 
        structure is known.
        
        Args:
            metadata (torch.Tensor): Metadata tensor of shape (num_features, metadata_dim).
        """
        self.feature_metadata = metadata
        self.pe = self.get_positional_encoding(metadata, self.metadata_dim)

