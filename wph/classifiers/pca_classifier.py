"""PCA/affine subspace classifier using sklearn for fitting with PyTorch parameter storage."""
import torch
from torch import nn
from typing import Optional, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAClassifier(nn.Module):
    """
    PCA/affine subspace classifier.
    
    Fits per-class PCA models using sklearn, then stores components as PyTorch parameters
    for differentiable forward passes. Supports gradient propagation through the classification.
    """
    
    def __init__(self, input_dim: int, num_classes: Optional[int] = None, 
                 n_components: Optional[int] = None, scale_features: bool = True, 
                 whiten: bool = False):
        """
        Initialize PCA classifier.
        
        Args:
            input_dim (int): Dimension of input features.
            num_classes (int, optional): Number of classes (set after fitting).
            n_components (int, optional): Number of PCA components to keep per class.
            scale_features (bool): Whether to scale features before PCA.
            whiten (bool): Whether to whiten the PCA components.
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.n_components = n_components
        self.scale_features = scale_features
        self.whiten = whiten
        self._is_fitted = False
        
        # Will be populated after fitting
        self.classes_ = None
        self._components = nn.ParameterDict()  # Store as PyTorch parameters
        self._means = nn.ParameterDict()
        self._scales = nn.ParameterDict()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit per-class PCA models using sklearn.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, input_dim).
            y (np.ndarray): Training labels of shape (n_samples,).
        """
        X, y = self._check_inputs(X, y)
        self.classes_ = np.unique(y)
        
        if self.num_classes is None:
            self.num_classes = len(self.classes_)
        
        # Clear existing parameters
        self._components = nn.ParameterDict()
        self._means = nn.ParameterDict()
        self._scales = nn.ParameterDict()
        
        # Fit per-class PCA using sklearn, then convert to PyTorch
        for label in self.classes_:
            X_class = X[y == label]
            
            # Optional scaling
            if self.scale_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_class)
                # Store scaling parameters as PyTorch tensors
                self._means[str(label)] = nn.Parameter(
                    torch.tensor(scaler.mean_, dtype=torch.float32), 
                    requires_grad=False
                )
                self._scales[str(label)] = nn.Parameter(
                    torch.tensor(scaler.scale_, dtype=torch.float32), 
                    requires_grad=False
                )
            else:
                X_scaled = X_class
            
            # Fit PCA
            pca = PCA(n_components=self.n_components, whiten=self.whiten)
            pca.fit(X_scaled)
            
            # Store PCA components as PyTorch parameters (transposed for easier use)
            components = torch.tensor(pca.components_, dtype=torch.float32).T
            self._components[str(label)] = nn.Parameter(components, requires_grad=True)
        
        self._is_fitted = True
        return self
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log probabilities with gradient support.
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, num_classes).
        """
        if not self._is_fitted:
            raise RuntimeError("PCAClassifier must be fitted before forward pass. Call fit() first.")
        
        scores = []
        for label in self.classes_:
            label_str = str(label)
            
            # Apply scaling if used
            x = features
            if self.scale_features:
                mean = self._means[label_str]
                scale = self._scales[label_str]
                x = (x - mean) / scale
            
            # Get PCA basis (already transposed, shape: [input_dim, n_components])
            basis = self._components[label_str]
            
            # Normalize basis vectors
            basis_norms = torch.linalg.norm(basis, dim=0, keepdim=True)
            basis_norms = torch.where(basis_norms == 0, torch.ones_like(basis_norms), basis_norms)
            basis_normed = basis / basis_norms
            
            # Project onto subspace and compute distance
            proj = x @ basis_normed @ basis_normed.T
            dists = torch.linalg.norm(proj - x, dim=1)
            
            # Negative distance as score (closer = higher score)
            scores.append(-dists)
        
        scores_tensor = torch.stack(scores, dim=1)
        return torch.nn.functional.log_softmax(scores_tensor, dim=1)
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            np.ndarray: Predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("PCAClassifier must be fitted before prediction. Call fit() first.")
        
        with torch.no_grad():
            log_probs = self.forward(features)
            argmax = torch.argmax(log_probs, dim=1).cpu().numpy()
        return self.classes_[argmax]
    
    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract features (pass-through for compatibility).
        
        Args:
            features (torch.Tensor): Input features of shape (batch_size, input_dim).
            
        Returns:
            torch.Tensor: Same as input features.
        """
        return features
    
    def count_parameters(self) -> int:
        """Count total number of stored parameters."""
        if not self._is_fitted:
            return 0
        
        total = 0
        for label in self.classes_:
            label_str = str(label)
            total += self._components[label_str].numel()
            if self.scale_features:
                total += self._means[label_str].numel()
                total += self._scales[label_str].numel()
        return total
    
    @staticmethod
    def _check_inputs(X, y):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y is None:
            raise ValueError("y cannot be None")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        return X, y
