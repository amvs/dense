import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim
from typing import Optional

class LinearDimReducerBase(nn.Module):
    def __init__(self, in_dim, out_dim:Optional[int]=None):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        # Register a dummy linear layer for submodule registration
        self.linear = nn.Identity()

    def reset_parameters(self):
        """Reset the fitted state and remove learned projection."""
        self.linear = nn.Identity()
        # Remove any buffers (e.g., pca_mean) if present
        for name in list(self._buffers.keys()):
            if name.startswith('pca_mean'):
                del self._buffers[name]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.linear(x)


class PCALinearDimReducer(LinearDimReducerBase):
    def __init__(self, in_dim, out_dim:Optional[int]=None):
        super().__init__(in_dim, out_dim)
        self.pca = PCA(n_components=self.out_dim)

    def fit(self, data: torch.Tensor):
        # data shape: (num_samples, in_dim)
        self.pca.fit(data.cpu().numpy())
        # Store as torch.nn.Linear
        W = torch.from_numpy(self.pca.components_.astype('float32'))  # (out_dim, in_dim)
        b = torch.from_numpy(self.pca.mean_.astype('float32'))        # (in_dim,)
        linear = nn.Linear(self.in_dim, self.out_dim, bias=False)
        linear.weight.data = W
        self.add_module('linear', linear)  # register as submodule
        self.linear = linear
        self.register_buffer('pca_mean', b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            raise RuntimeError("PCA model not fitted. Call fit() first.")
        x_centered = x - self.pca_mean
        return self.linear(x_centered)
    
class RandomProjLinearDimReducer(LinearDimReducerBase):
    def __init__(self, in_dim, out_dim:Optional[int]=None, n_samples: int=1000, eps: float=0.1):
        super().__init__(in_dim, out_dim)
        if out_dim is None:
            out_dim = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        self.random_proj = GaussianRandomProjection(n_components=out_dim)

    def fit(self, data: torch.Tensor):
        # data shape: (num_samples, in_dim)
        self.random_proj.fit(data.cpu().numpy())
        # Store as torch.nn.Linear
        W = torch.from_numpy(self.random_proj.components_.astype('float32'))  # (out_dim, in_dim)
        linear = nn.Linear(self.in_dim, self.out_dim, bias=False)
        linear.weight.data = W
        self.add_module('linear', linear)  # register as submodule
        self.linear = linear