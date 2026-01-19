from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from .base_encoder import BaseEncoder


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(x) + 1e-12
    return x / n


class FisherVectorEncoder(BaseEncoder):
    """Improved Fisher Vector encoder using diagonal-covariance GMM.

    Produces FV of dimension 2 * D * K (first and second order stats).
    Uses sklearn for fitting, then stores parameters as torch tensors for GPU eval.
    """

    def __init__(
        self,
        num_components: int = 64,
        pca_dim: Optional[int] = None,
        signed_sqrt_postprocess: bool = True,
        l2_postprocess: bool = True,
        random_state: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.pca_dim = pca_dim
        self.signed_sqrt_postprocess = signed_sqrt_postprocess
        self.l2_postprocess = l2_postprocess
        self.random_state = random_state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # fitted parameters (torch tensors)
        self.register_buffer("pca_mean", None)
        self.register_buffer("pca_components", None)
        self.register_buffer("gmm_weights", None)
        self.register_buffer("gmm_means", None)
        self.register_buffer("gmm_cov_diag", None)

    def fit(self, descriptors: np.ndarray) -> None:
        assert descriptors.ndim == 2, "descriptors must be (N, D)"
        X = descriptors
        pca_model: Optional[PCA] = None
        if self.pca_dim is not None:
            pca_model = PCA(n_components=self.pca_dim, whiten=False, random_state=self.random_state)
            X = pca_model.fit_transform(X)
            self.pca_mean = torch.from_numpy(pca_model.mean_.copy()).float().to(self.device)
            self.pca_components = torch.from_numpy(pca_model.components_.copy()).float().to(self.device)
        else:
            self.pca_mean = None
            self.pca_components = None

        gmm = GaussianMixture(
            n_components=self.num_components,
            covariance_type="diag",
            random_state=self.random_state,
            reg_covar=1e-6,
            init_params="kmeans",
            max_iter=200,
        )
        gmm.fit(X)
        # store as torch tensors for GPU use
        self.gmm_weights = torch.from_numpy(gmm.weights_.copy()).float().to(self.device)  # (K,)
        self.gmm_means = torch.from_numpy(gmm.means_.copy()).float().to(self.device)      # (K, D)
        # gmm.covariances_ is (K, D) for diag
        self.gmm_cov_diag = torch.from_numpy(gmm.covariances_.copy()).float().to(self.device)  # (K, D)

    def _project(self, X: torch.Tensor) -> torch.Tensor:
        if self.pca_components is None or self.pca_mean is None:
            return X
        Xc = X - self.pca_mean
        return torch.matmul(Xc, self.pca_components.t())

    def encode(self, descriptors: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute IFV for one image's local descriptors using torch (GPU-capable)."""
        assert self.gmm_means is not None, "Encoder not fitted"
        if isinstance(descriptors, np.ndarray):
            X = torch.from_numpy(descriptors).float().to(self.device)
        else:
            X = descriptors.float().to(self.device)

        X = self._project(X)
        N, D = X.shape
        K = self.gmm_weights.shape[0]

        pi = self.gmm_weights  # (K,)
        mu = self.gmm_means    # (K, D)
        var = self.gmm_cov_diag  # (K, D)
        sigma = torch.sqrt(var + 1e-12)

        # log prob per component: (N, K)
        # log N(x|mu,var) = -0.5 * ( (x-mu)^2/var + log(2pi) + log var ).sum
        x_exp = X.unsqueeze(1)  # (N,1,D)
        mu_exp = mu.unsqueeze(0)  # (1,K,D)
        var_exp = var.unsqueeze(0)
        log_det = torch.log(var_exp + 1e-12).sum(dim=2)  # (1,K)
        quad = ((x_exp - mu_exp) ** 2 / (var_exp + 1e-12)).sum(dim=2)  # (N,K)
        log_prob = -0.5 * (quad + log_det + D * torch.log(torch.tensor(2 * np.pi, device=self.device)))
        log_prob = log_prob + torch.log(pi + 1e-12)
        log_gamma = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
        gamma = torch.exp(log_gamma)  # (N,K)

        # z_{n,k} = (x_n - mu_k)/sigma_k
        z = (x_exp - mu_exp) / (sigma.unsqueeze(0) + 1e-12)  # (N,K,D)
        g_exp = gamma.unsqueeze(2)  # (N,K,1)
        u = (g_exp * z).sum(dim=0)  # (K,D)
        v = (g_exp * (z ** 2 - 1.0)).sum(dim=0)  # (K,D)

        scale_u = 1.0 / (N * torch.sqrt(pi + 1e-12))  # (K,)
        scale_v = 1.0 / (N * torch.sqrt(2.0 * (pi + 1e-12)))  # (K,)
        u = u * scale_u.unsqueeze(1)
        v = v * scale_v.unsqueeze(1)

        fv = torch.cat([u.reshape(-1), v.reshape(-1)], dim=0)
        if self.signed_sqrt_postprocess:
            fv = _signed_sqrt(fv)
        if self.l2_postprocess:
            fv = _l2_normalize(fv)
        return fv.detach().cpu().numpy()

    def encode_batch(self, descriptor_list: list[Union[np.ndarray, torch.Tensor]]) -> np.ndarray:
        codes = [self.encode(d) for d in descriptor_list]
        return np.stack(codes, axis=0)
