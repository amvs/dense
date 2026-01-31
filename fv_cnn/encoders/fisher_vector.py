from __future__ import annotations
from dense.helpers import LoggerManager
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .base_encoder import BaseEncoder


class GaussianMixtureModel(nn.Module):
    """Diagonal covariance GMM for Fisher Vector encoding."""
    
    def __init__(self, n_features: int, n_components: int):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components

        self.log_weights = nn.Parameter(torch.zeros(n_components))
        self.means = nn.Parameter(torch.randn(n_components, n_features) * 0.01)
        self.log_stdevs = nn.Parameter(torch.zeros(n_components, n_features))

    def forward(self, x):
        """Compute log probability of data under GMM."""
        weights = F.softmax(self.log_weights, dim=0)
        # Clamp log_stdevs to prevent variance collapse (min stdev ~ 0.3)
        log_stdevs_clamped = torch.clamp(self.log_stdevs, min=-1.2, max=3.0)
        stdevs = F.softplus(log_stdevs_clamped) + 1e-6
        variances = stdevs ** 2

        # Compute log probabilities manually (batch x components)
        # x: (N, D), means: (K, D), stdevs: (K, D)
        x_exp = x.unsqueeze(1)  # (N, 1, D)
        means_exp = self.means.unsqueeze(0)  # (1, K, D)
        var_exp = variances.unsqueeze(0)  # (1, K, D)
        
        # log N(x|mu,sigma^2) = -0.5 * [log(2*pi*sigma^2) + (x-mu)^2/sigma^2]
        log_det = torch.log(2 * 3.14159265359 * var_exp).sum(dim=2)  # (N, K)
        quad = ((x_exp - means_exp) ** 2 / var_exp).sum(dim=2)  # (N, K)
        log_probs = -0.5 * (log_det + quad)  # (N, K)
        log_probs = log_probs + torch.log(weights + 1e-12)  # (N, K)

        # Apply log-sum-exp trick
        max_log_prob = torch.max(log_probs, dim=-1, keepdim=True)[0]
        log_sum_exp = max_log_prob + torch.log(torch.sum(torch.exp(log_probs - max_log_prob), dim=-1, keepdim=True))

        return log_sum_exp.squeeze(-1)


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
        signed_sqrt_postprocess: bool = True,
        l2_postprocess: bool = True,
        random_state: int = 42,
        gmm_batch_size: Optional[int] = None,
        gmm_log_every_n_steps: int = 1000,
        gmm_enable_progress_bar: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.signed_sqrt_postprocess = signed_sqrt_postprocess
        self.l2_postprocess = l2_postprocess
        self.random_state = random_state
        self.gmm_batch_size = gmm_batch_size
        self.gmm_log_every_n_steps = gmm_log_every_n_steps
        self.gmm_enable_progress_bar = gmm_enable_progress_bar
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # fitted parameters (torch tensors)
        self.register_buffer("gmm_weights", None)
        self.register_buffer("gmm_means", None)
        self.register_buffer("gmm_cov_diag", None)

    def fit(self, descriptors: Union[np.ndarray, torch.Tensor]) -> None:

        logger = LoggerManager.get_logger()
        assert descriptors.ndim == 2, "descriptors must be (N, D)"
        if isinstance(descriptors, np.ndarray):
            X = torch.from_numpy(descriptors).float()
        else:
            X = descriptors.float()

        n_samples, n_features = X.shape
        
        # Initialize GMM model on GPU with data-driven initialization
        gmm = GaussianMixtureModel(n_features, self.num_components).to(self.device)
        
        # Initialize with data statistics to prevent collapse
        with torch.no_grad():
            # Sample some data points for initialization (keep on CPU)
            init_indices = torch.randperm(n_samples)[:min(10000, n_samples)]
            init_data = X[init_indices]
            
            # Initialize means with k-means++ style selection (move to GPU)
            kmeans_indices = torch.randperm(init_data.shape[0])[:self.num_components]
            gmm.means.data = init_data[kmeans_indices].to(self.device)
            
            # Initialize log_stdevs based on global data spread, not per-feature
            # Use a larger initial variance to prevent collapse
            global_std = torch.std(init_data)
            # Set initial stdev to 0.5 * global_std (conservative)
            init_log_std = torch.log(torch.clamp(global_std * 0.5, min=0.5))
            gmm.log_stdevs.data = init_log_std.expand(self.num_components, n_features).to(self.device)
        
        # Setup dataloader for mini-batch training (data stays on CPU)
        batch_size = self.gmm_batch_size if self.gmm_batch_size is not None else min(1024, n_samples)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Train GMM
        optimizer = torch.optim.Adam(gmm.parameters(), lr=5e-4)
        max_iter = 2000
        convergence_tolerance = 1e-4
        patience = 300
        
        losses = []
        for epoch in range(max_iter):
            epoch_loss = 0.0
            n_batches = 0
            
            for idx, (batch_X,) in enumerate(loader):
                # Move batch to GPU
                batch_X = batch_X.to(self.device)
                
                optimizer.zero_grad()
                loss = -gmm(batch_X).mean()
                
                # Check for numerical instability
                if loss.item() < -1000:
                    logger.warning(f"GMM training became unstable (loss={loss.item():.2f}). Stopping early.")
                    break
                
                if self.gmm_enable_progress_bar and (idx % self.gmm_log_every_n_steps == 0):
                    logger.info(f"GMM Epoch {epoch:4d} Batch {idx:4d} | Loss: {loss.item():.6f}")
                loss.backward()
                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_(gmm.parameters(), max_norm=10.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            # Log progress
            logger.info(f"GMM Epoch {epoch:4d} | Loss: {avg_loss:.6f}")
            
            # Check convergence
            if avg_loss < 0.1 and epoch > 2:
                if self.gmm_enable_progress_bar:
                    logger.info(f"GMM converged at epoch {epoch} (loss < 0.1)")
                break
            
            if len(losses) > patience:
                recent_std = np.std(losses[-patience:])
                if recent_std < convergence_tolerance:
                    if self.gmm_enable_progress_bar:
                        logger.info(f"GMM converged at epoch {epoch} (std < {convergence_tolerance})")
                    break
        
        # Extract fitted parameters
        gmm.eval()
        with torch.no_grad():
            weights = F.softmax(gmm.log_weights, dim=0)
            means = gmm.means
            log_stdevs_clamped = torch.clamp(gmm.log_stdevs, min=-1.2, max=3.0)
            stdevs = F.softplus(log_stdevs_clamped) + 1e-6
            variances = stdevs ** 2
        
        # Store as torch tensors for GPU use
        self.gmm_weights = weights.detach().float().to(self.device)  # (K,)
        self.gmm_means = means.detach().float().to(self.device)      # (K, D)
        self.gmm_cov_diag = variances.detach().float().to(self.device)  # (K, D)


    def encode(self, descriptors: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Compute IFV for one image's local descriptors using torch (GPU-capable)."""
        assert self.gmm_means is not None, "Encoder not fitted"
        if isinstance(descriptors, np.ndarray):
            X = torch.from_numpy(descriptors).float().to(self.device)
        else:
            X = descriptors.float().to(self.device)

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
