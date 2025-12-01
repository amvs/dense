import torch
import torch.fft as fft
import torch.nn.functional as F
from torch import nn
import math
from typing import Optional
from wph.layers.utils import periodic_rotate
import pdb

class WaveConvLayer(nn.Module):
    def __init__(
        self,
        J: int,
        L: int,
        A: int,
        M: int,
        N: int,
        num_channels: int = 1,
        filters: Optional[torch.Tensor] = None,
        share_rotations: bool = False,
        share_phases: bool = False,
        share_channels: bool = True,
    ):
        """
        Initializes the wavelet convolution layer.
        J: number of scales
        L: number of orientations
        A: number of phase shifts
        M, N: spatial dimensions of filters and input signals
        num_channels: number of input channels
        filters: precomputed filters. If None, filters are initialized randomly.
        share_scales: if True, all scales share the same filter (parameter shape: 1 instead of J)
        share_rotations: if True, all rotations share the same filter (parameter shape: 1 instead of L)
        share_phases: if True, all phases share the same filter (parameter shape: 1 instead of A)
        share_channels: if True, all channels share the same filters (parameter shape: 1 instead of num_channels)
        """
        super().__init__()
        self.J = J
        self.L = L
        self.A = A
        self.M = M
        self.N = N
        self.num_channels = num_channels
        self.share_rotations = share_rotations
        self.share_phases = share_phases
        self.share_channels = share_channels

        # Determine parameter shape based on sharing
        param_nc = 1 if share_channels else num_channels
        param_L = 1 if share_rotations else L
        param_A = 1 if share_phases else A

        if filters is None:
            # Initialize randomly
            real = torch.randn(param_nc, self.J, param_L, param_A, M, N)
            imag = torch.randn(param_nc, self.J, param_L, param_A, M, N)
            base_filters = torch.complex(real, imag)
        else:
            # Use precomputed filters (assume they're already in reduced form)
            expected_shape = (param_nc, self.J, param_L, param_A, M, N)
            assert (
                filters.shape == expected_shape
            ), f"filters shape {filters.shape} does not match expected shape {expected_shape}"
            base_filters = filters

        self.base_filters = nn.Parameter(base_filters)
        # Add caching for full filters
        self.register_buffer("full_filters", None)
        self.filters_cached = False
        # Register a hook to invalidate cache when base_filters are updated
        self.base_filters.register_hook(lambda grad: self._invalidate_cache())

    def _invalidate_cache(self):
        """Invalidates the full filters cache."""
        self.filters_cached = False

    def get_full_filters(self):
        """Reconstruct full filter bank from base filters."""
        if self.filters_cached and self.full_filters is not None:
            return self.full_filters

        filters = self.base_filters

        # Expand dimensions based on sharing settings
        if self.share_channels:
            filters = filters.expand(self.num_channels, -1, -1, -1, -1, -1)
        if self.share_rotations:
            expanded_filters = filters.expand(-1, -1, self.L, -1, -1, -1).clone()  # Clone to avoid in-place operations
            for l in range(self.L):
                angle = l / self.L * 180  # Convert to degrees
                rotated_filters = periodic_rotate(filters.flatten(end_dim=-3).clone(), angle)
                rotated_filters = torch.complex(rotated_filters.real, \
                                                torch.zeros_like(rotated_filters.imag)) # zero out imaginary part
                # kymatio filter bank also does this - claims it makes no difference
                # bc imag part is already zero (this doesn't seem to be true in practice)
                # but we zero it out to be consistent
                
                # Reconstruct complex filters
                expanded_filters[:, :, l, :, :, :] = rotated_filters.view_as(expanded_filters[:, :, l, :, :, :])
            filters = expanded_filters

        if self.share_phases:
            expanded_filters = filters.expand(-1, -1, -1, self.A, -1, -1).clone()  # Clone to avoid in-place operations
            for a in range(self.A):
                i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
                phase_shift = torch.exp(i * a * (2 * math.pi / self.A))
                expanded_filters[:, :, :, a, :, :] = expanded_filters[:, :, :, a, :, :] * phase_shift
            filters = expanded_filters

        # Cache the computed full filters
        self.full_filters = filters
        self.filters_cached = True
        return filters

    def forward(self, x):
        """
        Applies the wavelet convolution layer to the input tensor x.
        x: (nb, nc, M, N), real-valued input tensor
        returns: (nb, nc, J, L, A, M, N), complex-valued output tensor
        """
        nb = x.shape[0]
        num_channels = x.shape[1]

        # Get full filter bank (with sharing applied)
        filters = self.get_full_filters()  # (nc, J, L, A, M, N)

        hatx_c = fft.fft2(torch.complex(x, torch.zeros_like(x)))  # (nb, nc, M, N)
        hatpsi_la = filters.unsqueeze(0).conj()  # (1, nc, J, L, A, M, N)
        hatxpsi_bc = hatpsi_la * hatx_c.view(
            nb, num_channels, 1, 1, 1, self.M, self.N
        )  # (nb, nc, J, L, A, M, N)
        xpsi_bc = fft.ifft2(hatxpsi_bc)  # (nb, nc, J, L, A, M, N)
        return xpsi_bc
