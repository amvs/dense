import torch
import torch.fft as fft
from torch import nn
import math
from typing import Optional
from wph.layers.utils import periodic_rotate

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
            expanded_filters = filters.expand(
                -1, -1, self.L, -1, -1, -1
            ).clone()  # Clone to avoid in-place operations
            for l in range(self.L):
                angle = l / self.L * 180  # Convert to degrees
                rotated_filters = periodic_rotate(
                    filters.flatten(end_dim=-3).clone(), angle
                )
                rotated_filters = torch.complex(
                    rotated_filters.real, torch.zeros_like(rotated_filters.imag)
                )  # zero out imaginary part
                # kymatio filter bank also does this - claims it makes no difference
                # bc imag part is already zero (this doesn't seem to be true in practice)
                # but we zero it out to be consistent

                # Reconstruct complex filters
                expanded_filters[:, :, l, :, :, :] = rotated_filters.view_as(
                    expanded_filters[:, :, l, :, :, :]
                )
            filters = expanded_filters

        if self.share_phases:
            expanded_filters = filters.expand(
                -1, -1, -1, self.A, -1, -1
            ).clone()  # Clone to avoid in-place operations
            i = torch.tensor(1j, device=filters.device, dtype=filters.dtype)
            for a in range(self.A):
                phase_shift = torch.exp(i * a * (2 * math.pi / self.A))
                expanded_filters[:, :, :, a, :, :] = (
                    expanded_filters[:, :, :, a, :, :] * phase_shift
                )
            filters = expanded_filters

        # Cache the computed full filters only in eager mode
        if not torch.jit.is_scripting():
            self._cache_full_filters(filters)
        return filters

    @torch.jit.ignore
    def _cache_full_filters(self, filters):
        self.full_filters = filters
        self.filters_cached = True

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
        hatpsi_la = filters.unsqueeze(0)  # (1, nc, J, L, A, M, N)
        hatxpsi_bc = hatpsi_la * hatx_c.view(
            nb, num_channels, 1, 1, 1, self.M, self.N
        )  # (nb, nc, J, L, A, M, N)
        xpsi_bc = fft.ifft2(hatxpsi_bc)  # (nb, nc, J, L, A, M, N)
        return xpsi_bc


class WaveConvLayerDownsample(nn.Module):
    def __init__(
        self,
        J,
        L,
        A,
        T,
        num_channels=1,
        share_rotations=False,
        share_channels=False,
        share_phases=False,
        share_scales=False,
        init_filters=None, # optional: pass pre-computed base filters
    ):
        super().__init__()
        self.J = J
        self.L = L
        self.A = A
        self.T = T
        self.num_channels = num_channels
        self.share_rotations = share_rotations
        self.share_channels = share_channels
        self.share_phases = share_phases
        self.share_scales = share_scales

        # 1. Setup Base Parameters
        # We store the MINIMUM necessary parameters to allow gradient sharing.

        # Determine shape of the learnable parameter
        # If sharing rotations, we only need 1 orientation. Else L.
        param_L = 1 if share_rotations else L
        param_A = 1 if share_phases else A
        param_J = 1 if share_scales else J

        # For the Pyramid model, we assume filter size T is CONSTANT.
        if init_filters is None:
            # Random complex initialization
            # Shape: (param_J, param_L, param_A, T, T) (1 input channel per filter, usually)
            base_real = torch.randn(param_J, param_L, param_A, T, T)
            base_imag = torch.randn(param_J, param_L, param_A, T, T)
        else:
            # Ensure init_filters matches (param_J, param_L, param_A, T, T)
            assert init_filters.shape == (param_J, param_L, param_A, T, T), \
                f"init_filters shape {init_filters.shape} does not match expected shape {(param_J, param_L, param_A, T, T)}"
            base_real = init_filters.real
            base_imag = init_filters.imag

        self.base_real = nn.Parameter(base_real)
        self.base_imag = nn.Parameter(base_imag)
        self.filters_cached = False
        self.full_filters_real = None
        self.full_filters_imag = None
        

        # 2. Antialiasing Filter (Fixed buffer)
        self.register_buffer("aa_filter", self._create_gaussian_kernel())

        self.base_real.register_hook(lambda grad: self._invalidate_cache())
        self.base_imag.register_hook(lambda grad: self._invalidate_cache())

    def _invalidate_cache(self):
        """Invalidates the full filters cache."""
        self.filters_cached = False

    def _create_gaussian_kernel(self, sigma=0.8, kernel_size=3):
        # Create separable Gaussian
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        return gauss_2d.view(1, 1, kernel_size, kernel_size)

    def get_full_filters(self):
        """
        Constructs the full filter bank (L*A filters) from the base parameters.
        """
        # 1. Combine to complex for rotation/phase math
        # Shape: (param_J, param_L, param_A, T, T)
        # if sharing scales, don't need to expand, just use scale=0 filter
        # multiple times on progressively downsampled inputs
        # if not sharing scales, we have J filters already
        if self.filters_cached and self.full_filters_real is not None and self.full_filters_imag is not None:
            return torch.complex(self.full_filters_real, self.full_filters_imag)
        
        param_J = 1 if self.share_scales else self.J
        filters = torch.complex(self.base_real, self.base_imag)

        # 2. Handle Rotation Expansion
        if self.share_rotations:
            # We have 1 filter, we need L
            # filters: (1, 1, T, T) -> (L, 1, T, T)
            rotated_list = []
            for l in range(self.L):
                angle = l * (180.0 / self.L)
                # Ensure your rotation function supports gradients!
                rot_f = periodic_rotate(filters.flatten(start_dim=1, end_dim=-3), angle)
                rotated_list.append(rot_f.view_as(filters))
            filters = torch.cat(rotated_list, dim=1)  # (param_J, L, param_A, T, T)

        # 3. Handle Phase Expansion
        if self.share_phases:
            # We have L filters, we need L*A
            # filters: (L, 1, T, T) -> (L, A, T, T)
            # Ensure flexibility: only unsqueeze if the A dimension does not already exist
            if filters.dim() < 4:
                filters = filters.unsqueeze(1)
            filters = filters.repeat(1, self.A, 1, 1)

            # Create phase shifts
            phases = torch.arange(self.A, device=filters.device) * (
                2 * math.pi / self.A
            )
            phase_shift = torch.exp(1j * phases).view(1, self.A, 1, 1)

            filters = filters * phase_shift

            # Flatten to (L*A, 1, T, T)
            filters = filters.view(param_J, self.L * self.A, 1, self.T, self.T)

        filters = filters.reshape(param_J, self.L * self.A, 1, self.T, self.T)
        self.full_filters_real = filters.real
        self.full_filters_imag = filters.imag
        self.filters_cached = True
        return filters

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        nb, nc, m, n = x.shape
        results = []
        current_x = x

        # Generate filters for this pass (autograd-safe)
        # Shape: (L*A, 1, T, T)
        filters_c = self.get_full_filters()
        w_real = filters_c.real
        w_imag = filters_c.imag

        pad_amt = self.T // 2
        # F.pad format is (left, right, top, bottom)
        padding = (pad_amt, pad_amt, pad_amt, pad_amt)

        for j in range(self.J):
            # select filter for scale j
            if self.share_scales:
                j_real = w_real[0,...]
                j_imag = w_imag[0,...]
            else:
                j_real = w_real[j,...]
                j_imag = w_imag[j,...]

            # Expand for Grouped Convolution
            # We want to apply the bank of (L*A) filters to EACH input channel.
            # Standard trick: Repeat weights nc times
            # Shape: (nc * L * A, 1, T, T)
            j_real = j_real.repeat(nc, 1, 1, 1)
            j_imag = j_imag.repeat(nc, 1, 1, 1)
            # 1. Explicit Circular Padding
            x_pad = nn.functional.pad(current_x, padding, mode="circular")

            # 2. Convolution (Grouped)
            y_real = nn.functional.conv2d(x_pad, j_real, groups=nc)
            y_imag = nn.functional.conv2d(x_pad, j_imag, groups=nc)
            
            # 3. Reshape and Store
            # Output is (B, nc*L*A, H, W) -> (B, nc, L, A, H, W)
            m_j, n_j = y_real.shape[-2:]
            out = torch.complex(
                y_real.view(nb, nc, self.L, self.A, m_j, n_j),
                y_imag.view(nb, nc, self.L, self.A, m_j, n_j),
            )
            results.append(out)

            # 4. Downsample Logic (Pyramid)
            if j < self.J - 1:
                # Antialias (Gaussian)
                # Expand AA filter to match channels: (C, 1, 3, 3)
                aa_k = self.aa_filter.expand(nc, 1, -1, -1)

                # Pad for AA (Reflect implies continuity)
                x_aa_pad = nn.functional.pad(current_x, (1, 1, 1, 1), mode="reflect")
                x_blur = nn.functional.conv2d(x_aa_pad, aa_k, groups=nc)

                # Decimate
                current_x = x_blur[:, :, ::2, ::2]

        return results
