import math
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from wph.layers.utils import periodic_rotate
import warnings


class WaveConvLayerHybrid(nn.Module):
    """
    Hybrid wavelet convolution layer mixing signal downsampling and filter upsampling.
    All filters are stored in the spatial domain. Each scale j uses a user-provided
    downsample_splits[j] (number of signal downsampling steps); remaining j - splits
    steps are applied as filter upsampling.
    """

    def __init__(
        self,
        J: int,
        L: int,
        A: int,
        T: int,
        M: int,
        N: int,
        num_channels: int = 1,
        downsample_splits: Optional[List[int]] = None,
        share_rotations: bool = False,
        share_phases: bool = False,
        share_channels: bool = False,
        share_scales: bool = False,
        share_scale_pairs: bool = True,
        init_filters: Optional[torch.Tensor] = None,
        use_antialiasing: bool = False,
    ):
        super().__init__()
        self.J = J
        self.L = L
        self.A = A
        self.T = T
        self.M = M
        self.N = N
        self.num_channels = num_channels
        self.share_rotations = share_rotations
        self.share_phases = share_phases
        self.share_channels = share_channels
        self.share_scales = share_scales
        self.share_scale_pairs = True if share_scales else share_scale_pairs
        self.use_antialiasing = use_antialiasing

        # Validate downsample splits
        if downsample_splits is None:
            self.downsample_splits = [j // 2 for j in range(J)]
        else:
            assert len(downsample_splits) == J, "downsample_splits must have length J"
            self.downsample_splits = list(downsample_splits)
        for j, ds in enumerate(self.downsample_splits):
            if ds > j:
                raise ValueError(f"downsample_splits[{j}]={ds} cannot exceed scale index {j}")
        if any(self.downsample_splits[i] > self.downsample_splits[i + 1] for i in range(J - 1)):
            warnings.warn("downsample_splits is not non-decreasing; outputs may upsample implicitly between scales")

        # Parameter dimensions
        self.param_nc = 1 if share_channels else num_channels
        self.param_L = 1 if share_rotations else L
        self.param_A = 1 if share_phases else A
        self.param_J = 1 if self.share_scales else (J if self.share_scale_pairs else J * J)

        # Initialize base filters (T x T)
        if init_filters is None:
            base_real = torch.randn(self.param_J, self.param_L, self.param_A, T, T)
            base_imag = torch.randn(self.param_J, self.param_L, self.param_A, T, T)
        else:
            expected_shape = (self.param_J, self.param_L, self.param_A, T, T)
            assert init_filters.shape == expected_shape, (
                f"init_filters shape {init_filters.shape} does not match expected {expected_shape}"
            )
            base_real = init_filters.real
            base_imag = init_filters.imag
        self.base_real = nn.Parameter(base_real)
        self.base_imag = nn.Parameter(base_imag)

        # Upsampled filter parameters per scale (sharing-aware)
        self.upsampled_real: List[Optional[nn.Parameter]] = [None] * J
        self.upsampled_imag: List[Optional[nn.Parameter]] = [None] * J

        if self.share_scale_pairs:
            for j in range(J):
                up_pow = (j - self.downsample_splits[j])
                up_factor = 2 ** up_pow
                if up_factor > 1:
                    # Preserve odd filter growth: size = (T-1) * 2^r + 1
                    size = (T - 1) * up_factor + 1
                    up_real, up_imag = self._init_upsampled_params(j, size, up_factor)
                    self.upsampled_real[j] = nn.Parameter(up_real)
                    self.upsampled_imag[j] = nn.Parameter(up_imag)
                else:
                    self.upsampled_real[j] = None
                    self.upsampled_imag[j] = None

        # Pair-specific upsampled filters when share_scale_pairs is False
        self.pair_upsampled_real: List[Optional[nn.Parameter]] = []
        self.pair_upsampled_imag: List[Optional[nn.Parameter]] = []
        if not self.share_scale_pairs and not self.share_scales:
            for j in range(J):
                up_pow = (j - self.downsample_splits[j])
                up_factor = 2 ** up_pow
                if up_factor > 1:
                    size = (T - 1) * up_factor + 1
                    real_j, imag_j = self._init_pair_upsampled(j, size, up_factor)
                    self.pair_upsampled_real.append(nn.Parameter(real_j))
                    self.pair_upsampled_imag.append(nn.Parameter(imag_j))
                else:
                    self.pair_upsampled_real.append(None)
                    self.pair_upsampled_imag.append(None)

        # Caching
        self.filters_cached = False
        self.cache_full_filters = {}

        # Antialiasing filter
        self.register_buffer("aa_filter", self._create_gaussian_kernel())

        # Hooks to invalidate cache
        self.base_real.register_hook(lambda grad: self._invalidate_cache())
        self.base_imag.register_hook(lambda grad: self._invalidate_cache())
        for p in self.upsampled_real + self.upsampled_imag:
            if p is not None:
                p.register_hook(lambda grad: self._invalidate_cache())
        for p in self.pair_upsampled_real + self.pair_upsampled_imag:
            if p is not None:
                p.register_hook(lambda grad: self._invalidate_cache())

    def _init_upsampled_params(self, j: int, size: int, up_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.share_scales:
            base_real = self.base_real[0:1]
            base_imag = self.base_imag[0:1]
        else:
            base_real = self.base_real[j:j + 1] if self.share_scale_pairs else self.base_real
            base_imag = self.base_imag[j:j + 1] if self.share_scale_pairs else self.base_imag
        
        # Reshape to (batch, channels, H, W) for interpolation
        b, l, a, h, w = base_real.shape
        base_real_4d = base_real.view(b * l * a, 1, h, w)
        base_imag_4d = base_imag.view(b * l * a, 1, h, w)
        
        up_real_4d = F.interpolate(base_real_4d, size=(size, size), mode="nearest")
        up_imag_4d = F.interpolate(base_imag_4d, size=(size, size), mode="nearest")
        
        # Reshape back to (b, l, a, size, size)
        up_real = up_real_4d.view(b, l, a, size, size)
        up_imag = up_imag_4d.view(b, l, a, size, size)
        return up_real, up_imag

    def _init_pair_upsampled(self, j: int, size: int, up_factor: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_view_r = self.base_real.reshape(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
        base_view_i = self.base_imag.reshape(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
        subset_r = base_view_r[:, j:j + 1]  # shape (J,1, L, A, T, T)
        subset_i = base_view_i[:, j:j + 1]
        subset_r = subset_r.reshape(self.J, self.param_L, self.param_A, self.T, self.T)
        subset_i = subset_i.reshape(self.J, self.param_L, self.param_A, self.T, self.T)
        
        # Reshape for interpolation: (batch*L*A, 1, T, T)
        b, l, a, h, w = subset_r.shape
        subset_r_4d = subset_r.reshape(b * l * a, 1, h, w)
        subset_i_4d = subset_i.reshape(b * l * a, 1, h, w)
        
        up_real = F.interpolate(subset_r_4d, size=(size, size), mode="nearest")
        up_imag = F.interpolate(subset_i_4d, size=(size, size), mode="nearest")
        
        # Reshape back to (b, l, a, size, size)
        up_real = up_real.reshape(b, l, a, size, size)
        up_imag = up_imag.reshape(b, l, a, size, size)
        return up_real, up_imag

    def _invalidate_cache(self):
        self.filters_cached = False
        self.cache_full_filters.clear()

    def _create_gaussian_kernel(self, sigma=0.8, kernel_size=3):
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        gauss_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
        return gauss_2d.view(1, 1, kernel_size, kernel_size)

    def _expand_rotations(self, filters: torch.Tensor, T_j: int) -> torch.Tensor:
        if not self.share_rotations:
            return filters
        rotated_list = []
        for l in range(self.L):
            angle = l * (180.0 / self.L)
            rot_f = periodic_rotate(filters.flatten(end_dim=-3), angle)
            rot_f = rot_f.view_as(filters)
            rotated_list.append(rot_f)
        return torch.cat(rotated_list, dim=1)  # (param_J, L, param_A, T_j, T_j)

    def _expand_phases(self, filters: torch.Tensor) -> torch.Tensor:
        if not self.share_phases:
            return filters
        if filters.dim() < 5:
            filters = filters.unsqueeze(2)
        filters = filters.repeat(1, 1, self.A, 1, 1)
        phases = torch.arange(self.A, device=filters.device) * (2 * math.pi / self.A)
        phase_shift = torch.exp(1j * phases).view(1, 1, self.A, 1, 1)
        return filters * phase_shift

    def get_full_filters(self, j: int) -> torch.Tensor:
        if self.filters_cached and j in self.cache_full_filters:
            return self.cache_full_filters[j]

        up_pow = (j - self.downsample_splits[j])
        up_factor = 2 ** up_pow
        T_j = (self.T - 1) * up_factor + 1

        if not self.share_scale_pairs:
            # Pair-specific filters for current scale j (partners across dimension 0)
            if self.pair_upsampled_real:
                real = self.pair_upsampled_real[j] if self.pair_upsampled_real[j] is not None else None
                imag = self.pair_upsampled_imag[j] if self.pair_upsampled_imag[j] is not None else None
                if real is not None:
                    filters = torch.complex(real, imag)
                else:
                    base_view = self.base_real.view(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
                    base_view_i = self.base_imag.view(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
                    filters = torch.complex(
                        base_view[:, j, ...],
                        base_view_i[:, j, ...],
                    )
            else:
                base_view = self.base_real.view(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
                base_view_i = self.base_imag.view(self.J, self.J, self.param_L, self.param_A, self.T, self.T)
                filters = torch.complex(base_view[:, j, ...], base_view_i[:, j, ...])
            filters = filters.reshape(self.J, self.param_L, self.param_A, T_j, T_j)
        else:
            # Scale-specific filters
            real = self.upsampled_real[j] if self.upsampled_real[j] is not None else (
                self.base_real[0:1] if self.share_scales else self.base_real[j:j + 1]
            )
            imag = self.upsampled_imag[j] if self.upsampled_imag[j] is not None else (
                self.base_imag[0:1] if self.share_scales else self.base_imag[j:j + 1]
            )
            filters = torch.complex(real, imag)

        # Rotation and phase expansion
        filters = self._expand_rotations(filters, T_j)
        filters = self._expand_phases(filters)

        # Reshape for grouped conv: (param_J_effective, L*A, 1, T_j, T_j)
        filters = filters.reshape(filters.shape[0], self.L * self.A, 1, T_j, T_j)

        if not torch.jit.is_scripting():
            self.cache_full_filters[j] = filters
            self.filters_cached = True
        return filters

    def _downsample_signal(self, x: torch.Tensor, times: int, nc: int) -> torch.Tensor:
        current = x
        for _ in range(times):
            if self.use_antialiasing:
                aa_k = self.aa_filter.expand(nc, 1, -1, -1)
                x_pad = F.pad(current, (1, 1, 1, 1), mode="reflect")
                current = F.conv2d(x_pad, aa_k, groups=nc)
            current = current[:, :, ::2, ::2]
        return current

    def forward(self, x: torch.Tensor, scale_pairs: Optional[List[Tuple[int, int]]] = None):
        nb, nc, _, _ = x.shape
        results = []
        current_x = x
        cumulative_down = 0

        if self.share_scale_pairs:
            for j in range(self.J):
                desired_down = self.downsample_splits[j]
                if desired_down > cumulative_down:
                    current_x = self._downsample_signal(current_x, desired_down - cumulative_down, nc)
                    cumulative_down = desired_down

                filters_c = self.get_full_filters(j)
                up_pow = (j - self.downsample_splits[j])
                up_factor = 2 ** up_pow
                T_j = (self.T - 1) * up_factor + 1
                pad_amt = T_j // 2
                padding = (pad_amt, pad_amt, pad_amt, pad_amt)

                w_real = filters_c.real[0]
                w_imag = filters_c.imag[0]

                w_real = w_real.repeat(nc, 1, 1, 1)
                w_imag = w_imag.repeat(nc, 1, 1, 1)

                x_pad = F.pad(current_x, padding, mode="circular")
                y_real = F.conv2d(x_pad, w_real, groups=nc)
                y_imag = F.conv2d(x_pad, w_imag, groups=nc)
                m_j, n_j = y_real.shape[-2:]
                out = torch.complex(
                    y_real.view(nb, nc, self.L, self.A, m_j, n_j),
                    y_imag.view(nb, nc, self.L, self.A, m_j, n_j),
                )
                results.append(out)
            return results

        # pair-specific path
        pairs_by_j = {j: list(range(self.J)) for j in range(self.J)}
        if scale_pairs is not None:
            pairs_by_j = {j: [] for j in range(self.J)}
            for (j1, j2) in scale_pairs:
                if 0 <= j1 < self.J and 0 <= j2 < self.J:
                    pairs_by_j[j1].append(j2)
                    pairs_by_j[j2].append(j1)
            for j in range(self.J):
                pairs_by_j[j] = sorted(set(pairs_by_j[j]))

        results_nested: List[List[Optional[torch.Tensor]]] = []
        current_x = x
        cumulative_down = 0

        for j in range(self.J):
            desired_down = self.downsample_splits[j]
            if desired_down > cumulative_down:
                current_x = self._downsample_signal(current_x, desired_down - cumulative_down, nc)
                cumulative_down = desired_down

            up_pow = (j - self.downsample_splits[j])
            up_factor = 2 ** up_pow
            T_j = (self.T - 1) * up_factor + 1
            pad_amt = T_j // 2
            padding = (pad_amt, pad_amt, pad_amt, pad_amt)

            inner_list = [None] * self.J
            partners = pairs_by_j[j]
            for j2 in partners:
                pair_index = j2  # select along first dim of J partners
                filters_c = self.get_full_filters(j)
                w_real = filters_c.real[pair_index]
                w_imag = filters_c.imag[pair_index]
                w_real = w_real.repeat(nc, 1, 1, 1)
                w_imag = w_imag.repeat(nc, 1, 1, 1)

                x_pad = F.pad(current_x, padding, mode="circular")
                y_real = F.conv2d(x_pad, w_real, groups=nc)
                y_imag = F.conv2d(x_pad, w_imag, groups=nc)
                m_j, n_j = y_real.shape[-2:]
                out = torch.complex(
                    y_real.view(nb, nc, self.L, self.A, m_j, n_j),
                    y_imag.view(nb, nc, self.L, self.A, m_j, n_j),
                )
                inner_list[j2] = out
            results_nested.append(inner_list)

        return results_nested
