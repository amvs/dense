import torch
from torch import nn
import torch.nn.functional as F
import math
from wph.ops.backend import SubInitSpatialMean, DivInitStd


class ReluCenterLayer(nn.Module):
    def __init__(self, J: int, M: int, N: int, normalize: bool = True):
        """
        Initializes the ReLU center layer.
        J: number of scales
        M, N: spatial dimensions of input signals
        """
        super().__init__()
        self.J = J
        self.M = M
        self.N = N
        self.normalize = normalize
        masks = self.maskns(J, M, N)
        # shape to (1, 1, J, 1, 1, M, N) to broadcast over (nb, nc, J, L, A, M, N)
        masks = masks.view(1, 1, J, 1, 1, M, N)
        self.register_buffer("masks", masks)
        self.mean = SubInitSpatialMean()
        self.std = DivInitStd()

    def forward(self, x):
        """
        Apply ReLU center layer to input x.
        x: input tensor of shape (nb, nc, J, L, A, M, N)
        1. Normalize x by subtracting mean and dividing by std if normalize is True.
        2. Apply ReLU activation to the real part of x.
        3. Multiply by precomputed masks for aperiodic signals.
        returns: output tensor with shape (nb, nc, J, L, A, M, N)
        """
        nb, nc = x.shape[:2]
        assert x.shape[2] == self.J, f"Expected J={self.J}, but got {x.shape[2]}"
        assert x.shape[-2:] == (
            self.M,
            self.N,
        ), f"Expected spatial dimensions {(self.M, self.N)}, but got {x.shape[-2:]}"
        # subtract spatial mean
        x = self.mean(x)
        # optionally divide by precomputed std (does not re-center)
        if self.normalize:
            x = self.std(x)
        # take real part and apply ReLU
        x = F.relu(x.real)
        # apply masks per-scale and return
        x = x * self.masks.expand(nb, nc, -1, -1, -1, -1, -1)
        return x

    def maskns(self, J, M, N):
        """
        Create masks for aperiodic images
        """
        # Create a grid of coordinates
        x, y = torch.meshgrid(torch.arange(M), torch.arange(N))

        # Compute the mask using broadcasting
        masks = []
        for j in range(J):
            mask = (
                (x >= (2**j) // 2)
                & (y >= (2**j) // 2)
                & (x < M - (2**j) // 2)
                & (y < N - (2**j) // 2)
            )

            # Normalize the mask
            mask = mask.float()
            mask /= mask.sum(dim=(-1, -2), keepdim=True)
            mask *= M * N

            masks.append(mask)

        return torch.stack(masks, dim=0)

class ReluCenterLayerDownsample(ReluCenterLayer):
    def __init__(self, **kwargs):
        # Initialize Base without calling mask logic yet (or ignore base mask)
        super().__init__(**kwargs)
        
        # Overwrite masks with correct downsampled list
        self.masks = nn.ParameterList() 
        _masks_list = []
        
        for j in range(self.J):
            h_j = math.ceil(self.M / (2 ** j))
            w_j = math.ceil(self.N / (2 ** j))
            
            # differnt mask logic for downsampled feature maps
            # On the full grid, border is 2^j // 2.
            # On the downsampled grid (factor 2^j), this border size becomes:
            # (2^j // 2) / 2^j = 0.5 pixels.
            # So effectively, for j >= 1, we treat the border as 0 or 1 pixel.
            #  to be safe, reuse the J=0 logic (border=0) for all scales.
            
        
            # Using J=1 forces maskns to calculate for j=0 (border=0).
            m_stack = self.maskns(J=1, M=h_j, N=w_j) 
            mask = m_stack[0] # Take the only mask
            
            # Input item shape: (B, C, L, A, H, W)
            # Mask needs to be: (1, 1, 1, 1, H, W)
            mask = mask.view(1, 1, 1, 1, h_j, w_j)
            
            _masks_list.append(mask)

        # Register them properly so they move to GPU with the model
        for i, m in enumerate(_masks_list):
            self.register_buffer(f'mask_{i}', m)

    def get_mask(self, idx):
        # Helper to retrieve buffer by name
        return getattr(self, f'mask_{idx}')

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        x: list of J tensors
        """
        out = []
        for idx, feature_map in enumerate(x):
            # Checks — use ceiling division because downsampled sizes may round up
            expected_h = math.ceil(self.M / (2 ** idx))
            expected_w = math.ceil(self.N / (2 ** idx))
            assert feature_map.shape[-2:] == (expected_h, expected_w), (
                f"Expected spatial dims {(expected_h, expected_w)} for scale {idx}, "
                f"but got {feature_map.shape[-2:]}"
            )
            
            # 1. Norm
            feature_map = self.mean(feature_map)
            if self.normalize:
                feature_map = self.std(feature_map)
            
            # 2. ReLU (Real)
            if torch.is_complex(feature_map):
                feature_map = feature_map.real
                
            feature_map = F.relu(feature_map)
            
            # 3. Mask
            mask = self.get_mask(idx)
            feature_map = feature_map * mask
            
            out.append(feature_map)
        return out

class ReluCenterLayerDownsamplePairs(ReluCenterLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Build per-scale masks as in ReluCenterLayerDownsample
        _masks_list = []
        for j in range(self.J):
            h_j = math.ceil(self.M / (2 ** j))
            w_j = math.ceil(self.N / (2 ** j))
            m_stack = self.maskns(J=1, M=h_j, N=w_j)
            mask = m_stack[0]
            mask = mask.view(1, 1, 1, 1, h_j, w_j)
            _masks_list.append(mask)
        for i, m in enumerate(_masks_list):
            self.register_buffer(f'mask_{i}', m)

    def get_mask(self, idx):
        return getattr(self, f'mask_{idx}')

    def forward(self, x_nested):
        """
        x_nested: nested list/dict of tensors indexed by [j1][j2] (or {j2: ...}).
        Each tensor is (B, C, L, A, H_j1, W_j1) or complex.
        Returns a nested structure with the same sparsity pattern after ReLU+mask.
        """
        out_nested = []
        for j1, row in enumerate(x_nested):
            if row is None:
                out_nested.append(None)
                continue
            if isinstance(row, dict):
                inner_out = {}
                iterable = row.items()
            else:
                inner_out = [None] * len(row)
                iterable = enumerate(row)

            for j2, feature_map in iterable:
                if feature_map is None:
                    if isinstance(inner_out, dict):
                        inner_out[j2] = None
                    else:
                        inner_out[j2] = None
                    continue
                expected_h = math.ceil(self.M / (2 ** j1))
                expected_w = math.ceil(self.N / (2 ** j1))
                assert feature_map.shape[-2:] == (expected_h, expected_w), (
                    f"Expected spatial dims {(expected_h, expected_w)} for scale {j1}, got {feature_map.shape[-2:]}"
                )
                fm = self.mean(feature_map)
                if self.normalize:
                    fm = self.std(fm)
                if torch.is_complex(fm):
                    fm = fm.real
                fm = F.relu(fm)
                mask = self.get_mask(j1)
                fm = fm * mask
                if isinstance(inner_out, dict):
                    inner_out[j2] = fm
                else:
                    inner_out[j2] = fm
            out_nested.append(inner_out)
        return out_nested