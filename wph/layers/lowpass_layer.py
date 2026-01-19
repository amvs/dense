import torch
import torch.fft as fft
from torch import nn
from wph.ops.backend import DivInitStd, SubInitSpatialMean, maskns
from wph.layers.utils import create_masks_shift


class LowpassLayer(nn.Module):
    def __init__(
        self,
        J: int,
        M: int,
        N: int,
        hatphi: torch.Tensor,
        mask_angles: int = 4,
        mask_union: bool = True,
        num_channels: int = 3,
    ):
        super().__init__()
        self.J = J
        self.M = M
        self.N = N
        self.register_buffer("hatphi", hatphi)
        self.mask_angles = mask_angles
        self.mask_union = mask_union
        self.num_channels = num_channels
        self.divinitstdJ = DivInitStd()
        self.subinitmeanJ = SubInitSpatialMean()

        # create border masks for aperiodicity (used during forward)
        masks = maskns(self.J, self.M, self.N)
        if self.num_channels == 1:
            masks = masks.unsqueeze(1).unsqueeze(1)
        else:
            masks = masks.view(1, self.J, 1, 1, self.M, self.N)
        self.register_buffer("masks", masks)

        # create shift masks to select correlation positions (matching alpha_torch)
        masks_shift, _ = create_masks_shift(
            J=self.J,
            M=self.M,
            N=self.N,
            mask_union=self.mask_union,
            mask_angles=self.mask_angles,
        )
        self.register_buffer("masks_shift", masks_shift)

        # use the last shift mask (union of all shifts or largest scale depending on mask_union)
        mask_shift_last = self.masks_shift[-1, ...]
        n_shifts = int(mask_shift_last.sum().item())
        # number of low-pass moments: shift positions times all channel pairs
        self.nb_moments = n_shifts * (self.num_channels ** 2)

    def select_shifts(self, signal, mask=None):
        """Select only the shift positions from the signal (matching alpha_torch)."""
        if mask is None:
            mask = self.masks_shift[-1, ...]
        shape = signal.shape
        nb = shape[0]
        signal = signal.reshape(nb, -1)
        mask_flat = mask.expand(shape).reshape(nb, -1).bool()
        return signal[mask_flat].reshape(nb, -1)

    def flat_metadata(self):
        """Return metadata aligned with flattened output order."""
        mask = self.masks_shift[-1, ...]
        mask_positions = torch.nonzero(mask, as_tuple=False)
        n_shifts = len(mask_positions)
        
        meta = {
            "channel1": [],   # First channel in cross-correlation
            "channel2": [],   # Second channel in cross-correlation
            "mask_pos": [],   # Position in the mask
        }
        
        # Iterate in same order as forward pass: compute correlations between all (c1,c2) pairs across all positions
        # The forward pass applies a single lowpass filter to all channels, then correlates all channel pairs
        for c1 in range(self.num_channels):
            for c2 in range(self.num_channels):
                for pos_idx in range(n_shifts):
                    meta["channel1"].append(c1)
                    meta["channel2"].append(c2)
                    meta["mask_pos"].append(pos_idx)
        return meta

    def forward(self, hatx_c: torch.Tensor) -> torch.Tensor:
        nb, nc = hatx_c.shape[:2]
        hatxphi_c = hatx_c * self.hatphi.expand(nb, nc, -1, -1)  # (nb,nc,M,N)
        xphi_c = fft.ifft2(hatxphi_c)
        # apply border mask for aperiodicity (matching alpha_torch)
        xphi_c = xphi_c * self.masks[-1, -1, ...].view(1, 1, self.M, self.N)
        xphi0_c = self.subinitmeanJ(xphi_c)
        xphi0_c = self.divinitstdJ(xphi0_c)

        xphi0_c = xphi0_c.abs()

        # compute cross-correlations between all channel pairs -> (nb, nc*nc, M, N)
        z = xphi0_c.repeat(1, nc, 1, 1)
        z_ = torch.repeat_interleave(xphi0_c, self.num_channels, dim=1)
        xphi0_c = fft.ifft2(fft.fft2(z) * torch.conj(fft.fft2(z_)))
        xphi0_c = xphi0_c.real

        # select only the shift positions (not the full masked region)
        xphi0_c = self.select_shifts(xphi0_c)

        return xphi0_c
