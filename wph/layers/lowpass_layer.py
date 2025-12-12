import torch
import torch.fft as fft
from torch import nn
from wph.ops.backend import DivInitStd, SubInitSpatialMean, maskns


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
        self.nb_moments = M * N * num_channels**2

        self.divinitstdJ = DivInitStd()
        self.subinitmeanJ = SubInitSpatialMean()

        masks = maskns(self.J, self.M, self.N)
        if self.num_channels == 1:
            masks = masks.unsqueeze(1).unsqueeze(1)  # (J, M, N)
        else:
            masks = masks.view(1, self.J, 1, 1, self.M, self.N)
        self.register_buffer("masks", masks)

    def forward(self, hatx_c: torch.Tensor) -> torch.Tensor:
        nb, nc = hatx_c.shape[:2]
        hatxphi_c = hatx_c * self.hatphi.expand(nb, nc, -1, -1)  # (nb,nc,M,N)
        xphi_c = fft.ifft2(hatxphi_c)
        xphi_c.mul_(self.masks[-1, -1, ...].view(1, 1, self.M, self.N))
        xphi0_c = self.subinitmeanJ(xphi_c)
        xphi0_c = self.divinitstdJ(xphi0_c)

        xphi0_c = xphi0_c.abs()

        z = xphi0_c.repeat(1, nc, 1, 1)
        z_ = torch.repeat_interleave(xphi0_c, self.num_channels, dim=1)
        xphi0_c = fft.ifft2(fft.fft2(z) * torch.conj(fft.fft2(z_)))

        return xphi0_c.real
