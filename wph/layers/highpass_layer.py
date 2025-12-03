import torch
import torch.fft as fft
from torch import nn
from wph.layers.utils import create_masks_shift
from wph.ops.backend import DivInitStd, maskns


class HighpassLayer(nn.Module):
    def __init__(
        self,
        J: int,
        M: int,
        N: int,
        wavelets: str = "morlet",
        mask_angles: int = 4,
        mask_union: bool = True,
        num_channels: int = 3,
        mask_union_highpass: bool = True,
    ):
        super().__init__()
        self.wavelets = wavelets
        self.J = J
        self.M = M
        self.N = N
        self.mask_angles = mask_angles
        self.mask_union = mask_union
        self.num_channels = num_channels
        self.mask_union_highpass = mask_union_highpass
        self.build_haar(M=self.M, N=self.N)
        masks_shift, factr_shift = create_masks_shift(
            J=self.J,
            M=self.M,
            N=self.N,
            mask_union=self.mask_union,
            mask_angles=self.mask_angles,
        )
        self.register_buffer("masks_shift", masks_shift)
        self.factr_shift = factr_shift

        masks = maskns(self.J, self.M, self.N)
        if self.num_channels == 1:
            masks = masks.unsqueeze(1).unsqueeze(1)  # (J, M, N)
        else:
            masks = masks.view(1, self.J, 1, 1, self.M, self.N)
        self.register_buffer("masks", masks)
        if self.mask_union_highpass:
            mask = (self.masks_shift.sum(dim=0) > 0).to(dtype=torch.int32)
        else:
            mask = self.masks_shift[-1, ...]
        self.nb_moments = mask.sum().item() * num_channels**2

    def forward(self, hatx_c: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        """
        Applies a highpass filter to the input tensor x.
        x: input tensor of shape (batch_size, num_channels, M, N)
        Returns: highpass filtered tensor of the same shape
        """
        nb, nc = hatx_c.shape[:2]
        assert nc == self.num_channels
        out = []
        for hid1 in range(nc):
            for hid2 in range(nc):
                hatpsih_c = hatx_c[:, hid1, ...] * self.hathaar2d[hid2, ...].expand(
                    nb, -1, -1
                )
                xpsih_c = fft.ifft2(hatpsih_c)
                xpsih_c = self.divinitstdH[3 * hid1 + hid2](xpsih_c)
                xpsih_c = xpsih_c * self.masks[0, 0, ...].squeeze().expand(nb, -1, -1)
                xpsih_c = torch.complex(xpsih_c.abs(), torch.zeros_like(xpsih_c.real))
                xpsih_c = fft.fft2(xpsih_c)
                xpsih_c = fft.ifft2(xpsih_c * torch.conj(xpsih_c))
                xpsih_c = torch.real(xpsih_c) * self.masks_shift[-1].expand(nb, -1, -1)

                if self.mask_union_highpass:
                    mask = (self.masks_shift.sum(dim=0) > 0).to(dtype=torch.int32)
                else:
                    mask = self.masks_shift[-1, ...]
                if flatten:
                    out.append(self.select_shifts(xpsih_c.real, mask=mask))
                else:

                    xpsih_c = torch.real(xpsih_c) * mask.expand(nb, -1, -1)
                    out.append(xpsih_c)

        if flatten:
            xpsih_c = torch.cat(out, dim=1)
        else:
            xpsih_c = torch.cat(out, dim=0)
        return xpsih_c

    def build_haar(self, M, N):
        # add haar filters for high frequencies
        hathaar2d = torch.zeros(3, M, N, dtype=torch.cfloat)
        psi = torch.zeros(M, N, 2)
        psi[1, 1, 1] = 1 / 4
        psi[1, 2, 1] = -1 / 4
        psi[2, 1, 1] = 1 / 4
        psi[2, 2, 1] = -1 / 4
        hathaar2d[0, :, :] = fft.fft2(torch.view_as_complex(psi))

        psi[1, 1, 1] = 1 / 4
        psi[1, 2, 1] = 1 / 4
        psi[2, 1, 1] = -1 / 4
        psi[2, 2, 1] = -1 / 4
        hathaar2d[1, :, :] = fft.fft2(torch.view_as_complex(psi))

        psi[1, 1, 1] = 1 / 4
        psi[1, 2, 1] = -1 / 4
        psi[2, 1, 1] = -1 / 4
        psi[2, 2, 1] = 1 / 4
        hathaar2d[2, :, :] = fft.fft2(torch.view_as_complex(psi))
        self.register_buffer("hathaar2d", hathaar2d)

        self.divinitstdH = [None] * 3 * self.num_channels
        for hid in range(3 * self.num_channels):
            self.divinitstdH[hid] = DivInitStd()

    def select_shifts(self, signal, mask=None):
        if mask is None:
            mask = self.masks_shift[-1, ...]
        shape = signal.shape
        nb = shape[0]
        signal = signal.reshape(nb, -1)
        mask_flat = mask.expand(shape).reshape(nb, -1).bool()

        return signal[mask_flat].reshape(nb, -1)
