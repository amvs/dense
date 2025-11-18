import torch
from torch import nn
from typing import Optional, Literal
from torch.fft import fft2, ifft2

from dense import wavelets as wlets
from wph.layers.wave_conv_layer import WaveConvLayer
from wph.layers.relu_center_layer import ReluCenterLayer
from wph.layers.corr_layer import CorrLayer
from wph.layers.lowpass_layer import LowpassLayer
from wph.layers.highpass_layer import HighpassLayer


class WPHModel(nn.Module):
    def __init__(
        self,
        J: int,
        L: int,
        A: int,
        A_prime: int,
        M: int,
        N: int,
        filters: torch.Tensor,
        num_channels: int = 1,
        train_filters: bool = True,
        share_rotations: bool = False,
        share_phases: bool = False,
        share_channels: bool = True,
        normalize_relu: bool = True,
        delta_j: Optional[int] = None,
        delta_l: Optional[int] = None,
        shift_mode: Literal["samec", "all", "strict"] = "samec",
        mask_union: bool = False,
        mask_angles: int = 4,
        mask_union_highpass: bool = True,
        wavelets: Literal["morlet", "steer"] = 'morlet',
    ):
        super().__init__()
        self.J = J
        self.L = L
        self.A = A
        self.A_prime = A_prime
        self.M = M
        self.N = N
        self.num_channels = num_channels
        self.train_filters = train_filters
        self.share_rotations = share_rotations
        self.share_phases = share_phases
        self.share_channels = share_channels
        self.delta_j = delta_j if delta_j is not None else J
        self.delta_l = delta_l if delta_l is not None else L
        self.shift_mode = shift_mode
        self.mask_union = mask_union
        self.mask_angles = mask_angles
        if len(filters['hatpsi'].shape) != 6:
            raise ValueError("filters['hatpsi'] must have shape [nc, J, L, A, M, N], don't forget to add phase shifts before initializing")
        
        self.wave_conv = WaveConvLayer(
            J=J,
            L=L,
            A=A,
            M=M,
            N=N,
            num_channels=num_channels,
            filters=filters['hatpsi'],
            train_filters=train_filters,
            share_rotations=share_rotations,
            share_phases=share_phases,
            share_channels=share_channels,
        )
        self.relu_center = ReluCenterLayer(J=J, M=M, N=N, normalize=normalize_relu)
        self.corr = CorrLayer(
            J=J,
            L=L,
            A=A,
            A_prime=A_prime,
            M=M,
            N=N,
            num_channels=num_channels,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            shift_mode=shift_mode,
            mask_union=mask_union,
            mask_angles=mask_angles,
        )
        self.lowpass = LowpassLayer(
            J=J,
            M=M,
            N=N,
            num_channels=num_channels,
            hatphi=filters["hatphi"],
            mask_angles=mask_angles,
            mask_union=mask_union,
        )
        self.highpass = HighpassLayer(
            J=J,
            M=M,
            N=N,
            wavelets=wavelets,
            num_channels=num_channels,
            mask_angles=mask_angles,
            mask_union=mask_union,
            mask_union_highpass=mask_union_highpass,
        )

    def forward(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        nb = x.shape[0]
        xpsi = self.wave_conv(x)
        xrelu = self.relu_center(xpsi)
        xcorr = self.corr(xrelu.view(nb, self.num_channels * self.J * self.L * self.A, self.M, self.N), flatten=flatten)
        hatx_c = fft2(torch.complex(x, torch.zeros_like(x)))
        xlow = self.lowpass(hatx_c)
        xhigh = self.highpass(hatx_c)
        return xcorr, xlow, xhigh
