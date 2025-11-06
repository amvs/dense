"""
Author: Anneke von Seeger
Adapted from https://github.com/abrochar/wavelet-texture-synthesis/
to use PyTorch and to combine color/grayscale versions
"""

import torch
import torch.fft as fft
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import math
import sys
from typing import Optional, Literal

from wph.ops.backend import (
    SubInitSpatialMean,
    DivInitStd,
    padc,
    masks_subsample_shift,
    maskns,
)


class ALPHATorch(torch.nn.Module):
    def __init__(
        self,
        M=256,
        N=256,
        J=5,
        L=4,
        A=4,
        A_prime=1,
        delta_j=4,
        delta_l=4,
        nb_chunks=1,
        chunk_id=0,
        shift="all",
        wavelets="morlet",
        filters=None,
        num_channels=1,
        mask_union:bool = True,
        mask_pool:bool=False,
        mask_union_highpass=True,
        mask_select='j2',
        mask_angles =4,
        flatten=True
    ):
        """
        valid choices for shift and wavelets depend on whether working
        with grayscale or color images
        for grayscale images, shift should be "all" or "same"
            and wavelets can be "morlet" or "steer"
        for color images, shift can be "all" or "samec"
            and "morlet" is only valid choice for wavelets
        """
        super().__init__()
        self.M, self.N, self.J, self.L = M, N, J, L
        self.nb_chunks = nb_chunks  # number of chunks to cut whp cov
        self.chunk_id = chunk_id
        self.A = A
        self.A_prime = A_prime
        self.delta_j = delta_j
        self.delta_l = delta_l
        self.wavelets = wavelets
        self.shift = shift
        self.flatten = flatten
        assert 0 <= self.chunk_id < self.nb_chunks, "Invalid chunk_id"
        self.num_channels = num_channels
        self.mask_union = mask_union
        self.mask_pool = mask_pool
        self.mask_union_highpass = mask_union_highpass
        self.mask_select = mask_select
        self.mask_angles = mask_angles
        self.build(filters=filters)

    def build(self, filters):
        masks_shift = masks_subsample_shift(self.J, self.M, self.N, mask_union = self.mask_union, alpha = self.mask_angles)# , alpha = self.A)
        masks_shift = torch.cat((torch.zeros(1, self.M, self.N), masks_shift), dim=0)
        masks_shift[0, 0, 0] = 1.0

        self.register_buffer("masks_shift", masks_shift.clone().detach())
        # self.masks_shift = nn.Parameter(masks_shift, requires_grad=False)

        self.factr_shift = self.masks_shift.sum(dim=(-2, -1))

        self.filters_tensor(filters=filters)

        if self.num_channels == 1:
            self.idx_wph = self.compute_idx_grayscale()
        else:
            self.idx_wph = self.compute_index_color()

        this_wph = self.get_this_chunk(self.nb_chunks, self.chunk_id)
        # self.this_wph_la1 = nn.Parameter(this_wph["la1"], requires_grad=False)
        # self.this_wph_la2 = nn.Parameter(this_wph["la2"], requires_grad=False)
        # self.this_wph_shifted = nn.Parameter(this_wph["shifted"], requires_grad=False)
        self.register_buffer("this_wph_la1", this_wph["la1"].clone().detach())
        self.register_buffer("this_wph_la2", this_wph["la2"].clone().detach())
        self.register_buffer("this_wph_shifted", this_wph["shifted"].clone().detach())
        

        # define variables to store mean and std of observation
        self.subinitmean1 = SubInitSpatialMean()
        self.subinitmean2 = SubInitSpatialMean()
        self.divinitstd1 = DivInitStd()
        self.divinitstd2 = DivInitStd()

        self.divinitstdJ = DivInitStd()
        self.subinitmeanJ = SubInitSpatialMean()

        if self.num_channels == 1:
            if self.wavelets == "morlet":
                self.divinitstdH = [None, None, None]
                for hid in range(3):
                    self.divinitstdH[hid] = DivInitStd()
            if self.wavelets == "steer":
                self.subinitmean0 = SubInitSpatialMean()
                self.divinitstd0 = DivInitStd()
        else:
            if self.wavelets == "morlet":
                self.divinitstdH = [None] * 3 * self.num_channels
                for hid in range(3 * self.num_channels):
                    self.divinitstdH[hid] = DivInitStd()

    def filters_tensor(self, filters):
        J = self.J
        M = self.M
        N = self.N
        L = self.L

        if filters == None:
            if self.wavelets == "morlet":
                hatpsi = torch.load(
                    "./filters/morlet_N"
                    + str(N)
                    + "_J"
                    + str(J)
                    + "_L"
                    + str(L)
                    + ".pt"
                )  # (J,L,M,N)
                hatphi = torch.load(
                    "./filters/morlet_lp_N"
                    + str(N)
                    + "_J"
                    + str(J)
                    + "_L"
                    + str(L)
                    + ".pt"
                )  # (M,N)
        else:
            hatpsi = filters["hatpsi"]
            hatphi = filters["hatphi"]
        A = self.A
        A_prime = self.A_prime

        alphas_ = torch.arange(A, dtype=torch.float) / A * 2 * math.pi
        alphas = torch.complex(torch.cos(alphas_), torch.sin(alphas_))

        filt = torch.zeros(J, L, A, M, N, dtype=torch.cfloat)
        for alpha in range(A):
            for j in range(J):
                for theta in range(L):
                    psi_signal = hatpsi[j, theta, ...]
                    filt[j, theta, alpha, :, :] = alphas[alpha] * psi_signal

        self.register_buffer("hatphi", hatphi)
        self.register_buffer("hatpsi", filt)
        # self.hatphi = nn.Parameter(hatphi, requires_grad=False)
        # self.hatpsi = nn.Parameter(filt, requires_grad=False)

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

        # load masks for aperiodicity
        masks = self.maskns(J, M, N)
        if self.num_channels == 1:
            masks = masks.unsqueeze(1).unsqueeze(1)  # (J, M, N)
        else:
            masks = masks.view(1, J, 1, 1, M, N)

        self.register_buffer("masks", masks)
        # self.hathaar2d = nn.Parameter(hathaar2d, requires_grad=False)
        # self.masks = nn.Parameter(masks, requires_grad=False)

    def get_this_chunk(self, nb_chunks, chunk_id):
        # cut self.idx_wph into smaller pieces
        nb_cov = len(self.idx_wph["la1"])
        max_chunk = nb_cov // nb_chunks
        nb_cov_chunk = np.zeros(nb_chunks, dtype=np.int32)
        for idxc in range(nb_chunks):
            if idxc < nb_chunks - 1:
                nb_cov_chunk[idxc] = int(max_chunk)
            else:
                nb_cov_chunk[idxc] = int(nb_cov - max_chunk * (nb_chunks - 1))
                assert nb_cov_chunk[idxc] > 0

        this_wph = dict()
        offset = int(0)
        for idxc in range(nb_chunks):
            if idxc == chunk_id:
                this_wph["la1"] = self.idx_wph["la1"][
                    offset : offset + nb_cov_chunk[idxc]
                ].clone().detach()
                this_wph["la2"] = self.idx_wph["la2"][
                    offset : offset + nb_cov_chunk[idxc]
                ].clone().detach()
                this_wph["shifted"] = self.idx_wph["shifted"][
                    offset : offset + nb_cov_chunk[idxc]
                ].clone().detach()
            offset = offset + nb_cov_chunk[idxc]
        return this_wph

    def to_shift(self, j1, j2, l1, l2):
        if self.shift == "all":
            return True
        elif self.shift == "same":
            return (j1 == j2) and (l1 == l2)

    def compute_idx_grayscale(self):
        L = self.L
        J = self.J
        A = self.A
        A_prime = self.A_prime
        dj = self.delta_j
        dl = self.delta_l

        idx_la1 = []
        idx_la2 = []
        params_la1 = []
        params_la2 = []
        shifted = []
        nb_moments = 0

        for j1 in range(J):
            for j2 in range(j1, min(J, j1 + 1 + dj)):
                for l1 in range(L):
                    for l2 in range(max(0, l1 + 1 - dl), min(L, l1 + 1 + dl)):
                        for alpha1 in range(A):
                            for alpha2 in range(A_prime):
                                if self.to_shift(j1, j2, l1, l2):  # coeffs whith shifts
                                    idx_la1.append(A * L * j1 + A * l1 + alpha1)
                                    idx_la2.append(A * L * j2 + A * l2 + alpha2)
                                    if self.mask_union:
                                        idx = J
                                    elif self.mask_select == 'j1':
                                        idx = j1+1
                                    elif self.mask_select == 'j2':
                                        idx = j2+1
                                    shifted.append(idx)
                                    if self.mask_pool:
                                        nb_moments += 1
                                    else:
                                        nb_moments += int(self.factr_shift[idx])
                                else:
                                    idx_la1.append(A * L * j1 + A * l1 + alpha1)
                                    idx_la2.append(
                                        A * L * j2 + A * l2 + alpha2
                                    )
                                    shifted.append(0)
                                    nb_moments += 1
                                params_la1.append({'j': j1, 'l': l1, 'a': alpha1})
                                params_la2.append({'j': j2, 'l': l2, 'a': alpha2})

        if self.chunk_id == 0:
            print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).type(torch.long)
        idx_wph["la2"] = torch.tensor(idx_la2).type(torch.long)
        idx_wph["shifted"] = torch.tensor(shifted).type(torch.long)
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        return idx_wph

    def to_shift_color(self, c1, c2, j1, j2, l1, l2):
        if self.shift == "all":
            return True
        elif self.shift == "samec":
            return c1 == c2
        elif self.shift == "strict":
            return (c1 == c2) and (j1 == j2) and (l1 == l2)

    def compute_index_color(self):
        L = self.L
        J = self.J
        A = self.A
        A_prime = self.A_prime
        dj = self.delta_j
        dl = self.delta_l

        idx_la1 = []
        idx_la2 = []
        shifted = []
        params_la1 = []
        params_la2 = []
        nb_moments = 0

        for c1 in range(
            self.num_channels
        ):  # channels - confirmed by comparing with alpha_gray, which skips the outmost two loops
            for c2 in range(self.num_channels):  # channels
                for j1 in range(J):  # 0 to max scale
                    for j2 in range(
                        j1, min(j1 + 1 + dj, J)
                    ):  # previous scale to scale + delta_j OR max scale (so we don't get too large of a scale difference)
                        for l1 in range(L):  # from 0 to max # of rotations
                            for l2 in range(max(0, l1 + 1 - dl), min(L, l1 + 1 + dl)):  # same as l1
                                for a1 in range(A):  # from 0 to max number of phase shifts
                                    if self.to_shift_color(c1, c2, j1, j2, l1, l2):
                                        idx_la1.append(
                                            A * L * J * c1 + A * L * j1 + A * l1 + a1
                                        )
                                        idx_la2.append(
                                            A * L * J * c2 + A * L * j2 + A * l2
                                        )
                                        
                                        if self.mask_union:
                                            idx = J
                                        elif self.mask_select == 'j1':
                                            idx = j1+1
                                        elif self.mask_select == 'j2':
                                            idx = j2+1
                                        shifted.append(idx)
                                        if self.mask_pool:
                                            nb_moments += 1
                                        else:
                                            nb_moments += int(self.factr_shift[idx])
                                    else:
                                        idx_la1.append(
                                            A * L * J * c1 + A * L * j1 + A * l1 + a1
                                        )
                                        idx_la2.append(
                                            A * L * J * c2 + A * L * j2 + A * l2
                                        )
                                        shifted.append(0)
                                        nb_moments += 1
                                    params_la1.append({'j': j1, 'l': l1, 'a': a1, 'c': c1})
                                    params_la2.append({'j': j2, 'l': l2, 'a': 0, 'c': c2})
        if self.chunk_id == 0:
            print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).type(torch.long)
        idx_wph["la2"] = torch.tensor(idx_la2).type(torch.long)
        idx_wph["shifted"] = torch.tensor(shifted).type(torch.long)
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        return idx_wph

    def compute_wavelet_transform(self, hatx_c):
        nb = hatx_c.shape[0]
        # hatx_bc = hatx_c # [:, 0]  # (nb, M, N) - assuming grayscale
        hatpsi_la = self.hatpsi.unsqueeze(0).expand(
            nb, self.num_channels, -1, -1, -1, -1, -1
        )
        hatxpsi_bc = hatpsi_la * hatx_c.view(
            nb, self.num_channels, 1, 1, 1, self.M, self.N
        )  # (nb, nc, J,L,A,M,N)
        xpsi_bc = fft.ifft2(hatxpsi_bc)  # (nb, nc, J,L,A,M,N)
        return xpsi_bc

    def normalize_and_mask(self, xpsi_bc):
        nb = xpsi_bc.shape[0]
        xpsi_bc = torch.real(xpsi_bc).relu()  # (nb, J,L,A,M,N)
        # masks for non periodic images
        xpsi_bc = xpsi_bc.mul(
            self.masks.expand(nb, self.num_channels, -1, -1, -1, -1, -1)
        )  # (nb, J,L,A,M,N)
        # renorm by observation stats
        xpsi_bc = self.subinitmean1(xpsi_bc)
        xpsi_bc = self.divinitstd1(xpsi_bc)
        return xpsi_bc

    def compute_correlations(self, xpsi_bc, this_wph: Optional[dict]):
        if this_wph is None:
            xpsi_bc_la1 = xpsi_bc[
                :, self.this_wph_la1, ...
            ]  # xpsi_bc.index_select(1, self.this_wph_la1)
            xpsi_bc_la2 = xpsi_bc[:, self.this_wph_la2, ...]
        else:
            xpsi_bc_la1 = xpsi_bc[
                :, this_wph["la1"], ...
            ]  # xpsi_bc.index_select(1, self.this_wph_la1)
            xpsi_bc_la2 = xpsi_bc[:, this_wph["la2"], ...]

        hatconv_xpsi_bc = fft.fft2(xpsi_bc_la1) * torch.conj(fft.fft2(xpsi_bc_la2))
        conv_xpsi_bc = fft.ifft2(hatconv_xpsi_bc).real
        # print(conv_xpsi_bc.shape)
        return conv_xpsi_bc

    def mask_correlations(self, corr_bc, this_wph: Optional[dict], flatten=None):
        if flatten is None:   
            flatten=self.flatten
        nb = corr_bc.shape[0]
        n_coef = corr_bc.shape[1]
        if this_wph is None:
            shift_index = self.this_wph_shifted
        else:
            shift_index = this_wph["shifted"]
        masks_shift = torch.index_select(self.masks_shift, index=shift_index, dim=0)
        if flatten & self.mask_pool:
            corr_bc = corr_bc * masks_shift
            corr_bc = corr_bc.view(nb, -1, self.M * self.N)
            corr_bc = torch.avg_pool1d(corr_bc, kernel_size = corr_bc.shape[-1]).squeeze(-1)
        elif not self.mask_pool:
            masks_shift = masks_shift.bool()
            corr_bc = corr_bc[masks_shift.expand(nb, -1, -1, -1)].view(nb, -1)
        else:
            corr_bc = corr_bc * masks_shift
            corr_bc = corr_bc.view(nb, 1, -1, self.M, self.N)
            if self.mask_pool:
                corr_bc = torch.avg_pool2d(corr_bc, kernel_size=(self.M, self.N))
        return corr_bc

    def compute_lowpass_stats_color(self, hatx_c):
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatxphi_c = hatx_c * self.hatphi.expand(nb, nc, -1, -1)  # (nb,nc,M,N)
        xphi_c = fft.ifft2(hatxphi_c)
        xphi_c.mul_(self.masks[-1, -1, ...].view(1, 1, self.M, self.N))
        xphi0_c = self.subinitmeanJ(xphi_c)
        xphi0_c = self.divinitstdJ(xphi0_c)
        # xphi0_c = xphi0_c.real

        xphi0_c = (
            xphi0_c.abs()
        )  # torch.complex(xphi0_c.abs(), torch.zeros_like(xphi0_c))

        # xphi0_c = fft.fft2(xphi0_c)
        # xphi0_c = fft.ifft2(xphi0_c.mul_(torch.conj(xphi0_c)))
        z = xphi0_c.repeat(1, self.num_channels, 1, 1)

        z_ = torch.repeat_interleave(xphi0_c, self.num_channels, dim=1)
        xphi0_c = fft.ifft2(fft.fft2(z) * torch.conj(fft.fft2(z_)))

        return xphi0_c

    def compute_lowpass_stats_gray(self, hatx_c):
        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        hatxphi_c = hatx_c * self.hatphi.expand(nb, nc, -1, -1)  # (nb,nc,M,N)
        xphi_c = fft.ifft2(hatxphi_c)
        xphi0_c = self.subinitmeanJ(xphi_c)
        xphi0_c = self.divinitstdJ(xphi0_c)
        xphi0_c.mul_(self.masks[-1, -1, ...].view(1, 1, self.M, self.N))

        # xphi0_c = xphi0_c.real

        xphi0_c = (
            xphi0_c.abs()
        )  # torch.complex(xphi0_c.abs(), torch.zeros_like(xphi0_c))

        # xphi0_c = fft.fft2(xphi0_c)
        # xphi0_c = fft.ifft2(xphi0_c.mul_(torch.conj(xphi0_c)))
        z = xphi0_c.repeat(1, self.num_channels, 1, 1)

        z_ = torch.repeat_interleave(xphi0_c, self.num_channels, dim=1)
        xphi0_c = fft.ifft2(fft.fft2(z) * torch.conj(fft.fft2(z_)))

        return xphi0_c

    def compute_highpass_stats_gray(self, hatx_c, flatten  = None):
        if flatten is None:
            flatten=self.flatten
        nb = hatx_c.shape[0]
        if self.wavelets == "morlet":
            out = []
            for hid in range(3):
                hatxpsih_c = hatx_c * self.hathaar2d[hid, :, :].repeat(
                    nb, 1, 1, 1
                )  # (nb,nc,M,N)
                xpsih = fft.ifft2(hatxpsih_c)
                xpsih = self.divinitstdH[hid](xpsih)
                xpsih = torch.abs(xpsih)
                xpsih = xpsih * self.masks[0, ...].view(1, 1, self.M, self.N)
                # shifted correlations in Fourier domain
                xpsih = torch.complex(xpsih, torch.zeros_like(xpsih))
                xpsih = fft.fft2(xpsih)
                xpsih = fft.ifft2(xpsih * torch.conj(xpsih))
                
                if self.mask_union_highpass:
                    mask = (self.masks_shift.sum(dim=0) > 0).to(dtype = torch.int32)
                else:
                    mask = self.masks_shift[-1,...]            
                if flatten:
                    out.append(self.select_shifts(xpsih.real, mask=mask).view(nb, -1))
                else:
                    
                    xpsih = torch.real(xpsih) * mask.expand(
                        nb, -1, -1
                    )
                    out.append(xpsih)

            if flatten:
                xpsih = torch.cat(out, dim=1)
            else:
                xpsih = torch.cat(out, dim=0)
        # psi0 for steer
        if self.wavelets == "steer":
            hatxpsih = hatx_c * self.hatpsi0.repeat(nb, 1, 1, 1)  # (nb,1,M,N)
            xpsih = fft.ifft2(hatxpsih)
            xpsih = self.subinitmean0(xpsih)
            xpsih = self.divinitstd0(xpsih)
            xpsih = xpsih * self.masks[0, ...].repeat(nb, 1, 1, 1)
            xpsih = torch.view_as_complex(padc(xpsih.abs()))
            xpsih = fft.fft2(xpsih)
            xpsih = fft.ifft2(xpsih * torch.conj(xpsih))

            # if flatten:
            #     xpsi0 = torch.real(xpsi0)[self.masks_shift[-1,...].repeat(nb,1,1,1).nonzero(as_tuple=True)]
            #     Sout = torch.cat([Sout, xpsi0.view(nb, -1)], dim=1)
            # else:
            #     xpsi0 = torch.real(xpsi0[0,...]) * self.masks_shift[-1,...]
            #     Sout = torch.cat((Sout, xpsi0))
            #     # Sout[0, 0, -2,...] = xpsi0
        return xpsih

    def compute_highpass_stats_color(self, hatx_c, flatten=None):
        if flatten is None:
            flatten=self.flatten
        assert self.wavelets == "morlet"
        nb = hatx_c.shape[0]
        out = []
        for hid1 in range(self.num_channels):
            for hid2 in range(self.num_channels):
                hatpsih_c = hatx_c[:, hid1, ...] * self.hathaar2d[hid2, ...].expand(
                    nb, -1, -1
                )  # (nb, M,N)
                xpsih_c = fft.ifft2(hatpsih_c)
                xpsih_c = self.divinitstdH[3 * hid1 + hid2](xpsih_c)
                xpsih_c = xpsih_c * self.masks[0, 0, ...].squeeze().expand(nb, -1, -1)
                xpsih_c = torch.complex(xpsih_c.abs(), torch.zeros_like(xpsih_c.real))
                xpsih_c = fft.fft2(xpsih_c)
                xpsih_c = fft.ifft2(xpsih_c * torch.conj(xpsih_c))
                xpsih_c = torch.real(xpsih_c) * self.masks_shift[-1].expand(nb, -1, -1)

                if self.mask_union_highpass:
                    mask = (self.masks_shift.sum(dim=0) > 0).to(dtype = torch.int32)
                else:
                    mask = self.masks_shift[-1,...]            
                if flatten:
                    out.append(self.select_shifts(xpsih_c.real, mask=mask))
                else:
                    
                    xpsih_c = torch.real(xpsih_c) * mask.expand(
                        nb, -1, -1
                    )
                    out.append(xpsih_c)

        if flatten:
            xpsih_c = torch.cat(out, dim=1)
        else:
            xpsih_c = torch.cat(out, dim=0)
        return xpsih_c

    def select_shifts(self, signal, mask = None):
        if mask is None:
            mask = self.masks_shift[-1, ...]
        shape = signal.shape
        nb = shape[0]
        signal = signal.reshape(nb, -1)
        mask_flat = mask.expand(shape).reshape(nb, -1).bool()

        return signal[mask_flat].reshape(nb, -1)

    def forward_grayscale(
        self,
        input_tensor,
        flatten=None,
        this_wph: Optional[dict] = None,
        chunk_id: Optional[int] = None,
    ):
        if flatten is None:
            flatten=self.flatten
        x_c = torch.complex(input_tensor, torch.zeros_like(input_tensor))
        hatx_c = fft.fft2(x_c)

        nb = hatx_c.shape[0]
        nc = hatx_c.shape[1]
        assert nc == 1

        xpsi_bc = self.compute_wavelet_transform(hatx_c)
        xpsi_bc = self.normalize_and_mask(xpsi_bc=xpsi_bc)
        corr_bc = self.compute_correlations(
            xpsi_bc=xpsi_bc.view(
                nb, self.num_channels * self.J * self.L * self.A, self.M, self.N
            ),
            this_wph=this_wph,
        )
        Sout = self.mask_correlations(corr_bc, flatten=flatten, this_wph=this_wph)
        if chunk_id is not None:
            current_chunk_id = chunk_id
        else:
            current_chunk_id = self.chunk_id
        if current_chunk_id == self.nb_chunks - 1:
            highpass_stats = self.compute_highpass_stats_gray(hatx_c, flatten=flatten)
            if flatten:
                if self.wavelets == "steer":
                    highpass_stats = self.select_shifts(highpass_stats.real)
                    Sout = torch.cat((Sout, highpass_stats), dim=1)
                elif self.wavelets == "morlet":
                    Sout = torch.cat((Sout, highpass_stats), dim=1)
            else:
                highpass_stats = torch.real(highpass_stats) * self.masks_shift[-1, ...]
                if self.wavelets == "morlet":
                    Sout = torch.cat(
                        (Sout, highpass_stats.view(nb, 1, 3, self.M, self.N)), dim=2
                    )  # Sout[0, 0, -4:-1,...] = highpass_stats.squeeze()
                elif self.wavelets == "steer":
                    # Sout[0, 0, -2,...] = highpass_stats.squeeze()
                    Sout = torch.cat(
                        (Sout, highpass_stats.view(nb, 1, 1, self.M, self.N)), dim=2
                    )
                else:
                    raise ValueError("self.wavelets should be morlet or steer")

            xphi0_c = self.compute_lowpass_stats_gray(hatx_c)
            xphi0_c = xphi0_c.real
            if flatten:
                xphi0_c = self.select_shifts(xphi0_c)
                Sout = torch.cat((Sout, xphi0_c), dim=1)
            else:
                xphi0_c = xphi0_c.mul(self.masks_shift[-1, ...].repeat(nb, 1, 1, 1))
                Sout = torch.cat(
                    (Sout, xphi0_c.view(nb, 1, 1, self.M, self.N)), dim=2
                )  # Sout[0, 0, -1,...] = xphi0_c[0, 0, ...]

        return Sout

    def forward_color(
        self,
        input_tensor,
        flatten=None,
        this_wph: Optional[dict] = None,
        chunk_id: Optional[int] = None,
    ):
        if flatten is None:
            flatten=self.flatten
        x_c = torch.complex(input_tensor, torch.zeros_like(input_tensor))
        hatx_c = fft.fft2(x_c)

        nb, nc = x_c.shape[:2]
        assert nc == self.num_channels

        xpsi_bc = self.compute_wavelet_transform(hatx_c)

        xpsi_bc = self.normalize_and_mask(xpsi_bc=xpsi_bc)

        corr_bc = self.compute_correlations(
            xpsi_bc=xpsi_bc.view(
                nb, self.num_channels * self.J * self.L * self.A, self.M, self.N
            ),
            this_wph=this_wph,
        )
        Sout = self.mask_correlations(corr_bc, flatten=flatten, this_wph=this_wph)
        if chunk_id is not None:
            current_chunk_id = chunk_id
        else:
            current_chunk_id = self.chunk_id
        if current_chunk_id == self.nb_chunks - 1:
            # compute highpass
            # compute lowpass
            assert self.wavelets == "morlet"
            xphi0_c = self.compute_lowpass_stats_color(hatx_c=hatx_c)
            xphi0_c = xphi0_c.real
            if flatten:
                xphi0_c = self.select_shifts(xphi0_c)
                Sout = torch.cat((Sout, xphi0_c), dim=1)
            else:
                xphi0_c = xphi0_c.mul(self.masks_shift[-1, ...].repeat(nb, 1, 1, 1))
                Sout = torch.cat(
                    (Sout, xphi0_c.view(nb, 1, 3 * self.num_channels, self.M, self.N)),
                    dim=2,
                )  # Sout[0, 0, -1,...] = xphi0_c[0, 0, ...]

            highpass_stats = self.compute_highpass_stats_color(
                hatx_c=hatx_c, flatten=flatten
            )
            highpass_stats = highpass_stats.real
            if flatten:
                Sout = torch.cat((Sout, highpass_stats), dim=1)
            else:
                highpass_stats = torch.real(highpass_stats) * self.masks_shift[-1, ...]
                Sout = torch.cat(
                    (
                        Sout,
                        highpass_stats.view(
                            nb, 1, 3 * self.num_channels, self.M, self.N
                        ),
                    ),
                    dim=2,
                )  # Sout[0, 0, -4:-1,...] = highpass_stats.squeeze()

        return Sout

    def forward(
        self,
        input_tensor,
        flatten=None,
        this_wph: Optional[dict] = None,
        chunk_id: Optional[int] = None,
    ):
        if flatten is None:
            flatten=self.flatten
        nb, nc = input_tensor.shape[:2]
        assert nc == self.num_channels, f"Expected {self.num_channels} channels, got {nc}"
        if nc == 1:
            output = self.forward_grayscale(
                input_tensor=input_tensor,
                flatten=flatten,
                this_wph=this_wph,
                chunk_id=chunk_id,
            )
        else:
            output = self.forward_color(
                input_tensor=input_tensor,
                flatten=flatten,
                this_wph=this_wph,
                chunk_id=chunk_id,
            )
        return output

    def __call__(self, input_tensor):
        return self.forward(input_tensor=input_tensor)

    def maskns(self, J, M, N):
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
