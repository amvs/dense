import torch
import torch.fft as fft
from torch import nn
from typing import Optional, Literal
from wph.ops.backend import masks_subsample_shift

class CorrLayer(nn.Module):
    def __init__(self,
                 J: int,
                 L:int,
                 A:int,
                 A_prime:int,
                 M: int,
                 N: int,
                 num_channels: int = 1,
                 delta_j: Optional[int] = None,
                 delta_l: Optional[int] = None,
                 shift_mode: Literal["samec", 'all', 'strict'] = "samec",
                 mask_union: bool = False,
                 mask_angles: int = 4):
        """
        Initializes the Correlation layer.
        J: number of scales
        M, N: spatial dimensions of input signals
        """
        super().__init__()
        self.J = J
        self.M = M
        self.N = N
        self.L = L
        self.A = A
        self.A_prime = A_prime
        self.num_channels = num_channels
        self.delta_j = delta_j
        self.delta_l = delta_l
        self.shift_mode = shift_mode
        self.mask_union = mask_union
        self.mask_angles = mask_angles
        # precompute index mapping for filter pairs
        self.idx_wph = self.compute_idx()

        masks_shift = masks_subsample_shift(self.J, self.M, self.N, mask_union = self.mask_union, alpha = self.mask_angles)
        masks_shift = torch.cat((torch.zeros(1, self.M, self.N), masks_shift), dim=0)
        masks_shift[0, 0, 0] = 1.0

        self.register_buffer("masks_shift", masks_shift.clone().detach())
        self.factr_shift = self.masks_shift.sum(dim=(-2, -1))

        

    def to_shift_color(self, c1, c2, j1, j2, l1, l2):
        if self.shift_mode == "all":
            return True
        elif self.shift_mode == "samec":
            return c1 == c2
        elif self.shift_mode == "strict":
            return (c1 == c2) and (j1 == j2) and (l1 == l2)


    def forward(self):
        # Placeholder for computing indices
        L = self.L
        J = self.J
        A = self.A
        A_prime = self.A_prime
        dj = self.delta_j

        idx_la1 = []
        idx_la2 = []
        shifted = []
        params_la1 = []
        params_la2 = []
        nb_moments = 0

        for c1 in range(
            self.num_channels
        ):  # channels - signal 1
            for c2 in range(self.num_channels):  # channels - signal 2
                for j1 in range(J):  # 0 to max scale - scale, signal 1
                    for j2 in range(
                        j1, min(j1 + 1 + dj, J)
                    ):  # previous scale to scale + delta_j OR max scale (so we don't get too large of a scale difference)
                        for l1 in range(L):  # from 0 to max # of rotations - scale, signal 1
                            for l2 in range(L):  # same as l1, scale, signal 2
                                for a1 in range(A):  # phase shifts, signal 1
                                    for a2 in range(A_prime):  # phase shifts, signal 2
                                        if self.to_shift_color(c1, c2, j1, j2, l1, l2):
                                            idx_la1.append(
                                                A * L * J * c1 + A * L * j1 + A * l1 + a1
                                            )
                                            idx_la2.append(
                                                A * L * J * c2 + A * L * j2 + A * l2 + a2
                                            )
                                            if self.mask_union:
                                                idx = J # if we look at union of spatial shift masks, we always want last mask
                                            else:
                                                idx = j2+1 # else, we take mask corresponding to scale j2
                                            shifted.append(idx)
                                            nb_moments += int(self.factr_shift[idx])
                                        else: # if spatial shift conditions not satisfied, only keep self-correlation
                                            idx_la1.append(
                                                A * L * J * c1 + A * L * j1 + A * l1 + a1
                                            )
                                            idx_la2.append(
                                                A * L * J * c2 + A * L * j2 + A * l2 + a2  
                                            )
                                            shifted.append(0)
                                            nb_moments += 1
                                        params_la1.append({'j': j1, 'l': l1, 'a': a1, 'c': c1})
                                        params_la2.append({'j': j2, 'l': l2, 'a': a2, 'c': c2})
        print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).type(torch.long)
        idx_wph["la2"] = torch.tensor(idx_la2).type(torch.long)
        idx_wph["shifted"] = torch.tensor(shifted).type(torch.long)
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        return idx_wph

    def forward(self, xpsi, vmap_chunk_size: Optional[int] = None):
        """
        Compute cross-correlations using FFT and a vmapped per-pair multiply/ifft.

        Args:
            xpsi: real-valued tensor shaped (nb, C, M, N) where C = num_channels * J * L * A
            this_wph: optional dict with keys 'la1','la2' to restrict pairs; if None uses full idx_wph
            vmap_chunk_size: optional int passed to torch.vmap to limit internal batching (controls memory)

        Returns:
            Tensor of shape (nb, n_pairs, M, N) with real correlations.
        """
        la1 = self.idx_wph["la1"].to(xpsi.device)
        la2 = self.idx_wph["la2"].to(xpsi.device)

        # convert to complex and FFT along spatial dims
        x_c = torch.complex(xpsi, torch.zeros_like(xpsi))
        hatx = fft.fft2(x_c)  # (nb, C, M, N)

        # per-pair function: select two channels and compute ifft( hat1 * conj(hat2) )
        def _pair_corr(hatx_shared, i_idx, j_idx):
            # hatx_shared: (nb, C, M, N) passed as None (broadcasted)
            # i_idx, j_idx: 0-d LongTensors indexing channel dimension.
            # Avoid converting to Python ints inside vmap (no .item()/int()).
            # Use index_select with a 1D LongTensor to pick the channel dimension.
            idx1 = i_idx.unsqueeze(0).to(torch.long)
            idx2 = j_idx.unsqueeze(0).to(torch.long)
            hat1 = hatx_shared.index_select(1, idx1).squeeze(1)  # (nb, M, N)
            hat2 = hatx_shared.index_select(1, idx2).squeeze(1)
            prod = hat1 * torch.conj(hat2)
            corr = fft.ifft2(prod).real
            return corr

        # vmapped over pairs: in_dims (None, 0, 0) means hatx is shared, la1/la2 batched
        vmapped = torch.vmap(_pair_corr, in_dims=(None, 0, 0), out_dims=0, chunk_size=vmap_chunk_size)
        out = vmapped(hatx, la1, la2)  # shape (n_pairs, nb, M, N)
        out = out.permute(1, 0, 2, 3).contiguous()  # (nb, n_pairs, M, N)
        return out