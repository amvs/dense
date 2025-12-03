import torch
import torch.fft as fft
from torch import nn
from itertools import product
from typing import Optional, Literal
from .utils import create_masks_shift


class BaseCorrLayer(nn.Module):
    def __init__(
        self,
        J: int,
        L: int,
        A: int,
        A_prime: int,
        M: int,
        N: int,
        num_channels: int = 1,
        delta_j: Optional[int] = None,
        delta_l: Optional[int] = None,
        shift_mode: Literal["samec", "all", "strict"] = "samec",
        mask_angles: int = 4,
    ):
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
        self.delta_j = delta_j if delta_j is not None else J  # default to all scales
        self.delta_l = delta_l if delta_l is not None else L  # default to all rotations
        self.shift_mode = shift_mode
        self.mask_angles = mask_angles

        masks_shift, factr_shift = create_masks_shift(
            J=self.J,
            M=self.M,
            N=self.N,
            mask_union=self.uses_mask_union(),
            mask_angles=self.mask_angles,
        )
        self.register_buffer("masks_shift", masks_shift)
        self.factr_shift = factr_shift

        # precompute union mask and per-mask index mappings
        self.union_of_masks = (
            self.masks_shift.sum(dim=0) > 0
        ).flatten()  # (M*N,) bool - positions where any mask is nonzero
        self.n_union = int(self.union_of_masks.sum().item())  # total union positions

        # for each shift mask, map union positions to actual mask positions
        # mask_to_union[shift_idx] gives indices in union that correspond to this mask
        self.mask_to_union = {}
        for shift_idx in range(len(self.masks_shift)):
            mask_flat = self.masks_shift[shift_idx].flatten().bool()  # (M*N,)
            union_to_full = torch.where(self.union_of_masks)[
                0
            ]  # union idx -> full M*N idx
            mask_positions_in_full = torch.where(mask_flat)[
                0
            ]  # mask positions in full M*N
            # find which union indices map to this mask's positions
            mask_in_union = torch.tensor(
                [
                    i
                    for i, pos in enumerate(union_to_full)
                    if pos in mask_positions_in_full
                ]
            )
            self.mask_to_union[shift_idx] = mask_in_union

        # precompute index mapping for filter pairs
        self.idx_wph = self.compute_idx()

    def uses_mask_union(self):
        """Override in child classes if mask_union behavior differs."""
        return False

    def to_shift_color(self, c1, c2, j1, j2, l1, l2):
        if self.shift_mode == "all":
            return True
        elif self.shift_mode == "samec":
            return c1 == c2
        elif self.shift_mode == "strict":
            return (c1 == c2) and (j1 == j2) and (l1 == l2)
        else:
            return False

    def compute_idx(self):
        """Override in child classes to define index computation."""
        raise NotImplementedError("compute_idx must be implemented in child classes")

    def forward(self, xpsi, flatten: bool = True, vmap_chunk_size: Optional[int] = None):
        return self.compute_correlations(xpsi, flatten=flatten, vmap_chunk_size=vmap_chunk_size)

    def compute_correlations(self, xpsi, flatten: bool = True, vmap_chunk_size: Optional[int] = None):
        """Override in child classes to define correlation computation."""
        raise NotImplementedError("compute_correlations must be implemented in child classes")


class CorrLayer(BaseCorrLayer):
    def uses_mask_union(self):
        return True

    def compute_idx(self):
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

        for c1 in range(self.num_channels):  # channels - signal 1
            for c2 in range(self.num_channels):  # channels - signal 2
                for j1 in range(J):  # 0 to max scale - scale, signal 1
                    for j2 in range(
                        j1, min(j1 + 1 + dj, J)
                    ):  # previous scale to scale + delta_j OR max scale (so we don't get too large of a scale difference)
                        for l1 in range(
                            L
                        ):  # from 0 to max # of rotations - scale, signal 1
                            for l2 in range(
                                max(0, l1 + 1 - dl), min(L, l1 + 1 + dl)
                            ):  # constrained by delta_l
                                for a1 in range(A):  # phase shifts, signal 1
                                    for a2 in range(A_prime):  # phase shifts, signal 2
                                        if self.to_shift_color(c1, c2, j1, j2, l1, l2):
                                            idx_la1.append(
                                                A * L * J * c1
                                                + A * L * j1
                                                + A * l1
                                                + a1
                                            )
                                            idx_la2.append(
                                                A * L * J * c2
                                                + A * L * j2
                                                + A * l2
                                                + a2
                                            )
                                            if self.mask_union:
                                                idx = J  # if we look at union of spatial shift masks, we always want last mask
                                            else:
                                                idx = (
                                                    j2 + 1
                                                )  # else, we take mask corresponding to scale j2
                                            shifted.append(idx)
                                            nb_moments += int(self.factr_shift[idx])
                                        else:  # if spatial shift conditions not satisfied, only keep self-correlation
                                            idx_la1.append(
                                                A * L * J * c1
                                                + A * L * j1
                                                + A * l1
                                                + a1
                                            )
                                            idx_la2.append(
                                                A * L * J * c2
                                                + A * L * j2
                                                + A * l2
                                                + a2
                                            )
                                            shifted.append(0)
                                            nb_moments += 1
                                        params_la1.append(
                                            {"j": j1, "l": l1, "a": a1, "c": c1}
                                        )
                                        params_la2.append(
                                            {"j": j2, "l": l2, "a": a2, "c": c2}
                                        )
        print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).type(torch.long)
        idx_wph["la2"] = torch.tensor(idx_la2).type(torch.long)
        idx_wph["shifted"] = torch.tensor(shifted).type(torch.long)
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        return idx_wph

    # per-pair function: compute correlation and apply mask
    def _pair_corr(
        self, hatx_shared, masks_shared, i_idx, j_idx, shift_idx, flatten: bool = True
    ):
        idx1 = i_idx.unsqueeze(0).to(torch.long)
        idx2 = j_idx.unsqueeze(0).to(torch.long)
        shift_idx_1d = shift_idx.unsqueeze(0).to(torch.long)

        hat1 = hatx_shared.index_select(1, idx1).squeeze(1)  # (nb, M, N)
        hat2 = hatx_shared.index_select(1, idx2).squeeze(1)
        prod = hat1 * torch.conj(hat2)
        corr = fft.ifft2(prod).real  # (nb, M, N)

        # apply mask (zeros outside mask)
        mask = masks_shared.index_select(0, shift_idx_1d).squeeze(0)  # (M, N)
        corr_masked = corr * mask  # (nb, M, N)
        mask_downsample = masks_shared.sum(dim=0).bool()
        if flatten:
            return corr_masked[:, mask_downsample]  # (nb, n_masked)
        else:
            return corr_masked  # (nb, M, N)

    def compute_correlations(
        self, xpsi, flatten: bool = True, vmap_chunk_size: Optional[int] = None
    ):
        """
        Compute cross-correlations using FFT with vmapped per-pair computation.

        Args:
            xpsi: real tensor (nb, C, M, N) where C = num_channels * J * L * A
            flatten: if True, extract only masked values; if False, return full spatial grid
            vmap_chunk_size: optional int for vmap memory control

        Returns:
            (nb, n_corrs) with only masked correlation values if flatten=True
            (nb, n_pairs, M, N) with zeros outside masks if flatten=False
        """
        la1 = self.idx_wph["la1"].to(xpsi.device)
        la2 = self.idx_wph["la2"].to(xpsi.device)
        shifted = self.idx_wph["shifted"].to(xpsi.device)

        x_c = torch.complex(xpsi, torch.zeros_like(xpsi))
        hatx = fft.fft2(x_c)  # (nb, C, M, N)

        n_pairs = la1.shape[0]

        vmapped = torch.vmap(
            self._pair_corr,
            in_dims=(None, None, 0, 0, 0, None),
            out_dims=0,
            chunk_size=vmap_chunk_size,
        )
        out = vmapped(
            hatx, self.masks_shift, la1, la2, shifted, flatten
        )  # (n_pairs, nb, n_union) if flatten else (n_pairs, nb, M, N)
        if flatten:
            out = out.permute(1, 0, 2)  # (nb, n_pairs, n_union) or (nb, n_pairs, M, N)
        else:
            out = out.permute(1, 0, 2, 3)  # (nb, n_pairs, M, N)

        if flatten:
            # extract values for each pair's specific mask from union
            out_list = []
            for p in range(n_pairs):
                shift_idx = shifted[p].item()
                mask_indices = self.mask_to_union[
                    shift_idx
                ]  # positions in union for this mask
                out_list.append(out[:, p, mask_indices])  # (nb, n_masked_p)
            return torch.cat(out_list, dim=1)  # (nb, total_masked)
        else:
            return out  # (nb, n_pairs, M, N)


class CorrLayerDownsample(BaseCorrLayer):
    def compute_idx(self):
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

        for c1 in range(self.num_channels):  # channels - signal 1
            for c2 in range(self.num_channels):  # channels - signal 2
                for j1 in range(J):  # 0 to max scale - scale, signal 1
                    for j2 in range(
                        j1, min(j1 + 1 + dj, J)
                    ):  # previous scale to scale + delta_j OR max scale (so we don't get too large of a scale difference)
                        for l1 in range(
                            L
                        ):  # from 0 to max # of rotations - scale, signal 1
                            for l2 in range(
                                max(0, l1 + 1 - dl), min(L, l1 + 1 + dl)
                            ):  # constrained by delta_l
                                for a1 in range(A):  # phase shifts, signal 1
                                    for a2 in range(A_prime):  # phase shifts, signal 2
                                        if self.to_shift_color(c1, c2, j1, j2, l1, l2):
                                            idx_la1.append((j1, 
                                                A * L * c1
                                                + A * l1
                                                + a1
                                            ))
                                            idx_la2.append((j2,
                                                A * L * c2
                                                + A * l2
                                                + a2
                                            ))
                                            idx = (
                                                j2 + 1
                                            )  # we take mask corresponding to scale j2, which should always be bigger than j1
                                            shifted.append(idx)
                                            nb_moments += int(self.factr_shift[idx])
                                        else:  # if spatial shift conditions not satisfied, only keep self-correlation
                                            idx_la1.append((j1,
                                                A * L * c1
                                                + A * l1
                                                + a1
                                            ))
                                            idx_la2.append((j2,
                                                A * L * c2
                                                + A * l2
                                                + a2
                                            ))
                                            shifted.append(0)
                                            nb_moments += 1
                                        params_la1.append(
                                            {"j": j1, "l": l1, "a": a1, "c": c1}
                                        )
                                        params_la2.append(
                                            {"j": j2, "l": l2, "a": a2, "c": c2}
                                        )
        print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).type(torch.long)
        idx_wph["la2"] = torch.tensor(idx_la2).type(torch.long)
        idx_wph["shifted"] = torch.tensor(shifted).type(torch.long)
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        return idx_wph

    def compute_correlations(self, xpsi: list[torch.Tensor], flatten: bool = True, vmap_chunk_size: Optional[int] = None):
        """
        Match same scales, and pass to vmap together
        Do upsampling here before passing to _pair_corr in vmap
        """
        la1 = self.idx_wph["la1"].to(xpsi.device)
        la2 = self.idx_wph["la2"].to(xpsi.device)
        shifted = self.idx_wph["shifted"].to(xpsi.device)

        for (j1, j2) in product(range(self.J), range(self.J)):
            x1 = xpsi[j1].flatten(start_dim=1, end_dim=-3)
            x2 = xpsi[j2].flatten(start_dim=1, end_dim=-3)
            x1, x2 = self.upsample_correlations(x1, x2)
            x1 = torch.complex(x1, torch.zeros_like(x1))
            x2 = torch.complex(x2, torch.zeros_like(x2))
            hatx1 = fft.fft2(x1)  # (nb, C, M, N)
            hatx2 = fft.fft2(x2)  # (nb, C, M, N)
            mask_j1j2 = (la1[:, 0] == j1) & (la2[:, 0] == j2)
            j1j2_idxs = mask_j1j2.nonzero(as_tuple=True)[0]
            # if no pairs match these scales, skip
            if j1j2_idxs.numel() == 0:
                continue
            # la1/la2 store (j, idx) pairs; extract the actual index (second column)
            la1_j1 = la1[j1j2_idxs][:, 1]
            la2_j2 = la2[j1j2_idxs][:, 1]
            shifted_j1j2 = shifted[j1j2_idxs]
            vmapped = torch.vmap(self._pair_corr,
                                 in_dims = (None, None, None, 0,0,0,None),
                                 out_dims=0,
                                 chunk_size=vmap_chunk_size)
            out_j1j2 = vmapped(hatx1, hatx2, self.masks_shift, la1_j1, la2_j2, shifted_j1j2, flatten)
            # process out_j1j2 into final output


    def _pair_corr(self, hatx1_shared, hatx2_shared, masks_shared, i_idx, j_idx, shift_idx, flatten = True):
        """

        """
        idx1 = i_idx.unsqueeze(0).to(torch.long)
        idx2 = j_idx.unsqueeze(0).to(torch.long)
        shift_idx_1d = shift_idx.unsqueeze(0).to(torch.long)

        hatx1 = hatx1_shared.index_select(1, idx1).squeeze(1)  # (nb, M, N)
        hatx2 = hatx2_shared.index_select(1, idx2).squeeze(1)
        prod = hatx1 * torch.conj(hatx2)
        corr = fft.ifft2(prod).real  # (nb, M, N)

        # apply mask (zeros outside mask)
        mask = masks_shared.index_select(0, shift_idx_1d).squeeze(0)  # (M, N)
        corr_masked = corr * mask  # (nb, M, N)
        mask_downsample = masks_shared.sum(dim=0).bool()
        if flatten:
            return corr_masked[:, mask_downsample]  # (nb, n_masked)
        else:
            return corr_masked  # (nb, M, N)


    def upsample_correlations(self, x1: torch.Tensor, x2: torch.Tensor):
        if x1.shape == x2.shape:
            return x1, x2
        else:
            m1, n1 = x1.shape[-2:]
            m2, n2 = x2.shape[-2:]
            assert m1 > m2 and n1 > n2, "x1 must be larger spatially than x2 to upsample x2"
            x2 = nn.UpsamplingBilinear2d(size=(m1, n1))(x2)
            return x1, x2