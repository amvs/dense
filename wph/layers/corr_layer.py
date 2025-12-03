import torch
import torch.fft as fft
from torch import nn
import warnings
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
    def __init__(self, union_mask: bool = False, *args, **kwargs):
        self.mask_union = union_mask
        super().__init__(*args, **kwargs)
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
        
    def uses_mask_union(self):
        return self.mask_union

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self.masks_shift
        del self.factr_shift

        _masks_temp = []
        for j in range(self.J):
            h_j, w_j = self.M // (2 ** j), self.N // (2 ** j)
            m_shift, factr_shift_j = create_masks_shift(J = 1, M = h_j, N = w_j,
                                            mask_union=self.uses_mask_union(),
                                            mask_angles=self.mask_angles
                                            )
            _masks_temp.append(m_shift)
        for i, m in enumerate(_masks_temp):
            self.register_buffer(f'mask_scale_{i}', m)
            self.register_buffer(f'factr_scale_{i}', factr_shift_j)
            
        ref_masks = self.get_mask_for_scale(0) 
        
        # 1. Compute Union (Boolean)
        union_mask = ref_masks.sum(dim=0).bool() # (M, N)
        
        # 2. Get Linear Indices of the Union pixels
        # These are the pixels vmap will return.
        # We flatten spatial dims to match what happens inside _pair_corr(flatten=True)
        union_indices = torch.nonzero(union_mask.flatten(), as_tuple=True)[0]
        
        # 3. Map each specific mask to its position inside the Union vector
        self.mask_indices_map = {}
        
        for k in range(ref_masks.shape[0]):
            # Get pixels for this specific mask
            k_mask = ref_masks[k].flatten().bool()
            k_indices = torch.nonzero(k_mask, as_tuple=True)[0]
            
            # Find where k_indices appear within union_indices
            # This logic finds the intersection relative to the union
            # Logic: valid_positions = [i for i, u_idx in enumerate(union_indices) if u_idx in k_indices]
            
            # Optimized torch version using isin/searchsorted or simple boolean indexing
            # Since k_mask is a subset of union_mask:
            # We want indices in the COMPRESSED (union) vector that correspond to k.
            
            # Create a map from Full Grid -> Union Vector Index
            # Fill with -1 for invalid pixels
            grid_to_union_map = torch.full((ref_masks[0].numel(),), -1, dtype=torch.long)
            grid_to_union_map[union_indices] = torch.arange(len(union_indices))
            
            # Get the Union indices for the current mask's pixels
            mapped_indices = grid_to_union_map[k_indices]
            
            self.mask_indices_map[k] = mapped_indices

        # Register as buffers so they move to device
        for k, idxs in self.mask_indices_map.items():
            self.register_buffer(f'mask_idx_map_{k}', idxs)

    def get_mask_for_scale(self, j):
        return getattr(self, f'mask_scale_{j}'), getattr(self, f'factr_scale_{j}')
    
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
        pair_metadata = []
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
                                            shifted.append(1)
                                            factr_shift_j2 = self.get_mask_for_scale(j2)[1]
                                            nb_moments += int(factr_shift_j2[1])
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
                                        pair_metadata.append((j1,j2))
        print("number of moments (without low-pass and harr): ", nb_moments)

        idx_wph = dict()
        idx_wph["la1"] = torch.tensor(idx_la1).long()
        idx_wph["la2"] = torch.tensor(idx_la2).long()
        idx_wph["shifted"] = torch.tensor(shifted).long()
        self.params_la1 = params_la1
        self.params_la2 = params_la2
        self.nb_moments = nb_moments
        # Pre-group indices by (j1, j2) for batch processing
        grouped_indices = {}
        for global_idx, (j1, j2) in enumerate(pair_metadata):
            if (j1, j2) not in grouped_indices:
                grouped_indices[(j1, j2)] = []
            grouped_indices[(j1, j2)].append(global_idx)
        # convert lists to tensors for faster indexing later
        self.grouped_indices = {k: torch.tensor(v).long() for k, v in grouped_indices.items()}
        return idx_wph

    def compute_correlations(self, xpsi: list[torch.Tensor], flatten: bool = True, vmap_chunk_size: Optional[int] = None):
        """
        Match same scales, and pass to vmap together
        Do upsampling here before passing to _pair_corr in vmap
        """
        nb = xpsi[0].shape[0]
        device = xpsi[0].device
        # Get total number of pairs
        total_pairs = self.idx_wph["la1"].shape[0]

        # pre-compute ffts to save time
        hatx_list = []
        for x in xpsi:
            if not torch.is_complex(x):
                x_c = torch.complex(x, torch.zeros_like(x))
            hatx_list.append(fft.fft2(x_c))  # (nb, C, M, N)

        # List to store results before reordering
        results_storage = [None] * total_pairs

        # 3. Iterate over unique (j1, j2) pairs
        for (j1, j2), global_idxs in self.grouped_indices.items():
            global_idxs = global_idxs.to(device)
            
            # Extract specific indices for this batch
            # idx_wph["la1"] is (N, 2), we want column 1 (channel index)
            # We filter by the rows in global_idxs
            la1_batch = self.idx_wph["la1"][global_idxs, 1].to(device)
            la2_batch = self.idx_wph["la2"][global_idxs, 1].to(device)
            shifted_batch = self.idx_wph["shifted"][global_idxs].to(device)

            h1 = hatx_list[j1]
            h2 = hatx_list[j2]

            if j2 > j1:
                h2 = self.spectral_pad(h2, target_shape=h1.shape[-2:])
            elif j2 < j1:
                 # Should not happen based on j loop, but for safety:
                 warnings.warn("j2 < j1 encountered in CorrLayerDownsample; upsampling m1 instead of m2.")
                 h1 = self.spectral_pad(h1, target_shape=h2.shape[-2:])

            masks_current, _ = self.get_mask_for_scale(j1)
            vmapped = torch.vmap(self._pair_corr,
                                 in_dims = (None, None, None, 0,0,0,None),
                                 out_dims=0,
                                 chunk_size=vmap_chunk_size)
            # out_batch: (Batch_Pairs, nb, N_UNION_PIXELS)
            out_batch = vmapped(h1, h2, masks_current, la1_batch, la2_batch, shifted_batch, flatten)

            # reorder list to match original order
            # and if flattening throw away extra zeros from union to make sure all outputs same size
            for i, global_idx in enumerate(global_idxs):
                shift_type = shifted_batch[i].item()
                
                if flatten:
                    # 1. Retrieve the pre-calculated indices for this shift type
                    # (e.g., indices for center pixel if shift=0, or neighbors if shift=1)
                    valid_indices = getattr(self, f'mask_idx_map_{shift_type}')
                    cleaned_result = out_batch[i][:, valid_indices]
                    
                    results_storage[global_idx.item()] = cleaned_result
                else:
                    # If not flattening, we keep the full spatial grid (M, N)
                    results_storage[global_idx.item()] = out_batch[i]

        # Final Assembly
        if flatten:
            return torch.stack(results_storage, dim=1)  # (nb, total_masked)
        else:
            return results_storage


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


    def spectral_pad(self, x_hat, target_shape):
        """
        Zero-pads a Fourier representation to a larger size (Spectral Upsampling).
        x_hat: (..., M, N)
        target_shape: (M_new, N_new)
        """
        M, N = x_hat.shape[-2:]
        M_new, N_new = target_shape
        
        if M == M_new and N == N_new:
            return x_hat
        
        # use fftshift and ifftshift to handle corners correctly    
        x_shifted = fft.fftshift(x_hat, dim=(-2, -1))
        
        # Calculate padding
        pad_h = (M_new - M) // 2
        pad_w = (N_new - N) // 2
        # F.pad order: (left, right, top, bottom)
        # Handle odd/even sizes carefully if needed, but usually power of 2
        x_padded = torch.nn.functional.pad(x_shifted, (pad_w, pad_w, pad_h, pad_h))
        
        return fft.ifftshift(x_padded, dim=(-2, -1)) * (M_new * N_new) / (M * N) 
        