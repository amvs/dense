import os
import sys
import torch
import torch.fft as fft

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.alpha_torch import ALPHATorch
from wph.layers.corr_layer import CorrLayer


def make_filters(J, L, A, M, N, device="cpu"):
    # ALPHATorch expects hatpsi of shape (J, L, M, N) and will expand over A internally
    hatpsi = torch.randn(J, L, M, N, dtype=torch.cfloat, device=device)
    hatphi = torch.randn(M, N, dtype=torch.cfloat, device=device)
    return {"hatpsi": hatpsi, "hatphi": hatphi}


import os
import sys
import torch
import torch.fft as fft
import pytest

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.alpha_torch import ALPHATorch
from wph.layers.corr_layer import CorrLayer


def make_filters(J, L, A, M, N, device="cpu"):
    # ALPHATorch expects hatpsi of shape (J, L, M, N) and will expand over A internally
    hatpsi = torch.randn(J, L, M, N, dtype=torch.cfloat, device=device)
    hatphi = torch.randn(M, N, dtype=torch.cfloat, device=device)
    return {"hatpsi": hatpsi, "hatphi": hatphi}


# Parametrize tests across different configurations
@pytest.mark.parametrize("delta_j", [1, 4])
@pytest.mark.parametrize("mask_union", [False, True])
@pytest.mark.parametrize("num_channels,shift_mode,alpha_shift", [
    (1, "strict", "same"),   # num_channels=1: strict/same
    (1, "all", "all"),       # num_channels=1: all
    (3, "samec", "samec"),   # num_channels=3: samec
    (3, "all", "all"),       # num_channels=3: all
])
def test_compute_correlations_flatten_parametrized(delta_j, mask_union, num_channels, shift_mode, alpha_shift):
    """Test flatten=True mode matches ALPHATorch across different parameter combinations."""
    J, L, A, M, N = 2, 4, 3, 8, 8
    nb = 2
    filters = make_filters(J, L, A, M, N)

    # Create ALPHATorch model
    alpha = ALPHATorch(
        M=M, N=N, J=J, L=L, A=A, A_prime=1, 
        num_channels=num_channels,
        nb_chunks=1, chunk_id=0, 
        delta_j=delta_j, delta_l=1, 
        shift=alpha_shift, 
        filters=filters, 
        mask_union=mask_union
    )

    # Create test input
    img = torch.randn(nb, num_channels, M, N)
    x_c = torch.complex(img, torch.zeros_like(img))
    hatx = fft.fft2(x_c)
    xpsi = alpha.compute_wavelet_transform(hatx)
    xpsi = alpha.normalize_and_mask(xpsi)
    C = num_channels * J * L * A
    xpsi_flat = xpsi.view(nb, C, M, N)

    # ALPHATorch path: compute_correlations then mask_correlations with flatten
    corr_alpha = alpha.compute_correlations(xpsi_flat, this_wph=None)
    out_alpha = alpha.mask_correlations(corr_alpha, this_wph=None, flatten=True)

    # CorrLayer path: uses same indices and parameters
    corr_layer = CorrLayer(
        J=J, L=L, A=A, A_prime=1, M=M, N=N, 
        num_channels=num_channels,
        delta_j=delta_j, delta_l=1, 
        shift_mode=shift_mode, 
        mask_union=mask_union, 
        mask_angles=4
    )
    out_corr = corr_layer.compute_correlations(xpsi_flat, flatten=True, vmap_chunk_size=32)

    # Verify shapes match
    assert out_alpha.shape == out_corr.shape, (
        f"Shape mismatch for delta_j={delta_j}, mask_union={mask_union}, "
        f"shift={shift_mode}, num_channels={num_channels}: "
        f"{out_alpha.shape} vs {out_corr.shape}"
    )
    
    # Verify values match
    assert torch.allclose(out_alpha, out_corr, atol=1e-5, rtol=1e-4), (
        f"Values don't match for delta_j={delta_j}, mask_union={mask_union}, "
        f"shift={shift_mode}, num_channels={num_channels}"
    )


def test_compute_correlations_flatten_matches_alpha_torch():
    """Basic sanity check test - single configuration."""
    J, L, A, M, N = 2, 4, 3, 8, 8
    nb = 2
    num_channels = 1
    filters = make_filters(J, L, A, M, N)

    alpha = ALPHATorch(M=M, N=N, J=J, L=L, A=A, A_prime=1, num_channels=num_channels,
                       nb_chunks=1, chunk_id=0, delta_j=1, delta_l=1, shift='same', 
                       filters=filters, mask_union=False)

    img = torch.randn(nb, num_channels, M, N)
    x_c = torch.complex(img, torch.zeros_like(img))
    hatx = fft.fft2(x_c)
    xpsi = alpha.compute_wavelet_transform(hatx)
    xpsi = alpha.normalize_and_mask(xpsi)
    C = num_channels * J * L * A
    xpsi_flat = xpsi.view(nb, C, M, N)

    corr_alpha = alpha.compute_correlations(xpsi_flat, this_wph=None)
    out_alpha = alpha.mask_correlations(corr_alpha, this_wph=None, flatten=True)

    corr_layer = CorrLayer(J=J, L=L, A=A, A_prime=1, M=M, N=N, num_channels=num_channels,
                           delta_j=1, delta_l=1, shift_mode='strict', mask_union=False, mask_angles=4)
    out_corr = corr_layer.compute_correlations(xpsi_flat, flatten=True, vmap_chunk_size=32)

    assert out_alpha.shape == out_corr.shape, f"Shape mismatch: {out_alpha.shape} vs {out_corr.shape}"
    assert torch.allclose(out_alpha, out_corr, atol=1e-5, rtol=1e-4), "Values don't match for flatten=True"

test_compute_correlations_flatten_matches_alpha_torch()