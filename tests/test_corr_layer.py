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


def test_compute_correlations_matches_alpha_torch():
    # small sizes for a quick unit test
    J, L, A, M, N = 2, 2, 2, 8, 8
    nb = 7
    num_channels = 3

    filters = make_filters(J, L, A, M, N)

    # create ALPHATorch instance (original implementation)
    alpha = ALPHATorch(M=M, N=N, J=J, L=L, A=A, A_prime=A, num_channels=num_channels,
                       nb_chunks=1, chunk_id=0, delta_j=1, delta_l=1, filters=filters)

    # random input
    img = torch.randn(nb, num_channels, M, N)
    x_c = torch.complex(img, torch.zeros_like(img))
    hatx = fft.fft2(x_c)

    # follow alpha forward steps up to correlations
    xpsi = alpha.compute_wavelet_transform(hatx)
    xpsi = alpha.normalize_and_mask(xpsi)
    C = num_channels * J * L * A
    xpsi_flat = xpsi.view(nb, C, M, N)

    out_alpha = alpha.compute_correlations(xpsi_flat, this_wph=None)

    # prepare CorrLayer instance WITHOUT calling its __init__ (we'll pass indices explicitly)
    corr = CorrLayer.__new__(CorrLayer)

    # call CorrLayer.compute_correlations with alpha's this_wph buffers
    this_wph = {"la1": alpha.this_wph_la1, "la2": alpha.this_wph_la2}
    out_corr = CorrLayer.compute_correlations(corr, xpsi_flat, this_wph=this_wph, vmap_chunk_size=32)

    # compare shapes and values
    assert out_alpha.shape == out_corr.shape
    assert torch.allclose(out_alpha, out_corr, atol=1e-5, rtol=1e-4)
