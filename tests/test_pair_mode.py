import torch
import pytest

from wph.layers.wave_conv_layer import WaveConvLayerDownsample
from wph.layers.corr_layer import CorrLayerDownsample, CorrLayerDownsamplePairs
from wph.layers.relu_center_layer import ReluCenterLayerDownsample, ReluCenterLayerDownsamplePairs
from wph.wph_model import WPHModelDownsample


def test_wave_pair_mode_parameter_growth():
    J, L, A, T = 3, 2, 2, 5
    filters = {"psi": torch.randn(J, L, A, T, T, dtype=torch.complex64)}

    w_shared = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, share_scale_pairs=True, init_filters=filters["psi"])
    w_pairs = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, share_scale_pairs=False, init_filters=filters["psi"].new_zeros(J*J, L, A, T, T) + filters["psi"].mean())

    params_shared = sum(p.numel() for p in w_shared.parameters())
    params_pairs = sum(p.numel() for p in w_pairs.parameters())

    assert params_pairs > params_shared, "Pair-mode should increase parameter count."


def test_corr_outputs_same_size_pair_vs_shared():
    # same indexing parameters -> output size should match
    J, L, A, A_prime, M, N, T = 3, 2, 2, 2, 32, 32, 5
    B, C = 2, 1
    x = torch.randn(B, C, M, N)
    filters = {"psi": torch.randn(J, L, A, T, T, dtype=torch.complex64)}

    # Shared mode
    w_shared = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, share_scale_pairs=True, init_filters=filters["psi"])
    r_shared = ReluCenterLayerDownsample(J=J, M=M, N=N)
    c_shared = CorrLayerDownsample(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N)

    xpsi_shared = w_shared(x)
    xrelu_shared = r_shared(xpsi_shared)
    xcorr_shared = c_shared(xrelu_shared, flatten=True)

    # Pair mode: compute only needed pairs
    c_pairs = CorrLayerDownsamplePairs(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N)
    needed_pairs = sorted(c_pairs.grouped_indices.keys())

    w_pairs = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, share_scale_pairs=False, init_filters=filters["psi"].new_zeros(J*J, L, A, T, T) + filters["psi"].mean())
    xpsi_pairs = w_pairs(x, scale_pairs=needed_pairs)
    r_pairs = ReluCenterLayerDownsamplePairs(J=J, M=M, N=N)
    xrelu_pairs = r_pairs(xpsi_pairs)
    xcorr_pairs = c_pairs(xrelu_pairs, flatten=True)

    assert xcorr_shared.shape == xcorr_pairs.shape, "Pair-mode must not change number of correlation outputs."


def test_model_downsample_pair_mode_end_to_end():
    J, L, A, A_prime, M, N, T = 2, 2, 2, 2, 16, 16, 5
    B, C = 1, 1
    x = torch.randn(B, C, M, N)
    
    # Create base filters for shared mode
    psi_base = torch.randn(J, L, A, T, T, dtype=torch.complex64)
    
    # For pair mode, replicate the same filter for all JxJ pairs so outputs should match
    # Wave layer uses indexing: pair_index = j2 * J + j1
    # So we need to arrange filters in that order
    psi_pairs = torch.zeros(J*J, L, A, T, T, dtype=torch.complex64)
    for j1 in range(J):
        for j2 in range(J):
            pair_index = j2 * J + j1
            psi_pairs[pair_index] = psi_base[j1]
    
    hatphi = torch.randn(M, N, dtype=torch.complex64).real
    
    filters_shared = {"psi": psi_base, "hatphi": hatphi}
    filters_pairs = {"psi": psi_pairs, "hatphi": hatphi}

    model_shared = WPHModelDownsample(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N,
                                      num_channels=C, T=T, filters=filters_shared,
                                      share_scale_pairs=True)
    out_shared = model_shared(x, flatten=True)

    model_pairs = WPHModelDownsample(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N,
                                     num_channels=C, T=T, filters=filters_pairs,
                                     share_scale_pairs=False)
    out_pairs = model_pairs(x, flatten=True)

    assert out_shared.shape[0] == B
    assert out_pairs.shape[0] == B
    assert out_shared.shape[1] == out_pairs.shape[1], "End-to-end pair mode should keep feature dimension constant."
    
    # When initialized with identical filters, outputs should be close
    assert torch.allclose(out_shared, out_pairs, atol=1e-5), "Outputs should match when filters are identical."

@pytest.mark.parametrize("delta_j", [1, None])
@pytest.mark.parametrize("share_phases", [False, True])
@pytest.mark.parametrize("share_rotations", [False, True])
def test_param_grid_downsample(delta_j, share_phases, share_rotations):
    J, L, A, A_prime, M, N, T = 3, 3, 2, 2, 24, 24, 5
    B, C = 2, 1
    x = torch.randn(B, C, M, N)

    # Build filters respecting sharing flags for shared mode
    # Note: share_scales is always False in this test
    share_scales = False
    J_param_shared = 1 if share_scales else J
    L_param = 1 if share_rotations else L
    A_param = 1 if share_phases else A

    psi_shared = torch.randn(J_param_shared, L_param, A_param, T, T, dtype=torch.complex64)
    
    # For pair mode, replicate filters to all JxJ pairs so outputs match
    # Wave layer uses indexing: pair_index = j2 * J + j1
    # Replicate the base filter to all pairs with correct indexing
    psi_pairs = torch.zeros(J*J, L_param, A_param, T, T, dtype=torch.complex64)
    for j1 in range(J):
        for j2 in range(J):
            pair_index = j2 * J + j1
            psi_pairs[pair_index] = psi_shared[j1]
    
    hatphi = torch.randn(M, N, dtype=torch.complex64).real
    filters_shared = {"psi": psi_shared, "hatphi": hatphi}
    filters_pairs = {"psi": psi_pairs, "hatphi": hatphi}

    # Shared scale-pairs model
    model_shared = WPHModelDownsample(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N,
                                      num_channels=C, T=T, filters=filters_shared,
                                      share_scales=share_scales,
                                      share_rotations=share_rotations,
                                      share_phases=share_phases,
                                      delta_j=delta_j,
                                      share_scale_pairs=True)
    out_shared_flat = model_shared(x, flatten=True)
    out_shared_struct = model_shared(x, flatten=False)

    # Pair-mode model (note: share_scales=True overrides to shared pairs)
    model_pairs = WPHModelDownsample(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N,
                                     num_channels=C, T=T, filters=filters_pairs,
                                     share_scales=share_scales,
                                     share_rotations=share_rotations,
                                     share_phases=share_phases,
                                     delta_j=delta_j,
                                     share_scale_pairs=False)
    out_pairs_flat = model_pairs(x, flatten=True)
    out_pairs_struct = model_pairs(x, flatten=False)

    # Sanity: batch dims
    assert out_shared_flat.shape[0] == B
    assert out_pairs_flat.shape[0] == B

    # Feature dimension should be identical
    assert out_shared_flat.shape[1] == out_pairs_flat.shape[1]

    # Structured outputs should be tuples of three elements
    assert isinstance(out_shared_struct, tuple) and len(out_shared_struct) == 3
    assert isinstance(out_pairs_struct, tuple) and len(out_pairs_struct) == 3
    
    # When filters are identical, outputs should be close
    assert torch.allclose(out_shared_flat, out_pairs_flat, atol=1e-5), \
        f"Outputs should match when filters are identical (delta_j={delta_j}, share_scales={share_scales}, share_rotations={share_rotations})"
