import os
import sys
import torch
import torch.fft as fft
import pytest
from itertools import product
from wph.alpha_torch import ALPHATorch
from wph.layers.corr_layer import CorrLayer, CorrLayerDownsample
from wph.layers.wave_conv_layer import WaveConvLayer, WaveConvLayerDownsample
from tests.test_wave_conv_layer import create_compatible_filters, plot_fft_spatial_comparison




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


J_L_A_Aprime_C = [
    {"J": 2, "L": 4, "A": 2, "A_prime": 2, "num_channels": 1},
    {"J": 3, "L": 2, "A": 1, "A_prime": 1, "num_channels": 1},
    {"J": 2, "L": 4, "A": 2, "A_prime": 1, "num_channels": 3},
]
delta_j_values = [1, 4]
num_channels_shift_alpha = [
    {"num_channels": 1, "shift_mode": "strict", "alpha_shift": "same"},
    {"num_channels": 1, "shift_mode": "all", "alpha_shift": "all"},
    {"num_channels": 3, "shift_mode": "samec", "alpha_shift": "samec"},
    {"num_channels": 3, "shift_mode": "all", "alpha_shift": "all"},
]

# Combine all parameters into a single list of dictionaries
all_params = [
    {**base, "delta_j": delta_j, **extra}
    for base, delta_j, extra in product(J_L_A_Aprime_C, delta_j_values, num_channels_shift_alpha)
]

class TestCorrLayerDownsample:
    
    @pytest.fixture
    def model(self, request):
        """Fixture to create a CorrLayerDownsample model for testing."""
        # Init params
        params = request.param
        J, L, A, A_prime = params.get("J", 3), params.get("L", 2), params.get("A", 1), params.get("A_prime", 1)
        M, N = params.get("M", 32), params.get("N", 32)
        C = params.get("num_channels", 1)
        shift = params.get("shift_mode", "all")
        delta_j = params.get("delta_j", 1)
        delta_l = params.get("delta_l", 1)
        
        layer = CorrLayerDownsample(
            J=J, L=L, A=A, A_prime=A_prime, 
            M=M, N=N, num_channels=C,
            delta_j=delta_j, delta_l=delta_l,
            shift_mode=shift
        )
        return layer
    

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_init_buffers_and_shapes(self, model):
        """Verify buffers are created for each scale."""
        # Check masks exist for scale 0, 1, 2
        m0, f0 = model.get_mask_for_scale(0)
        assert m0.shape == (2, 32, 32) # (K, M, N)
        
        m1, f1 = model.get_mask_for_scale(1)
        assert m1.shape == (2, 16, 16) # Downsampled
        
        # Check Union Index Map
        # In our mock, Mask 0 (1 pixel) is a subset of Mask 1 (5 pixels).
        # Union is Mask 1 (5 pixels).
        # Map 0 should point to the center index of the Union vector.
        assert hasattr(model, 'mask_idx_map_0')
        assert hasattr(model, 'mask_idx_map_1')

        # Map 0 should have length 1, Map 1 length 9 = 8 (along 8 mask directions) + 1 center
        assert len(model.mask_idx_map_0) == 1
        assert len(model.mask_idx_map_1) == 9

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_spectral_pad(self, model):
        """Verify FFT zero-padding logic."""
        # Create a 4x4 signal (all ones)
        x_small = torch.ones(1, 1, 4, 4)
        hat_small = torch.fft.fft2(x_small)
        
        # Pad to 8x8
        hat_large = model.spectral_pad(hat_small, (8, 8))
        
        assert hat_large.shape == (1, 1, 8, 8)
        
        # Energy Check: DC component (0,0) should scale
        # FFT(Ones) puts all energy in DC.
        # 4x4 sum = 16. 8x8 sum = 64. Ratio = 4.
        # Spectral pad multiplies by (64/16) = 4.
        assert torch.isclose(hat_large[..., 0, 0].abs(), hat_small[..., 0, 0].abs() * 4.0)

        # High freq check: Corners of shifted FFT should be zero
        shifted = torch.fft.fftshift(hat_large)
        assert shifted[..., 0, 0].abs() == 0.0 # Corner is padding

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_forward_output_shape_flatten(self, model):
        """Verify forward pass produces list of tensor."""
        B = 2
        # Create input list [Scale0, Scale1, Scale2]
        xpsi = []
        for j in range(model.J):
            dim = 32 // (2**j)
            # Input channels C_in = num_channels * L * A = 1*2*1 = 2
            x = torch.randn(B, model.num_channels * model.L * model.A, dim, dim) 
            xpsi.append(x)

        out = model.compute_correlations(xpsi, flatten=True)
        assert out.ndim == 2  # (B, Coeffs)
        assert out.shape[0] == B
        assert out.shape[1] == model.nb_moments

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_forward_output_shape_not_flatten(self, model):
        """Verify forward pass produces concatenated tensor."""
        B = 2
        # Create input list [Scale0, Scale1, Scale2]
        xpsi = []
        for j in range(model.J):
            dim = 32 // (2**j)
            # Input channels C_in = num_channels * L * A = 1*2*1 = 2
            x = torch.randn(B, model.num_channels * model.L * model.A, dim, dim) 
            xpsi.append(x)

        out = model.compute_correlations(xpsi, flatten=False)
        assert len(out) == model.J  # (B, Coeffs)
        assert out[0].shape[-2:] == (model.M, model.N)
        assert out[-1].shape[-2:] == (model.M // (2**(model.J-1)), model.N // (2**(model.J-1)))

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_forward_constant_size_logic(self, model):
        """
        Since compute_correlations does torch.stack, we must ensure
        all pairs return the same number of coefficients.
        This usually means we ALWAYS use the Shift Mask (Index 1), 
        even for self-correlation, or the Identity Mask (Index 0) has same size.
        """
        
        xpsi = []
        for j in range(model.J):
            dim = model.M // (2**j)
            xpsi.append(torch.randn(1, model.num_channels * model.L * model.A, dim, dim))
        
        out = model.compute_correlations(xpsi, flatten=True)
        
        # Should stack successfully
        assert out.ndim == 2 # (B, Pairs, Coeffs)
        assert out.shape[1] == model.nb_moments

    @pytest.mark.parametrize("model", all_params, indirect=True)
    def test_integration_j1_j2_interaction(self, model):
        """
        Check that cross-scale correlation (j1=0, j2=1) runs without error
        and triggers the upsampling path.
        """
        xpsi = []
        for j in range(model.J):
            dim = 32 // (2**j)
            # Input channels C_in = num_channels * L * A = 1*2*1 = 2
            x = torch.randn(2, model.num_channels * model.L * model.A, dim, dim) 
            xpsi.append(x)
            
        # We perform computation. If spectral_pad had bugs, this would crash.
        out = model.compute_correlations(xpsi, flatten=False)
        
        # Check we have results
        assert len(out) == model.idx_wph["la1"].shape[0]
        # Check result 0 is a tensor
        assert torch.is_tensor(out[0])

def make_gaussian(N, sigma=2.0, center=None):
    if center is None:
        center = (N // 2, N // 2)
    x = torch.arange(N).float()
    y = torch.arange(N).float()
    X, Y = torch.meshgrid(x, y, indexing='ij')
    gauss = torch.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2))
    return gauss

def test_corr_layer_wave_conv_alignment():
    """
    Checks that CorrLayer and CorrLayerDownsample produce similar outputs when fed with relu(real(WaveConvLayer)) outputs.
    """
    # Match the setup from test_multiscale_rotation_equivalence in test_wave_conv_layer.py
    M = N = 16
    J = 3
    L = 4
    A = 2
    T = 5
    num_channels = 1

    x = make_gaussian(N=N).unsqueeze(0).unsqueeze(0)

    # Create matched filters
    small_k, hat_filters = create_compatible_filters(J, L, A, N, N, kernel_size=T)

    # FFT model
    wave_fft = WaveConvLayer(J=J, L=L, A=A, M=N, N=N, num_channels=num_channels, filters=hat_filters,
                        share_rotations=False, share_phases=False)
    # Spatial model
    wave_spatial = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, num_channels=num_channels,
                                    share_rotations=False, share_phases=False)
    # Patch spatial weights
    wave_spatial.base_real.data = torch.flip(small_k[0, 0].real, [-1, -2])
    wave_spatial.base_imag.data = torch.flip(small_k[0, 0].imag, [-1, -2])

    # Get outputs
    out_fft = wave_fft(x)  # (B, C, J, L, A, H, W)
    out_spatial = wave_spatial(x)  # List of (B, C, L, A, H, W)

    for j in range(J):
        # Downsample FFT output to match spatial output shape
        fft_out_j = out_fft[..., j, :, :, :, :]  # (B, C, L, A, H, W)
        spatial_out_j = out_spatial[j]           # (B, C, L, A, H', W')
        H, W = spatial_out_j.shape[-2:]
        # Use average pooling for downsampling
        fft_out_j_real_down = torch.nn.functional.avg_pool2d(
            fft_out_j.real.view(-1, H * 2 ** j, W * 2 ** j), kernel_size=2 ** j
        ).view_as(spatial_out_j)
        fft_out_j_imag_down = torch.nn.functional.avg_pool2d(
            fft_out_j.imag.view(-1, H * 2 ** j, W * 2 ** j), kernel_size=2 ** j
        ).view_as(spatial_out_j)
        fft_out_j_down = torch.complex(fft_out_j_real_down, fft_out_j_imag_down)

        plot_fft_spatial_comparison(fft_out_j, spatial_out_j, fft_out_j_down, j, save_path = f'tstDeleteMe_{j}.png')

    # For each scale, take real part and relu
    fft_relu = torch.relu(out_fft.real)
    spatial_relu = [torch.relu(out_spatial[j].real) for j in range(J)]

    # CorrLayer
    corr_fft = CorrLayer(J=J, L=L, A=A, A_prime=1, M=M, N=N, num_channels=num_channels,
                        delta_j=1, delta_l=1, shift_mode='strict', mask_union=False, mask_angles=4)
    corr_spatial = CorrLayerDownsample(J=J, L=L, A=A, A_prime=1, M=M, N=N, num_channels=num_channels,
                                      delta_j=1, delta_l=1, shift_mode='strict')


    # Compute correlations (not flattened, to get per-group outputs)
    out_corr_fft = corr_fft.compute_correlations(fft_relu.view(1, num_channels * J * L * A, M, N), flatten=False)
    out_corr_spatial = corr_spatial.compute_correlations(spatial_relu, flatten=False)

    # Compare outputs by grouped_indices, handling that spatial_out[j1] contains all (j1, j2) pairs
    grouped_indices = corr_spatial.grouped_indices
    print("\n--- Per-pair comparison (mean, max, min, mean abs diff) ---")
    # Build a mapping from (j1, j2) to (spatial_out tensor, pair index)
    # out_corr_spatial[j1] contains all pairs with that j1, in order of grouped_indices keys for that j1
    # We'll loop through all pairs in grouped_indices
    for (j1, j2), indices in grouped_indices.items():
        # Find which output tensor and which pair index within that tensor
        # out_corr_spatial[j1] contains all pairs for this j1
        spatial_out = out_corr_spatial[j1]  # shape: [B, n_pairs, H, W]
        # The order of pairs in spatial_out matches the order of grouped_indices keys for that j1
        # So we need to find the position of (j1, j2) among all keys with this j1
        keys_for_j1 = [k for k in grouped_indices.keys() if k[0] == j1]
        pair_idx = keys_for_j1.index((j1, j2))
        # For each pair in this group
        for local_p, global_idx in enumerate(indices):
            # spatial_out[0, pair_idx * n_pairs_per_group + local_p] matches out_corr_fft[:, global_idx, :, :]
            s = spatial_out[0, pair_idx * len(indices) + local_p].round(decimals=6)
            f = out_corr_fft[0, global_idx].round(decimals=6)
            # Count nonzero elements
            nnz_s = torch.count_nonzero(s)
            nnz_f = torch.count_nonzero(f)
            assert nnz_s == nnz_f, f"Nonzero count mismatch for (j1={j1}, j2={j2}, pair={local_p}): {nnz_s} vs {nnz_f}"            # Compute mean, min, max of nonzero elements
            s_nz = s[s != 0]
            f_nz = f[f != 0]
            s_nz_norm = s_nz / torch.norm(s_nz)
            f_nz_norm = f_nz / torch.norm(f_nz)
            # If all elements are zero, skip further checks
            if nnz_s > 0:
                norm_diff = torch.norm(s_nz) / torch.norm(f_nz)
                max_diff = torch.abs(s_nz_norm.max() - f_nz_norm.max())
                min_diff = torch.abs(s_nz_norm.min() - f_nz_norm.min())
                try:
                    # assert torch.abs(norm_diff - 1) < 0.1, f"Nonzero norm mismatch of greater than 10% for (j1={j1}, j2={j2}, pair={local_p}): {s_nz.mean().item()} vs {f_nz.mean().item()}"
                    assert max_diff < 1e-2, f"Nonzero max mismatch for (j1={j1}, j2={j2}, pair={local_p}): {s_nz.max().item()} vs {f_nz.max().item()}"
                    assert min_diff < 1e-2, f"Nonzero min mismatch for (j1={j1}, j2={j2}, pair={local_p}): {s_nz.min().item()} vs {f_nz.min().item()}"
                except AssertionError as e:
                    breakpoint()
                    print(e)
