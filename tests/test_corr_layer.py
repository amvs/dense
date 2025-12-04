import os
import sys
import torch
import torch.fft as fft
import pytest
from itertools import product
from wph.alpha_torch import ALPHATorch
from wph.layers.corr_layer import CorrLayer, CorrLayerDownsample


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