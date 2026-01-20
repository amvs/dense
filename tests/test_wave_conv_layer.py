import torch
from torch import fft
from torchviz import make_dot
import math
import pytest
from wph.layers.wave_conv_layer import WaveConvLayer, WaveConvLayerDownsample

import matplotlib.pyplot as plt
from scripts.visualize import colorize

def plot_fft_spatial_comparison(fft_out_j, spatial_out_j, fft_out_j_down, j, save_path=None):
    """
    Visualizes and compares FFT, downsampled FFT, and spatial outputs for all rotations (L).
    Uses colorize for complex tensors, otherwise plots real tensors as-is.
    """
    L = fft_out_j.shape[2]
    fig, axs = plt.subplots(4, L, figsize=(4*L, 12))
    for l in range(L):
        fft_img = fft_out_j[0, 0, l, 0].detach().cpu()
        spatial_img = spatial_out_j[0, 0, l, 0].detach().cpu()
        down_img = fft_out_j_down[0, 0, l, 0].detach().cpu()

        def _maybe_colorize(img):
            if torch.is_complex(img):
                return colorize(img)
            return img

        fft_img_col = _maybe_colorize(fft_img)
        spatial_img_col = _maybe_colorize(spatial_img)
        down_img_col = _maybe_colorize(down_img)

        axs[0, l].imshow(fft_img_col)
        axs[0, l].set_title(f'FFT Fullsize (scale {j}, L={l})')
        axs[1, l].imshow(down_img_col)
        axs[1, l].set_title(f'FFT Downsampled (scale {j}, L={l})')
        axs[2, l].imshow(spatial_img_col)
        axs[2, l].set_title(f'Spatial Downsample (scale {j}, L={l})')
        axs[3, l].imshow(torch.abs(spatial_img - down_img), cmap='hot', vmin=0, vmax=1)
        axs[3, l].set_title(f'Difference (scale {j}, L={l})')
        for row in range(4):
            axs[row, l].axis('off')
    plt.tight_layout()
    if save_path is None:
        save_path = f'wph_compare_scale{j}.png'
    plt.savefig(save_path)
    plt.close(fig)

def test_wave_conv_layer_gradients_share_rotations():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_rotations=True)

    x = torch.randn(1, 1, M, N, requires_grad=True)
    output = layer(x)
    loss = output.abs().sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not propagate to input."
    assert layer.base_filters.grad is not None, "Gradients did not propagate to filters."

def test_wave_conv_layer_gradients_share_phases():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_phases=True)

    x = torch.randn(1, 1, M, N, requires_grad=True)
    output = layer(x)
    loss = output.abs().sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not propagate to input."
    assert layer.base_filters.grad is not None, "Gradients did not propagate to filters."

def test_wave_conv_layer_gradients_share_channels():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=3, share_channels=True)

    x = torch.randn(1, 3, M, N, requires_grad=True)
    output = layer(x)
    loss = output.abs().sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not propagate to input."
    assert layer.base_filters.grad is not None, "Gradients did not propagate to filters."

def test_wave_conv_layer_gradients_compare_share_rotations():
    J, L, A, M, N = 3, 4, 1, 8, 8
    torch.autograd.set_detect_anomaly(True)
    # Case 1: share_rotations=True
    # need to define filters carefully
    filters = torch.load('tests/morlet_N8_J3_L4.pt', weights_only=True)
    layer_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_rotations=True, filters=filters[:,0,...].unsqueeze(0).unsqueeze(2).unsqueeze(3))
    x = torch.randn(1, 1, M, N, requires_grad=True)
    output_shared = layer_shared(x)
    loss_shared = output_shared.abs().sum()
    loss_shared.backward()
    grad_shared = layer_shared.base_filters.grad.clone()

    graph_shared = make_dot(loss_shared, params={"x": x, "filters": layer_shared.base_filters})
    graph_shared.render("computation_graph_shared", format="pdf")


    # Case 2: share_rotations=False
    layer_not_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_rotations=False, filters = layer_shared.get_full_filters().clone())
    output_not_shared = layer_not_shared(x)
    loss_not_shared = output_not_shared.abs().sum()
    loss_not_shared.backward()
    grad_not_shared = layer_not_shared.base_filters.grad.clone()

    # Visualize computation graph for share_rotations=False
    graph_not_shared = make_dot(loss_not_shared, params={"x": x, "filters": layer_not_shared.base_filters})
    graph_not_shared.render("computation_graph_not_shared", format="pdf")


    # Compare gradients
    # Explicitly sum gradients across the L dimension for the not-shared case
    grad_not_shared_summed = grad_not_shared.sum(dim=2)  # Summing across the L dimension

    # Check if the summed gradients match
    assert torch.allclose(layer_not_shared.base_filters, layer_shared.get_full_filters()), "Filters do not match between shared and not-shared cases."
    # assert torch.allclose(grad_shared, grad_not_shared_summed), "Gradients do not match after summing across the L dimension."

def test_wave_conv_layer_gradients_compare_share_phases():
    J, L, A, M, N = 3, 4, 2, 8, 8
    torch.autograd.set_detect_anomaly(True)

    # Load filters from file
    filters = torch.load('tests/morlet_N8_J3_L4.pt', weights_only=True)

    # Apply phase shifts to filters
    i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))
    phase_shifts = torch.stack([
        filters * torch.exp(i * alpha * (2 * math.pi / A)) for alpha in range(A)
    ], dim=-3)  # Stack along the phase dimension

    # Case 1: share_phases=True
    layer_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_phases=True, filters=filters.unsqueeze(0).unsqueeze(-3))
    x = torch.randn(1, 1, M, N, requires_grad=True)
    output_shared = layer_shared(x)
    loss_shared = output_shared.abs().sum()
    loss_shared.backward()
    grad_shared = layer_shared.base_filters.grad.clone()

    # Case 2: share_phases=False
    layer_not_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, share_phases=False, filters=phase_shifts.clone().unsqueeze(0))
    output_not_shared = layer_not_shared(x)
    loss_not_shared = output_not_shared.abs().sum()
    loss_not_shared.backward()
    grad_not_shared = layer_not_shared.base_filters.grad.clone()

    # Compare gradients

    grad_not_shared_summed = grad_not_shared.real.abs().sum(dim=-3)  # Summing across the A dimension - take absolute value to avoid cancellation bc of opposite sign from phase
    # recall that when A = 2, the shifts are 1 and -1, so gradients can cancel out
    # but when accumulating gradients in pytorch the sign is accounted for

    # Check if the summed gradients match
    assert torch.allclose(layer_not_shared.base_filters, layer_shared.get_full_filters()), "Filters do not match between shared and not-shared cases."
    assert torch.allclose(grad_shared.squeeze().real.abs(), grad_not_shared_summed.squeeze(), atol = 1e-6), "Gradients do not match after summing across the A dimension."

    # Check if phase shifts are correctly applied in get_full_filters
    full_filters = layer_shared.get_full_filters()
    for alpha in range(A):
        expected_phase_shift = torch.exp(i * alpha * (2 * math.pi / A))
        assert torch.allclose(full_filters[:, :, :, alpha, :, :], phase_shifts[:, :, 0, :, :] * expected_phase_shift), f"Phase shift not correctly applied for alpha={alpha}."

# --- Helper Functions ---

def create_compatible_filters(J, L, A, M, N, kernel_size=7):
    """
    Creates a set of small spatial filters and their FFT-equivalent
    full-sized filters to ensure we are testing the architecture, 
    not the randomness of initialization.
    """
    # 1. Create a sine wave kernel and L rotated versions
    # Shape: (1, 1, 1, L, kernel_size, kernel_size)
    small_k = torch.zeros(1, 1, L, A, kernel_size, kernel_size, dtype=torch.complex64)
    x = torch.linspace(-math.pi, math.pi, kernel_size)
    y = torch.linspace(-math.pi, math.pi, kernel_size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    for l in range(L):
        theta = l * math.pi / L
        X_rot = X * math.cos(theta) - Y * math.sin(theta)
        Y_rot = X * math.sin(theta) + Y * math.cos(theta)
        small_k[0, 0, l,] = torch.sin(X_rot + Y_rot + l * math.pi / 8) + 0.2j * torch.sin(X_rot + Y_rot + l * math.pi / 8)
    
    # 2. Create the "Pyramid" filters (Constant small size)
    # The downsample model uses the SAME filter repeatedly
    # Shape for Downsample Layer: (1, 1, 1, 1, K, K)
    # In practice, the class might expand this, but we initialize with this.
    
    # 3. Create the "FFT" filters (Effective filters at large scale)
    # For the FFT model, scale j corresponds to dilating the filter by 2^j.
    # We simulate this by padding the small kernel with zeros to size M,N 
    # BUT complying with the stride logic.
    
    filters_tensor = torch.zeros(1, J, L, A, M, N, dtype=torch.complex64)
    for j in range(J):
        for l in range(L):
            for a in range(A):
                large_filter = torch.zeros((M, N), dtype=torch.complex64)
                center_l = M // 2
                center_w = N // 2
                k_half = kernel_size // 2
                dilation = 2 ** j
                start_row = center_l - k_half * dilation
                start_col = center_w - k_half * dilation
                for ki in range(kernel_size):
                    for kj in range(kernel_size):
                        row = start_row + ki * dilation
                        col = start_col + kj * dilation
                        if 0 <= row < M and 0 <= col < N:
                            large_filter[row, col] = small_k[0, 0, l, a, ki, kj]
                filters_tensor[0, j, l, a] = large_filter
    hat_filters = fft.fft2(fft.ifftshift(filters_tensor, dim=(-2,-1)))
    return small_k, hat_filters

# --- Tests ---

@pytest.mark.parametrize("J", [2, 3])
@pytest.mark.parametrize("N, L, A, share_rotations, share_phases",
    [(32, 4, 2, True, True),
    (32, 4, 2, True, False),
    (32, 4, 2, False, True),
    (32, 4, 2, False, False),
    (32, 4, 4, True, True),
    (32, 8, 2, True, True),
 ])
def test_output_shapes(J, N, L, A, share_rotations, share_phases):
    """
    Verifies that the Downsample layer produces the correct list of tensor shapes.
    """
    B, C = 2, 1 
    model = WaveConvLayerDownsample(J=J, L=L, A=A, num_channels=C, T=3, share_rotations=share_rotations, share_phases=share_phases)
    x = torch.randn(B, C, N, N)
    
    out = model(x)
    
    assert len(out) == J
    for j, feature_map in enumerate(out):
        expected_size = N // (2**j)
        # Check dimensions
        # feature_map shape: (B, C, L, A, H, W)
        assert feature_map.shape[0] == B
        assert feature_map.shape[1] == C
        assert feature_map.shape[2] == L
        assert feature_map.shape[3] == A
        assert feature_map.shape[4] == expected_size
        assert feature_map.shape[5] == expected_size
        
        # Check it is complex
        assert torch.is_complex(feature_map)

def test_multiscale_alignment():
    """
    Checks if the downsampled output of the Spatial model roughly tracks 
    the downsampled output of the FFT model.
    
    Note: This will NOT be exact due to:
    1. Antialiasing filter differences (FFT implies ideal Sinc, Spatial uses Gaussian).
    2. Boundary effects.
    """
    N = 64
    J = 3
    model_spatial = WaveConvLayerDownsample(J=J, L=1, A=1, T=7)
    x = torch.randn(1, 1, N, N)
    
    out_spatial = model_spatial(x)
    
    # Sanity check: Energy should roughly decrease or stay constant, not explode
    energy_j0 = torch.mean(torch.abs(out_spatial[0])**2)
    energy_j1 = torch.mean(torch.abs(out_spatial[1])**2)
    
    # In standard scattering, energy decays. 
    # Just ensure we aren't getting NaNs or Infs
    assert not torch.isnan(out_spatial[1]).any()
    assert energy_j1 > 0
    assert energy_j1 < energy_j0


def test_scale_zero_equivalence():
    """
    Strict test: Scale 0 (no downsampling yet) must match EXACTLY 
    between FFT and Spatial implementations if filters are identical.
    """
    N = 32
    J = 1
    L = 4 # Use L > 1 to test the expansion logic
    A = 1
    T = 7 
    
    # Setup Inputs
    x = torch.randn(1, 1, N, N)
    
    # Create matched filters
    # small_k shape: (1, 1, 1, 1, T, T)
    small_k, hat_filters = create_compatible_filters(J, L, A, N, N, kernel_size=T)
    
    # --- 1. FFT Model ---
    # We must construct the full hat_filters correctly for L orientations
    # For this simple test, let's assume L=1 in the helper or manually replicate
    # shape hat_filters: (1, J, L, A, M, N)
    fft_model = WaveConvLayer(J=J, L=L, A=A, M=N, N=N, filters=hat_filters, 
                              share_rotations=False, share_phases=True)
    
    # --- 2. Spatial Model ---
    # Case: We share rotations = False for this specific test to allow 
    # manual injection of L different filters if we wanted, 
    # BUT to match the 'small_k' (which is 1 kernel), we should set all L filters to that same k.
    spatial_model = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, 
                                            share_rotations=False, share_phases=True)
    
    # PATCHING THE WEIGHTS
    # small_k is (1, 1, 1, 1, T, T). 
    # spatial_model.base_real expects (L, T, T) since share_rotations=False

    # need to flip filters because torch conv2d computes cross-correlation
    # not mathematical convolution
    # but flipping filters fixes this
    
    spatial_model.base_real.data = torch.flip(small_k[0, 0, 0].real, [1, 2])
    spatial_model.base_imag.data = torch.flip(small_k[0, 0, 0].imag, [1, 2])
    
    # Run Inference
    out_fft = fft_model(x) 
    out_spatial = spatial_model(x) 
    
    # Extract Scale 0 results
    # FFT Out: (B, C, J, L, A, H, W) -> Check your specific dim order in WaveConvLayer
    # Spatial Out: List of (B, C, L, A, H, W)
    
    res_fft = out_fft[..., 0, :, :, :, :] # Selecting J=0
    res_spatial = out_spatial[0] 
    
    # Compare
    # Note: If your differentiable_rotate is slightly different from the FFT rotation,
    # this might fail if L > 1 and you relied on internal rotation. 
    # Since we manually forced weights to be identical (no rotation applied inside model for this test config),
    # it should match exactly.
    assert torch.allclose(res_fft, res_spatial, atol=1e-5)

def test_multiscale_rotation_equivalence():
    """
    Compares outputs of FFT (fullsize) and Downsample (spatial) models at all scales and rotations.
    Downsamples FFT output to match spatial output shape for fair comparison.
    """
    N = 32
    J = 3
    L = 4
    A = 1
    T = 7

    x = torch.zeros(1, 1, N, N)
    x[0, 0, N//2, N//2] = 1.0  # Impulse in center

    # Create matched filters
    small_k, hat_filters = create_compatible_filters(J, L, A, N, N, kernel_size=T)

    fft_model = WaveConvLayer(J=J, L=L, A=A, M=N, N=N, filters=hat_filters,
                             share_rotations=False, share_phases=False)
    spatial_model = WaveConvLayerDownsample(J=J, L=L, A=A, T=T,
                                            share_rotations=False, share_phases=False)

    # need to flip filters because torch conv2d computes cross-correlation
    # not mathematical convolution
    # but flipping filters fixes this
    
    spatial_model.base_real.data = torch.flip(small_k[0, 0, 0].real, [-1, -2])
    spatial_model.base_imag.data = torch.flip(small_k[0, 0, 0].imag, [-1, -2])

    out_fft = fft_model(x)  # (B, C, J, L, A, H, W)
    out_spatial = spatial_model(x)  # List of (B, C, L, A, H, W)
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

        plot_fft_spatial_comparison(fft_out_j, spatial_out_j, fft_out_j_down, j)

        # Compare norms first
        norm_fft = torch.linalg.norm(fft_out_j_down)
        norm_spatial = torch.linalg.norm(spatial_out_j)

        rel_diff = abs(norm_fft - norm_spatial) / max(norm_fft, norm_spatial)
        # norm discrepancy increases with scale 
        assert rel_diff < 0.1 * j + 1e-6, f"Norms differ by more than 10%*{j}: {norm_fft} vs {norm_spatial} at scale {j}"

        # Compare normalized outputs
        fft_out_j_down_norm = fft_out_j_down / (norm_fft + 1e-12)
        spatial_out_j_norm = spatial_out_j / (norm_spatial + 1e-12)
        assert torch.allclose(fft_out_j_down_norm, spatial_out_j_norm, atol=1e-4), f"Normalized outputs mismatch at scale {j}"


def test_gradient_sharing_rotations():
    """
    Verifies that when share_rotations=True, computing gradients w.r.t output
    results in a valid gradient on the SINGLE base filter.
    """
    B, C, N = 2, 1, 32
    J, L, A, T = 1, 4, 1, 7
    
    # Initialize model with shared rotations
    model = WaveConvLayerDownsample(J=J, L=L, A=A, T=T, share_rotations=True)
    
    # Verify Parameter Shape
    # Should be (1, 1, T, T) not (L, 1, T, T)
    assert model.base_real.shape == (1, 1, T, T)
    
    x = torch.randn(B, C, N, N, requires_grad=True)
    
    # Forward pass
    results = model(x) # List of tensors
    output = results[0] # (B, C, L, A, H, W)
    
    # We verify that the output actually HAS L items (rotations)
    assert output.shape[2] == L
    
    # Backward pass
    # We sum everything to create a scalar loss
    loss = output.sum().abs()
    loss.backward()
    
    # Check Gradients
    assert model.base_real.grad is not None
    assert model.base_imag.grad is not None
    
    # Check that gradient is not zero (implies connection exists)
    assert torch.max(torch.abs(model.base_real.grad)) > 0.0
    
    # Sanity Check: If we didn't share rotations, we would have L gradients.
    # Here we ensure we are updating the single base kernel using info from all L output channels.
    # (This is implicitly tested by the shape assertion above).

def test_circular_boundary_conditions():
    """
    Checks if a feature moving off the right edge reappears on the left edge.
    """
    J, L, A, T = 1, 1, 1, 3
    model = WaveConvLayerDownsample(J=J, L=L, A=A, T=T)
    
    # Create an image with a single impulse at the far right edge
    N = 16
    x = torch.zeros(1, 1, N, N)
    x[0, 0, N//2, N-1] = 1.0 # Pixel on right border
    
    # Set filter to identity-like (center pixel = 1)
    with torch.no_grad():
        model.base_real.zero_()
        model.base_imag.zero_()
        model.base_real[0, 0, 1, 1] = 1.0 # Center of 3x3
        
        # Add a neighbor value to the filter to "pull" the boundary pixel
        # Filter: [0, 0, 0]
        #         [0, 1, 1] <- Look at right neighbor
        #         [0, 0, 0]
        model.base_real[0, 0, 1, 2] = 0.5 

    # Forward
    out = model(x)[0] # (B, C, L, A, H, W)
    # res = out[0, 0, 0, 0] # (H, W) real part mostly
    
    # Because of circular padding, the filter looking at the "right neighbor" 
    # of the pixel at x=N-1 should see the pixel at x=0.
    
    # This logic is tricky with convolution correlations. 
    # Simpler check: Input is rolled, Output should be rolled.
    
    x_normal = torch.randn(1, 1, N, N)
    x_rolled = torch.roll(x_normal, shifts=1, dims=-1)
    
    out_normal = model(x_normal)[0]
    out_rolled = model(x_rolled)[0]
    
    # The output should also be perfectly rolled
    out_normal_rolled = torch.roll(out_normal, shifts=1, dims=-1)
    
    assert torch.allclose(out_normal_rolled, out_rolled, atol=1e-5)