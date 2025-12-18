import pytest
import torch
import torch.nn as nn
from wph.layers.relu_center_layer import ReluCenterLayer, ReluCenterLayerDownsample


# --- Tests for original version ---

def test_original_initialization_shapes():
    """
    Verifies that the original layer creates a single 7D mask tensor
    that broadcasts correctly.
    """
    J, M, N = 4, 32, 32
    model = ReluCenterLayer(J=J, M=M, N=N)
    
    # Expected shape: (1, 1, J, 1, 1, M, N)
    assert model.masks.shape == (1, 1, J, 1, 1, M, N)
    assert model.masks.device == torch.device('cpu') # or check model device

def test_original_mask_pyramid_logic():
    """
    Crucial Test: In the original WPH, the grid is fixed (MxN), 
    but the wavelet grows. Therefore, the mask border MUST grow 
    exponentially (border ~= 2^j // 2) to handle boundary effects.
    """
    J, M, N = 4, 32, 32
    model = ReluCenterLayer(J=J, M=M, N=N)
    masks = model.masks.squeeze() # (J, M, N)
    
    # Scale j=0: border = 2^0 // 2 = 0
    # The corner pixel (0,0) should be VALID (non-zero)
    assert masks[0, 0, 0] > 0
    
    # Scale j=3: border = 2^3 // 2 = 4
    # The pixel at (3, 3) is inside the border region -> INVALID (zero)
    assert masks[3, 3, 3] == 0.0
    
    # The pixel at (4, 4) is just inside the valid region -> VALID
    assert masks[3, 4, 4] > 0.0

def test_original_forward_shapes_and_logic():
    """
    Verifies the forward pass handles the 7D tensor input correctly.
    """
    J, M, N = 3, 16, 16
    B, C, L, A = 2, 1, 4, 2
    model = ReluCenterLayer(J=J, M=M, N=N)
    
    
    # Input: (B, C, J, L, A, M, N) - ONE tensor, not a list
    x = torch.randn(B, C, J, L, A, M, N, dtype=torch.complex64)
    
    out = model(x)
    
    # 1. Check Output Shape (Should match input exactly)
    assert out.shape == x.shape
    
    # 2. Check Output is Real (ReLU strips complex)
    assert not torch.is_complex(out)

def test_original_relu_correctness():
    """
    Verifies ReLU is applied to the Real part.
    """
    J, M, N = 1, 4, 4
    model = ReluCenterLayer(J=J, M=M, N=N)
    model.mean = lambda x: x
    model.std = lambda x: x
    
    # Construct input with specific values
    # (B=1, C=1, J=1, L=1, A=1, M=4, N=4)
    x_val = torch.zeros(1, 1, 1, 1, 1, 4, 4)
    
    # Pixel 0: Negative Real -> 0
    x_val[..., 0, 0] = -5.0
    # Pixel 1: Positive Real -> unchanged
    x_val[..., 0, 1] = 5.0
    
    # Make complex
    x_input = torch.complex(x_val, torch.randn_like(x_val))
    
    out = model(x_input)
    
    assert out[..., 0, 0].item() == 0.0
    assert out[..., 0, 1].item() == 5.0

def test_original_mask_broadcasting():
    """
    Verifies that the mask (1,1,J,1,1,M,N) correctly broadcasts 
    over batches and orientations (B,C,J,L,A,M,N).
    """
    J, M, N = 2, 8, 8
    model = ReluCenterLayer(J=J, M=M, N=N)
    model.mean = lambda x: x
    model.std = lambda x: x
    
    # Manually zero out the mask at a specific location
    # Let's target Scale 1, Pixel (4,4)
    with torch.no_grad():
        model.masks.fill_(1.0)
        model.masks[0, 0, 1, 0, 0, 4, 4] = 0.0
        
    x = torch.ones(2, 1, 2, 2, 1, 8, 8) # B=2, L=2
    
    out = model(x)
    
    # Check Batch 0, Scale 1, Orient 0, Pixel (4,4) -> Should be 0
    assert out[0, 0, 1, 0, 0, 4, 4] == 0.0
    # Check Batch 1, Scale 1, Orient 1, Pixel (4,4) -> Should also be 0 (Broadcast)
    assert out[1, 0, 1, 1, 0, 4, 4] == 0.0
    
    # Check Scale 0 (Should be 1.0)
    assert out[0, 0, 0, 0, 0, 4, 4] == 1.0

# --- Tests for downsample version ---

def test_initialization_shapes():
    J, M, N = 3, 32, 32
    model = ReluCenterLayerDownsample(J=J, M=M, N=N)
    
    # Check if masks were created with correct shrinking sizes
    assert model.get_mask(0).shape[-2:] == (32, 32)
    assert model.get_mask(1).shape[-2:] == (16, 16)
    assert model.get_mask(2).shape[-2:] == (8, 8)
    
    # Check broadcasting dimensions (1, 1, 1, 1, H, W)
    assert model.get_mask(0).ndim == 6

def test_forward_output_shapes():
    J, M, N = 3, 32, 32
    B, C, L, A = 2, 1, 4, 2
    model = ReluCenterLayerDownsample(J=J, M=M, N=N)
    
    # Create input: List of Multi-Scale Tensors
    inputs = []
    for j in range(J):
        h, w = M // 2**j, N // 2**j
        # Input is usually complex coming from WaveConv
        inp = torch.randn(B, C, L, A, h, w, dtype=torch.complex64)
        inputs.append(inp)
        
    outputs = model(inputs)
    
    assert len(outputs) == J
    
    # 1. Check Output Sizes
    for j, out in enumerate(outputs):
        h, w = M // 2**j, N // 2**j
        assert out.shape == (B, C, L, A, h, w)
        
    # 2. Check Output is Real (ReLU output)
    assert not torch.is_complex(outputs[0])

def test_relu_logic_and_masking():
    """
    Verifies that negative real parts are zeroed, 
    and the mask is applied.
    """
    J, M, N = 1, 4, 4
    model = ReluCenterLayerDownsample(J=J, M=M, N=N)
    
    # Mock Identity normalization to test ReLU values purely
    model.mean = lambda x: x # Identity
    model.std = lambda x: x  # Identity
    
    # Input: (1, 1, 1, 1, 4, 4)
    # 1st pixel: Positive Real -> Should stay
    # 2nd pixel: Negative Real -> Should become 0
    # 3rd pixel: Complex with Pos Real -> Should keep Real part
    x_val = torch.tensor([
        [1.0, -5.0, 10.0, 10.0],
        [1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0],
        [1.0,  1.0,  1.0,  1.0]
    ])
    x_complex = torch.complex(x_val, torch.zeros_like(x_val))
    # Shape it (B, C, L, A, H, W)
    x_input = [x_complex.view(1, 1, 1, 1, 4, 4)]
    
    # --- Manually set mask to verify it works ---
    # Set mask to 0 at the last pixel (3,3)
    with torch.no_grad():
        mask_0 = model.get_mask(0)
        mask_0.fill_(1.0) 
        mask_0[..., 3, 3] = 0.0
    
    out = model(x_input)[0]
    
    # Check ReLU
    assert out[..., 0, 0].item() == 1.0
    assert out[..., 0, 1].item() == 0.0 # -5 became 0
    
    # Check Mask
    assert out[..., 3, 3].item() == 0.0 # Masked out

def test_complex_handling():
    """
    Ensures the layer correctly takes the Real part of complex inputs
    before ReLU.
    """
    J, M, N = 1, 4, 4
    model = ReluCenterLayerDownsample(J=J, M=M, N=N)
    model.mean = lambda x: x
    model.std = lambda x: x
    
    # Input: Real=Positive, Imag=Negative
    # If we accidentally took abs(), Imag would contribute.
    # We want Re(x).
    x_val = torch.complex(torch.tensor([2.0]), torch.tensor([-50.0]))
    x_input = [x_val.view(1, 1, 1, 1, 1, 1).expand(-1, -1, -1, -1, 4, 4)]
    
    out = model(x_input)[0]
    
    # Should be 2.0 (Real part), not approx 50 (Modulus)
    assert out[0, 0, 0, 0, 0, 0] == 2.0

def test_mask_normalization_property():
    """
    The WPH mask is normalized such that mean(mask) * size = size.
    Essentially, the mask weights sum up to M*N (conserving energy roughly).
    """
    J, M, N = 2, 16, 16
    model = ReluCenterLayerDownsample(J=J, M=M, N=N)
    
    # Scale 0
    m0 = model.get_mask(0)
    # Logic in maskns: mask /= mask.sum(); mask *= M*N
    # So mask.sum() should be M*N
    assert torch.isclose(m0.sum(), torch.tensor(float(M*N)))
    
    # Scale 1
    m1 = model.get_mask(1)
    h1, w1 = M//2, N//2
    assert torch.isclose(m1.sum(), torch.tensor(float(h1*w1)))