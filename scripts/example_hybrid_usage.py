"""
Example usage of WPHModelHybrid for wavelet-based image classification.

This script demonstrates how to use the hybrid wavelet convolution layer,
which combines signal downsampling and filter upsampling for efficient
multi-scale feature extraction.
"""

import torch
import torch.nn as nn
from wph.wph_model import WPHModelHybrid, WPHClassifier
from dense.wavelets import filter_bank


def example_basic_usage():
    """
    Basic usage: Create a hybrid WPH model and perform forward pass.
    
    The hybrid layer mixes signal downsampling and filter upsampling at each scale.
    Key concepts:
    - downsample_splits[j]: Number of signal downsampling steps at scale j
    - upsample_factor = 2^(j - downsample_splits[j]): Remaining scale is filter upsampling
    - Default: downsample_splits[j] = j // 2 balances downsampling and upsampling
    """
    # Configure wavelet parameters
    J = 3  # Max scale (number of octaves)
    L = 4  # Number of orientations
    A = 4  # Number of phases
    T = 8  # Filter spatial size (T x T)
    M, N = 28, 28  # Image dimensions (28x28 like MNIST)
    
    # Generate filter bank (Morlet wavelets)
    wavelet_params = {"S": 7, "T": T, "w0": 1.09955}
    filters = filter_bank(
        J=J, L=L, A=A,
        S=wavelet_params["S"],
        T=T,
        w0=wavelet_params["w0"]
    )
    
    # Create hybrid WPH model with default downsample_splits
    model = WPHModelHybrid(
        T=T,
        filters=filters,
        J=J,
        L=L,
        A=A,
        M=M,
        N=N,
        num_channels=1,
        share_rotations=False,
        share_phases=False,
        share_channels=True,
        share_scales=False,
        share_scale_pairs=True,
        # downsample_splits defaults to [0, 0, 1] for J=3 (j//2 for each j)
    )
    
    # Forward pass on a batch of images
    batch_size = 4
    x = torch.randn(batch_size, 1, M, N)
    output = model(x)
    
    # Output structure
    print(f"Output keys: {output.keys()}")
    # 'coeffs_psi': list of complex tensors, one per scale
    # 'coeffs_phi': lowpass coefficients
    
    coeffs = output['coeffs_psi']
    print(f"Number of scales: {len(coeffs)}")
    for j, coeff in enumerate(coeffs):
        print(f"  Scale {j}: {coeff.shape} (spatial size {coeff.shape[2:]})")


def example_custom_downsample_splits():
    """
    Custom downsampling strategy: Explicitly set downsample_splits.
    For entry j of downsampling list, must choose an int between 0 and j.
    
    Different split strategies:
    1. [0, 0, 0]: No signal downsampling (all via upsampled filters)
       - Keeps full resolution at all scales
       - Uses larger upsampled filters for coarse scales
    
    2. [0, 0, 1]: Balanced (default-like)
       - Mild downsampling of signal, mild upsampling of filters
    
    3. [0, 1, 2] for J=3: Progressive (no downsample at j=0, max at j=2)
       - More aggressive compression at coarse scales
    """
    J, L, A, T, M, N = 3, 4, 4, 8, 28, 28
    filters = filter_bank(J=J, L=L, A=A, S=7, T=T, w0=1.09955)
    
    # Strategy 1: All filter upsampling (no signal downsampling)
    print("\n=== Strategy 1: No Signal Downsampling ===")
    model1 = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=1,
        downsample_splits=[0, 0, 0],  # All scales: j - 0 = j
        share_channels=True
    )
    x = torch.randn(1, 1, M, N)
    output1 = model1(x)
    for j, coeff in enumerate(output1['coeffs_psi']):
        print(f"  Scale {j}: {coeff.shape[2:]} (full resolution)")
    
    # Strategy 2: Progressive downsampling
    print("\n=== Strategy 2: Progressive Downsampling ===")
    model2 = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=1,
        downsample_splits=[0, 1, 2],  # Signal downsampled by [1, 2, 4]
        share_channels=True
    )
    output2 = model2(x)
    for j, coeff in enumerate(output2['coeffs_psi']):
        print(f"  Scale {j}: {coeff.shape[2:]} (downsampled by 2^{j})")
    
   
def example_with_antialiasing():
    """
    Enable antialiasing Gaussian blur before downsampling.
    
    Antialiasing prevents aliasing artifacts when downsampling the signal.
    Uses a 3x3 Gaussian kernel with sigma=0.8 before each 2x decimation step.
    """
    J, L, A, T, M, N = 3, 4, 4, 8, 28, 28
    filters = filter_bank(J=J, L=L, A=A, S=7, T=T, w0=1.09955)
    
    model = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=1,
        downsample_splits=[0, 1, 2],
        share_channels=True,
        use_antialiasing=True,  # Enable antialiasing
    )
    
    x = torch.randn(4, 1, M, N)
    output = model(x)
    print("Forward pass with antialiasing completed")
    print(f"Output shape scale 0: {output['coeffs_psi'][0].shape}")


def example_training_integration():
    """
    Integration with PyTorch training loop.
    
    Shows how to use hybrid model as feature extractor in a classifier.
    """
    J, L, A, T, M, N = 3, 4, 4, 8, 28, 28
    num_classes = 10
    
    filters = filter_bank(J=J, L=L, A=A, S=7, T=T, w0=1.09955)
    
    # Feature extractor (hybrid WPH model)
    feature_extractor = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=1,
        downsample_splits=[0, 1, 2],
        share_channels=True,
    )
    
    # Classifier on top of features
    classifier = WPHClassifier(
        feature_extractor=feature_extractor,
        num_classes=num_classes,
        use_batch_norm=True,
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # Dummy batch
    batch_size = 32
    x = torch.randn(batch_size, 1, M, N, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Forward pass and loss
    logits = classifier(x)
    loss = loss_fn(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions shape: {logits.shape}")
    print("Training step completed")


def example_parameter_sharing():
    """
    Different parameter sharing strategies for memory efficiency.
    
    - share_channels: Use same filters for all input channels
    - share_rotations: Share filter coefficients across orientations
    - share_phases: Share filter coefficients across phase shifts
    - share_scales: Use same filters for all scales (not recommended)
    """
    J, L, A, T, M, N = 3, 4, 4, 8, 28, 28
    filters = filter_bank(J=J, L=L, A=A, S=7, T=T, w0=1.09955)
    
    # Minimal parameters: share as much as possible
    model = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=3,
        downsample_splits=[0, 1, 2],
        share_rotations=True,   # Single rotation coefficient
        share_phases=True,      # Single phase coefficient
        share_channels=True,    # All channels share filters
        share_scales=False,     # Different filters per scale
        share_scale_pairs=True, # Paired scales share filters
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters with aggressive sharing: {total_params}")
    
    # More parameters: less sharing
    model2 = WPHModelHybrid(
        T=T, filters=filters, J=J, L=L, A=A, M=M, N=N,
        num_channels=3,
        downsample_splits=[0, 1, 2],
        share_rotations=False,
        share_phases=False,
        share_channels=False,
        share_scales=False,
        share_scale_pairs=True,
    )
    
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"Total parameters with minimal sharing: {total_params2}")


if __name__ == "__main__":
    print("=" * 70)
    print("WPHModelHybrid Examples")
    print("=" * 70)
    
    example_basic_usage()
    example_custom_downsample_splits()
    example_with_antialiasing()
    example_training_integration()
    example_parameter_sharing()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
