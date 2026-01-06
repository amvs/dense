"""
Test for normalization layers to ensure they work per-sample, not batch-averaged.
"""
import torch
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.ops.backend import SubInitSpatialMean, DivInitStd


def test_subinit_spatial_mean_per_sample():
    """Test that SubInitSpatialMean computes mean per sample, not batch-averaged."""
    # Create module
    mean_layer = SubInitSpatialMean()
    
    # Create input with different means for each sample in batch
    batch_size = 4
    channels = 2
    height, width = 8, 8
    
    # Create samples with very different spatial means
    inputs = []
    for i in range(batch_size):
        sample = torch.randn(1, channels, height, width) + (i * 10.0)  # Different offsets
        inputs.append(sample)
    
    batch_input = torch.cat(inputs, dim=0)
    
    # Initialize with the batch
    output = mean_layer(batch_input)
    
    # Check that each sample is centered independently
    for i in range(batch_size):
        sample_output = output[i:i+1]
        # After subtracting spatial mean, the spatial mean should be close to 0
        spatial_mean = torch.mean(sample_output, dim=(-2, -1), keepdim=True)
        assert torch.allclose(spatial_mean, torch.zeros_like(spatial_mean), atol=1e-5), \
            f"Sample {i} spatial mean is not close to 0: {spatial_mean.mean().item()}"
    
    # Verify that minput has per-sample spatial means (not batch-averaged)
    # minput should have shape [batch_size, channels, 1, 1]
    expected_shape = (batch_size, channels, 1, 1)
    assert mean_layer.minput.shape == expected_shape, \
        f"Expected minput shape {expected_shape}, got {mean_layer.minput.shape}"
    
    print("✓ SubInitSpatialMean per-sample test passed")


def test_divinit_std_per_sample():
    """Test that DivInitStd computes std per sample, not batch-averaged."""
    # Create module
    std_layer = DivInitStd()
    
    # Create input with different variances for each sample in batch
    batch_size = 4
    channels = 2
    height, width = 8, 8
    
    # Create samples with very different spatial variances
    inputs = []
    for i in range(batch_size):
        scale = (i + 1) * 2.0  # Different scales
        sample = torch.randn(1, channels, height, width) * scale
        inputs.append(sample)
    
    batch_input = torch.cat(inputs, dim=0)
    
    # Initialize with the batch
    output = std_layer(batch_input)
    
    # Check that each sample is normalized independently
    for i in range(batch_size):
        sample_output = output[i:i+1]
        # After dividing by std, the spatial std should be close to 1
        # First center the sample
        centered = sample_output - torch.mean(sample_output, dim=(-2, -1), keepdim=True)
        spatial_std = torch.norm(centered, dim=(-2, -1), keepdim=True) / \
                      torch.sqrt(torch.tensor(height * width, dtype=centered.dtype))
        # Should be close to 1.0 after normalization
        assert torch.allclose(spatial_std, torch.ones_like(spatial_std), atol=0.1), \
            f"Sample {i} spatial std is not close to 1: {spatial_std.mean().item()}"
    
    # Verify that stdinput has per-sample spatial stds (not batch-averaged)
    # stdinput should have shape [batch_size, channels, 1, 1]
    expected_shape = (batch_size, channels, 1, 1)
    assert std_layer.stdinput.shape == expected_shape, \
        f"Expected stdinput shape {expected_shape}, got {std_layer.stdinput.shape}"
    
    print("✓ DivInitStd per-sample test passed")


def test_batch_independence():
    """Test that normalization is independent of batch composition."""
    # Create modules
    mean_layer1 = SubInitSpatialMean()
    mean_layer2 = SubInitSpatialMean()
    
    # Create two samples
    sample1 = torch.randn(1, 2, 8, 8) + 5.0
    sample2 = torch.randn(1, 2, 8, 8) - 3.0
    
    # Process samples together in a batch
    batch = torch.cat([sample1, sample2], dim=0)
    output_batch = mean_layer1(batch)
    
    # Process samples separately
    output1 = mean_layer2(sample1)
    # Need to reinitialize for sample2 since it will overwrite
    mean_layer3 = SubInitSpatialMean()
    output2 = mean_layer3(sample2)
    
    # The outputs for each sample should be the same whether processed
    # together or separately (within numerical tolerance)
    assert torch.allclose(output_batch[0:1], output1, atol=1e-5), \
        "Sample 1 output differs when processed in batch vs separately"
    assert torch.allclose(output_batch[1:2], output2, atol=1e-5), \
        "Sample 2 output differs when processed in batch vs separately"
    
    print("✓ Batch independence test passed")


def test_shape_check():
    """Test that shape check validates full shape including batch size."""
    mean_layer = SubInitSpatialMean()
    
    # Initialize with one batch size and shape
    input1 = torch.randn(2, 3, 8, 8)
    _ = mean_layer(input1)
    
    # Try with same shape - should NOT trigger reinit warning
    input2 = torch.randn(2, 3, 8, 8)
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mean_layer(input2)
        # Should NOT have warned
        overwrite_warnings = [x for x in w if "overwriting" in str(x.message).lower()]
        assert len(overwrite_warnings) == 0, "Should not warn when shape is the same"
    
    # Try with different batch size - should trigger reinit warning
    input3 = torch.randn(4, 3, 8, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = mean_layer(input3)
        # Should have warned since batch size changed
        overwrite_warnings = [x for x in w if "overwriting" in str(x.message).lower()]
        assert len(overwrite_warnings) > 0, "Should warn when batch size changes"
    
    print("✓ Shape check test passed")


if __name__ == "__main__":
    test_subinit_spatial_mean_per_sample()
    test_divinit_std_per_sample()
    test_batch_independence()
    print("\n✅ All normalization layer tests passed!")
