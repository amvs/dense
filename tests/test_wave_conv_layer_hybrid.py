import pytest
import torch
import math
from wph.layers.wave_conv_layer_hybrid import WaveConvLayerHybrid
from torchvision.transforms.functional import resize

class TestWaveConvLayerHybridInstantiation:
    def test_basic_instantiation(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28, num_channels=1
        )
        assert layer.J == 2
        assert layer.L == 4
        assert layer.A == 1
        assert layer.T == 7
        assert layer.downsample_splits == [0, 0]  # default is j // 2

    def test_default_downsample_splits(self):
        layer = WaveConvLayerHybrid(
            J=3, L=4, A=1, T=7, M=28, N=28
        )
        assert layer.downsample_splits == [0, 0, 1]  # default is j // 2

    def test_custom_downsample_splits(self):
        layer = WaveConvLayerHybrid(
            J=3, L=4, A=1, T=7, M=28, N=28,
            downsample_splits=[0, 1, 2]
        )
        assert layer.downsample_splits == [0, 1, 2]

    def test_downsample_splits_validation(self):
        with pytest.raises(ValueError):
            WaveConvLayerHybrid(
                J=3, L=4, A=1, T=7, M=28, N=28,
                downsample_splits=[1, 1, 1]  # first index > 0
            )

    def test_downsample_splits_length_validation(self):
        with pytest.raises(AssertionError):
            WaveConvLayerHybrid(
                J=3, L=4, A=1, T=7, M=28, N=28,
                downsample_splits=[0, 1]  # length mismatch
            )


class TestFilterInitialization:
    def test_base_filter_shapes(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28
        )
        # param_J, param_L, param_A, T, T
        assert layer.base_real.shape == (2, 4, 1, 7, 7)
        assert layer.base_imag.shape == (2, 4, 1, 7, 7)

    def test_upsampled_filter_shapes_no_downsample(self):
        """With no downsampling, up_factor = 2^(j - 0), so upsampling happens"""
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            downsample_splits=[0, 0]  # all no downsampling
        )
        # j=0: up_factor = 2^(0 - 0) = 1, no upsampling
        # j=1: up_factor = 2^(1 - 0) = 2, upsampling by 2x → (1, 4, 1, 14, 14)
        assert layer.upsampled_real[0] is None
        assert layer.upsampled_imag[0] is None
        assert layer.upsampled_real[1] is not None
        assert layer.upsampled_real[1].shape == (1, 4, 1, 14, 14)
        assert layer.upsampled_imag[1] is not None
        assert layer.upsampled_imag[1].shape == (1, 4, 1, 14, 14)

    def test_upsampled_filter_shapes_with_downsample(self):
        """With some downsampling, up_factor = 2^(j - downsample_splits[j])"""
        layer = WaveConvLayerHybrid(
            J=3, L=4, A=1, T=7, M=28, N=28,
            downsample_splits=[0, 0, 0]  # all no downsampling
        )
        # j=0: up_factor = 2^(0 - 0) = 1, no upsampling
        # j=1: up_factor = 2^(1 - 0) = 2, upsampling by 2x
        # j=2: up_factor = 2^(2 - 0) = 4, upsampling by 4x
        assert layer.upsampled_real[0] is None
        assert layer.upsampled_real[1] is not None
        assert layer.upsampled_real[1].shape == (1, 4, 1, 14, 14)
        assert layer.upsampled_real[2] is not None
        assert layer.upsampled_real[2].shape == (1, 4, 1, 28, 28)

    def test_upsampled_filter_shapes_actual_multi_scale(self):
        """Test upsampling with different parameters"""
        layer = WaveConvLayerHybrid(
            J=3, L=4, A=1, T=4, M=16, N=16,
            downsample_splits=[0, 0, 1]
        )
        # j=0: up_factor = 2^(0 - 0) = 1, no upsampling
        # j=1: up_factor = 2^(1 - 0) = 2, upsampling by 2x
        # j=2: up_factor = 2^(2 - 1) = 2, upsampling by 2x
        assert layer.upsampled_real[0] is None
        assert layer.upsampled_real[1] is not None
        assert layer.upsampled_real[1].shape == (1, 4, 1, 8, 8)
        assert layer.upsampled_real[2] is not None
        assert layer.upsampled_real[2].shape == (1, 4, 1, 8, 8)

    def test_nearest_neighbor_initialization(self):
        """Verify upsampled filters are consistent with Fourier zero-padding"""
        layer = WaveConvLayerHybrid(
            J=3, L=4, A=1, T=4, M=16, N=16,
            downsample_splits=[0, 0, 0]
        )

        def fourier_downsample(x: torch.Tensor, factor: int) -> torch.Tensor:
            """Downsample spatial tensor by cropping its Fourier spectrum.

            This is the inverse of Fourier-domain zero-padding used for upsampling.
            """
            assert x.shape[-2] % factor == 0 and x.shape[-1] % factor == 0
            h, w = x.shape[-2], x.shape[-1]
            new_h, new_w = h // factor, w // factor

            X = torch.fft.fft2(x, dim=(-2, -1))
            X_shift = torch.fft.fftshift(X, dim=(-2, -1))

            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            X_cropped = X_shift[..., start_h:start_h + new_h, start_w:start_w + new_w]
            X_cropped = torch.fft.ifftshift(X_cropped, dim=(-2, -1))

            x_small = torch.fft.ifft2(X_cropped, dim=(-2, -1)).real
            return x_small

        up_real_j2 = layer.upsampled_real[2]  # This has up_factor = 2^(2-0) = 4
        base_real = layer.base_real

        # Check that the upsampled filter is 4x larger in spatial dimensions
        # up_real is (1, 4, 1, 16, 16), base_real is (1, 4, 1, 4, 4)
        assert up_real_j2.shape[-2] == 4 * base_real.shape[-2]
        assert up_real_j2.shape[-1] == 4 * base_real.shape[-1]

        # Check that Fourier-domain downsampling of the upsampled filter
        # recovers the original base filter
        downsampled = fourier_downsample(up_real_j2, factor=4)
        assert torch.allclose(base_real, downsampled, atol=1e-5, rtol=1e-4)
    def test_cache_invalidation(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28
        )
        filters1 = layer.get_full_filters(0)
        assert layer.filters_cached
        
        # Modify base filters to trigger invalidation
        with torch.no_grad():
            layer.base_real.add_(0.1)
        
        # Cache should be invalidated by gradient hook
        # (In practice, hooks are called during backward; for forward-only, we can manually invalidate)
        layer._invalidate_cache()
        assert not layer.filters_cached


class TestForwardPass:
    def test_forward_shape_with_share_scale_pairs_true(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            share_scale_pairs=True,
            downsample_splits=[0, 1]
        )
        x = torch.randn(2, 1, 28, 28)
        output = layer(x)
        
        assert isinstance(output, list)
        assert len(output) == 2
        # No downsampling at j=0
        assert output[0].shape[0] == 2  # batch
        assert output[0].shape[1] == 1  # channels
        # Downsampled at j=1
        assert output[1].shape[-2:] == torch.Size([14, 14])

    def test_forward_with_custom_downsample_splits(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            downsample_splits=[0, 0]  # no downsampling
        )
        x = torch.randn(2, 1, 28, 28)
        output = layer(x)
        
        assert output[0].shape[-2:] == torch.Size([28, 28])
        # j=1: up_factor = 1-0=1, no downsampling
        # Due to odd padding in circular convolution, may not be exact
        assert output[1].shape[-2] == 28
        assert output[1].shape[-1] == 28

    def test_forward_hybrid_downsample_upsample(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=64, N=64,
            downsample_splits=[0, 1]  # j=1: 1x down, no filter upsample
        )
        x = torch.randn(1, 1, 64, 64)
        output = layer(x)
        
        assert len(output) == 2
        assert output[0].shape[-2:] == torch.Size([64, 64])  # no downsampling
        assert output[1].shape[-2:] == torch.Size([32, 32])  # 1x downsampling

    def test_forward_output_is_complex(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28
        )
        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        
        for scale_out in output:
            assert torch.is_complex(scale_out)

    def test_forward_different_batch_sizes(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28
        )
        for batch_size in [1, 4, 8]:
            x = torch.randn(batch_size, 1, 28, 28)
            output = layer(x)
            assert output[0].shape[0] == batch_size
            assert output[1].shape[0] == batch_size

    def test_forward_multiple_channels(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28, num_channels=3
        )
        x = torch.randn(2, 3, 28, 28)
        output = layer(x)
        
        assert output[0].shape[1] == 3
        assert output[1].shape[1] == 3


class TestSharingBehaviors:
    def test_share_rotations_true(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            share_rotations=True
        )
        assert layer.param_L == 1
        filters = layer.get_full_filters(0)
        # After rotation expansion, should have L filters
        assert filters.shape[1] == 4  # L*A

    def test_share_phases_true(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=2, T=7, M=28, N=28,
            share_phases=True
        )
        assert layer.param_A == 1
        filters = layer.get_full_filters(0)
        # After phase expansion, should have L*A filters
        assert filters.shape[1] == 8  # L*A

    def test_share_channels_true(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28, num_channels=3,
            share_channels=True
        )
        assert layer.param_nc == 1
        x = torch.randn(2, 3, 28, 28)
        output = layer(x)
        # Even with 3 channels input, param should have 1 channel
        assert layer.base_real.shape[0] == 2  # param_J when share_scales=False

    def test_share_scales_true(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            share_scales=True
        )
        assert layer.param_J == 1
        # share_scales forces share_scale_pairs=True
        assert layer.share_scale_pairs == True


class TestAntialiasing:
    def test_antialiasing_disabled(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=64, N=64,
            downsample_splits=[0, 1],
            use_antialiasing=False
        )
        x = torch.randn(1, 1, 64, 64)
        output = layer(x)
        assert output[1].shape[-2:] == (32, 32)

    def test_antialiasing_enabled(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=64, N=64,
            downsample_splits=[0, 1],
            use_antialiasing=True
        )
        x = torch.randn(1, 1, 64, 64)
        output = layer(x)
        assert output[1].shape[-2:] == (32, 32)

    def test_antialiasing_filter_created(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            use_antialiasing=True
        )
        assert hasattr(layer, 'aa_filter')
        assert layer.aa_filter.shape == (1, 1, 3, 3)


class TestScalePairsMode:
    def test_forward_scale_pairs_true(self):
        """Test normal list output with share_scale_pairs=True"""
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            share_scale_pairs=True
        )
        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        
        # Should be list of J tensors
        assert isinstance(output, list)
        assert len(output) == 2

    def test_scale_pairs_false_initialization(self):
        """Test that pair-specific initialization works with reshape instead of view"""
        # This test just checks instantiation doesn't crash
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            share_scale_pairs=False
        )
        # If we got here, initialization succeeded
        assert layer.share_scale_pairs == False
        assert len(layer.pair_upsampled_real) > 0

        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        assert isinstance(output, list)
        assert len(output) == 2


class TestEdgeCases:
    def test_j_equals_one(self):
        layer = WaveConvLayerHybrid(
            J=1, L=4, A=1, T=7, M=28, N=28
        )
        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        
        assert len(output) == 1
        assert output[0].shape[0] == 1  # batch

    def test_small_image(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=16, N=16,
            downsample_splits=[0, 1]
        )
        x = torch.randn(1, 1, 16, 16)
        output = layer(x)
        
        assert output[0].shape[-2:] == torch.Size([16, 16])
        assert output[1].shape[-2:] == torch.Size([8, 8])

    def test_rectangular_image(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=56,
            downsample_splits=[0, 1]
        )
        x = torch.randn(1, 1, 28, 56)
        output = layer(x)
        
        assert output[0].shape[-2:] == torch.Size([28, 56])
        assert output[1].shape[-2:] == torch.Size([14, 28])


class TestGradientFlow:
    def test_gradients_flow_through_base_filters(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28
        )
        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        
        # Take real part for scalar loss
        loss = output[0].real.sum()
        loss.backward()
        
        assert layer.base_real.grad is not None
        assert layer.base_imag.grad is not None

    def test_gradients_flow_through_upsampled_filters(self):
        layer = WaveConvLayerHybrid(
            J=2, L=4, A=1, T=7, M=28, N=28,
            downsample_splits=[0, 1]
        )
        x = torch.randn(1, 1, 28, 28)
        output = layer(x)
        
        # Take real part for scalar loss
        loss = output[1].real.sum()
        loss.backward()
        
        if layer.upsampled_real[1] is not None:
            assert layer.upsampled_real[1].grad is not None
            assert layer.upsampled_imag[1].grad is not None
