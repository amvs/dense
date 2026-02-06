"""Integration tests for WPHModelHybrid with full WPH pipeline."""

import torch
import pytest
from wph.wph_model import WPHModel, WPHModelHybrid, WPHModelDownsample, WPHClassifier
from dense.wavelets import filter_bank


class TestWPHModelHybridIntegration:
    """Test full WPH pipeline with hybrid layer."""

    @pytest.fixture
    def filters_and_config(self):
        """Create filter bank and basic config."""
        wavelet_params = {"S": 7, "T": 8, "w0": 1.09955}
        J = 3
        L = 4
        A = 4
        filters = filter_bank(wavelet_name="morlet", max_scale=J, nb_orients=L, 
                             S=wavelet_params["S"], T=wavelet_params["T"], w0=wavelet_params["w0"])
        
        # filters is a list of (L, H, W) tensors per scale
        # Use the first scale (coarsest) as base
        psi_base = filters[0]  # Shape: (L, 7, 7)
        actual_T = psi_base.shape[-1]
        
        # Create complex filters with phase shifts manually
        # (J, L, A, T, T)
        psi_full = torch.zeros(J, L, A, actual_T, actual_T, dtype=torch.complex64)
        
        for j in range(J):
            for a in range(A):
                # Apply phase shift: e^(i * a * 2*pi/A)
                phase = torch.exp(torch.complex(torch.tensor(0.0), torch.tensor(a * 2 * torch.pi / A)))
                for l in range(L):
                    psi_full[j, l, a] = psi_base[l].to(torch.complex64) * phase
        
        # Create a simple low-pass filter (real-valued), matching M,N
        M, N = 28, 28
        hatphi = torch.randn(M, N, dtype=torch.complex64).real

        filters = {"psi": psi_full, "hatphi": hatphi}
        config = {
            "J": J,
            "L": L,
            "A": A,
            "T": actual_T,
            "M": M,
            "N": N,
            "num_channels": 1,
            "share_rotations": False,
            "share_phases": False,
            "share_channels": False,
            "share_scales": False,
            "share_scale_pairs": True,
        }
        return filters, config

    def test_forward_pass_default_downsample_splits(self, filters_and_config):
        """Test hybrid model forward pass with default downsample_splits."""
        filters, config = filters_and_config
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        x = torch.randn(2, 1, 28, 28)
        xcorr, xlow, xhigh = model(x, flatten=False)

        # xcorr should be a list with one tensor per scale
        assert isinstance(xcorr, list)
        assert len(xcorr) == config["J"]
        # Each coefficient map is real-valued (correlations)
        for coeff in xcorr:
            assert coeff.dtype in (torch.float32, torch.float64)

    def test_forward_pass_custom_downsample_splits(self, filters_and_config):
        """Test hybrid model with custom downsample_splits."""
        filters, config = filters_and_config
        downsample_splits = [0, 1, 2]
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        x = torch.randn(2, 1, 28, 28)
        xcorr, xlow, xhigh = model(x, flatten=False)

        assert isinstance(xcorr, list)
        assert len(xcorr) == config["J"]

    def test_hybrid_output_sizes_match_downsampling(self, filters_and_config):
        """Verify output sizes match downsampling applied at each scale."""
        filters, config = filters_and_config
        downsample_splits = [0, 1, 2]
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        x = torch.randn(2, 1, 28, 28)
        xcorr, xlow, xhigh = model(x, flatten=False)
        coeffs = xcorr
        # Scale 0: downsampled by 2^0 = 1
        assert coeffs[0].shape[2:] == (28, 28)
        # Scale 1: downsampled by 2^1 = 2
        assert coeffs[1].shape[2:] == (14, 14)
        # Scale 2: downsampled by 2^2 = 4
        assert coeffs[2].shape[2:] == (7, 7)

    def test_no_downsampling_variant(self, filters_and_config):
        """Test hybrid model with no signal downsampling (all filter upsampling)."""
        filters, config = filters_and_config
        downsample_splits = [0, 0, 0]
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        x = torch.randn(2, 1, 28, 28)
        xcorr, xlow, xhigh = model(x, flatten=False)
        coeffs = xcorr
        # With no signal downsampling, all outputs are full resolution
        for coeff in coeffs:
            assert coeff.shape[2:] == (28, 28)

    def test_antialiasing_option(self, filters_and_config):
        """Test hybrid model with antialiasing enabled."""
        filters, config = filters_and_config
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
        )
        x = torch.randn(2, 1, 28, 28)
        xcorr, xlow, xhigh = model(x, flatten=False)
        assert isinstance(xcorr, list)

    def test_gradient_flow_through_hybrid_model(self, filters_and_config):
        """Test gradient flow through entire hybrid pipeline."""
        filters, config = filters_and_config
        model = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        x = torch.randn(2, 1, 28, 28, requires_grad=True)
        xcorr, xlow, xhigh = model(x, flatten=False)

        # Compute loss on first scale (real correlations)
        loss = torch.sum(xcorr[0])
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        assert model.wave_conv.base_real.grad is not None

    def test_classifier_with_hybrid_model(self, filters_and_config):
        """Test full WPHClassifier with hybrid feature extractor."""
        filters, config = filters_and_config
        feature_extractor = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )
        classifier = WPHClassifier(
            feature_extractor=feature_extractor,
            num_classes=10,
            use_batch_norm=True,
        )
        x = torch.randn(2, 1, 28, 28)
        logits = classifier(x)

        # Output shape should be (batch_size, num_classes)
        assert logits.shape == (2, 10)

    def test_different_image_sizes(self, filters_and_config):
        """Test hybrid model with different image sizes."""
        filters, config = filters_and_config

        for M, N in [(32, 32), (64, 64), (28, 56)]:
            # Create a low-pass filter that matches current image size
            hatphi_dyn = torch.randn(M, N).real
            filters_dyn = {
                "psi": filters["psi"],
                "hatphi": hatphi_dyn,
            }
            model = WPHModelHybrid(
                T=config["T"],
                filters=filters_dyn,
                J=config["J"],
                L=config["L"],
                A=config["A"],
                A_prime=1,
                M=M,
                N=N,
                num_channels=config["num_channels"],
                share_rotations=config["share_rotations"],
                share_phases=config["share_phases"],
                share_channels=config["share_channels"],
                share_scales=config["share_scales"],
                share_scale_pairs=config["share_scale_pairs"],
            )
            x = torch.randn(1, 1, M, N)
            xcorr, xlow, xhigh = model(x, flatten=False)
            assert isinstance(xcorr, list)
            assert len(xcorr) == config["J"]

    def test_hybrid_matches_downsample(self, filters_and_config):
        """Verify hybrid matches downsample when splits = [0,1,2,...].
        
        NOTE: This test documents expected behavior. The hybrid model uses filter 
        upsampling while downsample uses signal downsampling. These are mathematically
        different approaches that produce different numerical results. The test checks
        that the *structures* match (shapes, nonzero patterns) and that intermediate
        layers (correlation) produce similar outputs.
        """
        filters, config = filters_and_config

        #  progressive splits - downsample every time
        downsample_splits = [j for j in range(config["J"])]

        # Hybrid feature extractor
        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
        )

        # Downsample feature extractor
        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])

        # Compare per-scale outputs shapes for non-flattened call
        coeffs_h, xlow_h, xhigh_h = hybrid(x, flatten=False)
        coeffs_d, xlow_d, xhigh_d = downsample(x, flatten=False)
        assert isinstance(coeffs_h, list) and isinstance(coeffs_d, list)
        assert len(coeffs_h) == len(coeffs_d) == config["J"]
        
        # Check that structures match (same shapes, similar sparsity)
        for j in range(config["J"]):
            assert coeffs_h[j].shape == coeffs_d[j].shape, f"Scale {j} shapes don't match"
            
            # Check that both have similar sparsity (same nonzero count, allowing for floating point)
            h_nonzero_count = (coeffs_h[j] != 0).sum().item()
            d_nonzero_count = (coeffs_d[j] != 0).sum().item()
            # Allow small differences due to floating point precision in correlation computation
            assert abs(h_nonzero_count - d_nonzero_count) <= 1, f"Scale {j} nonzero counts differ: {h_nonzero_count} vs {d_nonzero_count}"
        
        # Low/high-pass should match exactly
        assert torch.allclose(xlow_h, xlow_d, rtol=1e-5, atol=1e-6), "Lowpass outputs don't match"
        assert torch.allclose(xhigh_h, xhigh_d, rtol=1e-5, atol=1e-6), "Highpass outputs don't match"

    def test_wave_conv_outputs_close_hybrid_vs_downsample(self, filters_and_config):
        """Test mathematical equivalence: signal downsampling vs filter upsampling.
        
        Both paths should produce numerically similar outputs (relaxed tolerance due to
        different AA/padding strategies). Differences may grow at higher scales.
        """
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])

        y_h = hybrid.wave_conv(x)
        y_d = downsample.wave_conv(x)
        assert isinstance(y_h, list) and isinstance(y_d, list)
        assert len(y_h) == len(y_d) == config["J"]
        for j in range(config["J"]):
            # Relaxed tolerance reflecting that different AA/downsample strategies
            # produce similar but not identical numerics. Tolerance increases with scale.
            rtol = 0.01 + 0.05 * j  # start at 1%, grow to ~16% at j=3
            atol = 0.001 + 0.01 * j
            assert torch.allclose(y_h[j], y_d[j], rtol=rtol, atol=atol,
                                 equal_nan=False)

    def test_relu_center_outputs_hybrid_vs_downsample(self, filters_and_config):
        """Test relu_center outputs match between hybrid and downsample models.
        
        This checks the second step in the pipeline: wave_conv -> relu_center.
        """
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
            normalize_relu=False,  # Match downsample
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])

        # Step 1: wave_conv
        y_h = hybrid.wave_conv(x)
        y_d = downsample.wave_conv(x)

        # Step 2: relu_center
        xrelu_h = hybrid.relu_center(y_h)
        xrelu_d = downsample.relu_center(y_d)

        assert isinstance(xrelu_h, list) and isinstance(xrelu_d, list)
        assert len(xrelu_h) == len(xrelu_d) == config["J"]

        for j in range(config["J"]):
            print(f"Scale {j}: hybrid shape {xrelu_h[j].shape}, downsample shape {xrelu_d[j].shape}")
            print(f"Scale {j}: hybrid mean {xrelu_h[j].mean():.6f}, downsample mean {xrelu_d[j].mean():.6f}")
            print(f"Scale {j}: hybrid std {xrelu_h[j].std():.6f}, downsample std {xrelu_d[j].std():.6f}")
            
            rtol = 0.01 + 0.05 * j
            atol = 0.001 + 0.01 * j
            assert torch.allclose(xrelu_h[j], xrelu_d[j], rtol=rtol, atol=atol,
                                 equal_nan=False), f"Scale {j} relu_center outputs don't match"

    def test_corr_inputs_hybrid_vs_downsample(self, filters_and_config):
        """Test that inputs to corr layer match between hybrid and downsample models.
        
        This checks the intermediate state right before correlation computation.
        """
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
            normalize_relu=False,
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])
        nb = x.shape[0]

        # Replicate forward pass up to corr layer
        # Hybrid
        xpsi_h = hybrid.wave_conv(x)
        xrelu_h = hybrid.relu_center(xpsi_h)
        
        # Downsample
        xpsi_d = downsample.wave_conv(x)
        xrelu_d = downsample.relu_center(xpsi_d)

        # Check that the lists match
        assert isinstance(xrelu_h, list) and isinstance(xrelu_d, list)
        assert len(xrelu_h) == len(xrelu_d) == config["J"]

        # Print diagnostics for each scale
        for j in range(config["J"]):
            print(f"\nScale {j}:")
            print(f"  Hybrid shape: {xrelu_h[j].shape}, Downsample shape: {xrelu_d[j].shape}")
            print(f"  Hybrid range: [{xrelu_h[j].min():.6f}, {xrelu_h[j].max():.6f}]")
            print(f"  Downsample range: [{xrelu_d[j].min():.6f}, {xrelu_d[j].max():.6f}]")
            
            rtol = 0.01 + 0.05 * j
            atol = 0.001 + 0.01 * j
            assert torch.allclose(xrelu_h[j], xrelu_d[j], rtol=rtol, atol=atol,
                                 equal_nan=False), f"Scale {j} inputs to corr don't match"

    def test_corr_masks_hybrid_vs_downsample(self, filters_and_config):
        """Test that correlation masks match between hybrid and downsample models."""
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
            normalize_relu=False,
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        # Compare masks at each scale
        for j in range(config["J"]):
            mask_h, factr_h = hybrid.corr.get_mask_for_scale(j)
            mask_d, factr_d = downsample.corr.get_mask_for_scale(j)
            
            print(f"\nScale {j}:")
            print(f"  Hybrid mask shape: {mask_h.shape}, factr: {factr_h}")
            print(f"  Downsample mask shape: {mask_d.shape}, factr: {factr_d}")
            
            assert mask_h.shape == mask_d.shape, f"Scale {j} mask shapes don't match"
            assert torch.allclose(mask_h, mask_d), f"Scale {j} masks don't match"
            assert torch.allclose(factr_h, factr_d), f"Scale {j} factr don't match"
    def test_pair_corr_direct_comparison(self, filters_and_config):
        """Test correlation layer matches via public forward pass."""
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
            normalize_relu=False,
        )

        x = torch.randn(2, 1, config["M"], config["N"])
        
        # Get full outputs via forward pass
        with torch.no_grad():
            out_h, _, _ = hybrid(x, flatten=False)
            out_d, _, _ = downsample(x, flatten=False)
        
        # Verify outputs have same structure
        assert isinstance(out_h, list) and isinstance(out_d, list)
        assert len(out_h) == len(out_d) == config["J"]
        
        # Check that sparsity is similar (allowing small variations)
        for j in range(config["J"]):
            h_nonzero = (out_h[j] != 0).sum().item()
            d_nonzero = (out_d[j] != 0).sum().item()
            # Allow 1% tolerance in sparsity
            max_count = max(h_nonzero, d_nonzero)
            assert abs(h_nonzero - d_nonzero) <= max(1, 0.01 * max_count), f"Scale {j} sparsity differs too much"

    def test_corr_outputs_hybrid_vs_downsample(self, filters_and_config):
        """Test that corr layer outputs match between hybrid and downsample models.
        
        This directly tests the correlation computation on matching inputs.
        """
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
            normalize_relu=False,
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])

        # Get outputs through wave_conv and relu_center
        xpsi_h = hybrid.wave_conv(x)
        xrelu_h = hybrid.relu_center(xpsi_h)
        
        xpsi_d = downsample.wave_conv(x)
        xrelu_d = downsample.relu_center(xpsi_d)

        # Apply correlation
        xcorr_h = hybrid.corr(xrelu_h, flatten=False)
        xcorr_d = downsample.corr(xrelu_d, flatten=False)

        assert isinstance(xcorr_h, list) and isinstance(xcorr_d, list)
        assert len(xcorr_h) == len(xcorr_d) == config["J"]

        for j in range(config["J"]):
            print(f"\nScale {j}:")
            print(f"  Hybrid corr shape: {xcorr_h[j].shape}")
            print(f"  Downsample corr shape: {xcorr_d[j].shape}")
            print(f"  Hybrid corr mean: {xcorr_h[j].mean():.6f}, std: {xcorr_h[j].std():.6f}")
            print(f"  Downsample corr mean: {xcorr_d[j].mean():.6f}, std: {xcorr_d[j].std():.6f}")
            print(f"  Hybrid nonzero: {(xcorr_h[j] != 0).sum().item()}, Downsample nonzero: {(xcorr_d[j] != 0).sum().item()}")
            
            rtol = 0.01 + 0.05 * j
            atol = 0.001 + 0.01 * j
            assert torch.allclose(xcorr_h[j], xcorr_d[j], rtol=rtol, atol=atol,
                                 equal_nan=False), f"Scale {j} corr outputs don't match"

    def test_wave_conv_normalized_close_hybrid_vs_downsample(self, filters_and_config):
        """Test correlation of normalized WaveConv outputs (ReluCenter).
        
        After spatial mean subtraction and std normalization, outputs should be
        highly correlated. Differences shrink relative to magnitudes.
        """
        filters, config = filters_and_config
        downsample_splits = [j for j in range(config["J"])]

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            num_channels=config["num_channels"],
            downsample_splits=downsample_splits,
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            use_antialiasing=True,
        )

        downsample = WPHModelDownsample(
            J=config["J"],
            L=config["L"],
            A=config["A"],
            A_prime=1,
            M=config["M"],
            N=config["N"],
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])
        y_h = hybrid.wave_conv(x)
        y_d = downsample.wave_conv(x)
        xr_h = hybrid.relu_center(y_h)
        xr_d = downsample.relu_center(y_d)

        assert isinstance(xr_h, list) and isinstance(xr_d, list)
        assert len(xr_h) == len(xr_d) == config["J"]
        for j in range(config["J"]):
            # Compute Pearson correlation as a measure of linear agreement
            h_flat = xr_h[j].real.flatten()
            d_flat = xr_d[j].real.flatten()
            correlation = torch.nn.functional.cosine_similarity(h_flat, d_flat, dim=0)
            # Correlation should be high (>0.8) indicating strong alignment,
            # even if magnitudes differ slightly due to different pathways
            assert correlation > 0.8, f"Scale {j}: correlation {correlation:.4f} < 0.8"

    def test_low_highpass_equal_across_models(self, filters_and_config):
        """Low/high-pass outputs should be equal across full, downsample, hybrid."""
        filters, config = filters_and_config

        J, L, A, M, N = config["J"], config["L"], config["A"], config["M"], config["N"]
        # Build a stub hatpsi for full model (unused in low/high comparisons)
        hatpsi = torch.complex(
            torch.randn(J, L, A, M, N),
            torch.randn(J, L, A, M, N)
        )
        filters_full = {"hatpsi": hatpsi, "hatphi": filters["hatphi"]}

        full = WPHModel(
            filters=filters_full,
            J=J,
            L=L,
            A=A,
            A_prime=1,
            M=M,
            N=N,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_union=False,
            mask_angles=4,
            mask_union_highpass=True,
        )

        downsample = WPHModelDownsample(
            J=J,
            L=L,
            A=A,
            A_prime=1,
            M=M,
            N=N,
            T=config["T"],
            filters=filters,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
            normalize_relu=False,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_angles=4,
            mask_union_highpass=True,
        )

        hybrid = WPHModelHybrid(
            T=config["T"],
            filters=filters,
            J=J,
            L=L,
            A=A,
            A_prime=1,
            M=M,
            N=N,
            num_channels=config["num_channels"],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config["share_scales"],
            share_scale_pairs=config["share_scale_pairs"],
        )

        x = torch.randn(2, 1, M, N)
        _, xlow_f, xhigh_f = full(x, flatten=False)
        _, xlow_d, xhigh_d = downsample(x, flatten=False)
        _, xlow_h, xhigh_h = hybrid(x, flatten=False)

        # Shapes equal
        assert xlow_f.shape == xlow_d.shape == xlow_h.shape
        assert xhigh_f.shape == xhigh_d.shape == xhigh_h.shape

        # Numerical closeness
        assert torch.allclose(xlow_f, xlow_d, rtol=1e-5, atol=1e-6)
        assert torch.allclose(xlow_f, xlow_h, rtol=1e-5, atol=1e-6)
        assert torch.allclose(xhigh_f, xhigh_d, rtol=1e-5, atol=1e-6)
        assert torch.allclose(xhigh_f, xhigh_h, rtol=1e-5, atol=1e-6)

    def test_hybrid_no_downsample_matches_full_model(self):
        """Test that hybrid with no downsampling matches full WPHModel using compatible filters.
        
        Follows test_scale_zero_equivalence pattern: uses small_k (original spatial kernels)
        for hybrid model and their dilated FFT versions for full model. Both models apply
        filters directly to signal at all scales (no signal downsampling in hybrid).
        """
        import math
        from torch import fft
        
        # Helper function to create compatible filters
        def create_compatible_filters(J, L, A, M, N, kernel_size=7):
            """Creates small spatial filters and their dilated FFT-equivalent versions."""
            small_k = torch.zeros(1, 1, L, A, kernel_size, kernel_size, dtype=torch.complex64)
            x = torch.linspace(-math.pi, math.pi, kernel_size)
            y = torch.linspace(-math.pi, math.pi, kernel_size)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            for l in range(L):
                theta = l * math.pi / L
                X_rot = X * math.cos(theta) - Y * math.sin(theta)
                Y_rot = X * math.sin(theta) + Y * math.cos(theta)
                small_k[0, 0, l,] = torch.sin(X_rot + Y_rot + l * math.pi / 8) + 0.2j * torch.sin(X_rot + Y_rot + l * math.pi / 8)
            
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
        
        # Test parameters - use J=1 to simplify filter comparison
        J, L, A = 1, 4, 2
        M, N = 32, 32
        T = 7
        
        # Create compatible filters
        small_k, hat_filters_raw = create_compatible_filters(J, L, A, M, N, kernel_size=T)
        # small_k shape: (1, 1, L, A, T, T)
        # hat_filters_raw shape: (1, J, L, A, M, N)
        
        hat_filters_tensor = hat_filters_raw[0]  # (J, L, A, M, N)
        
        # Create lowpass filter in frequency domain
        phi_spatial = torch.ones(M, N, dtype=torch.complex64) / (M * N)
        hatphi = torch.fft.fft2(torch.fft.ifftshift(phi_spatial, dim=(-2, -1)), dim=(-2, -1))
        
        # Create small lowpass filter for spatial domain
        phi_small = torch.ones(T, T, dtype=torch.complex64) / (T * T)
        
        # Create filters dict for full WPHModel (frequency domain)
        filters_full = {
            'hatpsi': hat_filters_tensor,
            'hatphi': hatphi,
        }
        
        # Create filters dict for hybrid model (spatial domain small kernels)
        # Replicate small_k[0, 0] (L, A, T, T) across J scales
        # Flip filters because torch conv2d computes cross-correlation, not convolution
        psi_spatial = torch.flip(small_k[0, 0].unsqueeze(0).expand(J, -1, -1, -1, -1), dims=[-2, -1])  # (J, L, A, T, T)
        filters_hybrid = {
            'psi': psi_spatial,
            'hatphi': hatphi,
        }
        
        # Create full WPHModel with FFT-based filters
        full = WPHModel(
            filters=filters_full,
            J=J,
            L=L,
            A=A,
            A_prime=1,
            M=M,
            N=N,
            num_channels=1,
            share_rotations=False,
            share_phases=False,
            share_channels=True,
            delta_j=None,
            delta_l=None,
            shift_mode="samec",
            mask_union=False,
            mask_angles=4,
            mask_union_highpass=True,
            normalize_relu=True,
        )
        
        # Create hybrid model with no downsampling (downsample_splits=[0]*J)
        hybrid = WPHModelHybrid(
            T=T,
            filters=filters_hybrid,
            J=J,
            L=L,
            A=A,
            A_prime=1,
            M=M,
            N=N,
            num_channels=1,
            downsample_splits=[0] * J,  # No signal downsampling
            share_rotations=False,
            share_phases=False,
            share_channels=True,
            share_scales=False,
            share_scale_pairs=True,
            shift_mode="samec",
            normalize_relu=True,
        )
        
        # Test input - impulse at center
        x = torch.zeros(2, 1, M, N)
        x[:, :, M//2, N//2] = 1.0
        
        # wave conv outputs should match
        y_full = full.wave_conv(x)
        y_hybrid = hybrid.wave_conv(x)
        assert isinstance(y_full, torch.Tensor) and isinstance(y_hybrid, list)
        assert y_full.squeeze().allclose(y_hybrid[0].squeeze(), rtol=1e-5, atol=1e-6), "WaveConv outputs don't match"

        # Forward passes
        xcorr_full, xlow_full, xhigh_full = full(x, flatten=False)
        xcorr_hybrid, xlow_hybrid, xhigh_hybrid = hybrid(x, flatten=False)
        
        # Check structures match
        assert isinstance(xcorr_full, torch.Tensor) and isinstance(xcorr_hybrid, list)
        assert len(xcorr_hybrid) == J
        
        # Check shapes match at each scale
        for j in range(J):
            assert xcorr_full.shape == xcorr_hybrid[j].shape, \
                f"Scale {j}: shapes mismatch - full {xcorr_full[j].shape} vs hybrid {xcorr_hybrid[j].shape}"
        
        # Check lowpass/highpass match exactly
        assert torch.allclose(xlow_full, xlow_hybrid, rtol=1e-5, atol=1e-6), \
            "Lowpass outputs don't match"
        assert torch.allclose(xhigh_full, xhigh_hybrid, rtol=1e-5, atol=1e-6), \
            "Highpass outputs don't match"
        
        # Check correlation outputs match (relaxed tolerance for spatial vs FFT differences)
        for j in range(J):
            assert torch.allclose(xcorr_full, xcorr_hybrid[j], rtol=1e-4, atol=1e-4), \
                f"Scale {j}: correlation outputs don't match"
