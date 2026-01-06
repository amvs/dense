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

    def test_hybrid_matches_downsample_default_splits(self, filters_and_config):
        """Verify hybrid matches downsample when splits = [0,1,2,...]."""
        filters, config = filters_and_config

        # Default progressive splits
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
            shift_mode="default",
            mask_angles=4,
            mask_union_highpass=True,
        )

        x = torch.randn(2, 1, config["M"], config["N"])

        # Compare per-scale outputs shapes for non-flattened call
        coeffs_h, xlow_h, xhigh_h = hybrid(x, flatten=False)
        coeffs_d, xlow_d, xhigh_d = downsample(x, flatten=False)
        assert isinstance(coeffs_h, list) and isinstance(coeffs_d, list)
        assert len(coeffs_h) == len(coeffs_d) == config["J"]
        for j in range(config["J"]):
            assert coeffs_h[j].shape == coeffs_d[j].shape

        # Corr-layer per-scale output shapes should match across models

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
            shift_mode="default",
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
            shift_mode="default",
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
            shift_mode="default",
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
            shift_mode="default",
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
