import os
import sys
import torch

# Ensure repository root is on sys.path so `import wph...` works under pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.layers.wave_conv_layer import WaveConvLayer
from wph.layers.relu_center_layer import ReluCenterLayer


def test_waveconv_shapes_and_dtype():
    J, L, A, M, N = 2, 3, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1)
    x = torch.randn(2, 1, M, N)
    out = layer(x)
    assert out.shape == (2, 1, J, L, A, M, N)
    assert torch.is_complex(out), "Expected complex output from WaveConvLayer"


def test_waveconv_sharing_and_get_full_filters():
    J, L, A, M, N = 2, 3, 2, 8, 8
    num_channels = 2
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=num_channels,
                          share_rotations=True, share_channels=False)
    # base filters should have reduced rotation dimension
    expected_base_shape = (num_channels, J, 1, A, M, N)
    assert tuple(layer.base_filters.shape) == expected_base_shape
    full = layer.get_full_filters()
    assert tuple(full.shape) == (num_channels, J, L, A, M, N)


def test_relu_center_forward():
    J, L, A, M, N = 2, 3, 2, 8, 8
    nb, nc = 2, 1
    real = torch.randn(nb, nc, J, L, A, M, N)
    imag = torch.randn_like(real)
    x = torch.complex(real, imag)

    relu = ReluCenterLayer(J=J, M=M, N=N, normalize=False)
    out = relu(x)
    assert out.shape == (nb, nc, J, L, A, M, N)
    assert torch.is_floating_point(out), "Expected real-valued output after ReLU"
    # ReLU output should be non-negative
    assert out.min() >= -1e-6


def test_integration_waveconv_relu():
    J, L, A, M, N = 2, 3, 2, 8, 8
    nb, nc = 2, 1
    img = torch.randn(nb, nc, M, N)
    wave = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=nc)
    xpsi = wave(img)
    relu = ReluCenterLayer(J=J, M=M, N=N, normalize=True)
    out = relu(xpsi)
    assert out.shape == (nb, nc, J, L, A, M, N)
    assert out.min() >= -1e-6
