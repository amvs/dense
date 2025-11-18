import torch
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from wph.wph_model import WPHModel


def test_wph_model_default():
    # Default initialization
    J, L, A, A_prime, M, N = 3, 4, 2, 1, 8, 8
    filters = {"hatpsi": torch.randn(1, J, L, A, M, N), "hatphi": torch.randn(M, N)}
    model = WPHModel(J=J, L=L, A=A, A_prime=A_prime, M=M, N=N, filters=filters)

    x = torch.randn(1, 1, M, N)
    xcorr, xlow, xhigh = model(x, flatten=True)

    assert xcorr.shape[0] == 1  # Batch size
    assert xlow.shape[0] == 1  # Batch size
    assert xhigh.shape[0] == 1  # Batch size


def test_wph_model_custom_params():
    # Custom initialization
    J, L, A, A_prime, M, N = 2, 3, 2, 1, 16, 16
    # Use shared rotations and phases
    filters = {"hatpsi": torch.randn(1, J, 1, 1, M, N), "hatphi": torch.randn(M, N)}
    model = WPHModel(
        J=J,
        L=L,
        A=A,
        A_prime=A_prime,
        M=M,
        N=N,
        filters=filters,
        share_rotations=True,
        share_phases=True,
    )

    x = torch.randn(1, 1, M, N)
    xcorr, xlow, xhigh = model(x, flatten=False)

    assert xcorr.shape[0] == 1  # Batch size
    assert xlow.shape[0] == 1  # Batch size
    assert xhigh.shape[0] == 1  # Batch size


def test_wph_model_multiple_channels():
    # Test with multiple channels
    J, L, A, A_prime, M, N = 3, 4, 2, 1, 8, 8
    filters = {"hatpsi": torch.randn(1, J, L, A, M, N), "hatphi": torch.randn(M, N)}
    model = WPHModel(
        J=J, L=L, A=A, A_prime=A_prime, M=M, N=N, filters=filters, num_channels=3
    )

    x = torch.randn(1, 3, M, N)
    xcorr, xlow, xhigh = model(x, flatten=True)

    assert xcorr.shape[0] == 1  # Batch size
    assert xlow.shape[0] == 1  # Batch size
    assert xhigh.shape[0] == 1  # Batch size


if __name__ == "__main__":
    test_wph_model_default()
    test_wph_model_custom_params()
    test_wph_model_multiple_channels()
    print("All tests passed.")
