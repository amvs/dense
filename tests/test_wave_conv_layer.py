import torch
import os
import sys
from torchviz import make_dot
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add the project root directory to the Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.layers.wave_conv_layer import WaveConvLayer

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
    assert torch.allclose(grad_shared, grad_not_shared_summed), "Gradients do not match after summing across the L dimension."

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
