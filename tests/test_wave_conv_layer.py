import torch
import os
import sys
import pdb
from torchviz import make_dot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dense.wavelets import morlet

# Add the project root directory to the Python path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wph.layers.wave_conv_layer import WaveConvLayer

def test_wave_conv_layer_gradients_share_rotations():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, train_filters=True, share_rotations=True)

    x = torch.randn(1, 1, M, N, requires_grad=True)
    output = layer(x)
    loss = output.abs().sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not propagate to input."
    assert layer.base_filters.grad is not None, "Gradients did not propagate to filters."

def test_wave_conv_layer_gradients_share_phases():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, train_filters=True, share_phases=True)

    x = torch.randn(1, 1, M, N, requires_grad=True)
    output = layer(x)
    loss = output.abs().sum()
    loss.backward()

    assert x.grad is not None, "Gradients did not propagate to input."
    assert layer.base_filters.grad is not None, "Gradients did not propagate to filters."

def test_wave_conv_layer_gradients_share_channels():
    J, L, A, M, N = 3, 4, 2, 8, 8
    layer = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=3, train_filters=True, share_channels=True)

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
    layer_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, train_filters=True, share_rotations=True, filters=filters[:,0,...].unsqueeze(0).unsqueeze(2).unsqueeze(3))
    x = torch.randn(1, 1, M, N, requires_grad=True)
    output_shared = layer_shared(x)
    loss_shared = output_shared.abs().sum()
    loss_shared.backward()
    grad_shared = layer_shared.base_filters.grad.clone()

    graph_shared = make_dot(loss_shared, params={"x": x, "filters": layer_shared.base_filters})
    graph_shared.render("computation_graph_shared", format="pdf")


    # Case 2: share_rotations=False
    layer_not_shared = WaveConvLayer(J=J, L=L, A=A, M=M, N=N, num_channels=1, train_filters=True, share_rotations=False, filters = filters.unsqueeze(2).unsqueeze(0))
    output_not_shared = layer_not_shared(x)
    loss_not_shared = output_not_shared.abs().sum()
    loss_not_shared.backward()
    grad_not_shared = layer_not_shared.base_filters.grad.clone()

    # Visualize computation graph for share_rotations=False
    graph_not_shared = make_dot(loss_not_shared, params={"x": x, "filters": layer_not_shared.base_filters})
    graph_not_shared.render("computation_graph_not_shared", format="pdf")


    # Compare gradients
    assert torch.allclose(layer_not_shared.base_filters, layer_shared.get_full_filters())
    assert torch.allclose(grad_shared.sum(), grad_not_shared.sum()), "Summed gradients do not match between share_rotations=True and False."

if __name__ == "__main__":
    # test_wave_conv_layer_gradients_share_rotations()
    # test_wave_conv_layer_gradients_share_phases()
    # test_wave_conv_layer_gradients_share_channels()
    test_wave_conv_layer_gradients_compare_share_rotations()
    print("All tests passed.")