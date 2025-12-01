import torch
from torch import nn
import torch.nn.functional as F
from wph.ops.backend import SubInitSpatialMean, DivInitStd


class ReluCenterLayer(nn.Module):
    def __init__(self, J: int, M: int, N: int, normalize: bool = True):
        """
        Initializes the ReLU center layer.
        J: number of scales
        M, N: spatial dimensions of input signals
        """
        super().__init__()
        self.J = J
        self.M = M
        self.N = N
        self.normalize = normalize
        masks = self.maskns(J, M, N)
        # shape to (1, 1, J, 1, 1, M, N) to broadcast over (nb, nc, J, L, A, M, N)
        masks = masks.view(1, 1, J, 1, 1, M, N)
        self.register_buffer("masks", masks)
        self.mean = SubInitSpatialMean()
        self.std = DivInitStd()

    def forward(self, x):
        """
        Apply ReLU center layer to input x.
        x: input tensor of shape (nb, nc, J, L, A, M, N)
        1. Normalize x by subtracting mean and dividing by std if normalize is True.
        2. Apply ReLU activation to the real part of x.
        3. Multiply by precomputed masks for aperiodic signals.
        returns: output tensor with shape (nb, nc, J, L, A, M, N)
        """
        nb, nc = x.shape[:2]
        assert x.shape[2] == self.J, f"Expected J={self.J}, but got {x.shape[2]}"
        assert x.shape[-2:] == (
            self.M,
            self.N,
        ), f"Expected spatial dimensions {(self.M, self.N)}, but got {x.shape[-2:]}"
        # subtract spatial mean
        x = self.mean(x)
        # optionally divide by precomputed std (does not re-center)
        if self.normalize:
            x = self.std(x)
        # take real part and apply ReLU
        x = F.relu(x.real)
        # apply masks per-scale and return
        x = x * self.masks.expand(nb, nc, -1, -1, -1, -1, -1)
        return x

    def maskns(self, J, M, N):
        """
        Create masks for aperiodic images
        """
        # Create a grid of coordinates
        x, y = torch.meshgrid(torch.arange(M), torch.arange(N))

        # Compute the mask using broadcasting
        masks = []
        for j in range(J):
            mask = (
                (x >= (2**j) // 2)
                & (y >= (2**j) // 2)
                & (x < M - (2**j) // 2)
                & (y < N - (2**j) // 2)
            )

            # Normalize the mask
            mask = mask.float()
            mask /= mask.sum(dim=(-1, -2), keepdim=True)
            mask *= M * N

            masks.append(mask)

        return torch.stack(masks, dim=0)
