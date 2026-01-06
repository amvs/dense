"""
Adapted from https://github.com/abrochar/wavelet-texture-synthesis/
"""

import torch
import warnings

def maskns(J, M, N):
    m = torch.ones(J, M, N)
    for j in range(J):
        for x in range(M):
            for y in range(N):
                if (x<(2**j)//2 or y<(2**j)//2 \
                or x+1>M-(2**j)//2 or y+1>N-(2**j)//2):
                    m[j, x, y] = 0
    m = m.type(torch.float)
    m = m / m.sum(dim=(-1,-2), keepdim=True)
    m = m*M*N
    return m

def masks_subsample_shift(J,M,N, alpha=4, mask_union=True):
    m = torch.zeros(J,M,N).type(torch.float)
    m[:,0,0] = 1.
    angles = torch.arange(2*alpha).type(torch.float)
    angles = angles/(2*alpha)*2*torch.pi
    for j in range(J):
        for theta in range(len(angles)):
            x = int(torch.round((2**j)*torch.cos(angles[theta])))
            y = int(torch.round((2**j)*torch.sin(angles[theta])))
            if mask_union:
                for j_ in range(j,J):
                    m[j_,x,y] = 1.
            else:
                m[j,x,y] = 1.
    return m


from torch import nn

class SubInitSpatialMean(nn.Module):
    def __init__(self):
        super().__init__()
        # Use an empty tensor as initial value
        self.register_buffer('minput', torch.empty(0), persistent=False)

    def forward(self, input):
        if self.minput.numel() == 0:
            minput = input.clone().detach()
            minput = torch.mean(minput, -1, True)
            minput = torch.mean(minput, -2, True)
            self.minput = minput
        elif self.minput.shape[1:-2] != input.shape[1:-2]:
            warnings.warn('overwriting minput')
            minput = input.clone().detach()
            minput = torch.mean(minput, -1, True)
            minput = torch.mean(minput, -2, True)
            self.minput = minput
        output = input - self.minput
        return output



class DivInitStd(nn.Module):
    def __init__(self, stdcut=1e-9):
        super().__init__()
        # Use an empty tensor as initial value
        self.register_buffer('stdinput', torch.empty(0), persistent=False)
        self.eps = stdcut

    def forward(self, input):
        if self.stdinput.numel() == 0:
            stdinput = input.clone().detach()  # input size:(...,M,N)
            m = torch.mean(torch.mean(stdinput, -1, True), -2, True)
            stdinput = stdinput - m
            d = input.shape[-1]*input.shape[-2]
            stdinput = torch.norm(stdinput, dim=(-2,-1), keepdim=True)
            self.stdinput = stdinput / torch.sqrt(torch.tensor(d, dtype=stdinput.dtype, device=stdinput.device))
            self.stdinput = self.stdinput + self.eps
        elif self.stdinput.shape[1:-2] != input.shape[1:-2]:
            warnings.warn('overwriting stdinput')
            stdinput = input.clone().detach()  # input size:(...,M,N)
            m = torch.mean(torch.mean(stdinput, -1, True), -2, True)
            stdinput = stdinput - m
            d = input.shape[-1]*input.shape[-2]
            stdinput = torch.norm(stdinput, dim=(-2,-1), keepdim=True)
            self.stdinput = stdinput / torch.sqrt(torch.tensor(d, dtype=stdinput.dtype, device=stdinput.device))
            self.stdinput = self.stdinput + self.eps

        output = input/self.stdinput
        return output


def padc(x):
    x_ = x.clone()
    return torch.stack((x_, torch.zeros_like(x_)), dim=-1)
