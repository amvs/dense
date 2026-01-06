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

    def forward(self, input):
        minput = input.clone().detach()
        minput = torch.mean(minput, dim=(-2, -1), keepdim=True)
        output = input - minput
        return output



class DivInitStd(nn.Module):
    def __init__(self, stdcut=1e-9):
        super().__init__()
        self.eps = stdcut

    def forward(self, input):
        stdinput = input.clone().detach()
        m = torch.mean(stdinput, dim=(-2, -1), keepdim=True)
        stdinput = stdinput - m
        d = input.shape[-1] * input.shape[-2]
        stdinput = torch.norm(stdinput, dim=(-2, -1), keepdim=True)
        stdinput = stdinput / torch.sqrt(torch.tensor(d, dtype=stdinput.dtype, device=stdinput.device))
        stdinput = stdinput + self.eps
        output = input / stdinput
        return output


def padc(x):
    x_ = x.clone()
    return torch.stack((x_, torch.zeros_like(x_)), dim=-1)
