import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedConv2d(nn.Module):
    def __init__(self, weight, groups):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.groups = groups

    def forward(self, x):
        return F.conv2d(
            x,
            self.weight,
            bias = None,
            padding = 'same',
            groups = self.groups
        )

def wavelet2d(filters: torch.Tensor, 
              in_channels: int, 
              stride: int = 1, 
              dilation: int = 1, 
              kernel_dtype = torch.complex64,
              share_channels: bool = False) -> nn.Conv2d:
    '''
    Create nn.Conv2d with
    - bias = False
    - weights set to `filters`
    - padding to same size
    - group convolution: each image channel is convolved separately 
        with each oriented filter.
    
    Arguments:
    share_channels -- if True, all in_channels share the same set of filters.
                if False, each in_channel has its own set of filters in the memory.

    filters must have shape [C_filter, S, S]
    for band pass filter, C_filter = nb_orients

    Example:
    image <-- tensor of shape [1, 3, 128, 128], one color image of size 128*128
    filters <-- tensor of shape [4, 3, 3], 4 oriented wavelet filters of size 3*3
    conv2d = wavelet2d(filters, image.shape[1])
    result = conv2d(image) -> shape [1, 12, 128, 128]
    '''
    nb_orients, S, _ = filters.shape
    if share_channels:
        weight = filters.unsqueeze(0).expand(in_channels, -1, -1, -1).reshape(nb_orients * in_channels, 1, S, S)
        conv = SharedConv2d(weight, groups=in_channels)
    else:
        weight = filters.unsqueeze(1).repeat_interleave(in_channels, dim=0)
        out_channels = weight.shape[0] # in_channels * nb_orients
        size = filters.shape[-1]
        padding = dilation*(size-1)//2 # always same size padding
        conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = size,
            stride = stride,
            padding = padding,
            padding_mode = 'circular',
            dilation = dilation,
            groups = in_channels,
            dtype = kernel_dtype,
            bias = False
        )
        with torch.no_grad():
            conv.weight.copy_(weight)
    return conv
