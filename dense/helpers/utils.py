import torch
import torch.nn as nn
import torch.nn.functional as F

class MyConv2d(nn.Module):
    '''
    A custom convolution layer that decouples the trainable parameters
    and the kernels used in convolution operation.

    The convolution is performed as group convolution: 
    Given an input signal composed of multiple channels, each channel
    is convolved separately with a set of kernels. No bias is used and 
    padding is set to maintain the same size.

    When initialized, the trainable parameters are constructed from the
    provided filters. Then, the convolution kernels are constructed from the
    trainable parameters depending on whether all input channels share the
    same set of kernels or not.

    Arrtibutes:
    param  -- nn.Parameter, the trainable parameters used to construct the kernels
    kernel -- torch.Tensor, the convolution kernels constructed from param depending
                on whether all input channels share the same set of kernels or not.

    Arguments:
    filters -- tensor of shape [nb_orients, S, S], the set of convolution kernels
    in_channel -- int, number of input channels
    share_channels 
        -- bool, if True, all input channels share the same set of kernels.
           param will have shape [1, nb_orients, S, S]
        -- if False, each input channel has its own set of kernels.
           param will have shape [in_channel * nb_orients, 1, S, S]

    '''
    def __init__(self, filters, in_channel, share_channels: bool = False):
        super().__init__()
        nb_orients, S, _ = filters.shape
        self.S = S
        self.in_channel = in_channel
        self.share_kernels = share_channels
        self.out_channel = nb_orients * in_channel
        if share_kernels:
            weight = filters.unsqueeze(0)
            self.param = nn.Parameter(weight) # shape [1, nb_orients, S, S]
        else:
            weight = filters.unsqueeze(1).repeat_interleave(in_channel, dim=0)
            self.param = nn.Parameter(weight) # shape [in_channel * nb_orients, 1, S, S]

    def forward(self, x):
        if self.share_kernels:
            kernel = self.param.expand(self.in_channel, -1, -1, -1).reshape(self.out_channel, 1, self.S, self.S)
        else:
            kernel = self.param

        return F.conv2d(
            x,
            kernel,
            bias = None,
            padding = 'same',
            groups = self.in_channel
        )
