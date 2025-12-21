import torch
import torch.nn as nn
from dataclasses import dataclass
from .wavelets import filter_bank
from torch import Tensor
import torch.nn.functional as F
import random

from .helpers import checkpoint
@dataclass
class ScatterParams:
    n_scale: int
    n_orient: int
    in_channels: int
    wavelet: str
    n_class: int
    share_channels: bool
    in_size: int
    out_size: int
    random: bool = False

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

    Attributes:
    param  -- nn.Parameter, the trainable parameters used to construct the kernels

    Arguments:
    filters -- tensor of shape [nb_orients, S, S], the set of convolution kernels
    in_channels -- int, number of input channels
    share_channels 
        -- bool, if True, all input channels share the same set of kernels.
           param will have shape [1, nb_orients, S, S]
        -- if False, each input channel has its own set of kernels.
           param will have shape [in_channels * nb_orients, 1, S, S]

    '''
    def __init__(
        self, filters: Tensor, in_channels: int, share_channels: bool = False
    ):
        super().__init__()
        nb_orients, S, _ = filters.shape
        self.S = S
        self.in_channels = in_channels
        self.share_channels = share_channels
        self.out_channel = nb_orients * in_channels
        if share_channels:
            weight = filters.unsqueeze(0)
            self.param = nn.Parameter(weight) # shape [1, nb_orients, S, S]
        else:
            weight = filters.unsqueeze(1).repeat_interleave(in_channels, dim=0)
            self.param = nn.Parameter(weight) # shape [in_channels * nb_orients, 1, S, S]

    def forward(self, x):
        if self.share_channels:
            kernel = self.param.expand(self.in_channels, -1, -1, -1).reshape(self.out_channel, 1, self.S, self.S)
        else:
            kernel = self.param

        return F.conv2d(
            x,
            kernel,
            bias = None,
            padding = 'same',
            groups = self.in_channels
        )

class ScatterBlock(nn.Module):

    def __init__(
        self, filters: Tensor, in_channels:int, share_channels: bool = False
    ):
        super().__init__()
        self.conv = MyConv2d(
            filters=filters,
            in_channels=in_channels,
            share_channels=share_channels
        )

    def forward(self, *inputs):
        imgs = torch.cat(inputs, dim=1)
        imgs = imgs.to(torch.complex64)
        result = torch.abs(self.conv(imgs))
        return result

class dense(nn.Module):

    def __init__(self, params: ScatterParams):
        super().__init__()
        for k, v in vars(params).items():
            setattr(self, k, v)
        
        self.set_filters()

        self.pooling = nn.AvgPool2d(2, 2)

        self.in_channels_per_block = []
        in_channels = params.in_channels
        for _ in range(self.n_scale):
            self.in_channels_per_block.append(in_channels)
            in_channels = in_channels * (self.n_orient + 1)
        self.out_channels = in_channels
        self.blocks = nn.ModuleList(
            [
                ScatterBlock(
                    filters=self.filters[0],
                    in_channels=self.in_channels_per_block[scale],
                    share_channels=self.share_channels
                )
                for scale in range(self.n_scale)
            ]
        )
        out_size = self.out_size
        self.out_dim = self.out_channels * out_size**2 #* (self.in_size // 2**self.n_scale)**2
        self.global_pool = nn.AdaptiveAvgPool2d((out_size,out_size))
        self.linear = nn.Linear(self.out_dim, self.n_class)

    def set_filters(self):
        self.filters = filter_bank(self.wavelet, self.n_scale, self.n_orient)
        if self.random:
            random_filters = []
            seeds = [42, 123, 999, 2025, 7]

            # Randomly select one seed
            seed = random.choice(seeds)

            # Set the seed for reproducibility
            torch.manual_seed(seed)

            # If you use CUDA
            torch.cuda.manual_seed_all(seed)

            for filt in self.filters:
                temp = torch.randn_like(filt.real) + 1j * torch.randn_like(filt.real)
                random_filters.append(temp.to(dtype=filt.dtype))
            self.filters = random_filters


    def forward(self, img):
        inputs = [img]
        for ind, block in enumerate(self.blocks):
            #out = checkpoint(block, *inputs) if ind != 0 else block(*inputs)
            out = block(*inputs)
            inputs.append(out)
            inputs = [self.pooling(i) for i in inputs]
        features = self.global_pool(torch.cat(inputs, dim=1)).reshape(img.shape[0], -1)
        return self.linear(features) #self.linear(F.normalize(features, p=2, dim=1))


    def train_classifier(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = True

    def train_conv(self):
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = False

    def full_train(self):
        for param in self.parameters():
            param.requires_grad = True

    def fine_tuned_params(self):
        '''
        Interface to get the parameters of convolution layers for fine-tuning.
        '''
        return self.blocks.parameters()

    def n_tuned_params(self):
        total_params = 0
        for param in self.blocks.parameters():
            total_params += param.numel()
        return total_params
