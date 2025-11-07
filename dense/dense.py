import torch
from torch import nn
from .helpers import checkpoint, wavelet2d
from .wavelets import filter_bank
class dense(nn.Module):
    def __init__(self, max_scale:int, nb_orients:int, # number of scales and orientations
                       image_shape:tuple,        # image_channel, Height, Width
                       wavelet:str = "morlet",   # Choose wavelet to use
                       nb_class:int = None,      # nb of classification class
                       efficient:bool = True,     # enable memory efficient
                       isShared:bool = False
                ):
        super().__init__()
        self.max_scale = max_scale
        self.nb_orients = nb_orients
        self.image_channel, self.image_size, self.image_size2 = image_shape
        self.wavelet = wavelet
        self.efficient = efficient
        self.isShared = isShared
        # check valid parameter
        if self.image_size != self.image_size2:
            raise ValueError("Images are not square size")
        
        # prepare filters of different scales and angles
        self.set_filters()

        # prepare convolution operations
        self.set_convs()

        # optional linear layer
        if nb_class != None:
            self.linear = nn.Linear(self.out_dim, nb_class)
        else:
            self.linear = None

    def set_filters(self):
        self.filters = filter_bank(self.wavelet, self.max_scale, self.nb_orients)

    def set_convs(self):
        max_scale = self.max_scale
        nb_orients = self.nb_orients
        # set up convolution parameters
        self.sequential_conv = nn.ModuleList()
        in_channel = self.image_channel
        for j in range(max_scale):
            conv = wavelet2d(self.filters[0], in_channel, isShared=self.isShared)
            self.sequential_conv.append(conv)
            in_channel = in_channel * (nb_orients + 1) # non_linear doesn't increase the channel, only conv
        
        # chainning conv, nonlinear operation
        self.module_list = []
        for conv in self.sequential_conv:
            module = self.funcFactory(conv)
            self.module_list.append(module)

        # pooling on the last layer, as low pass filter    
        self.pooling = nn.AvgPool2d(2, 2)
        self.out_dim = in_channel * (self.image_size//2**max_scale)**2


    def nonLinear(self, imgs):
        return torch.abs(imgs)  
    
    def funcFactory(self, conv):
        def module(*inputs):
            imgs = torch.cat(inputs, dim=1)
            img_c = imgs.to(torch.complex64)
            result = self.nonLinear(conv(img_c))
            return result
        return module    

    def train_classifier(self):
        for conv in self.sequential_conv:
            for param in conv.parameters():
                param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = True 

    def train_conv(self):
        for conv in self.sequential_conv:
            for param in conv.parameters():
                param.requires_grad = True
        for param in self.linear.parameters():
            param.requires_grad = False       
        
    def forward(self, img):
        inputs = [img]
        for index, module in enumerate(self.module_list):
            # the first operation must not be checkpointed, to avoid no gradients in all cases
            result = checkpoint(module, *inputs) if self.efficient and index != 0 else module(*inputs)
            inputs.append(result)
            inputs = [self.pooling(inp) for inp in inputs]
        features = torch.cat(inputs, dim=1)
        if self.linear:
            features = self.linear(features.reshape(img.shape[0], -1))
        return features

