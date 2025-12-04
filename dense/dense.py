import torch
from torch import nn
from .helpers import checkpoint, MyConv2d
from .wavelets import filter_bank
class dense(nn.Module):
    def __init__(self, max_scale:int, nb_orients:int, # number of scales and orientations
                       image_shape:tuple,        # image_channel, Height, Width
                       wavelet:str = "morlet",   # Choose wavelet to use
                       nb_class:int = None,      # nb of classification class
                       efficient:bool = True,     # enable memory efficient
                       share_channels:bool = False,
                       random:bool = False
                ):
        super().__init__()
        self.max_scale = max_scale
        self.nb_orients = nb_orients
        self.image_channel, self.image_size, self.image_size2 = image_shape
        self.wavelet = wavelet
        self.efficient = efficient
        self.share_channels = share_channels
        self.random = random
        # check valid parameter
        if self.image_size != self.image_size2:
            raise ValueError("Images are not square size")
        
        # prepare filters of different scales and angles
        self.set_filters()

        # construct scattering layers
        self.build_layers()

        # optional linear layer
        if nb_class != None:
            self.linear = nn.Linear(self.out_dim, nb_class)
        else:
            self.linear = None

    def set_filters(self):
        self.filters = filter_bank(self.wavelet, self.max_scale, self.nb_orients)
        if self.random:
            random_filters = []
            torch.manual_seed(42)
            for filt in self.filters:
                temp = torch.randn_like(filt.real) + 1j * torch.randn_like(filt.real)
                random_filters.append(temp.to(dtype=filt.dtype))
            self.filters = random_filters

    def build_layers(self):
        """
        Build the scattering transform layers.

        This method performs three major tasks:

        (1) Build a list of convolution layers, one for each scale.
            Each convolution is implemented using MyConv2d.

            The convolution layers are stored in `self.sequential_conv`
            as an nn.ModuleList so that PyTorch properly registers
            parameters and supports `.to(device)` as well as obtaining
            learnable parameters via `model.sequential_conv.parameters()`
            or `model.fine_tuned_params()`.

        (2) Wrap each convolution in a module produced by `funcFactory`,
            which concat all input features and then applies the convolution 
            followed by a nonlinear operation. These modules are stored in 
            `self.module_list`.
            The way allows activation checkpointing across modules for
            reducing memory usage during forward.

        (3) Define an average-pooling layer and compute the flattened output
            dimension in `self.out_dim`. The average-pooling layer is used
            for spatial downsampling at every scale, and as data flow to last
            layer, the culmulative effect also serves as a low-pass filter.

        After calling this function:
            - `self.sequential_conv[j]` contains the j-th convolution.
              Only the learnable parameters inside each MyConv2d are optimized
              when the optimizer is given `model.sequential_conv.parameters()`
              or `model.fine_tuned_params()`.
            - `self.module_list[j]` contains the full conv + nonlinearity.

        """
        max_scale = self.max_scale
        nb_orients = self.nb_orients

        self.sequential_conv = nn.ModuleList()
        in_channel = self.image_channel
        for j in range(max_scale):
            # initialize convolution layer with wavelet filters
            conv = MyConv2d(
                filters=self.filters[0], 
                in_channel=in_channel, 
                share_channels=self.share_channels
            )
            self.sequential_conv.append(conv)
            in_channel = in_channel * (nb_orients + 1) # non_linear doesn't increase the channel, only conv
        
        # construct modules for each conv + non linear
        self.module_list = []
        for conv in self.sequential_conv:
            module = self.funcFactory(conv)
            self.module_list.append(module)
            
        # downsampling by average pooling
        # for each scale and as low pass filter  
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
        for param in self.sequential_conv.parameters():
            param.requires_grad = False
        for param in self.linear.parameters():
            param.requires_grad = True 

    def train_conv(self):
        for param in self.sequential_conv.parameters():
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
        return self.sequential_conv.parameters()
        
    def forward(self, img):
        # maintain a list of all previous layer outputs
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

