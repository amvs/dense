#!/usr/bin/env python
# coding: utf-8

# **Create a wrapper of conv2d that accepts wavelet filters and other parameters. It create a Conv2d layer and assign the weight as wavelet filters, disable gradient.**

# In[8]:


import torch
from torch import nn
from create_filters import filter_bank
def wavelet2d(filters: torch.Tensor, in_channels: int, stride: int = 1, dilation: int = 1, kernel_dtype = torch.complex64) -> nn.Conv2d:
    '''
    Create nn.Conv2d with
    - bias = False
    - weights set to `filters`

    filters must have shape [C_filter, S, S]
    for band pass filter, C_filter = nb_angles

    Example:
    image <-- tensor of shape [1, 3, 128, 128], one color image of size 128*128
    filters <-- tensor of shape [4, 3, 3], 4 oriented wavelet filters of size 3*3
    conv2d = wavelet2d(filters, image.shape[1])
    result = conv2d(image) -> shape [1, 12, 128, 128], each rbg channel is convolved separately with each oriented filter.
    '''
    weight = filters.unsqueeze(1).repeat_interleave(in_channels, dim=0)
    out_channels = weight.shape[0] # in_channels * C_filter
    size = filters.shape[-1]
    padding = dilation*(size-1)//2 # always same size padding
    conv = nn.Conv2d(
        in_channels = in_channels,
        out_channels = out_channels,
        kernel_size = size,
        stride = stride,
        padding = padding,
        dilation = dilation,
        groups = in_channels,
        dtype = kernel_dtype,
        bias = False
    )
    with torch.no_grad():
        conv.weight.copy_(weight)
    return conv


# 

# In[9]:


class dense(nn.Module):
    def __init__(self, scale_J:int,       # number of scales
                       angle_K:int,       # number of orientation
                       image_shape:tuple, # image_channel, square_image_size
                       kernel_size:int,   # square_filter_size
                       nb_class:int = None  # nb of classification class
                ):
        super().__init__()
        self.scale_J = scale_J
        self.angle_K = angle_K
        self.image_channel, self.image_size = image_shape
        self.kernel_size = kernel_size

        # check valid parameter

        self.load_filters()
        self.sequential_conv = nn.ModuleList()
        #self.sequential_pooling = nn.ModuleList()
        in_channel = self.image_channel
        for j in range(scale_J):
            conv = wavelet2d(self.filters[j], in_channel)
            #pool = nn.AvgPool2d(2, 2)
            self.sequential_conv.append(conv)
            #self.sequential_pooling.append(pool)
            in_channel = in_channel * (angle_K + 1) # non_linear doesn't increase the channel, only conv
        self.out_dim = in_channel * (self.image_size//2**scale_J)**2
        self.pooling = nn.AvgPool2d(2**scale_J, 2**scale_J)
        if nb_class != None:
            self.linear = nn.Linear(self.out_dim, nb_class)
        else:
            self.linear = None

    def non_linear(self, imgs):
        return torch.abs(imgs)

    def cuda(self):
        # for conv in self.sequential_conv:
        #     conv.to("cuda")
        for wavelet_filter in self.filters:
            wavelet_filter.to("cuda")
        if self.linear:
            self.linear.to("cuda")
        return self                

    def load_filters(self):
        # [J, K, N, N], with different scale factor!
        # self.filters = torch.zeros((self.scale_J, self.angle_K, self.kernel_size, self.kernel_size), dtype=torch.complex64)
        # wavelet_filter = torch.load('./filters/morlet_S'+str(self.kernel_size)+'_K'+str(self.angle_K)+'.pt')
        # for j in range(self.scale_J):
        #     self.filters[j] = wavelet_filter * (2**j)
        S = self.kernel_size
        L = self.angle_K
        self.filters = []
        for j in range(self.scale_J):
            wavelet_filter = filter_bank(8, S, L) / (2**(2*j)) # at scale j, the filter is scaled by 2^j
            S = 2 * S - 1  # next scale has size 2*S - 1
            self.filters.append(wavelet_filter)  # scale the filter by 2^j


    def forward(self, img):
        for conv in self.sequential_conv:
            result = self.non_linear(conv(img.to(torch.complex64)))
            img = torch.cat([img, result], dim=1)
        img = self.pooling(img)
        if self.linear:
            img = self.linear(img.reshape(img.shape[0], -1))
        return img


# In[10]:


from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# In[11]:


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == targets).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


# In[12]:


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def run_mnist_experiment(scale_J, angle_K, kernel_size, model, image_shape=(1, 28), nb_class=10):

    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    return model


# In[13]:


if __name__ == '__main__':
    # Try different combinations
    models = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    J_params = [2]
    K_params = [3]
    kernel_size_params = [9]
    for J in J_params:
        for K in K_params:
            for ks in kernel_size_params:
                print(f"\nRunning: J={J}, K={K}, kernel={ks}")
                model = dense(J, K, (1, 28), ks, 10).to(device)
                for conv in model.sequential_conv:
                    for param in conv.parameters():
                        param.requires_grad = False
                for param in model.linear.parameters():
                    param.requires_grad = True
                models.append(run_mnist_experiment(J, K, ks, model))



# In[14]:
if __name__ == '__main__':

    count = 0
    result_models = []
    for J in J_params:
            for K in K_params:
                for ks in kernel_size_params:
                    print(f"\nRunning: J={J}, K={K}, kernel={ks}")
                    model = models[count]
                    for conv in model.sequential_conv:
                        for param in conv.parameters():
                            param.requires_grad = True
                    for param in model.linear.parameters():
                        param.requires_grad = False
                    print(model)
                    result_models.append(run_mnist_experiment(J, K, ks, model=model))
                    count = count+1


# In[ ]:




