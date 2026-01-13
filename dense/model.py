import torch
import torch.nn as nn
from dataclasses import dataclass
from .wavelets import filter_bank
from torch import Tensor
import torch.nn.functional as F
import random
from collections import defaultdict

from .helpers import checkpoint
@dataclass
class ScatterParams:
    n_scale: int
    n_orient: int
    n_copies: int
    in_channels: int
    wavelet: str
    n_class: int
    share_channels: bool
    in_size: int
    out_size: int # -1 means no global pooling on scatter features
    # PCA
    pca_dim: int
    # depth
    depth: int = -1  # -1 means full scatter
    #
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
            weight = filters.unsqueeze(1).repeat(in_channels, 1, 1, 1)
            self.param = nn.Parameter(weight) # shape [in_channels * nb_orients, 1, S, S]

    def forward(self, x):
        if self.share_channels:
            raise Exception("Bug")
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
        self, filters: Tensor, in_channels:int, n_copies: int, share_channels: bool = False
    ):
        super().__init__()
        self.conv = MyConv2d(
            filters=filters,
            in_channels=in_channels,
            share_channels=share_channels
        )
        self.n_copies = n_copies

    def forward(self, *inputs):
        imgs = torch.cat(inputs, dim=1)
        imgs = imgs.to(torch.complex64)
        result = torch.abs(self.conv(imgs))
        # Reshape and average over n_copies
        return result.reshape(imgs.shape[0], self.conv.out_channel//self.n_copies, 
                             self.n_copies, imgs.shape[-1], imgs.shape[-1]).mean(dim=2)

class dense(nn.Module):

    def __init__(self, params: ScatterParams):
        super().__init__()
        for k, v in vars(params).items():
            setattr(self, k, v)
        
        self.set_filters()

        self.pooling = nn.AvgPool2d(2, 2)

        self.in_channels_per_block = []
        in_channels = params.in_channels
        if in_channels != 1:
            raise ValueError("Only support grayscale input (in_channels=1)")

        for _ in range(self.n_scale):
            self.in_channels_per_block.append(in_channels)
            in_channels = in_channels * (self.n_orient + 1)
        self.out_channels = in_channels

        if self.depth != -1 and self.depth != self.n_scale: # 相等时是完整 scatter，没必要裁剪
            if self.depth < 2 or self.depth > self.n_scale:
                raise ValueError("depth should be in [2, n_scale]")
            keep_idx, keep_len = self.build_keep_idx()
            for i, idx in enumerate(keep_idx):
                self.register_buffer(f"keep_idx_{i}", idx)
            self.in_channels_per_block = keep_len[:-1]  # 最后一层不需要
            self.out_channels = keep_len[-1]        
        # print("in_channels_per_block:", self.in_channels_per_block)
        # print("out_channels:", self.out_channels)
        self.blocks = nn.ModuleList(
            [
                ScatterBlock(
                    filters=self.filters[0],
                    in_channels=self.in_channels_per_block[scale],
                    n_copies=self.n_copies,
                    share_channels=self.share_channels
                )
                for scale in range(self.n_scale)
            ]
        )
        out_size = self.out_size
        if out_size != -1:
            self.out_dim = self.out_channels * out_size**2 #* (self.in_size // 2**self.n_scale)**2 
            self.global_pool = nn.AdaptiveAvgPool2d((out_size,out_size))
        else:
            self.out_dim = self.out_channels * (self.in_size // 2**self.n_scale)**2
        self.linear = nn.Linear(self.out_dim, self.n_class)

        # self.register_buffer("class_means", None)   # [C, D]
        # self.register_buffer("pca_bases", None)     # [C, D, k]

    
    @torch.no_grad()
    def fit(self, dataloader, device):
        self.eval()

        # running statistics
        count = torch.zeros(self.n_class, device=device)
        sum_feat = None

        # ---------- pass 1: class means ----------
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = self.scatter_forward(imgs)  # [B, D]

            if sum_feat is None:
                D = feats.shape[1]
                sum_feat = torch.zeros(self.n_class, D, device=device)

            for c in range(self.n_class):
                mask = labels == c
                if mask.any():
                    sum_feat[c] += feats[mask].sum(dim=0)
                    count[c] += mask.sum()

        class_means = sum_feat / count.unsqueeze(1)

        # ---------- pass 2: covariance ----------
        self.pca_bases = self._compute_pca_bases_randomized(
            dataloader, class_means, device
        )
        self.class_means = class_means

    @torch.no_grad()
    def _compute_pca_bases_randomized(
        self, dataloader, class_means, device
    ):
        C = self.n_class
        k = self.pca_dim
        oversample = 5  # 常用设置

        # 每个类别一个 list
        feats_by_class = defaultdict(list)

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            feats = self.scatter_forward(imgs)

            for c in range(C):
                mask = labels == c
                if mask.any():
                    x = feats[mask] - class_means[c]
                    feats_by_class[c].append(x)

        pca_bases = []

        for c in range(C):
            X = torch.cat(feats_by_class[c], dim=0)  # [Nc, D]

            # ---------- randomized SVD ----------
            q = k + oversample
            R = torch.randn(X.shape[1], q, device=device)

            Y = X @ R                  # [Nc, q]
            Q, _ = torch.linalg.qr(Y)  # [Nc, q]

            B = Q.T @ X                # [q, D]
            _, _, Vh = torch.linalg.svd(B, full_matrices=False)

            V = Vh[:k].T               # [D, k]
            pca_bases.append(V)

        return torch.stack(pca_bases)  # [C, D, k]

    def build_keep_idx(self):
        d = self.depth
        all_out_groups, input_lens = self.compute_out_groups()

        keep_idx = []
        for groups in all_out_groups:
            idx = []
            for dep, start, end in groups:
                if dep <= d - 1:
                    idx.extend(range(start, end))
            keep_idx.append(torch.tensor(idx, dtype=torch.long))
            
        return keep_idx, input_lens

    def compute_out_groups(self):
        J = self.n_scale
        dmax = self.depth
        K = self.n_orient
        # inputs: list of (depth, channels)
        inputs_groups = [(0, 1)]  # img first, channel is gray

        all_out_groups = [[(0, 0, 1)]]  # first layer: only img
        inputs_len = [1]
        for ind in range(J):
            out_groups = []
            ch_ptr = 0

            for dep, ch in inputs_groups:
                new_dep = dep + 1
                new_ch = ch * K
                out_groups.append((new_dep, ch_ptr, ch_ptr + new_ch))
                ch_ptr += new_ch

            all_out_groups.append(out_groups)

            # build next inputs
            next_inputs = []

            for new_dep, start, end in out_groups:
                if new_dep <= dmax - 1:
                    next_inputs.append((new_dep, end - start))

            inputs_groups.extend(next_inputs)
            inputs_len.append(sum([ch for _, ch in inputs_groups]))
        return all_out_groups, inputs_len


    def set_filters(self):
        self.filters = filter_bank(self.wavelet, self.n_scale, self.n_orient)
        # self.filters[0] = torch.ones(2, 7, 7)
        # self.filters[0][0] = self.filters[0][0]*2
        # print(self.filters[0].shape)
        # print(self.filters[0])
        self.filters[0] = self.repeat_interleave_with_noise(self.filters[0])
        # print(self.filters[0].shape)
        # print(self.filters[0])
        if self.random:
            random_filters = []
            seeds = [42, 123, 999, 2025, 7]

            # Randomly select one seed
            seed = random.choice(seeds)

            # Set the seed for reproducibility
            torch.manual_seed(seed)

            torch.cuda.manual_seed_all(seed)

            # for filt in self.filters:
            #     temp = torch.randn_like(filt.real) + 1j * torch.randn_like(filt.real)
            #     random_filters.append(temp.to(dtype=filt.dtype))
            self.filters[0] = torch.randn_like(self.filters[0])
        

    def repeat_interleave_with_noise(self, x):
        """
        Repeat each filter n_copies times and add small random noise to each copy.
        The first K copies are kept unchanged to preserve the original filters.
        """
        K, S, _ = x.shape

        x = x.repeat(self.n_copies, 1, 1)
        noise = torch.randn_like(x) * 1e-3
        noise[:K, :, :] = 0.0
        y = x + noise
        y = y.view(self.n_copies, K, S, S).permute(1,0,2,3).reshape(K * self.n_copies, S, S)
        return y


    def scatter_forward(self, img):
        inputs = [img]
        outputs = [img]
        for ind, block in enumerate(self.blocks):
            #out = checkpoint(block, *inputs) if ind != 0 else block(*inputs)
            out = block(*inputs)
            outputs.append(out) # save results from all blocks outputs[-1]
            # 若depth无限制，则全部输出传递到下一层
            if self.depth == -1:
                outputs = [self.pooling(i) for i in outputs]
                inputs = outputs
                continue
            # 若depth有限制
            ## 当前尺度的输出不在最后一层，也把全部输出传递到下一层
            if ind + 1 < self.depth:
                next_in = outputs[-1]
            else: ## 否则只传递部分输出到下一层
                idx = getattr(self, f"keep_idx_{ind+1}")
                next_in = outputs[-1][:, idx, :, :]
            inputs.append(next_in)
            # 池化输出，低通和保持shape一样
            outputs = [self.pooling(i) for i in outputs]    
            inputs = [self.pooling(i) for i in inputs]
        if self.out_size != -1:
            features = self.global_pool(torch.cat(outputs, dim=1)).reshape(img.shape[0], -1)
        else:
            features = torch.cat(outputs, dim=1).reshape(img.shape[0], -1)
        return features #self.linear(features) #self.linear(F.normalize(features, p=2, dim=1))

    def forward(self, imgs):
        """
        imgs: [B, ...]
        return: logits [B, C] (负重构误差)
        """
        feats = self.scatter_forward(imgs)  # [B, D]

        B, D = feats.shape

        C = self.n_class

        scores = feats.new_empty(B, C)

        for c in range(C):
            mean = self.class_means[c]     # [D]
            V = self.pca_bases[c]          # [D, k]

            x = feats - mean               # [B, D]

            # low-dim coordinates: [B, k]
            alpha = x @ V                  # O(B D k)

            # reconstruction: [B, D]
            recon = alpha @ V.T

            residual = x - recon
            scores[:, c] = -residual.norm(dim=1)

        # for c in range(C):
        #     x = feats - self.class_means[c]    # [B, D]
        #     alpha = x @ self.pca_bases[c]      # [B, k]

        #     dist2 = (x ** 2).sum(dim=1) - (alpha ** 2).sum(dim=1)
        #     scores[:, c] = -torch.sqrt(dist2 + 1e-8)

        return scores


    def train_classifier(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        # for param in self.linear.parameters():
        #     param.requires_grad = True

    def train_conv(self):
        for param in self.blocks.parameters():
            param.requires_grad = True
        # for param in self.linear.parameters():
        #     param.requires_grad = False

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
