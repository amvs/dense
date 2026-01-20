import torch
import torch.nn as nn
from dataclasses import dataclass
from .wavelets import filter_bank
from torch import Tensor
import torch.nn.functional as F
import random
from collections import defaultdict
import math
import numpy as np
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
    # depth
    depth: int = -1  # -1 means full scatter
    #
    random: bool = False
    # Classifier type: 'hypernetwork', or 'attention'
    classifier_type: str = 'hypernetwork'
    # Hypernetwork parameters
    hypernet_hidden_dim: int = 64
    # Attention parameters
    attention_d_model: int = 128
    attention_num_heads: int = 4
    attention_num_layers: int = 2

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


class HypernetworkClassifier(nn.Module):
    """
    Method 1: Hypernetwork Classifier
    Maps metadata to classifier weights, then uses those weights for classification.
    This reduces the number of parameters compared to a full linear classifier.
    """
    def __init__(self, num_features: int, num_classes: int, 
                feature_dim: int,
                metadata_dim: int = 5, 
                hidden_dim: int = 64):
        """
        Args:
            num_features: Number of feature maps (channels)
            num_classes: Number of classification classes
            feature_dim: Spatial dimension of each feature map (N*N)
            metadata_dim: Dimension of metadata vector (default 5: [depth, scale_1, angle_1, scale_2, angle_2])
            hidden_dim: Hidden dimension for the hypernetwork.
            
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # Hypernetwork: maps metadata to weight vectors
        # Output: weight matrix of shape [num_features, num_classes]
        self.hypernet = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim * num_classes)
        )
        
        # Optional bias
        self.bias = nn.Parameter(torch.zeros(num_classes))
        
    def forward(self, features, metadata):
        """
        Args:
            features: Tensor of shape [batch, num_features, feature_dim] - feature maps
            metadata: Tensor of shape [num_features, metadata_dim] - metadata for each feature map
        Returns:
            logits: Tensor of shape [batch, num_classes]
        """
        # Generate weights for each feature map using hypernetwork
        # weights: [num_features, feature_dim, num_classes]
        weights = self.hypernet(metadata).view(self.num_features, self.feature_dim, self.num_classes)
        
        # Apply generated weights: [batch, num_features, num_classes]
        per_node_logits = torch.einsum('bnd,ndc->bnc', features, weights)

        logits = per_node_logits.sum(dim=1) + self.bias
        
        return logits


class AttentionClassifier(nn.Module):
    """
    Method 2: Attention-based Classifier
    Treats feature maps as an unordered set, using metadata-aware attention mechanism.
    """
    def __init__(self, num_features: int, num_classes: int, metadata_dim: int = 5,
                 d_model: int = 128, num_heads: int = 4, num_layers: int = 2):
        """
        Args:
            num_features: Number of feature maps (channels)
            num_classes: Number of classification classes
            metadata_dim: Dimension of metadata vector (default 5)
            d_model: Model dimension for attention
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.d_model = d_model
        
        # Metadata encoding: maps metadata to d_model-dimensional embeddings
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Feature projection: maps feature maps to d_model
        self.feature_proj = nn.Linear(1, d_model)  # Each feature map is a single value after pooling
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )
        
        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, features, metadata):
        """
        Args:
            features: Tensor of shape [batch, num_features, H, W] - feature maps
            metadata: Tensor of shape [num_features, metadata_dim] - metadata for each feature map
        Returns:
            logits: Tensor of shape [batch, num_classes]
        """
        batch_size = features.shape[0]
        
        # Global average pooling: [batch, num_features, H, W] -> [batch, num_features]
        features_pooled = features.mean(dim=[2, 3])  # [batch, num_features]
        
        # Encode metadata: [num_features, metadata_dim] -> [num_features, d_model]
        metadata_emb = self.metadata_encoder(metadata)  # [num_features, d_model]
        
        # Project features: [batch, num_features] -> [batch, num_features, d_model]
        features_emb = self.feature_proj(features_pooled.unsqueeze(-1))  # [batch, num_features, d_model]
        
        # Combine feature and metadata embeddings
        # Option 1: Add them
        combined = features_emb + metadata_emb.unsqueeze(0)  # [batch, num_features, d_model]
        
        # Apply transformer: [batch, num_features, d_model] -> [batch, num_features, d_model]
        encoded = self.transformer(combined)
        
        # Aggregate using learned query (attention pooling)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        # Compute attention scores
        attn_scores = torch.bmm(query, encoded.transpose(1, 2))  # [batch, 1, num_features]
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Weighted sum
        aggregated = torch.bmm(attn_weights, encoded)  # [batch, 1, d_model]
        aggregated = aggregated.squeeze(1)  # [batch, d_model]
        
        # Classify
        logits = self.classifier(aggregated)  # [batch, num_classes]
        
        return logits



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
                   
        print("in_channels_per_block:", self.in_channels_per_block)
        print("out_channels:", self.out_channels)
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
        
        # Initialize classifier based on type
        self._init_classifier()

    def _init_classifier(self):
        """Initialize the classifier based on classifier_type."""
        # Compute metadata for all feature maps
        metadata = self.get_feature_metadata()
        num_features = metadata.shape[0]
        
        # Compute feature dimension after pooling (spatial size)
        final_size = self.in_size // (2 ** self.n_scale)
        feature_dim = final_size * final_size
        
        if self.classifier_type == 'hypernetwork':
            hypernet_hidden_dim = self.hypernet_hidden_dim
            self.classifier = HypernetworkClassifier(
                num_features=num_features,
                num_classes=self.n_class,
                feature_dim=feature_dim,
                metadata_dim=5,
                hidden_dim=hypernet_hidden_dim
            )
            # Register metadata as buffer so it's moved to device with model
            self.register_buffer('feature_metadata', metadata)
        elif self.classifier_type == 'attention':
            attention_d_model = self.attention_d_model
            attention_num_heads = self.attention_num_heads
            attention_num_layers = self.attention_num_layers
            self.classifier = AttentionClassifier(
                num_features=num_features,
                num_classes=self.n_class,
                metadata_dim=5,
                d_model=attention_d_model,
                num_heads=attention_num_heads,
                num_layers=attention_num_layers
            )
            # Register metadata as buffer
            self.register_buffer('feature_metadata', metadata)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")
    
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
        self.out_channels = sum([out_group[-1][-1] for out_group in all_out_groups])
        return keep_idx, input_lens

    def compute_out_groups(self):
        J = self.n_scale
        dmax = self.depth if self.depth != -1 else J
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
        # Convert numpy arrays to torch tensors if needed (e.g., for morlet wavelet)
        if isinstance(self.filters[0], np.ndarray):
            self.filters[0] = torch.tensor(self.filters[0], dtype=torch.complex64)
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


    def forward(self, img):
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
        features = torch.cat(outputs, dim=1)
        
        # Apply classifier
        if self.classifier_type == 'hypernetwork':
            # Reshape features: [batch, num_features, H, W] -> [batch, num_features, H*W]
            batch_size = features.shape[0]
            num_features = features.shape[1]
            H, W = features.shape[2], features.shape[3]
            features_reshaped = features.view(batch_size, num_features, H * W)
            # Use metadata-aware classifier
            metadata = self.feature_metadata
            return self.classifier(features_reshaped, metadata)
        elif self.classifier_type == 'attention':
            # Attention classifier expects [batch, num_features, H, W] and does pooling internally
            metadata = self.feature_metadata
            return self.classifier(features, metadata)
        else:
            raise ValueError(f"Unknown classifier_type: {self.classifier_type}")



    def train_classifier(self):
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def train_conv(self):
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
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

    def n_classifier_params(self):
        total_params = 0
        for param in self.classifier.parameters():
                total_params += param.numel()
        return total_params
    

    def get_feature_metadata(self):
        """
        Compute metadata for each feature map channel.
        Returns a tensor of shape [num_features, 5] where each row is:
        [depth, scale_1, angle_1, scale_2, angle_2]
        
        Features are organized as concatenated outputs from all blocks:
        - outputs[0]: depth=0 (input image, 1 channel)
        - outputs[1]: from block 0 (scale 0)
        - outputs[2]: from block 1 (scale 1)
        - etc.
        
        Uses compute_out_groups to track the exact organization.
        """
        J = self.n_scale
        K = self.n_orient
        dmax = self.depth if self.depth != -1 else J
        
        # Build metadata by tracking paths through the scattering tree
        # Track: (depth, scales_path, angles_path) for each input group
        input_paths = [(0, [], [])]  # Start with depth=0 input image
        
        all_metadata = []
        
        # Depth 0: input image
        all_metadata.append((0, [], []))
        
        for scale in range(J):
            output_paths = []
            
            # Process each input path
            for dep, scales_path, angles_path in input_paths:
                if dep >= dmax:
                    continue
                
                # Each input path produces K output paths (one per orientation)
                for angle in range(K):
                    new_depth = dep + 1
                    new_scales = scales_path + [scale]
                    new_angles = angles_path + [angle]
                    output_paths.append((new_depth, new_scales, new_angles))
                    all_metadata.append((new_depth, new_scales, new_angles))
            
            # Update input paths for next scale
            if scale < J - 1:
                # Keep all previous paths (they continue as low-pass)
                input_paths = [
                    (dep, scales, angles) 
                    for dep, scales, angles in input_paths 
                    if dep < dmax
                ]
                # Add new output paths
                input_paths.extend(output_paths)
        
        # Convert to tensor format [depth, scale_1, angle_1, scale_2, angle_2]
        metadata_tensor = torch.zeros(len(all_metadata), 5, dtype=torch.long)
        for idx, (depth, scales, angles) in enumerate(all_metadata):
            metadata_tensor[idx, 0] = depth
            if len(scales) >= 1:
                metadata_tensor[idx, 1] = scales[0]
                metadata_tensor[idx, 2] = angles[0]
            else:
                metadata_tensor[idx, 1] = -1
                metadata_tensor[idx, 2] = -1
            if len(scales) >= 2:
                metadata_tensor[idx, 3] = scales[1]
                metadata_tensor[idx, 4] = angles[1]
            else:
                metadata_tensor[idx, 3] = -1
                metadata_tensor[idx, 4] = -1
        
        return metadata_tensor