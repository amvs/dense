import torch
from torch import nn
from typing import Optional, Literal, Union, Tuple
from torch.fft import fft2, ifft2

from wph.layers.wave_conv_layer import WaveConvLayer
from wph.layers.relu_center_layer import ReluCenterLayer
from wph.layers.corr_layer import CorrLayer
from wph.layers.lowpass_layer import LowpassLayer
from wph.layers.highpass_layer import HighpassLayer


class WPHModel(nn.Module):
    def __init__(
        self,
        J: int,
        L: int,
        A: int,
        A_prime: int,
        M: int,
        N: int,
        filters: torch.Tensor,
        num_channels: int = 1,
        share_rotations: bool = False,
        share_phases: bool = False,
        share_channels: bool = True,
        normalize_relu: bool = True,
        delta_j: Optional[int] = None,
        delta_l: Optional[int] = None,
        shift_mode: Literal["samec", "all", "strict"] = "samec",
        mask_union: bool = False,
        mask_angles: int = 4,
        mask_union_highpass: bool = True,
        wavelets: Literal["morlet", "steer"] = 'morlet',
    ):
        super().__init__()
        self.J = J
        self.L = L
        self.A = A
        self.A_prime = A_prime
        self.M = M
        self.N = N
        self.num_channels = num_channels
        self.share_rotations = share_rotations
        self.share_phases = share_phases
        self.share_channels = share_channels
        self.delta_j = delta_j if delta_j is not None else J
        self.delta_l = delta_l if delta_l is not None else L
        self.shift_mode = shift_mode
        self.mask_union = mask_union
        self.mask_angles = mask_angles
        self.wavelets = wavelets
        self.mask_union_highpass = mask_union_highpass
        A_param = 1 if share_phases else A
        
        assert filters['hatpsi'].shape[-3] == A_param, "filters['hatpsi'] must have A phase shifts (or shape 1 if sharing phase shifts), has shape {}".format(filters['hatpsi'].shape)
        if len(filters['hatpsi'].shape) == 5:
            filters['hatpsi'] = filters['hatpsi'].unsqueeze(0).repeat(num_channels, 1, 1, 1, 1, 1)
        
        self.wave_conv = WaveConvLayer(
            J=J,
            L=L,
            A=A,
            M=M,
            N=N,
            num_channels=num_channels,
            filters=filters['hatpsi'],
            share_rotations=share_rotations,
            share_phases=share_phases,
            share_channels=share_channels,
        )
        self.relu_center = ReluCenterLayer(J=J, M=M, N=N, normalize=normalize_relu)
        self.corr = CorrLayer(
            J=J,
            L=L,
            A=A,
            A_prime=A_prime,
            M=M,
            N=N,
            num_channels=num_channels,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            shift_mode=shift_mode,
            mask_union=mask_union,
            mask_angles=mask_angles,
        )
        self.lowpass = LowpassLayer(
            J=J,
            M=M,
            N=N,
            num_channels=num_channels,
            hatphi=filters["hatphi"],
            mask_angles=mask_angles,
            mask_union=mask_union,
        )
        self.highpass = HighpassLayer(
            J=J,
            M=M,
            N=N,
            wavelets=self.wavelets,
            num_channels=num_channels,
            mask_angles=self.mask_angles,
            mask_union=self.mask_union,
            mask_union_highpass=self.mask_union_highpass,
        )
        self.nb_moments = self.corr.nb_moments + self.lowpass.nb_moments + self.highpass.nb_moments
        
    def forward(self, x: torch.Tensor, flatten: bool = True, vmap_chunk_size=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        nb = x.shape[0]
        xpsi = self.wave_conv(x)
        xrelu = self.relu_center(xpsi)
        xcorr = self.corr(xrelu.view(nb, self.num_channels * self.J * self.L * self.A, self.M, self.N), flatten=flatten, vmap_chunk_size=vmap_chunk_size)
        hatx_c = fft2(x)
        xlow = self.lowpass(hatx_c)
        xhigh = self.highpass(hatx_c)
        
        if flatten:
            return torch.cat([xcorr, xlow.flatten(start_dim=1), xhigh], dim=1)
        else:
            return xcorr, xlow, xhigh

class WPHClassifier(nn.Module):
    def __init__(self, feature_extractor: nn.Module, num_classes: int, use_batch_norm: bool = False):
        """
        A wrapper class for classification using WPHModel as a feature extractor.

        Args:
            feature_extractor (nn.Module): The feature extractor model (e.g., WPHModel).
            num_classes (int): Number of classes for classification.
            use_batch_norm (bool): Whether to include a batch normalization layer before the classifier.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm

        # Define the batch normalization layer (optional)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.feature_extractor.nb_moments)
        else:
            self.batch_norm = None

        # Define the classifier layer
        self.classifier = nn.Linear(self.feature_extractor.nb_moments, num_classes)

    def forward(self, x: torch.Tensor, vmap_chunk_size=None) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            x (torch.Tensor): Input tensor.
            vmap_chunk_size (int, optional): Chunk size for vmap operations when computing correlations.

        Returns:
            torch.Tensor: Classification logits.
        """
        # Always flatten features for classifier
        features = self.feature_extractor(x, flatten=True, vmap_chunk_size=vmap_chunk_size)

        # Apply batch normalization if enabled
        if self.batch_norm is not None:
            features = self.batch_norm(features)
        logits = self.classifier(features)
        return logits

    def set_trainable(self, parts: dict):
        """
        Set trainable status for different parts of the model.

        Args:
            parts (dict): A dictionary with keys 'feature_extractor' and 'classifier',
                          and boolean values indicating whether each part should be trainable.
        """
        if 'feature_extractor' in parts:
            for param in self.feature_extractor.parameters():
                param.requires_grad = parts['feature_extractor']

        if 'classifier' in parts:
            for param in self.classifier.parameters():
                param.requires_grad = parts['classifier']

    def fine_tuned_params(self):
        """
        Get parameters of the feature extractor that are set to be trainable.

        Returns:
            list: List of trainable parameters from the feature extractor.
        """
        return [param for param in self.feature_extractor.parameters() if param.requires_grad]