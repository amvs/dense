import torch
from torch import nn
from typing import Optional, Literal, Union, Tuple
from torch.fft import fft2, ifft2
import copy
import warnings
import numpy as np

from wph.layers.wave_conv_layer import WaveConvLayer, WaveConvLayerDownsample
from wph.layers.relu_center_layer import ReluCenterLayer, ReluCenterLayerDownsample, ReluCenterLayerDownsamplePairs
from wph.layers.corr_layer import CorrLayer, CorrLayerDownsample, CorrLayerDownsamplePairs
from wph.layers.lowpass_layer import LowpassLayer
from wph.layers.highpass_layer import HighpassLayer
from wph.layers.spatial_attent_layer import SpatialAttentionLayer
from dense.helpers import LoggerManager

class WPHFeatureBase(nn.Module):
    def __init__(self,
                 J: int,
                L: int,
                A: int,
                A_prime: int,
                M: int,
                N: int,
                num_channels: int = 1,
                share_rotations: bool = False,
                share_phases: bool = False,
                share_channels: bool = True,
                share_scales: bool = False,
                normalize_relu: bool = True,
                delta_j: Optional[int] = None,
                delta_l: Optional[int] = None,
                shift_mode: Literal["samec", "all", "strict"] = "samec",
                mask_angles: int = 4,
                mask_union_highpass: bool = True,
                spatial_attn: bool = False,
                grad_checkpoint:  bool = False,
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
        self.share_scales = share_scales
        if self.share_channels:
            logger = LoggerManager.get_logger()
            logger.warning("share_channels=True is not implemented yet; defaulting to share_channels=False")
        self.delta_j = delta_j if delta_j is not None else J
        self.delta_l = delta_l if delta_l is not None else L
        self.shift_mode = shift_mode
        self.mask_angles = mask_angles
        self.mask_union_highpass = mask_union_highpass
        self.normalize_relu = normalize_relu
        self.spatial_attn = spatial_attn
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Subclasses should implement this!")

class WPHModel(WPHFeatureBase):
    def __init__(
        self,
        filters: dict[str, torch.Tensor],
        mask_union: bool = False,
        mask_union_highpass: bool = False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_union = mask_union
        self.mask_union_highpass = mask_union_highpass
        if self.share_scales:
            logger = LoggerManager.get_logger()
            logger.warning("share_scales=True is not implemented; defaulting to share_scales=False")
        A_param = 1 if self.share_phases else self.A
        assert filters['hatpsi'].shape[-3] == A_param, "filters['hatpsi'] must have A phase shifts (or shape 1 if sharing phase shifts), has shape {}".format(filters['hatpsi'].shape)
        if len(filters['hatpsi'].shape) == 5:
            filters['hatpsi'] = filters['hatpsi'].unsqueeze(0).repeat(self.num_channels, 1, 1, 1, 1, 1)
        
        self.wave_conv = WaveConvLayer(
            J=self.J,
            L=self.L,
            A=self.A,
            M=self.M,
            N=self.N,
            num_channels=self.num_channels,
            filters=filters['hatpsi'],
            share_rotations=self.share_rotations,
            share_phases=self.share_phases,
            share_channels=self.share_channels,
        )
        self.relu_center = ReluCenterLayer(J=self.J, M=self.M, N=self.N, normalize=self.normalize_relu)
        self.corr = CorrLayer(
            J=self.J,
            L=self.L,
            A=self.A,
            A_prime=self.A_prime,
            M=self.M,
            N=self.N,
            num_channels=self.num_channels,
            delta_j=self.delta_j,
            delta_l=self.delta_l,
            shift_mode=self.shift_mode,
            mask_union=self.mask_union,
            mask_angles=self.mask_angles,
        )
        self.lowpass = LowpassLayer(
            J=self.J,
            M=self.M,
            N=self.N,
            num_channels=self.num_channels,
            hatphi=filters["hatphi"],
            mask_angles=self.mask_angles,
            mask_union=self.mask_union,
        )
        self.highpass = HighpassLayer(
            J=self.J,
            M=self.M,
            N=self.N,
            num_channels=self.num_channels,
            mask_angles=self.mask_angles,
            mask_union=self.mask_union,
            mask_union_highpass=self.mask_union_highpass,
        )
        if self.spatial_attn:
            self.attent = SpatialAttentionLayer()
        self.nb_moments = self.corr.nb_moments + self.lowpass.nb_moments + self.highpass.nb_moments
        
    def flat_metadata(self):
        """Combine metadata from all three layers (corr, lowpass, highpass) with layer indicator.
        
        Returns consolidated metadata with common keys:
        - scale1, scale2: scales (scale2=-1 for lowpass/highpass)
        - rotation1, rotation2: rotations (rotation2=-1 for lowpass/highpass, rotation1=haar_idx for highpass)
        - phase1, phase2: phases (-1 for lowpass/highpass)
        - channel1, channel2: channels (channel2=-1 for highpass)
        - mask_pos: mask position index
        - layer: 0=corr, 1=lowpass, 2=highpass
        - layer_feature_idx: index within that layer
        """
        corr_meta = self.corr.flat_metadata()
        lowpass_meta = self.lowpass.flat_metadata()
        highpass_meta = self.highpass.flat_metadata()
        
        # Define all common keys
        combined_meta = {
            "layer": [],
            "layer_feature_idx": [],
            "scale1": [],
            "scale2": [],
            "rotation1": [],
            "rotation2": [],
            "phase1": [],
            "phase2": [],
            "channel1": [],
            "channel2": [],
            "mask_pos": [],
        }
        
        # Corr layer features (layer_id=0) - has all dimensions
        n_corr = len(corr_meta["scale1"])
        for i in range(n_corr):
            combined_meta["layer"].append(0)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(corr_meta["scale1"][i])
            combined_meta["scale2"].append(corr_meta["scale2"][i])
            combined_meta["rotation1"].append(corr_meta["rotation1"][i])
            combined_meta["rotation2"].append(corr_meta["rotation2"][i])
            combined_meta["phase1"].append(corr_meta["phase1"][i])
            combined_meta["phase2"].append(corr_meta["phase2"][i])
            combined_meta["channel1"].append(-1)  # Corr doesn't track channels explicitly
            combined_meta["channel2"].append(-1)
            combined_meta["mask_pos"].append(corr_meta["mask_pos"][i])
        
        # Lowpass layer features (layer_id=1) - has channel1, channel2 (no scale)
        n_lowpass = len(lowpass_meta["channel1"])
        for i in range(n_lowpass):
            combined_meta["layer"].append(1)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(-1)
            combined_meta["scale2"].append(-1)
            combined_meta["rotation1"].append(-1)
            combined_meta["rotation2"].append(-1)
            combined_meta["phase1"].append(-1)
            combined_meta["phase2"].append(-1)
            combined_meta["channel1"].append(lowpass_meta["channel1"][i])
            combined_meta["channel2"].append(lowpass_meta["channel2"][i])
            combined_meta["mask_pos"].append(lowpass_meta["mask_pos"][i])
        
        # Highpass layer features (layer_id=2) - has rotation1 (haar idx), channel1
        n_highpass = len(highpass_meta["rotation1"])
        for i in range(n_highpass):
            combined_meta["layer"].append(2)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(-1)
            combined_meta["scale2"].append(-1)
            combined_meta["rotation1"].append(highpass_meta["rotation1"][i])  # Haar filter index
            combined_meta["rotation2"].append(-1)
            combined_meta["phase1"].append(-1)
            combined_meta["phase2"].append(-1)
            combined_meta["channel1"].append(highpass_meta["channel1"][i])
            combined_meta["channel2"].append(-1)
            combined_meta["mask_pos"].append(highpass_meta["mask_pos"][i])
        return combined_meta
        
    def forward(self, x: torch.Tensor, flatten: bool = True, vmap_chunk_size=None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        nb = x.shape[0]
        xpsi = self.wave_conv(x)
        if self.spatial_attn:
            xpsi = self.attent(xpsi)
        xrelu = self.relu_center(xpsi)
        xcorr = self.corr(xrelu.view(nb, self.num_channels * self.J * self.L * self.A, self.M, self.N), flatten=flatten, vmap_chunk_size=vmap_chunk_size, use_checkpoint=self.grad_checkpoint)
        hatx_c = fft2(x)
        xlow = self.lowpass(hatx_c)
        xhigh = self.highpass(hatx_c)
        
        if flatten:
            return torch.cat([xcorr, xlow.flatten(start_dim=1), xhigh], dim=1)
        else:
            return xcorr, xlow, xhigh


class WPHModelDownsample(WPHFeatureBase):
    def __init__(
            self,
            T: int,
            filters: dict[str, torch.Tensor],
            hatphi: torch.Tensor = None,
            share_scale_pairs: bool = True,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.T = T
        # expose pair sharing; share_scales overrides to True behavior
        self.share_scale_pairs = True if self.share_scales else share_scale_pairs
        A_param = 1 if self.share_phases else self.A
        L_param = 1 if self.share_rotations else self.L
        J_param = 1 if self.share_scales else (self.J if self.share_scale_pairs else self.J * self.J)
        assert filters['psi'].shape == (J_param, L_param, A_param, T, T), "filters['psi'] must have shape (L or 1, A or 1, T, T), has shape {}".format(filters['psi'].shape)
        
        self.wave_conv = WaveConvLayerDownsample(J = self.J,
                                                       L = self.L,
                                                       A = self.A,
                                                       T=self.T,
                                                       num_channels = self.num_channels,
                                                       share_rotations= self.share_rotations,
                                                       share_phases=self.share_phases,
                                                       share_channels=self.share_channels,
                                                       share_scales=self.share_scales,
                                                       share_scale_pairs=self.share_scale_pairs,
                                                       init_filters = filters['psi'])
        
        if self.share_scale_pairs:
            self.relu_center = ReluCenterLayerDownsample(J=self.J,
                                                          M=self.M,
                                                          N=self.N,
                                                          normalize=self.normalize_relu)
            self.corr = CorrLayerDownsample(J=self.J,
                                                L=self.L,
                                                A=self.A,
                                                A_prime=self.A_prime,
                                                M=self.M,
                                                N=self.N,
                                                num_channels=self.num_channels,
                                                delta_j=self.delta_j,
                                                delta_l=self.delta_l,
                                                shift_mode=self.shift_mode,
                                                mask_angles=self.mask_angles)
        else:
            self.relu_center = ReluCenterLayerDownsamplePairs(J=self.J,
                                                          M=self.M,
                                                          N=self.N,
                                                          normalize=self.normalize_relu)
            self.corr = CorrLayerDownsamplePairs(J=self.J,
                                                L=self.L,
                                                A=self.A,
                                                A_prime=self.A_prime,
                                                M=self.M,
                                                N=self.N,
                                                num_channels=self.num_channels,
                                                delta_j=self.delta_j,
                                                delta_l=self.delta_l,
                                                shift_mode=self.shift_mode,
                                                mask_angles=self.mask_angles)
        self.highpass = HighpassLayer(J = self.J,
                                       M = self.M,
                                       N = self.N,
                                       num_channels=self.num_channels,
                                       mask_angles=self.mask_angles,
                                       mask_union=False,
                                       mask_union_highpass=self.mask_union_highpass)
        self.lowpass = LowpassLayer(J = self.J,
                                     M = self.M,
                                     N = self.N,
                                     num_channels=self.num_channels,
                                     hatphi=filters["hatphi"],
                                     mask_angles=self.mask_angles,
                                     mask_union=False)
        if self.spatial_attn:
            self.attent = SpatialAttentionLayer()
        self.nb_moments = self.corr.nb_moments + self.lowpass.nb_moments + self.highpass.nb_moments
    
    def flat_metadata(self):
        """Combine metadata from all three layers (corr, lowpass, highpass) with layer indicator.
        
        Returns consolidated metadata with common keys:
        - scale1, scale2: scales (scale2=-1 for lowpass/highpass)
        - rotation1, rotation2: rotations (rotation2=-1 for lowpass/highpass, rotation1=haar_idx for highpass)
        - phase1, phase2: phases (-1 for lowpass/highpass)
        - channel1, channel2: channels (channel2=-1 for highpass)
        - mask_pos: mask position index
        - layer: 0=corr, 1=lowpass, 2=highpass
        - layer_feature_idx: index within that layer
        """
        corr_meta = self.corr.flat_metadata()
        lowpass_meta = self.lowpass.flat_metadata()
        highpass_meta = self.highpass.flat_metadata()
        
        # Define all common keys
        combined_meta = {
            "layer": [],
            "layer_feature_idx": [],
            "scale1": [],
            "scale2": [],
            "rotation1": [],
            "rotation2": [],
            "phase1": [],
            "phase2": [],
            "channel1": [],
            "channel2": [],
            "mask_pos": [],
        }
        
        # Corr layer features (layer_id=0) - has all dimensions
        n_corr = len(corr_meta["scale1"])
        for i in range(n_corr):
            combined_meta["layer"].append(0)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(corr_meta["scale1"][i])
            combined_meta["scale2"].append(corr_meta["scale2"][i])
            combined_meta["rotation1"].append(corr_meta["rotation1"][i])
            combined_meta["rotation2"].append(corr_meta["rotation2"][i])
            combined_meta["phase1"].append(corr_meta["phase1"][i])
            combined_meta["phase2"].append(corr_meta["phase2"][i])
            combined_meta["channel1"].append(-1)  # Corr doesn't track channels explicitly
            combined_meta["channel2"].append(-1)
            combined_meta["mask_pos"].append(corr_meta["mask_pos"][i])
        
        # Lowpass layer features (layer_id=1) - has channel1, channel2 (no scale)
        n_lowpass = len(lowpass_meta["channel1"])
        for i in range(n_lowpass):
            combined_meta["layer"].append(1)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(-1)
            combined_meta["scale2"].append(-1)
            combined_meta["rotation1"].append(-1)
            combined_meta["rotation2"].append(-1)
            combined_meta["phase1"].append(-1)
            combined_meta["phase2"].append(-1)
            combined_meta["channel1"].append(lowpass_meta["channel1"][i])
            combined_meta["channel2"].append(lowpass_meta["channel2"][i])
            combined_meta["mask_pos"].append(lowpass_meta["mask_pos"][i])
        
        # Highpass layer features (layer_id=2) - has rotation1 (haar filter idx), channel1
        n_highpass = len(highpass_meta["rotation1"])
        for i in range(n_highpass):
            combined_meta["layer"].append(2)
            combined_meta["layer_feature_idx"].append(i)
            combined_meta["scale1"].append(-1)
            combined_meta["scale2"].append(-1)
            combined_meta["rotation1"].append(highpass_meta["rotation1"][i])  # Haar filter index
            combined_meta["rotation2"].append(-1)
            combined_meta["phase1"].append(-1)
            combined_meta["phase2"].append(-1)
            combined_meta["channel1"].append(highpass_meta["channel1"][i])
            combined_meta["channel2"].append(-1)
            combined_meta["mask_pos"].append(highpass_meta["mask_pos"][i])
        return combined_meta
            
    def forward(self, x: torch.Tensor, flatten: bool = True, vmap_chunk_size=None) -> torch.Tensor:
        if self.share_scale_pairs:
            xpsi = self.wave_conv(x)
            if self.spatial_attn:
                xpsi = [self.attent(x) for x in xpsi]
            xrelu = self.relu_center(xpsi)
            xcorr = self.corr(xrelu, flatten=flatten, vmap_chunk_size=vmap_chunk_size, use_checkpoint=self.grad_checkpoint)
        else:
            # compute only required pairs
            # warm up indices by accessing property (built in __init__)
            needed_pairs = sorted(self.corr.grouped_indices.keys())
            xpsi_nested = self.wave_conv(x, scale_pairs=needed_pairs)
            if self.spatial_attn:
                xpsi_nested = [self.attent(x) for x in xpsi_nested]
            xrelu = self.relu_center(xpsi_nested)
            xcorr = self.corr(xrelu, flatten=flatten, vmap_chunk_size=vmap_chunk_size, use_checkpoint=self.grad_checkpoint)
        hatx_c = fft2(x)
        xlow = self.lowpass(hatx_c)
        xhigh = self.highpass(hatx_c)
        
        if flatten:
            return torch.cat([xcorr, xlow.flatten(start_dim=1), xhigh], dim=1)
        else:
            return xcorr, xlow, xhigh
        

class WPHClassifier(nn.Module):
    def __init__(self, feature_extractor: WPHFeatureBase, classifier: nn.Module, use_batch_norm: bool = False, copies: int = 1, noise_std: float = 0.01):
        """
        Lightweight wrapper for classification using WPHModel as a feature extractor.

        Args:
            feature_extractor (nn.Module): The feature extractor model (e.g., WPHModel or WPHModelDownsample).
            classifier (nn.Module): The classifier module (e.g., LinearClassifier, HyperNetworkClassifier, etc.).
            use_batch_norm (bool): Whether to use batch normalization. Default is False.
            noise_std (float): Standard deviation of noise added to filters for each copy. Default is 0.01.
        """
        super().__init__()
        self.copies = copies
        self.noise_std = noise_std
        self.classifier = classifier
        self.use_batch_norm = use_batch_norm
        # store copies of feature extractor in module list
        self.feature_extractors = nn.ModuleList([feature_extractor])

        if copies > 1:
            extra_copies = [self._deep_copy_feature_extractor(feature_extractor) for _ in range(copies - 1)]
            for fe_copy in extra_copies:
                self._add_noise_to_copy(fe_copy, noise_std)
            self.feature_extractors.extend(extra_copies)

            # Trainable averaging layer initialized to uniform weights (1/copies)
            self.ensemble_weights = nn.Linear(copies, 1, bias=False)
            with torch.no_grad():
                self.ensemble_weights.weight.fill_(1.0 / copies)
        else:
            self.ensemble_weights = None
        
        # Define the batch normalization layer (optional)
        nb_moments_int = int(feature_extractor.nb_moments)
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(nb_moments_int)
        else:
            self.batch_norm = None

    @staticmethod
    def _deep_copy_feature_extractor(feature_extractor):
        """Create a deep copy of the feature extractor."""
        return copy.deepcopy(feature_extractor)

    @staticmethod
    def _add_noise_to_copy(feature_extractor, noise_std: float):
        """
        Add small random noise to the filters of a feature extractor copy.
        Handles both WPHModel and WPHModelDownsample architectures.
        Applies noise to base_filters (WaveConvLayer) or base_real/base_imag (WaveConvLayerDownsample).
        Also applies noise to hatphi in LowpassLayer for both.
        """
        if not hasattr(feature_extractor, 'wave_conv'):
            return
        
        wave_conv = feature_extractor.wave_conv
        
        # Add noise to wave_conv filters - handle both layer types
        if hasattr(wave_conv, 'base_filters'):
            # WaveConvLayer (used by WPHModel)
            with torch.no_grad():
                wave_conv.base_filters.add_(torch.randn_like(wave_conv.base_filters) * noise_std)
            if hasattr(wave_conv, '_invalidate_cache'):
                wave_conv._invalidate_cache()

        # WaveConvLayerDownsample uses base_real/base_imag
        if hasattr(wave_conv, 'base_real'):
            with torch.no_grad():
                wave_conv.base_real.add_(torch.randn_like(wave_conv.base_real) * noise_std)
            if hasattr(wave_conv, '_invalidate_cache'):
                wave_conv._invalidate_cache()

        if hasattr(wave_conv, 'base_imag'):
            with torch.no_grad():
                wave_conv.base_imag.add_(torch.randn_like(wave_conv.base_imag) * noise_std)
            if hasattr(wave_conv, '_invalidate_cache'):
                wave_conv._invalidate_cache()
        
        # Add noise to lowpass filter (hatphi) for both architectures
        if hasattr(feature_extractor, 'lowpass'):
            lowpass = feature_extractor.lowpass
            if hasattr(lowpass, 'hatphi'):
                with torch.no_grad():
                    noisy_hatphi = lowpass.hatphi + torch.randn_like(lowpass.hatphi) * noise_std
                # Remove buffer and register as parameter
                if 'hatphi' in lowpass._buffers:
                    del lowpass._buffers['hatphi']
                lowpass.register_parameter('hatphi', nn.Parameter(noisy_hatphi))

    def extract_features(self, x: torch.Tensor, vmap_chunk_size=None) -> torch.Tensor:
        """
        Extract features from the input using the feature extractor(s).
        
        Args:
            x (torch.Tensor): Input tensor.
            vmap_chunk_size (int, optional): Chunk size for vmap operations.
            
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, nb_moments).
        """
        features_list = [fe(x, flatten=True, vmap_chunk_size=vmap_chunk_size) for fe in self.feature_extractors]

        if self.ensemble_weights is None:
            return features_list[0]
        
        if len(features_list) == 1:
            return features_list[0]
        
        # Stack features: (batch_size, nb_moments, copies)
        features_stack = torch.stack(features_list, dim=2)
        # Apply trainable averaging: (batch_size, nb_moments, 1)
        return self.ensemble_weights(features_stack).squeeze(2)

    def forward(self, x: torch.Tensor, vmap_chunk_size=None, return_feats=False) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            x (torch.Tensor): Input tensor.
            vmap_chunk_size (int, optional): Chunk size for vmap operations when computing correlations.
            return_feats (bool): Whether to return features along with logits.

        Returns:
            torch.Tensor: Classification logits (or tuple if return_feats=True).
        """
        features = self.extract_features(x, vmap_chunk_size=vmap_chunk_size)
        if self.batch_norm is not None:
            features = self.batch_norm(features)
        
        # Pass features to classifier
        logits = self.classifier(features)
        
        if return_feats:
            return logits, features
        else:
            return logits

    def set_trainable(self, parts: dict):
        """
        Set trainable status for different parts of the model.

        Args:   
            parts (dict): A dictionary with keys 'feature_extractor', 'classifier', and optionally 'spatial_attn',
                          and boolean values indicating whether each part should be trainable.
        """
        if 'feature_extractor' in parts:
            trainable = parts['feature_extractor']
            for fe in self.feature_extractors:
                for param in fe.parameters():
                    param.requires_grad = trainable

        if 'classifier' in parts:
            for param in self.classifier.parameters():
                param.requires_grad = parts['classifier']

        if 'spatial_attn' in parts and hasattr(self.feature_extractors[0], 'spatial_attn') and self.feature_extractors[0].spatial_attn:
            trainable = parts['spatial_attn']
            for fe in self.feature_extractors:
                if hasattr(fe, 'attent'):
                    for param in fe.attent.parameters():
                        param.requires_grad = trainable

    def fine_tuned_params(self):
        """
        Get parameters of the feature extractor that are set to be trainable.

        Returns:
            list: List of trainable parameters from the feature extractor.
        """
        params = []
        for fe in self.feature_extractors:
            params.extend([param for param in fe.parameters() if param.requires_grad])
        return params
    
    @property
    def nb_moments(self):
        """Get the number of moments/features from the feature extractor."""
        return self.feature_extractors[0].nb_moments