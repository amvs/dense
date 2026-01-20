"""
Factory functions for creating WPH models.
Used by train_wph.py and train_wph_svm.py.
"""
import os
import torch
from wph.wph_model import WPHModel, WPHModelDownsample
from dense.helpers import LoggerManager


def construct_filters_fullsize(config, image_shape):
    """
    Constructs the filters based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        image_shape (tuple): Shape of the input image.

    Returns:
        dict: Dictionary containing the constructed filters.
    """
    from dense.wavelets import filter_bank
    from wph.layers.utils import apply_phase_shifts
    
    logger = LoggerManager.get_logger()
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    num_phases = config["num_phases"]
    share_rotations = config.get("share_rotations", False)
    share_channels = config.get("share_channels", True)
    share_phases = config.get("share_phases", False)
    num_channels = image_shape[0]

    # Determine parameter shape based on sharing
    param_nc = 1 if share_channels else num_channels
    param_L = 1 if share_rotations else nb_orients
    param_A = 1 if share_phases else num_phases

    filter_dir = os.path.join(os.path.dirname(__file__), "../filters")
    hatpsi_path = os.path.join(filter_dir, f"morlet_N{image_shape[1]}_J{max_scale}_L{nb_orients}.pt")
    hatphi_path = os.path.join(filter_dir, f"morlet_lp_N{image_shape[1]}_J{max_scale}_L{nb_orients}.pt")

    if config.get("random_filters", False):
        logger.info("Initializing filters randomly.")
        filters = {
            "hatpsi": torch.complex(
                torch.randn(max_scale, param_L, image_shape[1], image_shape[2]),
                torch.randn(max_scale, param_L, image_shape[1], image_shape[2])
            ),
            "hatphi": torch.complex(
                torch.randn(1, image_shape[1], image_shape[2]),
                torch.randn(1, image_shape[1], image_shape[2])
            )
        }
    else:
        # Check if filters exist, otherwise generate them
        if not os.path.exists(hatpsi_path) or not os.path.exists(hatphi_path):
            logger.info("Filters not found. Generating filters...")
            build_filters_script = os.path.join(os.path.dirname(__file__), "../wph/ops/build-filters.py")
            os.system(f"python {build_filters_script} --N {image_shape[1]} --J {max_scale} --L {nb_orients} --wavelets morlet")
            
        # Load precomputed filters
        filters = {
            "hatpsi": torch.load(hatpsi_path, weights_only=True),
            "hatphi": torch.load(hatphi_path, weights_only=True)
        }

    # Apply phase shifts to the filters
    if share_rotations:
        filters['hatpsi'] = filters['hatpsi'][:,0,...].unsqueeze(1)
    filters["hatpsi"] = apply_phase_shifts(filters["hatpsi"], A=param_A)
    return filters


def construct_filters_downsample(config, image_shape):
    """
    Constructs the filters for the downsampled WPH model based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        image_shape (tuple): Shape of the input image.
    Returns:
        dict: Dictionary containing the constructed filters.
    """
    from dense.wavelets import filter_bank
    from wph.layers.utils import apply_phase_shifts
    
    logger = LoggerManager.get_logger()
    nb_orients = config["nb_orients"]
    num_phases = config["num_phases"]
    max_scale = config['max_scale']
    share_rotations = config.get("share_rotations", False)
    share_channels = config.get("share_channels", True)
    share_phases = config.get("share_phases", False)
    share_scales = config.get("share_scales", True)
    share_scale_pairs = config.get("share_scale_pairs", True)
    num_channels = image_shape[0]

    # Determine parameter shape based on sharing
    param_nc = 1 if share_channels else num_channels
    param_L = 1 if share_rotations else nb_orients
    param_A = 1 if share_phases else num_phases
    # If share_scales=True, overrides to share pairs; otherwise respect share_scale_pairs
    param_J = 1 if share_scales else (max_scale if share_scale_pairs else max_scale * max_scale)

    if config.get("random_filters", False):
        logger.info("Initializing filters randomly.")
        T = config.get("wavelet_params", {}).get("S", 3)
        filters = {
            "psi": torch.complex(
                torch.randn(param_J, param_L, param_A, T, T),
                torch.randn(param_J, param_L, param_A, T, T)
            ),
            "hatphi": torch.complex(
                torch.randn(1, image_shape[1], image_shape[2]),
                torch.randn(1, image_shape[1], image_shape[2])
            )
        }
    else:
        logger.info(f"Generating filters with base wavelet: {config.get('wavelet', 'morlet')}")
        filters = {}
        filters['psi'] = filter_bank(
            wavelet_name=config.get("wavelet", "morlet"),
            max_scale=1,
            nb_orients=nb_orients,
            **config.get("wavelet_params", {})
        )[0]  # Get first scale
        filter_dir = os.path.join(os.path.dirname(__file__), "../filters")
        hatphi_path = os.path.join(filter_dir, f"morlet_lp_N{image_shape[1]}_J1_L{nb_orients}.pt")
        if not os.path.exists(hatphi_path):
            logger.info("Filters not found. Generating filters...")
            build_filters_script = os.path.join(os.path.dirname(__file__), "../wph/ops/build-filters.py")
            os.system(f"python {build_filters_script} --N {image_shape[1]} --J {1} --L {nb_orients} --wavelets morlet")
            
        # Load precomputed filters
        filters["hatphi"] = torch.load(hatphi_path, weights_only=True)
        filters["psi"] = apply_phase_shifts(filters["psi"], A=param_A).squeeze(0) # squeeze J dim
        T = filters['psi'].shape[-1]
        if share_scales:
            filters['psi'] = filters['psi'].unsqueeze(0)  # Add J dim: (1, L, A, T, T)
        elif share_scale_pairs:
            # Replicate same filter across J scales: (J, L, A, T, T)
            filters['psi'] = torch.stack([filters['psi'].clone() for _ in range(max_scale)], dim=0)
        else:
            # Create J*J filters for pair mode with indexing: pair_index = j2 * J + j1
            base_filter = filters['psi']  # (L, A, T, T)
            psi_pairs = torch.zeros(max_scale * max_scale, param_L, param_A, T, T, dtype=base_filter.dtype)
            for j1 in range(max_scale):
                for j2 in range(max_scale):
                    pair_index = j2 * max_scale + j1
                    psi_pairs[pair_index] = base_filter.clone()
            filters['psi'] = psi_pairs
    return filters


def create_wph_feature_extractor(config, image_shape, device):
    """
    Create WPH feature extractor (full or downsampled) based on config.
    
    Args:
        config: Configuration dictionary
        image_shape: Tuple of (channels, height, width)
        device: torch.device for model placement
    
    Returns:
        tuple: (feature_extractor, filters)
    """
    logger = LoggerManager.get_logger()
    downsample = config.get("downsample", False)
    
    if downsample:
        filters = construct_filters_downsample(config, image_shape)
        T = filters['psi'].shape[-1]
        logger.info(f"Using downsampled WPH model with filter size T={T}")
        feature_extractor = WPHModelDownsample(
            J=config["max_scale"],
            L=config["nb_orients"],
            A=config["num_phases"],
            A_prime=config.get("num_phases_prime", 1),
            M=image_shape[1],
            N=image_shape[2],
            T=T,
            filters=filters,
            num_channels=image_shape[0],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            share_scales=config.get("share_scales", True),
            share_scale_pairs=config.get("share_scale_pairs", True),
            normalize_relu=config["normalize_relu"],
            delta_j=config.get("delta_j"),
            delta_l=config.get("delta_l"),
            shift_mode=config["shift_mode"],
            mask_angles=config["mask_angles"],
            mask_union_highpass=config["mask_union_highpass"],
            spatial_attn=config.get("spatial_attn", False),
            grad_checkpoint=config.get("grad_checkpoint", False),
        ).to(device)
    else:
        if config.get("wavelet", "morlet") != "morlet":
            logger.warning("Full-size WPHModel only supports 'morlet' wavelet. Overriding to 'morlet'.")
            config["wavelet"] = "morlet"
        filters = construct_filters_fullsize(config, image_shape)
        feature_extractor = WPHModel(
            J=config["max_scale"],
            L=config["nb_orients"],
            A=config["num_phases"],
            A_prime=config.get("num_phases_prime", 1),
            M=image_shape[1],
            N=image_shape[2],
            filters=filters,
            num_channels=image_shape[0],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            normalize_relu=config["normalize_relu"],
            delta_j=config.get("delta_j"),
            delta_l=config.get("delta_l"),
            shift_mode=config["shift_mode"],
            mask_union=config["mask_union"],
            mask_angles=config["mask_angles"],
            mask_union_highpass=config["mask_union_highpass"],
            grad_checkpoint=config.get('grad_checkpoint', False),
            spatial_attn=config.get("spatial_attn", False),
        ).to(device)
    
    return feature_extractor, filters
