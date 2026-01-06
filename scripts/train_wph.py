import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import sys
import random
import numpy as np
import time
import platform
from functools import partial
from dotenv import load_dotenv
load_dotenv()

from configs import load_config, save_config, apply_overrides
from training.datasets import get_loaders, split_train_val
from training import train_one_epoch, evaluate
from wph.wph_model import WPHModel, WPHModelDownsample, WPHClassifier
from dense.helpers import LoggerManager
from dense.wavelets import filter_bank
from wph.layers.utils import apply_phase_shifts



def parse_args():
    parser = argparse.ArgumentParser(description="Train WPHClassifier with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
    parser.add_argument("--override", nargs="*", default=[],
                        help="List of key=value pairs to override config")
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="If this is a sweep job, specify the sweep output dir")
    parser.add_argument("--skip-classifier-training", action='store_true',
                        help="If set, skip the initial classifier training phase")
    parser.add_argument("--skip-finetuning", action='store_true',
                        help="If set, skip the fine-tuning feature extractor phase")
    parser.add_argument("--wandb_project", type=str, default="WPHWavelet",)
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)

def worker_init_fn(worker_id, seed=None):
    """Ensure deterministic behavior in DataLoader workers."""
    if seed is not None:
        np.random.seed(seed + worker_id)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, logger, phase, configs, exp_dir, original_params=None):
    """
    A reusable function for training the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler): Learning rate scheduler.
        criterion (Loss): Loss function.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs to train.
        logger (Logger): Logger for logging progress.
        phase (str): Phase of training (e.g., 'classifier', 'feature_extractor').
        configs (dict): Configuration dictionary.
        original_params (list[torch.Tensor], optional): Original parameters for regularization during fine-tuning.

    Returns:
        None
    """
    best_acc = 0.0  # Initialize best accuracy
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            base_loss=criterion,
            device=device,
            lambda_reg=configs['lambda_reg'] if phase == 'feature_extractor' else 0.0,
            original_params=original_params if phase == 'feature_extractor' else None,
            vmap_chunk_size=configs.get('vmap_chunk_size', None),
            normalize_reg=configs.get('normalize_reg', True)
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.log("Phase: {} Epoch: {}".format(phase, epoch+1))
        
        # Step scheduler based on validation accuracy
        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = [group['lr'] for group in optimizer.param_groups]
            logger.log(f"Epoch={epoch+1} LR={current_lr[0]:.6e} LR_Conv={current_lr[1]:.6e}")
        
        logger.log(f"Epoch={epoch+1} Train_Acc={train_metrics['accuracy']:.4f} Val_Acc={val_acc:.4f} Base_Loss={train_metrics['base_loss']:.4e} Reg_Loss={train_metrics['reg_loss']:.4e} Total_Loss={train_metrics['total_loss']:.4e}", data=True)

        # Track best accuracy and save state_dict only when validation improves
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(exp_dir, f"best_{phase}_model_state.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"Saved best {phase} model state_dict to {best_model_path}")

        # Log L2 norm distance for feature extractor phase
        if phase == 'feature_extractor' and original_params is not None:
            l2_norm = sum((p - o).norm().item() for p, o in zip(model.feature_extractor.parameters(), original_params))
        else:
            l2_norm = 0.0
        logger.log(f"Epoch={epoch+1} L2_Norm_Distance={l2_norm:.4f}", data=True)

    # Save final model after training completes
    final_model_path = os.path.join(exp_dir, f"final_{phase}_model_state.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.log(f"Saved final {phase} model state_dict to {final_model_path}")

    return best_acc, l2_norm

def construct_filters_fullsize(config, image_shape):
    """
    Constructs the filters based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        image_shape (tuple): Shape of the input image.

    Returns:
        dict: Dictionary containing the constructed filters.
    """
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
        logger.info(f"Generating filters with base wavelet: {config.get("wavelet", "morlet")}")
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
    return(filters)




def log_model_parameters(model, logger):
    """Logs the number of trainable parameters in the feature extractor and classifier."""
    feature_extractor_params = sum(p.numel() for p in model.feature_extractor.parameters() if p.requires_grad)
    classifier_params = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    total_params = feature_extractor_params + classifier_params

    logger.log(f"Feature_Extractor_Params={feature_extractor_params}", data=True)
    logger.log(f"Classifier_Params={classifier_params}", data=True)
    logger.log(f"Total_Params={total_params}", data=True)

def main():
    # Parse arguments
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    # Set random seed for reproducibility
    seed = config.get("seed", None)  # Default seed is None if not provided
    set_seed(seed)

    # Create output folder
    train_ratio = config["train_ratio"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.sweep_dir is not None:
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, f"{train_ratio=}", f"run-{timestamp}")
    else:
        exp_dir = os.path.join("experiments", f"{train_ratio=}", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize logger
    logger = LoggerManager.get_logger(log_dir=exp_dir,
                                      wandb_project=args.wandb_project,
                                      config=config, name=f"{train_ratio=}(run-{timestamp})")
    sys.excepthook = LoggerManager.log_uncaught_exceptions
    logger.log("===== Start log =====")

    # Log environment details
    logger.log(f"Python_Version={platform.python_version()}")
    logger.log(f"PyTorch_Version={torch.__version__}")
    logger.log(f"CUDA_Available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log(f"CUDA_Version={torch.version.cuda}")
        logger.log(f"GPU={torch.cuda.get_device_name(0)}")
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using_Device={device}")

    # Get data loaders
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    test_ratio = config["test_ratio"]
    train_val_ratio = config.get("train_val_ratio", 4)
    # Create a worker_init_fn with seed bound using functools.partial
    worker_init_with_seed = partial(worker_init_fn, seed=seed)
    if dataset == "mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, batch_size=batch_size, train_ratio=1-test_ratio, worker_init_fn=worker_init_with_seed
        )
    else:
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, resize=resize, deeper_path=deeper_path,
            batch_size=batch_size, train_ratio=1-test_ratio, worker_init_fn=worker_init_with_seed
        )
    train_loader, val_loader = split_train_val(
        train_loader.dataset, train_ratio=train_ratio, train_val_ratio=train_val_ratio, batch_size=batch_size, drop_last=True
    )

    # Initialize models
    downsample = config.get("downsample", False)
    if downsample:
        filters = construct_filters_downsample(config, image_shape)
        T = filters['psi'].shape[-1]
        logger.info(f"Using downsampled WPH model with filter size T={T}")
        feature_extractor = WPHModelDownsample(J=config["max_scale"],
            L=config["nb_orients"],
            A=config["num_phases"],
            A_prime=config.get("num_phases_prime", 1),
            M=image_shape[1],
            N=image_shape[2],
            T = T,
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
        ).to(device)
    model = WPHClassifier(
        feature_extractor, 
        num_classes=config["num_classes"],
        use_batch_norm=config.get("use_batch_norm", False)
    ).to(device)

    # Log model architecture
    logger.log("Model_Architecture:")
    logger.log(str(model))

    # Log model parameters
    log_model_parameters(model, logger)


    # Training setup
    # Support separate learning rates for classifier and conv (feature extractor) phases
    lr_classifier = float(config.get("lr_classifier", config.get("lr", 1e-3)))
    lr_conv = float(config.get("lr_conv", config.get("lr", 1e-3) * 0.01))
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": lr_classifier},
        {"params": model.feature_extractor.parameters(), "lr": lr_conv}
    ])
    
    # Create learning rate scheduler
    scheduler_config = config.get("scheduler", {"mode": "max", "factor": 0.5, "patience": 2, "min_lr": 1e-7})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get("mode", "max"),
        factor=scheduler_config.get("factor", 0.5),
        patience=scheduler_config.get("patience", 5),
        min_lr=scheduler_config.get("min_lr", 1e-7),
        verbose=True
    )
    logger.log(f"Scheduler: ReduceLROnPlateau(mode={scheduler_config.get('mode', 'max')}, factor={scheduler_config.get('factor', 0.5)}, patience={scheduler_config.get('patience', 5)})")

    criterion = nn.CrossEntropyLoss()

    # Clone original parameters for regularization during fine-tuning
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.parameters()]

    # Train classifier
    if not args.skip_classifier_training:
        logger.log("Training classifier...")
        model.set_trainable({"feature_extractor": False, "classifier": True})
        start_time = time.time()
        best_acc_classifier, _ = train_model(model=model, train_loader=train_loader,
                                          val_loader=val_loader, optimizer=optimizer,
                                          scheduler=scheduler, criterion=criterion, device=device, epochs=classifier_epochs,
                                          logger=logger, phase='classifier', configs=config, exp_dir=exp_dir)
        elapsed_time = time.time() - start_time
        logger.log(f"Classifier training completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping classifier training phase.")
        best_acc_classifier = 0.0

    # Load best classifier model and evaluate
    if not args.skip_classifier_training:
        best_classifier_path = os.path.join(exp_dir, "best_classifier_model_state.pt")
        if os.path.exists(best_classifier_path):
            model.load_state_dict(torch.load(best_classifier_path, weights_only=True))
            logger.log(f"Loaded best classifier model from {best_classifier_path}")
    test_loss, classifier_test_acc = evaluate(model, test_loader, criterion, device)
    logger.log(f"Classifier Test Accuracy: {classifier_test_acc:.4f}")

    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Saved original model state_dict to {save_original}")

    # Fine-tune feature extractor
    if not args.skip_finetuning:
        logger.log("Fine-tuning feature extractor...")
        model.set_trainable({"feature_extractor": True, "classifier": False})
        
        # Reset scheduler for feature extractor phase
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "max"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
            min_lr=scheduler_config.get("min_lr", 1e-7),
            verbose=True
        )
        logger.log("Reset scheduler for feature extractor phase")
        
        start_time = time.time()
        best_acc_feature_extractor, l2_norm = train_model(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, criterion=criterion, device=device, epochs=conv_epochs, logger=logger, phase='feature_extractor', configs=config, exp_dir=exp_dir, original_params=original_params)
        elapsed_time = time.time() - start_time
        logger.log(f"Feature extractor fine-tuning completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping fine-tuning phase.")
        best_acc_feature_extractor = 0.0

    # Load best feature extractor model and evaluate
    if not args.skip_finetuning:
        best_fe_path = os.path.join(exp_dir, "best_feature_extractor_model_state.pt")
        if os.path.exists(best_fe_path):
            model.load_state_dict(torch.load(best_fe_path, weights_only=True))
            logger.log(f"Loaded best feature extractor model from {best_fe_path}")
    test_loss, feature_extractor_test_acc = evaluate(model, test_loader, criterion, device)
    logger.log(f"Feature Extractor Test Accuracy={feature_extractor_test_acc:.4f}", data=True)

    # Repeat test accuracies at the end of the log
    logger.log("===== Final Test Accuracies =====")
    logger.log(f"Classifier_Test_Accuracy={classifier_test_acc:.4f}", data=True)
    logger.log(f"Feature_Extractor_Test_Accuracy={feature_extractor_test_acc:.4f}", data=True)

    
    # Update config with final accuracies
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["nb_moments"] = model.feature_extractor.nb_moments
    config["l2_norm_finetuning"] = l2_norm
    config["feature_extractor_test_acc"] = feature_extractor_test_acc
    config["classifier_test_acc"] = classifier_test_acc
    config["classifier_last_acc"] = classifier_test_acc
    config["classifier_best_acc"] = best_acc_classifier
    config["feature_extractor_last_acc"] = feature_extractor_test_acc
    config["feature_extractor_best_acc"] = best_acc_feature_extractor
    config["feature_extractor_params"] = sum(p.numel() for p in model.feature_extractor.parameters())
    config["finetuning_gain"] = feature_extractor_test_acc - classifier_test_acc
    config["classifier_params"] = sum(p.numel() for p in model.classifier.parameters())
    config["device"] = str(device)
    config["model_type"] = "wph"
    filter_dir = os.path.join(os.path.dirname(__file__), "../filters")
    config["filters"] = {
        "hatpsi": os.path.join(filter_dir, f"morlet_N{image_shape[1]}_J{config['max_scale']}_L{config['nb_orients']}.pt"),
        "hatphi": os.path.join(filter_dir, f"morlet_lp_N{image_shape[1]}_J{config['max_scale']}_L{config['nb_orients']}.pt")
    }
    # Save updated config
    save_config(exp_dir, config)

    # Plotting and visualization (similar to train.py)
    from plot_before_and_after import plot_kernels_wph_base_filters
    from visualize import visualize_main
    try:
        logger.log("Plotting kernels before and after training...")
        base_filter_names = ['feature_extractor.wave_conv.base_real', 'feature_extractor.wave_conv.base_imag'] if config['downsample'] else ['feature_extractor.wave_conv.base_filters']
        # img_file_names = plot_kernels_wph_base_filters(exp_dir, trained_filename='best_feature_extractor_model_state.pt', base_filters_key=base_filter_names)
        # # Log kernel image if available
        # for f in img_file_names:
        #     kernel_img_path = os.path.join(exp_dir, f)
        #     if os.path.exists(kernel_img_path):
        #         logger.send_file("kernels_before_after", kernel_img_path, "image")
        logger.log("Visualizing filters and activations...")
        visualize_main(exp_dir, tuned_filename='best_feature_extractor_model_state.pt', model_type='wph', filters=filters)
        # Log activation image if available
        activation_img_path = os.path.join(exp_dir, "activations.png")
        if os.path.exists(activation_img_path):
            logger.send_file("activations", activation_img_path, "image")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.log(f"Plotting/visualization failed: {e}")


  
    logger.finish()

if __name__ == "__main__":
    main()