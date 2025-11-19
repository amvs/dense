import argparse
import os
from datetime import datetime
import pdb
import torch
import torch.nn as nn
from configs import load_config, save_config, apply_overrides
from training.datasets import get_loaders, split_train_val
from training import train_one_epoch, evaluate
from wph.wph_model import WPHModel, WPHClassifier
from dense.helpers import LoggerManager
from wph.layers.utils import apply_phase_shifts
import sys

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
    return parser.parse_args()

def train_model(model, train_loader, val_loader, optimizer, criterion, device, epochs, logger, phase):
    """
    A reusable function for training the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        criterion (Loss): Loss function.
        device (torch.device): Device to train on.
        epochs (int): Number of epochs to train.
        logger (Logger): Logger for logging progress.
        phase (str): Phase of training (e.g., 'classifier', 'feature_extractor').

    Returns:
        None
    """
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        logger.info(f"[{phase}] Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

def construct_filters(config, image_shape, logger):
    """
    Constructs the filters based on the configuration.

    Args:
        config (dict): Configuration dictionary.
        image_shape (tuple): Shape of the input image.
        logger (Logger): Logger for logging progress.

    Returns:
        dict: Dictionary containing the constructed filters.
    """
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
                torch.randn(param_nc, max_scale, param_L, 1, image_shape[1], image_shape[2]),
                torch.randn(param_nc, max_scale, param_L, 1, image_shape[1], image_shape[2])
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
            os.system(f"python /projects/standard/lermang/vonse006/wph_collab/dense/wph/ops/build-filters.py --N {image_shape[1]} --J {max_scale} --L {nb_orients} --wavelets morlet")

        # Load precomputed filters
        filters = {
            "hatpsi": torch.load(hatpsi_path, weights_only=True),
            "hatphi": torch.load(hatphi_path, weights_only=True)
        }

    # Apply phase shifts to the filters
    filters["hatpsi"] = apply_phase_shifts(filters["hatpsi"], A=param_A)
    return filters

def main():
    # Parse arguments
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    # Create output folder
    val_ratio = config["val_ratio"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.sweep_dir is not None:
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, f"{val_ratio=}", f"run-{timestamp}")
    else:
        exp_dir = os.path.join("experiments", f"{val_ratio=}", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Initialize logger
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    sys.excepthook = LoggerManager.log_uncaught_exceptions
    logger.info("===== Start log =====")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Get data loaders
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    if dataset == "mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, batch_size=batch_size, train_ratio=train_ratio
        )
    else:
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, resize=resize, deeper_path=deeper_path,
            batch_size=batch_size, train_ratio=train_ratio
        )
    train_loader, val_loader = split_train_val(
        train_loader.dataset, val_ratio=val_ratio, batch_size=batch_size
    )

    # Initialize models
    filters = construct_filters(config, image_shape, logger)
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
        wavelets=config["wavelet"]
    ).to(device)
    model = WPHClassifier(
        feature_extractor, 
        num_classes=config["num_classes"],
        use_batch_norm=config.get("use_batch_norm", False)
    ).to(device)

    # Training setup
    lr = float(config["lr"])
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": lr},
        {"params": model.feature_extractor.parameters(), "lr": lr * 0.01}
    ])

    # Ensure filters remain complex-valued and updates propagate correctly
    if config.get("train_filters", False):
        resolved_filters = {
            key: value.resolve_conj() if value.is_conj() else value
            for key, value in filters.items()
        }
        optimizer.add_param_group({"params": resolved_filters.values(), "lr": lr * 0.001})
        # Replace original filters with resolved filters to ensure updates propagate
        filters.update(resolved_filters)

    criterion = nn.CrossEntropyLoss()

    # Train classifier
    if not args.skip_classifier_training:
        logger.info("Training classifier...")
        model.train_feature_extractor(False)
        model.train_classifier(True)
        train_model(model, train_loader, val_loader, optimizer, criterion, device, classifier_epochs, logger, phase='classifier')
    else:
        logger.info("Skipping classifier training phase.")
    # Save intermediate model
    save_path = os.path.join(exp_dir, "classifier_trained.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved classifier-trained model to {save_path}")

    # Fine-tune feature extractor
    if not args.skip_finetuning:
        logger.info("Fine-tuning feature extractor...")
        model.train_feature_extractor(True)
        model.train_classifier(False)
        train_model(model, train_loader, val_loader, optimizer, criterion, device, conv_epochs, logger, phase='feature_extractor')
    else:
        logger.info("Skipping fine-tuning phase.")
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Save final model
    save_path = os.path.join(exp_dir, "fine_tuned.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved fine-tuned model to {save_path}")

    # Save config
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["test_acc"] = test_acc
    save_config(exp_dir, config)

    # Add error handling for missing configuration keys
    required_keys = ["val_ratio", "dataset", "batch_size", "train_ratio", "max_scale", "nb_orients", "wavelet", "lr", "classifier_epochs", "conv_epochs"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required configuration key: {key}")

    # Log all relevant configuration details
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"{key}: {value}")

    # Save additional information to the config file
    config["device"] = str(device)
    filter_dir = os.path.join(os.path.dirname(__file__), "../filters")
    config["filters"] = {
        "hatpsi": os.path.join(filter_dir, f"morlet_N{image_shape[1]}_J{config['max_scale']}_L{config['nb_orients']}.pt"),
        "hatphi": os.path.join(filter_dir, f"morlet_lp_N{image_shape[1]}_J{config['max_scale']}_L{config['nb_orients']}.pt")
    }

if __name__ == "__main__":
    main()