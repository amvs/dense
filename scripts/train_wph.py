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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from configs import load_config, save_config, apply_overrides, AutoConfig
from training import train_one_epoch, evaluate
from training.base_trainer import recompute_bn_running_stats
from training.experiment_utils import setup_experiment, log_model_parameters
from training.data_utils import load_and_split_data
from wph.wph_model import WPHClassifier
from wph.classifiers import LinearClassifier, HyperNetworkClassifier
from wph.model_factory import create_wph_feature_extractor
from dense.helpers import LoggerManager




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

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device, epochs, logger, phase, configs, exp_dir, original_fe_params=None, original_classifier_params=None, test_loader=None, freeze_classifier=True):
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
        original_fe_params (list[torch.Tensor], optional): Original feature-extractor parameters for regularization during fine-tuning.
        original_classifier_params (list[torch.Tensor], optional): Original classifier parameters for regularization during fine-tuning.
        freeze_classifier (bool): Whether classifier is frozen during feature extractor phase.

    Returns:
        None
    """
    best_acc = 0.0  # Initialize best accuracy
    # Accuracy and loss history per epoch for plotting
    train_acc_hist, val_acc_hist, test_acc_hist = [], [], []
    base_loss_hist, reg_loss_hist, total_loss_hist = [], [], []
    l2_norm_hist = []
    for epoch in range(epochs):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            base_loss=criterion,
            device=device,
            lambda_reg=configs['lambda_reg'] if phase == 'feature_extractor' else 0.0,
            original_params=original_fe_params if phase == 'feature_extractor' else None,
            vmap_chunk_size=configs.get('vmap_chunk_size', None),
            normalize_reg=configs.get('normalize_reg', True)
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        # Optionally evaluate on test set each epoch (for tracking only)
        if test_loader is not None:
            test_loss_epoch, test_acc_epoch = evaluate(model, test_loader, criterion, device)
        else:
            test_acc_epoch = None
        logger.log("Phase: {} Epoch: {}".format(phase, epoch+1))        
        # Step scheduler based on validation accuracy
        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = [group['lr'] for group in optimizer.param_groups]
            # logger.log(f"Epoch={epoch+1} LR={current_lr[0]:.6e} LR_Conv={current_lr[1]:.6e}")
        
        # Record history metrics
        train_acc_hist.append(float(train_metrics.get('accuracy', float('nan'))))
        val_acc_hist.append(float(val_acc))
        test_acc_hist.append(float(test_acc_epoch) if test_acc_epoch is not None else float('nan'))

        # Build log message with metrics and L2 norms
        log_msg = f"Epoch={epoch+1} Train_Acc={train_metrics['accuracy']:.4f} Val_Acc={val_acc:.4f} Base_Loss={train_metrics['base_loss']:.4e} Reg_Loss={train_metrics['reg_loss']:.4e} Total_Loss={train_metrics['total_loss']:.4e}"
        if test_acc_epoch is not None:
            log_msg += f" Test_Acc={test_acc_epoch:.4f}"
        
        # Add L2 norms if in feature extractor phase
        if phase == 'feature_extractor' and original_fe_params is not None:
            current_fe_params = list(model.feature_extractors.parameters())
            l2_norm_fe = sum((p - o).norm().item() for p, o in zip(current_fe_params, original_fe_params))
            l2_norm = l2_norm_fe
            log_msg += f" L2_Norm_FE={l2_norm_fe:.4f}"
            
            # Also add classifier L2 norm if classifier is trainable (not frozen)
            if not freeze_classifier and original_classifier_params is not None:
                current_classifier_params = list(model.classifier.parameters())
                l2_norm_classifier = sum((p - o).norm().item() for p, o in zip(current_classifier_params, original_classifier_params))
                log_msg += f" L2_Norm_Classifier={l2_norm_classifier:.4f}"
        else:
            l2_norm = 0.0
        
        logger.log(log_msg, data=True)

        # Track best model by validation accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(exp_dir, f"best_{phase}_model_state.pt")
            torch.save(model.state_dict(), best_model_path)
            # Log a clean metrics line for cloud parsing and a human-readable message separately
            logger.log(f"Best_Val_Acc={val_acc:.4f}", data=True)
            logger.log(f"New best {phase} model saved to {best_model_path}")

        base_loss_hist.append(float(train_metrics.get('base_loss', float('nan'))))
        reg_loss_hist.append(float(train_metrics.get('reg_loss', float('nan'))))
        total_loss_hist.append(float(train_metrics.get('total_loss', float('nan'))))
        l2_norm_hist.append(float(l2_norm))
    
    # Save final model after training completes
    final_model_path = os.path.join(exp_dir, f"final_{phase}_model_state.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.log(f"Saved final {phase} model state_dict to {final_model_path}")

    # Return accuracy history for plotting
    history = {
        'train_acc': train_acc_hist,
        'val_acc': val_acc_hist,
        'test_acc': test_acc_hist,
        'base_loss': base_loss_hist,
        'reg_loss': reg_loss_hist,
        'total_loss': total_loss_hist,
        'l2_norm': l2_norm_hist,
    }
    return best_acc, l2_norm, history


def main():
    # Parse arguments
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    # Wrap config so that any use of config.get(key, default) will
    # record the default into the config dictionary for later saving.
    config = AutoConfig(config)

    # Set up experiment (creates exp_dir, logger, device)
    exp_dir, logger, device, seed = setup_experiment(args, config, args.wandb_project)
    
    # Set random seed for reproducibility
    set_seed(seed)

    # Get data loaders
    # Create a worker_init_fn with seed bound using functools.partial
    worker_init_with_seed = partial(worker_init_fn, seed=seed)
    train_loader, val_loader, test_loader, nb_class, image_shape = load_and_split_data(
        config, worker_init_with_seed
    )

    # Initialize models
    feature_extractor, filters = create_wph_feature_extractor(config, image_shape, device)
    
    # Create linear classifier
    nb_moments = int(feature_extractor.nb_moments)
    model_type = config.get("model_type", "wph")
    if model_type == "wph":
        classifier = LinearClassifier(
            input_dim=nb_moments,
            num_classes=config["num_classes"],
        )
    elif model_type == "wph_hypernetwork":
        classifier = HyperNetworkClassifier(
            num_classes=config["num_classes"],
            metadata_dim=config.get("metadata_dim", 10),
            hidden_dim=config.get("hidden_dim", 128)
        )
        feature_metadata = feature_extractor.flat_metadata()
        feature_metadata = list(feature_metadata.values())
        classifier.set_feature_metadata(metadata = torch.tensor(feature_metadata))
    elif model_type == "wph_pca":
        raise ValueError("PCA classifier should be trained via train_wph_pca.py")
    elif model_type == "wph_svm":
        raise ValueError("SVM classifier should be trained via train_wph_svm.py")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=classifier,
        copies=int(config.get("copies", 1)),
        noise_std=float(config.get("noise_std", 0.01)),
        use_batch_norm=config.get("use_batch_norm", False)
    ).to(device)

    # Log model architecture
    logger.log("Model_Architecture:")
    logger.log(str(model))

    # Log model parameters
    log_model_parameters(model, model.classifier, logger)


    # Training setup
    # Support separate learning rates for classifier and conv (feature extractor) phases
    lr_classifier = float(config.get("lr_classifier", config.get("lr", 1e-3)))
    lr_conv = float(config.get("lr_conv", config.get("lr", 1e-3) * 0.01))
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam([
        {"params": model.classifier.parameters(), "lr": lr_classifier},
        {"params": model.feature_extractors.parameters(), "lr": lr_conv}
    ])
    
    # Create learning rate scheduler
    scheduler_config = config.get("scheduler", {"mode": "max", "factor": 0.5, "patience": 5, "min_lr": 1e-7})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_config.get("mode", "max"),
        factor=scheduler_config.get("factor", 0.5),
        patience=scheduler_config.get("patience", 5),
        min_lr=scheduler_config.get("min_lr", 1e-7),
        verbose=True
    )
    logger.log(f"Scheduler: ReduceLROnPlateau(mode={scheduler_config.get('mode', 'max')}, factor={scheduler_config.get('factor', 0.5)}, patience={scheduler_config.get('patience', 5)})")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("ce_smooth", 0.0))

    # Clone original parameters for regularization during fine-tuning
    with torch.no_grad():
        original_fe_params = [p.clone().detach() for p in model.feature_extractors.parameters()]
        original_classifier_params = [p.clone().detach() for p in model.classifier.parameters()]
    l2_norm = 0.0

    # Train classifier
    if not args.skip_classifier_training:
        logger.log("Training classifier...")
        model.set_trainable({"feature_extractor": False, "classifier": True, "spatial_attn": config.get("spatial_attn", False)})
        # Warm up BatchNorm running stats before classifier training
        bn_warmup_batches = int(config.get("bn_warmup_batches", 100))
        try:
            t0 = time.time()
            recompute_bn_running_stats(model, train_loader, device, max_batches=bn_warmup_batches, logger=logger, momentum=0.9)
            logger.log(f"BN warmup took {time.time()-t0:.2f}s", data=True)
        except Exception as e:
            logger.log(f"BN warmup failed: {e}", data=True)
        start_time = time.time()
        best_acc_classifier, _, hist_cls = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epochs=classifier_epochs,
            logger=logger,
            phase='classifier',
            configs=config,
            exp_dir=exp_dir,
            test_loader=test_loader,
        )
        elapsed_time = time.time() - start_time
        logger.log(f"Classifier training completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping classifier training phase.")
        best_acc_classifier = 0.0
        hist_cls = {
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'base_loss': [], 'reg_loss': [], 'total_loss': [], 'l2_norm': []
        }

    # Load best classifier model and evaluate
    if not args.skip_classifier_training:
        best_classifier_path = os.path.join(exp_dir, "best_classifier_model_state.pt")
        if os.path.exists(best_classifier_path):
            model.load_state_dict(torch.load(best_classifier_path, weights_only=True))
            logger.log(f"Loaded best classifier model from {best_classifier_path}")
    # Evaluate classifier phase (validation and test)
    val_loss, classifier_val_acc = evaluate(model, val_loader, criterion, device)
    test_loss, classifier_test_acc = evaluate(model, test_loader, criterion, device)
    logger.log(f"Classifier Val Accuracy: {classifier_val_acc:.4f}")
    logger.log(f"Classifier Test Accuracy: {classifier_test_acc:.4f}")

    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Saved original model state_dict to {save_original}")

    # Fine-tune feature extractor
    if not args.skip_finetuning:
        logger.log("Fine-tuning feature extractor...")
        # Config option to control whether the classifier remains frozen during finetuning.
        # Default: keep previous behavior (freeze classifier)
        freeze_classifier = config.get("freeze_classifier", True)
        classifier_trainable = not bool(freeze_classifier)
        model.set_trainable({"feature_extractor": True, "classifier": classifier_trainable})
        logger.log(f"Finetune: classifier_trainable={classifier_trainable}")
        
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
        best_acc_feature_extractor, l2_norm, hist_fe = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            epochs=conv_epochs,
            logger=logger,
            phase='feature_extractor',
            configs=config,
            exp_dir=exp_dir,
            original_fe_params=original_fe_params,
            original_classifier_params=original_classifier_params,
            test_loader=test_loader,
            freeze_classifier=freeze_classifier,
        )
        elapsed_time = time.time() - start_time
        logger.log(f"Feature extractor fine-tuning completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping fine-tuning phase.")
        best_acc_feature_extractor = 0.0
        hist_fe = {
            'train_acc': [], 'val_acc': [], 'test_acc': [],
            'base_loss': [], 'reg_loss': [], 'total_loss': [], 'l2_norm': []
        }

    # Aggregate history across phases
    train_acc_all = (hist_cls['train_acc'] if hist_cls else []) + (hist_fe['train_acc'] if hist_fe else [])
    val_acc_all = (hist_cls['val_acc'] if hist_cls else []) + (hist_fe['val_acc'] if hist_fe else [])
    test_acc_all = (hist_cls['test_acc'] if hist_cls else []) + (hist_fe['test_acc'] if hist_fe else [])

    # Evaluate feature extractor phase
    # Load best feature extractor model and evaluate
    if not args.skip_finetuning:
        best_fe_path = os.path.join(exp_dir, "best_feature_extractor_model_state.pt")
        if os.path.exists(best_fe_path):
            model.load_state_dict(torch.load(best_fe_path, weights_only=True))
            logger.log(f"Loaded best feature extractor model from {best_fe_path}")
    test_loss, feature_extractor_test_acc = evaluate(model, test_loader, criterion, device)
    val_loss_fe, feature_extractor_val_acc = evaluate(model, val_loader, criterion, device)
    logger.log(f"Feature Extractor Val Accuracy={feature_extractor_val_acc:.4f}", data=True)
    logger.log(f"Feature Extractor Test Accuracy={feature_extractor_test_acc:.4f}", data=True)

    # Repeat test accuracies at the end of the log
    logger.log("===== Final Test Accuracies =====")
    logger.log(f"Classifier_Test_Accuracy={classifier_test_acc:.4f}", data=True)
    logger.log(f"Feature_Extractor_Test_Accuracy={feature_extractor_test_acc:.4f}", data=True)

    # Plot accuracy and losses history across epochs
    try:
        total_epochs = len(train_acc_all)
        if total_epochs > 0:
            epochs_axis = list(range(1, total_epochs + 1))
            # Aggregate loss/l2 histories
            base_loss_all = (hist_cls['base_loss'] if hist_cls else []) + (hist_fe['base_loss'] if hist_fe else [])
            reg_loss_all = (hist_cls['reg_loss'] if hist_cls else []) + (hist_fe['reg_loss'] if hist_fe else [])
            total_loss_all = (hist_cls['total_loss'] if hist_cls else []) + (hist_fe['total_loss'] if hist_fe else [])
            l2_norm_all = (hist_cls['l2_norm'] if hist_cls else []) + (hist_fe['l2_norm'] if hist_fe else [])

            fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 15))
            # Top: accuracy
            axs[0].plot(epochs_axis, train_acc_all, label='Train', color='tab:blue', linewidth=2)
            axs[0].plot(epochs_axis, val_acc_all, label='Val', color='tab:orange', linewidth=2)
            if any([not np.isnan(v) for v in test_acc_all]):
                test_plot = [v if not np.isnan(v) else None for v in test_acc_all]
                axs[0].plot(epochs_axis, test_plot, label='Test', color='tab:green', linewidth=2)
            axs[0].set_ylabel('Accuracy')
            axs[0].set_title('Accuracy per Epoch')
            axs[0].legend()
            axs[0].grid(True, alpha=0.3)

            # Bottom: losses and l2 norm
            axs[1].plot(epochs_axis, base_loss_all, label='Base Loss', color='tab:red', linewidth=1.5)
            axs[1].plot(epochs_axis, total_loss_all, label='Total Loss', color='tab:brown', linewidth=1.5)
            axs[1].set_ylabel('Loss')
            axs[1].set_xlabel('Epoch')
            axs[1].set_yscale('log')
            axs[1].legend()
            axs[1].grid(True, alpha=0.3)

             # Bottom: losses and l2 norm
            axs[2].plot(epochs_axis, reg_loss_all, label='Reg Loss', color='tab:purple', linewidth=1.5)
            axs[2].plot(epochs_axis, l2_norm_all, label='L2 Norm', color='tab:gray', linestyle='--', linewidth=1.5)
            axs[2].set_ylabel('Loss / L2')
            axs[2].set_xlabel('Epoch')
            axs[2].set_yscale('log')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)

            plt.tight_layout()
            acc_plot_path = os.path.join(exp_dir, 'accuracy.png')
            plt.savefig(acc_plot_path)
            plt.close()
            logger.log(f"Saved accuracy+loss plot to {acc_plot_path}")

            # Save per-epoch metrics as a DataFrame (CSV)
            df = pd.DataFrame({
                'epoch': epochs_axis,
                'train_acc': train_acc_all,
                'val_acc': val_acc_all,
                'test_acc': test_acc_all,
                'base_loss': base_loss_all,
                'reg_loss': reg_loss_all,
                'total_loss': total_loss_all,
                'l2_norm': l2_norm_all,
            })
            acc_csv_path = os.path.join(exp_dir, 'accuracy.csv')
            df.to_csv(acc_csv_path, index=False)
            logger.log(f"Saved accuracy+loss dataframe to {acc_csv_path}")
        else:
            logger.log("No epochs recorded; skipping accuracy plot.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.log(f"Failed to create/save accuracy+loss plot: {e}")

    
    # Update config with final accuracies
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["nb_moments"] = model.feature_extractors[0].nb_moments
    config["l2_norm_finetuning"] = l2_norm
    config["feature_extractor_test_acc"] = feature_extractor_test_acc
    config["classifier_test_acc"] = classifier_test_acc
    config["classifier_last_acc"] = classifier_val_acc
    config["classifier_best_acc"] = best_acc_classifier
    config["feature_extractor_last_acc"] = feature_extractor_val_acc
    config["feature_extractor_best_acc"] = best_acc_feature_extractor
    config["feature_extractor_params"] = sum(p.numel() for fe in model.feature_extractors for p in fe.parameters())
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
        base_filter_names = ['feature_extractors.0.wave_conv.base_real', 'feature_extractors.0.wave_conv.base_imag'] if config['downsample'] else ['feature_extractors.0.wave_conv.base_filters']
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