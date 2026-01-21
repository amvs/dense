import argparse
from configs import load_config, save_config, apply_overrides
import os
from datetime import datetime
import torch
import torch.nn as nn
from training.datasets import get_loaders, split_train_val
from training import train_one_epoch, evaluate
from dense import dense, ScatterParams
from dense.helpers import LoggerManager
from plot_before_and_after import plot_kernels
from visualize import visualize_main
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
    parser.add_argument("--wandb_project", type=str, default="DenseWavelet",
                    help="Weights and Biases project name for logging")
    parser.add_argument("--override", nargs="*", default=[],
                    help="List of key=value pairs to override config")
    parser.add_argument("--sweep_dir", type=str,  default=None,
                    help="if this is sweep job, specify the sweep output dir")
    return parser.parse_args()

def main():
    # Read config
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    # Create output folder
    train_ratio = config.get("train_ratio", None)
    example_per_class = config.get("example_per_class", None)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Use example_per_class for naming if available, otherwise train_ratio
    if example_per_class is not None:
        exp_name = f"epc={example_per_class}"
    elif train_ratio is not None:
        exp_name = f"{train_ratio=}"
    else:
        exp_name = "full_data"
    
    if args.sweep_dir is not None: # if this is a sweep job, save to the sweep dir
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, exp_name, f"run-{timestamp}")
    else:
        exp_dir = os.path.join("experiments", exp_name, f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir, 
                            wandb_project=args.wandb_project, 
                            config=config, name=f"{exp_name}(run-{timestamp})")
    
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")    

    # get data loader
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    test_ratio = config.get("test_ratio", 0.15)
    train_ratio = config.get("train_ratio", None)  # Optional for KTH dataset
    train_val_ratio = config.get("train_val_ratio", 4)
    seed = config["seed"]
    
    if dataset == "mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, 
            batch_size=batch_size, 
            train_ratio=1-test_ratio
        )
        train_loader, val_loader = split_train_val(
            train_loader.dataset,
            train_ratio=train_ratio,
            batch_size=batch_size,
            train_val_ratio=train_val_ratio,
            seed=seed
        )
    elif dataset == "kthtips2b":
        # KTH-TIPS2b returns train_loader, val_loader, test_loader already
        resize = config["resize"]
        kth_root_dir = config["kth_root_dir"]
        fold = config.get("fold", None)  # Optional fold parameter for cross-validation
        example_per_class = config.get("example_per_class", None)  # New: examples per class
        use_balanced_batches = config.get("use_balanced_batches", True)  # Use balanced batches
        use_scale_augmentation = config.get("use_scale_augmentation", False)  # Scale augmentation for training
        
        # If example_per_class is not set, try to use train_ratio (backward compatibility)
        if example_per_class is None and train_ratio is not None and train_ratio < 1.0:
            # Note: train_ratio is deprecated for KTH dataset when using balanced batches
            # Will use all available data if example_per_class is not specified
            pass
        
        loaders_result = get_loaders(
            dataset=dataset,
            root_dir=kth_root_dir,
            resize=resize,
            batch_size=batch_size,
            fold=fold,
            train_ratio=train_ratio if example_per_class is None else None,  # Only use if example_per_class not set
            example_per_class=example_per_class,
            drop_last=False,
            seed=seed,
            use_balanced_batches=use_balanced_batches,
            use_scale_augmentation=use_scale_augmentation
        )
        
        # Handle return value: new version returns stats as 6th element
        if len(loaders_result) == 6:
            train_loader, val_loader, test_loader, nb_class, image_shape, dataset_stats = loaders_result
        else:
            # Backward compatibility: old version doesn't return stats
            train_loader, val_loader, test_loader, nb_class, image_shape = loaders_result
            dataset_stats = None
        
        # Write dataset statistics to config if available
        if dataset_stats is not None:
            config["dataset_stats"] = {
                "num_classes": dataset_stats["num_classes"],
                "examples_per_class": dataset_stats["examples_per_class"],
                "min_examples_per_class": dataset_stats["min_examples_per_class"],
                "max_examples_per_class": dataset_stats["max_examples_per_class"],
                "is_balanced": dataset_stats["is_balanced"],
                "train_examples_per_class": dataset_stats.get("train_examples_per_class", None),
                "train_total_examples": dataset_stats.get("train_total_examples", len(train_loader.dataset))
            }
            logger.log(f"Dataset statistics: {config['dataset_stats']}")
        
        # Note: KTH loader already handles train/val/test split, so no need for split_train_val
    elif dataset == "kthtips":
        # KTH-TIPS (original) returns train_loader, val_loader, test_loader already
        resize = config["resize"]
        kth_root_dir = config["kth_root_dir"]
        example_per_class = config.get("example_per_class", None)  # Examples per class
        use_balanced_batches = config.get("use_balanced_batches", True)  # Use balanced batches
        use_scale_augmentation = config.get("use_scale_augmentation", False)  # Scale augmentation for training
        
        loaders_result = get_loaders(
            dataset=dataset,
            root_dir=kth_root_dir,
            resize=resize,
            batch_size=batch_size,
            example_per_class=example_per_class,
            drop_last=False,
            seed=seed,
            use_balanced_batches=use_balanced_batches,
            use_scale_augmentation=use_scale_augmentation
        )
        
        # Handle return value: new version returns stats as 6th element
        if len(loaders_result) == 6:
            train_loader, val_loader, test_loader, nb_class, image_shape, dataset_stats = loaders_result
        else:
            # Backward compatibility: old version doesn't return stats
            train_loader, val_loader, test_loader, nb_class, image_shape = loaders_result
            dataset_stats = None
        
        # Write dataset statistics to config if available
        if dataset_stats is not None:
            config["dataset_stats"] = {
                "num_classes": dataset_stats["num_classes"],
                "examples_per_class": dataset_stats["examples_per_class"],
                "min_examples_per_class": dataset_stats["min_examples_per_class"],
                "max_examples_per_class": dataset_stats["max_examples_per_class"],
                "is_balanced": dataset_stats["is_balanced"],
                "train_examples_per_class": dataset_stats.get("train_examples_per_class", None),
                "train_total_examples": dataset_stats.get("train_total_examples", None)
            }
            logger.log(f"Dataset statistics: {config['dataset_stats']}")
        
        # Note: KTH loader already handles train/val/test split, so no need for split_train_val
    elif dataset == "outex":
        # Outex returns train_loader, val_loader, test_loader already
        resize = config["resize"]
        outex_root_dir = config["outex_root_dir"]
        problem_id = config.get("problem_id", "000")  # Default problem ID
        train_val_ratio = config.get("train_val_ratio", 4)  # Default train:val ratio
        train_loader, val_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            root_dir=outex_root_dir,
            resize=resize,
            batch_size=batch_size,
            problem_id=problem_id,
            train_ratio=train_ratio,  # Outex loader handles train_ratio internally
            train_val_ratio=train_val_ratio,
            drop_last=False
        )
        # Note: Outex loader already handles train/val/test split, so no need for split_train_val
    else:  # Kaggle datasets
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset, 
            resize=resize,
            deeper_path=deeper_path,
            batch_size=batch_size, 
            train_ratio=1-test_ratio
        )
        train_loader, val_loader = split_train_val(
            train_loader.dataset,
            train_ratio=train_ratio,
            batch_size=batch_size,
            train_val_ratio=train_val_ratio,
            seed=seed
        )
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    share_channels = config.get("share_channels", False)
    n_copies = config["n_copies"]
    depth = config.get("depth", -1)
    random = config.get("random", False)
    use_original_filters = config["use_original_filters"]
    use_log_transform = config.get("use_log_transform", False)
    
    # Classifier parameters
    classifier_type = config["classifier_type"]
    hypernet_hidden_dim = config.get("hypernet_hidden_dim", 64)
    attention_d_model = config.get("attention_d_model", 128)
    attention_num_heads = config.get("attention_num_heads", 4)
    attention_num_layers = config.get("attention_num_layers", 2)
    pca_dim = config.get("pca_dim", 50)
    lowrank_pca_rank = config.get("lowrank_pca_rank", 100)
    
    params = ScatterParams(
        n_scale=max_scale,
        n_orient=nb_orients,
        n_copies=n_copies,
        in_channels=image_shape[0],
        wavelet=wavelet,
        n_class=nb_class,
        share_channels=share_channels,
        in_size=image_shape[1],
        depth=depth,
        random=random,
        use_original_filters=use_original_filters,
        use_log_transform=use_log_transform,
        classifier_type=classifier_type,
        hypernet_hidden_dim=hypernet_hidden_dim,
        attention_d_model=attention_d_model,
        attention_num_heads=attention_num_heads,
        attention_num_layers=attention_num_layers,
        pca_dim=pca_dim,
        lowrank_pca_rank=lowrank_pca_rank,
    )
    model = dense(params).to(device)

    # Train classifier
    lr = float(config["lr"])
    #lr = min(lr, radius * 0.5)
    linear_lr = float(config["linear_lr"])
    weight_decays = float(config["weight_decays"]) / linear_lr # cancel lr effect in weight decay
    # radius = float(config["radius"])
    lambda_reg = float(config["lambda_reg"])
    
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    fine_tune_mode = config["fine_tune_mode"]  # Options: 'extractor_only', 'both'

    ##
    classifier_patience = config.get("classifier_patience", 8)
    conv_patience = config.get("conv_patience", 8)
    # Fine-tuning learning rate multiplier (default: 0.1, meaning use 10% of original LR)
    ft_lr_multiplier = config.get("ft_lr_multiplier", 0.1)
    # Gradient clipping max norm (default: 1.0, set to 0 to disable)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    ##

    base_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.fine_tuned_params()]
    n_tuned_params = model.n_tuned_params()
    n_linear_params = model.n_classifier_params()
    logger.log(f"n_tuned_params={n_tuned_params} n_classifier_params={n_linear_params}", data=False)
    logger.log(f"Fine-tuning mode: {fine_tune_mode}")
    
    # Stage 1: Train classifier (same for both modes)
    # Note: No regularization needed here because:
    # - Extractor is frozen (parameters don't change)
    # - Classifier is trained from scratch (no original parameters to regularize against)
    classifier_type = config["classifier_type"]
    
    if classifier_type == 'pca':
        # PCA classifier: fit on training data (non-parametric, no gradient training)
        logger.log("Fitting PCA classifier (extractor frozen)...")
        model.train_classifier()  # Freeze extractor
        model.eval()  # Set to eval mode for feature extraction
        
        # Fit PCA classifier on training data
        logger.log("Computing class means and PCA bases...")
        model.classifier.fit(model, train_loader, device)
        logger.log("PCA classifier fitted successfully.")
    elif classifier_type == 'trainable_pca':
        # Trainable PCA: First fit PCA, then initialize trainable version
        logger.log("Initializing Trainable PCA classifier from PCA solution...")
        model.train_classifier()  # Freeze extractor
        model.eval()  # Set to eval mode for feature extraction
        
        # Step 1: Fit a temporary PCA classifier
        from dense.model import PCAClassifier
        temp_pca = PCAClassifier(num_classes=nb_class, pca_dim=pca_dim)
        temp_pca = temp_pca.to(device)
        logger.log("Fitting PCA to initialize trainable classifier...")
        temp_pca.fit(model, train_loader, device)
        
        # Step 2: Initialize trainable PCA from PCA solution
        with torch.no_grad():
            model.classifier.class_means.data = temp_pca.class_means.clone()
            model.classifier.pca_bases.data = temp_pca.pca_bases.clone()
        logger.log("Trainable PCA initialized from PCA solution.")
        
        # Step 3: Switch to train mode and train the classifier
        model.train()
        model.train_classifier()  # Ensure classifier is trainable
    elif classifier_type == 'lowrank_pca':
        # Low-rank PCA: First fit PCA, then initialize low-rank version
        logger.log("Initializing Low-Rank PCA classifier from PCA solution...")
        model.train_classifier()  # Freeze extractor
        model.eval()  # Set to eval mode for feature extraction
        
        # Step 1: Fit a temporary PCA classifier
        from dense.model import PCAClassifier
        temp_pca = PCAClassifier(num_classes=nb_class, pca_dim=pca_dim)
        temp_pca = temp_pca.to(device)
        logger.log("Fitting PCA to initialize low-rank classifier...")
        temp_pca.fit(model, train_loader, device)
        
        # Step 2: Initialize low-rank PCA from PCA solution
        logger.log(f"Initializing low-rank approximation (rank={lowrank_pca_rank})...")
        model.classifier._init_from_pca(temp_pca)
        logger.log("Low-rank PCA initialized from PCA solution.")
        
        # Log parameter count comparison
        n_params = model.n_classifier_params()
        full_pca_params = nb_class * (model.classifier.feature_dim * (1 + pca_dim))
        logger.log(f"Low-rank PCA parameters: {n_params:,} (vs full PCA: {full_pca_params:,}, "
                  f"reduction: {100*(1-n_params/full_pca_params):.1f}%)")
        
        # Step 3: Switch to train mode and train the classifier
        model.train()
        model.train_classifier()  # Ensure classifier is trainable
        
        # Evaluate on train and validation sets
        train_loss, train_acc = evaluate(model, train_loader, base_loss, device)
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
        
        logger.log(
            f"PCA Classifier Results: "
            f"Train_Acc={train_acc:.4f} Train_Loss={train_loss:.4f} "
            f"Val_Acc={val_acc:.4f} Val_Loss={val_loss:.4f}",
            data=True,
        )
        
        # Log metrics
        logger.log_metrics({
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        }, prefix="classifier/")
        
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_train_acc = train_acc
        best_train_loss = train_loss
        
    elif classifier_type == 'trainable_pca':
        # Trainable PCA: train the classifier (already initialized from PCA)
        logger.log("Training trainable PCA classifier (extractor frozen)...")
        model.train_classifier()
        
        optimizer_cls = torch.optim.Adam(
            model.classifier.parameters(),
            lr=linear_lr,  # Use same LR as other classifiers
        )
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_train_acc = 0.0
        best_train_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(classifier_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer_cls, base_loss, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, base_loss, device
            )

            logger.log(
                f"Epoch={epoch+1}/{classifier_epochs} "
                f"Train_Acc={train_metrics['accuracy']:.4f} "
                f"Train_Loss={train_metrics['total_loss']:.4f} "
                f"Base_Loss={train_metrics['base_loss']:.4f} "
                f"Val_Acc={val_acc:.4f} "
                f"Val_Loss={val_loss:.4f}",
                data=False,
            )
            logger.log_metrics({
                "train_accuracy": train_metrics['accuracy'],
                "train_loss": train_metrics['total_loss'],
                "train_base_loss": train_metrics['base_loss'],
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, step=epoch+1, prefix="classifier/")
            logger.track_metric("classifier", "train_accuracy", train_metrics['accuracy'], epoch+1)
            logger.track_metric("classifier", "val_accuracy", val_acc, epoch+1)
            logger.track_metric("classifier", "train_loss", train_metrics['total_loss'], epoch+1)
            logger.track_metric("classifier", "val_loss", val_loss, epoch+1)
            logger.flush_metrics()

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_metrics['accuracy']
                best_train_loss = train_metrics['base_loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= classifier_patience:
                    logger.log("Early stopping classifier training.")
                    break
        
        if best_state is None:
            logger.log("Warning: No improvement during classifier training. Using final model state.")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        logger.log(
            f"Finish trainable PCA classifier training. "
            f"Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}, "
            f"Best Train Acc: {best_train_acc:.4f}, Best Train Loss: {best_train_loss:.4f}"
        )
    elif classifier_type == 'lowrank_pca':
        # Low-rank PCA: train the classifier (already initialized from PCA)
        logger.log("Training low-rank PCA classifier (extractor frozen)...")
        model.train_classifier()
        
        optimizer_cls = torch.optim.Adam(
            model.classifier.parameters(),
            lr=linear_lr,  # Use same LR as other classifiers
        )
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_train_acc = 0.0
        best_train_loss = float("inf")
        best_state = None
        patience_counter = 0
        
        for epoch in range(classifier_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer_cls, base_loss, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, base_loss, device
            )

            logger.log(
                f"Epoch={epoch+1}/{classifier_epochs} "
                f"Train_Acc={train_metrics['accuracy']:.4f} "
                f"Train_Loss={train_metrics['total_loss']:.4f} "
                f"Base_Loss={train_metrics['base_loss']:.4f} "
                f"Val_Acc={val_acc:.4f} "
                f"Val_Loss={val_loss:.4f}",
                data=False,
            )
            logger.log_metrics({
                "train_accuracy": train_metrics['accuracy'],
                "train_loss": train_metrics['total_loss'],
                "train_base_loss": train_metrics['base_loss'],
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, step=epoch+1, prefix="classifier/")
            logger.track_metric("classifier", "train_accuracy", train_metrics['accuracy'], epoch+1)
            logger.track_metric("classifier", "val_accuracy", val_acc, epoch+1)
            logger.track_metric("classifier", "train_loss", train_metrics['total_loss'], epoch+1)
            logger.track_metric("classifier", "val_loss", val_loss, epoch+1)
            logger.flush_metrics()

            # Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_metrics['accuracy']
                best_train_loss = train_metrics['base_loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= classifier_patience:
                    logger.log("Early stopping classifier training.")
                    break
        
        if best_state is None:
            logger.log("Warning: No improvement during classifier training. Using final model state.")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        logger.log(
            f"Finish low-rank PCA classifier training. "
            f"Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}, "
            f"Best Train Acc: {best_train_acc:.4f}, Best Train Loss: {best_train_loss:.4f}"
        )
        
    else:
        # Gradient-based classifiers (hypernetwork, attention)
        logger.log("Training classifier (extractor frozen)...") 
        model.train_classifier()
        
        optimizer_cls = torch.optim.Adam(
            model.classifier.parameters(),
            lr=linear_lr, # classifier needs larger lr
            #weight_decay=weight_decays,
        )
        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_train_acc = 0.0
        best_train_loss = float("inf")
        best_state = None
        patience_counter = 0
        for epoch in range(classifier_epochs):
            train_metrics = train_one_epoch(
                model, train_loader, optimizer_cls, base_loss, device
            )
            val_loss, val_acc = evaluate(
                model, val_loader, base_loss, device
            )

            # Log to console/file (no wandb call)
            logger.log(
                f"Epoch={epoch+1}/{classifier_epochs} "
                f"Train_Acc={train_metrics['accuracy']:.4f} "
                f"Train_Loss={train_metrics['total_loss']:.4f} "
                f"Base_Loss={train_metrics['base_loss']:.4f} "
                f"Val_Acc={val_acc:.4f} "
                f"Val_Loss={val_loss:.4f}",
                data=False,  # Changed to False - metrics logged via log_metrics instead
            )
            # Enhanced wandb logging with structured metrics (batched, flushed at epoch end)
            logger.log_metrics({
                "train_accuracy": train_metrics['accuracy'],
                "train_loss": train_metrics['total_loss'],
                "train_base_loss": train_metrics['base_loss'],
                "val_accuracy": val_acc,
                "val_loss": val_loss,
            }, step=epoch+1, prefix="classifier/")
            # Track metrics for plotting (in-memory only, no wandb call)
            logger.track_metric("classifier", "train_accuracy", train_metrics['accuracy'], epoch+1)
            logger.track_metric("classifier", "val_accuracy", val_acc, epoch+1)
            logger.track_metric("classifier", "train_loss", train_metrics['total_loss'], epoch+1)
            logger.track_metric("classifier", "val_loss", val_loss, epoch+1)
            
            # Flush metrics to wandb at end of epoch (non-blocking async)
            logger.flush_metrics()

            # ---- Early stopping on validation loss
            if val_loss < best_val_loss:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_train_acc = train_metrics['accuracy']
                best_train_loss = train_metrics['base_loss']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= classifier_patience:
                    logger.log("Early stopping classifier training.")
                    break
        
        # Restore best classifier
        if best_state is None:
            logger.log("Warning: No improvement during classifier training. Using final model state.")
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        logger.log(
            f"Finish classifier training task. "
            f"Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}, "
            f"Best Train Acc: {best_train_acc:.4f}, Best Train Loss: {best_train_loss:.4f}"
        )

    ini_test_loss, ini_test_acc = evaluate(model, test_loader, base_loss, device)
    logger.log(f"classifier_test_acc={ini_test_acc:.4f} classifier_test_loss={ini_test_loss:.4f}", data=False)
    # Flush any remaining metrics before logging final metrics
    logger.flush_metrics()
    # Log classifier stage final metrics
    logger.log_metrics({
        "test_accuracy": ini_test_acc,
        "test_loss": ini_test_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_train_accuracy": best_train_acc,
        "best_train_loss": best_train_loss,
    }, prefix="classifier/final/")
    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Save model to {save_original}")

    # Evaluate validation set right before fine-tuning to establish baseline
    model.eval()
    val_loss_before_ft, val_acc_before_ft = evaluate(model, val_loader, base_loss, device)
    logger.log(f"Validation before fine-tuning: Acc={val_acc_before_ft:.4f}, Loss={val_loss_before_ft:.4f}", data=False)
    logger.log_metrics({
        "val_accuracy": val_acc_before_ft,
        "val_loss": val_loss_before_ft,
    }, prefix="fine_tune/baseline/")

    # Stage 2: Fine-tuning based on mode
    # Note: PCA classifier doesn't support fine-tuning (it's non-parametric)
    # Trainable PCA can be fine-tuned like other classifiers
    if classifier_type == 'pca':
        logger.log("PCA classifier is non-parametric. Skipping fine-tuning stage.")
        # Final evaluation
        final_test_loss, final_test_acc = evaluate(model, test_loader, base_loss, device)
        logger.log(f"Final Test Acc: {final_test_acc:.4f}, Final Test Loss: {final_test_loss:.4f}", data=False)
        logger.flush_metrics()  # Flush before finishing
        
        # Save final metrics
        config["dist"] = 0.0  # No parameter changes for PCA
        config["nb_class"] = nb_class
        config["n_tuned_params"] = n_tuned_params
        config["n_linear_params"] = n_linear_params
        config["classifier_test_acc"] = ini_test_acc
        config["classifier_test_loss"] = ini_test_loss
        config["image_shape"] = list(image_shape)
        config["best_train_acc"] = best_train_acc
        config["best_val_acc"] = best_val_acc
        config["test_acc"] = final_test_acc
        config["test_loss"] = final_test_loss
        config["best_train_loss"] = best_train_loss
        config["best_val_loss"] = best_val_loss
        save_config(exp_dir, config)
        
        logger.finish()
        return
    
    # Fine-tuning for gradient-based classifiers
    # Ensure model is in train mode before fine-tuning
    model.train()
    
    if fine_tune_mode == "extractor_only":
        # Option 1: Freeze classifier and fine-tune extractor
        # Regularization is applied to prevent extractor parameters from drifting too far
        # from their original values, which could hurt the pre-trained classifier
        logger.log("Fine tuning extractor (classifier frozen)...")
        model.train_conv()
        # Use a lower learning rate for fine-tuning to avoid instability
        # Fine-tuning typically needs smaller LR than initial training to prevent
        # large parameter changes that can hurt the pre-trained classifier
        ft_lr = lr * ft_lr_multiplier
        logger.log(f"Using fine-tuning learning rate: {ft_lr} (original lr: {lr}, multiplier: {ft_lr_multiplier})")
        optimizer_ft = torch.optim.Adam(
            model.fine_tuned_params(),
            lr=ft_lr,
        )
        # Scale lambda_reg by learning rate to make regularization strength independent of LR
        # This ensures consistent regularization effect regardless of learning rate magnitude
        lambda_reg = lambda_reg / ft_lr # cancel lr effect in regularization
    elif fine_tune_mode == "both":
        # Option 2: Fine-tune both classifier and extractor together
        # Regularization is applied only to extractor parameters (not classifier)
        # because classifier was just trained and has no "original" parameters to regularize against
        logger.log("Fine tuning both classifier and extractor together...")
        model.full_train()
        # Use lower learning rates for fine-tuning
        # Extractor gets reduced LR
        ft_lr_extractor = lr * ft_lr_multiplier
        
        # Classifier LR: Use much smaller LR for PCA-based classifiers to prevent destroying PCA initialization
        # For lowrank_pca/trainable_pca, the classifier is initialized from optimal PCA solution,
        # so it needs a VERY small LR to avoid catastrophic forgetting
        if classifier_type in ['lowrank_pca', 'trainable_pca']:
            # Use same multiplier as extractor (very small) to preserve PCA initialization
            ft_lr_classifier = linear_lr * ft_lr_multiplier
            logger.log(f"Using PCA-preserving classifier LR (same as extractor multiplier) for {classifier_type}")
        else:
            # For other classifiers, use moderate reduction
            ft_lr_classifier = linear_lr * max(ft_lr_multiplier * 2, 0.5)  # At least 50% of original, or 2x extractor multiplier
        
        logger.log(f"Using fine-tuning learning rates: extractor={ft_lr_extractor:.10f}, classifier={ft_lr_classifier:.10f}")
        # Use parameter groups with different learning rates
        optimizer_ft = torch.optim.Adam([
            {'params': model.classifier.parameters(), 'lr': ft_lr_classifier},
            {'params': model.fine_tuned_params(), 'lr': ft_lr_extractor}
        ])
        # Scale lambda_reg by extractor learning rate (regularization only applies to extractor)
        # This ensures consistent regularization effect regardless of learning rate magnitude
        lambda_reg = lambda_reg / ft_lr_extractor # cancel lr effect in regularization
    else:
        raise ValueError(f"Unknown fine_tune_mode: {fine_tune_mode}. Must be 'extractor_only' or 'both'")
    
    # Initialize best metrics with baseline values
    best_val_loss = val_loss_before_ft
    best_val_acc = val_acc_before_ft
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(conv_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer_ft,
            base_loss,
            device,
            original_params = original_params,
            #r=radius,
            lambda_reg=lambda_reg,
            max_grad_norm=max_grad_norm,
        )

        val_loss, val_acc = evaluate(
            model, val_loader, base_loss, device
        )

        # Log training progress (console/file only, metrics batched)
        logger.log(
            f"Epoch={classifier_epochs+epoch+1}/{classifier_epochs+conv_epochs} "
            f"Train_Acc={train_metrics['accuracy']:.4f} "
            f"Train_Loss={train_metrics['total_loss']:.4f} "
            f"Base_Loss={train_metrics['base_loss']:.4f} "
            f"Reg_Loss={train_metrics['reg_loss']:.4f} "
            f"Val_Acc={val_acc:.4f} "
            f"Val_Loss={val_loss:.4f}",
            data=False,  # Changed to False - metrics logged via log_metrics instead
        )
        # Calculate parameter distance for this epoch
        epoch_dist_sq = 0.0
        max_param_diff = 0.0
        for p, p0 in zip(model.fine_tuned_params(), original_params):
            if p.requires_grad:
                diff = p - p0
                epoch_dist_sq += torch.sum(torch.abs(diff) ** 2)
                max_param_diff = max(max_param_diff, torch.abs(diff).max().item())
        epoch_dist = torch.sqrt(epoch_dist_sq) if epoch_dist_sq > 0 else torch.tensor(0.0)
        
        # Warn if parameter distance is growing too large
        if epoch_dist.item() > 100.0:
            logger.log(f"WARNING: Large parameter distance detected: {epoch_dist.item():.2f}. Consider reducing learning rate or increasing regularization.")
        if max_param_diff > 10.0:
            logger.log(f"WARNING: Large individual parameter change detected: {max_param_diff:.2f}. Consider reducing learning rate.")
        
        # Enhanced wandb logging with structured metrics
        logger.log_metrics({
            "train_accuracy": train_metrics['accuracy'],
            "train_loss": train_metrics['total_loss'],
            "train_base_loss": train_metrics['base_loss'],
            "train_reg_loss": train_metrics['reg_loss'],
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "parameter_distance": epoch_dist.item(),
        }, step=classifier_epochs+epoch+1, prefix="fine_tune/")
        # Track metrics for plotting
        logger.track_metric("fine_tune", "train_accuracy", train_metrics['accuracy'], classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "val_accuracy", val_acc, classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "train_loss", train_metrics['total_loss'], classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "val_loss", val_loss, classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "train_base_loss", train_metrics['base_loss'], classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "train_reg_loss", train_metrics['reg_loss'], classifier_epochs+epoch+1)
        logger.track_metric("fine_tune", "parameter_distance", epoch_dist.item(), classifier_epochs+epoch+1)
        
        # Flush metrics to wandb at end of epoch (non-blocking async)
        logger.flush_metrics()

        
        # ---- Early stopping on validation loss
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_loss = train_metrics['base_loss']
            best_train_acc = train_metrics['accuracy']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= conv_patience:
                logger.log("Early stopping fine-tuning.")
                break

    # Restore best model
    if best_state is None:
        logger.log("Warning: No improvement during fine-tuning. Using final model state.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    
    # Calculate distance from original parameters
    dist_sq = 0.0
    for p, p0 in zip(model.fine_tuned_params(), original_params):
        if p.requires_grad:
            diff = p - p0
            dist_sq += torch.sum(torch.abs(diff) ** 2)
    dist = torch.sqrt(dist_sq) if dist_sq > 0 else torch.tensor(0.0)

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, base_loss, device)

    logger.log(f"Finish fine-tuning task. Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")
    logger.log(
        f"Final Results - "
        f"Test_Acc={test_acc:.4f} "
        f"Classifier_Test_Acc={ini_test_acc:.4f} "
        f"Train_Ratio={train_ratio:.4f} "
        f"Best_Val_Acc={best_val_acc:.4f} "
        f"Best_Train_Acc={best_train_acc:.4f} "
        f"Test_Loss={test_loss:.4f} "
        f"Best_Train_Loss={best_train_loss:.4f} "
        f"Best_Val_Loss={best_val_loss:.4f} "
        f"lambda_reg={lambda_reg:.6f} "
        f"LR={lr:.6f} "
        f"dist={dist.item():.6f}",
        data=True
    )
    
    # Enhanced wandb logging: Log fine-tuning final metrics
    logger.log_metrics({
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_train_accuracy": best_train_acc,
        "best_train_loss": best_train_loss,
        "parameter_distance": dist.item(),
    }, prefix="fine_tune/final/")
    
    # Log comparison metrics: classifier vs fine-tuned performance
    if ini_test_acc is not None:
        improvement = test_acc - ini_test_acc
        logger.log_comparison({
            "classifier_test_acc": ini_test_acc,
            "fine_tuned_test_acc": test_acc,
            "improvement": improvement,
            "classifier_test_loss": ini_test_loss,
            "fine_tuned_test_loss": test_loss,
            "train_ratio": train_ratio,
            "fine_tune_mode": fine_tune_mode,
            "parameter_distance": dist.item(),
        }, title="FineTuning_Comparison")
        
        # Create before/after comparison bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Classifier", "Fine-tuned"]
        accuracies = [ini_test_acc, test_acc]
        colors = ['skyblue', 'lightgreen']
        bars = ax.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Before/After Fine-tuning Comparison\nImprovement: {improvement:.4f}")
        ax.set_ylim([0, max(accuracies) * 1.1])
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.4f}', ha='center', va='bottom')
        plt.tight_layout()
        comparison_plot_path = os.path.join(exp_dir, "plots", "before_after_comparison.png")
        os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
        plt.savefig(comparison_plot_path, dpi=150)
        plt.close()
        logger.send_file("before_after_comparison", comparison_plot_path, "image")
    
    # Create all experiment-level plots
    logger.create_experiment_plots(exp_dir)
    #
    save_fine_tuned = os.path.join(exp_dir, "trained.pt")
    torch.save(model.state_dict(), save_fine_tuned)
    logger.log(f"Save model to {save_fine_tuned}")

    # back up config
    config["dist"] = dist.item()
    config["nb_class"] = nb_class
    config["n_tuned_params"] = n_tuned_params
    config["n_linear_params"] = n_linear_params
    config["classifier_test_acc"] = ini_test_acc
    config["classifier_test_loss"] = ini_test_loss
    config["fine_tune_mode"] = fine_tune_mode
    config["image_shape"] = list(image_shape)
    # Fine-tuning stage metrics
    config["best_train_acc"] = best_train_acc
    config["best_val_acc"] = best_val_acc
    config["test_acc"] = test_acc
    config["test_loss"] = test_loss
    config["best_train_loss"] = best_train_loss
    config["best_val_loss"] = best_val_loss
    # config["lr"] = lr
    config["random"] = False
    
    # Save comparison metrics for sweep analysis
    if ini_test_acc is not None:
        config["classifier_test_acc"] = ini_test_acc
        config["classifier_test_loss"] = ini_test_loss
        config["fine_tuned_test_acc"] = test_acc
        config["fine_tuned_test_loss"] = test_loss
        config["improvement"] = test_acc - ini_test_acc
        config["FineTuning_Comparison/classifier_test_acc"] = ini_test_acc
        config["FineTuning_Comparison/fine_tuned_test_acc"] = test_acc
        config["FineTuning_Comparison/improvement"] = test_acc - ini_test_acc
    
    save_config(exp_dir, config)



    logger.finish()
    
if __name__ == "__main__":
    main()