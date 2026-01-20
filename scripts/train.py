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
    # Create output folder, divides by train ratio
    train_ratio = config["train_ratio"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.sweep_dir is not None: # if this is a sweep job, save to the sweep dir
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, f"{train_ratio=}", f"run-{timestamp}")
    else:
        exp_dir = os.path.join("experiments", f"{train_ratio=}", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir, 
                            wandb_project=args.wandb_project, 
                            config=config, name=f"{train_ratio=}(run-{timestamp})")
    
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")    

    # get data loader
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    test_ratio = config["test_ratio"]
    train_ratio = config["train_ratio"]
    train_val_ratio = config["train_val_ratio"]
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
        train_loader, val_loader, test_loader, nb_class, image_shape = get_loaders(
            dataset=dataset,
            root_dir=kth_root_dir,
            resize=resize,
            batch_size=batch_size,
            fold=fold,
            train_ratio=train_ratio,  # KTH loader handles train_ratio internally
            drop_last=True
        )
        # Note: KTH loader already handles train/val/test split, so no need for split_train_val
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
    
    # Classifier parameters
    classifier_type = config["classifier_type"]
    hypernet_hidden_dim = config["hypernet_hidden_dim"]
    attention_d_model = config["attention_d_model"]
    attention_num_heads = config["attention_num_heads"]
    attention_num_layers = config["attention_num_layers"]
    
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
        classifier_type=classifier_type,
        hypernet_hidden_dim=hypernet_hidden_dim,
        attention_d_model=attention_d_model,
        attention_num_heads=attention_num_heads,
        attention_num_layers=attention_num_layers,
    )
    model = dense(params).to(device)

    # Train classifier
    lr = float(config["lr"])
    #lr = min(lr, radius * 0.5)
    linear_lr = float(config["linear_lr"])
    weight_decays = float(config["weight_decays"]) / linear_lr # cancel lr effect in weight decay
    # radius = float(config["radius"])
    lambda_reg = float(config["lambda_reg"]) / lr # cancel lr effect in regularization
    
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    fine_tune_mode = config["fine_tune_mode"]  # Options: 'extractor_only', 'both'

    ##
    classifier_patience = config.get("classifier_patience", 8)
    conv_patience = config.get("conv_patience", 8)
    ##

    base_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.fine_tuned_params()]
    n_tuned_params = model.n_tuned_params()
    n_linear_params = model.n_classifier_params()
    logger.log(f"n_tuned_params={n_tuned_params} n_classifier_params={n_linear_params}", data=True)
    logger.log(f"Fine-tuning mode: {fine_tune_mode}")
    
    # Stage 1: Freeze extractor and train classifier (same for both modes)
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

        logger.log(
            f"Epoch={epoch+1}/{classifier_epochs} "
            f"Train_Acc={train_metrics['accuracy']:.4f} "
            f"Train_Loss={train_metrics['total_loss']:.4f} "
            f"Base_Loss={train_metrics['base_loss']:.4f} "
            f"Val_Acc={val_acc:.4f} "
            f"Val_Loss={val_loss:.4f}",
            data=True,
        )
        # Enhanced wandb logging with structured metrics
        logger.log_metrics({
            "train_accuracy": train_metrics['accuracy'],
            "train_loss": train_metrics['total_loss'],
            "train_base_loss": train_metrics['base_loss'],
            "val_accuracy": val_acc,
            "val_loss": val_loss,
        }, step=epoch+1, prefix="classifier/")
        # Track metrics for plotting
        logger.track_metric("classifier", "train_accuracy", train_metrics['accuracy'], epoch+1)
        logger.track_metric("classifier", "val_accuracy", val_acc, epoch+1)
        logger.track_metric("classifier", "train_loss", train_metrics['total_loss'], epoch+1)
        logger.track_metric("classifier", "val_loss", val_loss, epoch+1)

        # ---- Early stopping on validation accuracy
        if val_acc > best_val_acc:
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
    logger.log(f"classifier_test_acc={ini_test_acc:.4f} classifier_test_loss={ini_test_loss:.4f}", data=True)
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

    # Stage 2: Fine-tuning based on mode
    if fine_tune_mode == "extractor_only":
        # Option 1: Freeze classifier and fine-tune extractor
        logger.log("Fine tuning extractor (classifier frozen)...")
        model.train_conv()
        optimizer_ft = torch.optim.Adam(
            model.fine_tuned_params(),
            lr=lr,
        )
    elif fine_tune_mode == "both":
        # Option 2: Fine-tune both classifier and extractor together
        logger.log("Fine tuning both classifier and extractor together...")
        model.full_train()
        # Use parameter groups with different learning rates
        optimizer_ft = torch.optim.Adam([
            {'params': model.classifier.parameters(), 'lr': linear_lr},
            {'params': model.fine_tuned_params(), 'lr': lr}
        ])
    else:
        raise ValueError(f"Unknown fine_tune_mode: {fine_tune_mode}. Must be 'extractor_only' or 'both'")
    
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_acc = 0.0
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
        )

        val_loss, val_acc = evaluate(
            model, val_loader, base_loss, device
        )

        # Log training progress (test evaluation only every N epochs or at end to save time)
        logger.log(
            f"Epoch={classifier_epochs+epoch+1}/{classifier_epochs+conv_epochs} "
            f"Train_Acc={train_metrics['accuracy']:.4f} "
            f"Train_Loss={train_metrics['total_loss']:.4f} "
            f"Base_Loss={train_metrics['base_loss']:.4f} "
            f"Reg_Loss={train_metrics['reg_loss']:.4f} "
            f"Val_Acc={val_acc:.4f} "
            f"Val_Loss={val_loss:.4f}",
            data=True,
        )
        # Calculate parameter distance for this epoch
        epoch_dist_sq = 0.0
        for p, p0 in zip(model.fine_tuned_params(), original_params):
            if p.requires_grad:
                diff = p - p0
                epoch_dist_sq += torch.sum(torch.abs(diff) ** 2)
        epoch_dist = torch.sqrt(epoch_dist_sq) if epoch_dist_sq > 0 else torch.tensor(0.0)
        
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

        # ---- Early stopping on validation accuracy
        if val_acc > best_val_acc:
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