import logging
import os
import wandb
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from types import MethodType

class LoggerManager:
    """
    Singleton manager for project-wide logging.
    Supports a global logger or per-run log files.
    """
    _logger = None
    _cloud = False

    @staticmethod
    def get_logger(log_dir='.', name="run", level=logging.INFO, wandb_project=None, config=None):
        """
        Returns a singleton logger instance for the project.
        """
        # Return existing logger if already created
        if LoggerManager._logger is not None:
            return LoggerManager._logger
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Avoid adding multiple handlers if logger already exists
        if not logger.hasHandlers():
            # File handler
            log_file = os.path.join(log_dir, f"{name}.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(fh)

            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(level)
            ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
            logger.addHandler(ch)

        if wandb_project and not LoggerManager._cloud:
            # Additional cloud logging setup can be added here
            wandb_login()
            wandb.init(project=wandb_project, config=config, name=name)
            LoggerManager._cloud = True

        def send_file(title, path, type):
            '''
            Log images to wandb
            Example:
                logger.send_file("sample_image", "path/to/image.png", "image")
            Supported types: "image", "csv"
            '''
            if LoggerManager._cloud:
                if type == "image":
                    wandb.log({title: wandb.Image(path)})
                elif type == "csv":
                    artifact = wandb.Artifact(title, type="dataset")
                    artifact.add_file(path)
                    wandb.log_artifact(artifact)
                elif type == "table":
                    table = wandb.Table(dataframe=pd.read_csv(path))
                    wandb.log({title: table})
                else:
                    logger.error(f"Unsupported file type for wandb logging: {type}")
        
        def finish():
            '''
            Finish the wandb run
            '''
            logger.info("Finishing log...")
            if LoggerManager._cloud:
                wandb.finish()

        def log(message: str, data: bool = False):
            '''
            Unified log method
            - If data=False, logs plain text message
            - If data=True, logs text and parse key=value pairs for wandb
            example for data=True:
                logger.log("epoch=1 loss=0.345 acc=0.89", data=True)
            '''
            logger.info(message)
            if data and LoggerManager._cloud:
                json_data = {}
                try:
                    parts = message.strip().split()
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            # Try to convert to float, fallback to string if fails
                            try:
                                json_data[key] = float(value)
                            except ValueError:
                                json_data[key] = value
                except Exception as e:
                    logger.error(f"Failed to parse log message for cloud logging: {e}")
                    return
                wandb.log(json_data)
        
        def log_metrics(metrics: dict, step: int = None, prefix: str = ""):
            '''
            Log structured metrics to wandb with optional prefix for grouping.
            
            Args:
                metrics: Dictionary of metric_name -> value
                step: Optional step/epoch number
                prefix: Optional prefix for metric names (e.g., "classifier/", "fine_tune/")
            
            Example:
                logger.log_metrics({"accuracy": 0.95, "loss": 0.1}, step=5, prefix="train/")
                # Logs as: train/accuracy=0.95, train/loss=0.1 at step 5
            '''
            if LoggerManager._cloud:
                prefixed_metrics = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
                if step is not None:
                    prefixed_metrics["epoch"] = step
                wandb.log(prefixed_metrics)
        
        def log_comparison(metrics: dict, title: str = "Comparison"):
            '''
            Log comparison metrics (e.g., before/after fine-tuning).
            Creates a wandb summary entry for easy comparison.
            
            Args:
                metrics: Dictionary of comparison_name -> value
                title: Title for the comparison
            
            Example:
                logger.log_comparison({
                    "classifier_test_acc": 0.85,
                    "fine_tuned_test_acc": 0.92,
                    "improvement": 0.07
                })
            '''
            if LoggerManager._cloud:
                # Log as summary metrics for easy comparison
                for key, value in metrics.items():
                    # Convert torch tensors to Python scalars if needed
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else value.tolist()
                    wandb.run.summary[f"{title}/{key}"] = value
                # Also log as regular metrics (only numeric values)
                numeric_metrics = {f"{title}/{k}": v for k, v in metrics.items() 
                                 if isinstance(v, (int, float)) or 
                                 (isinstance(v, torch.Tensor) and v.numel() == 1)}
                if numeric_metrics:
                    wandb.log(numeric_metrics)
        
        # Track metrics for plotting (experiment-level)
        _metrics_history = defaultdict(list)
        
        def track_metric(prefix: str, metric_name: str, value: float, epoch: int):
            '''
            Track a metric value for plotting purposes.
            
            Args:
                prefix: Stage prefix (e.g., "classifier", "fine_tune")
                metric_name: Name of the metric (e.g., "train_accuracy")
                value: Metric value
                epoch: Epoch number
            '''
            key = f"{prefix}/{metric_name}"
            _metrics_history[key].append({"epoch": epoch, "value": value})
        
        def create_experiment_plots(exp_dir: str):
            '''
            Create experiment-level plots for training analysis.
            Saves plots to experiment directory and uploads to wandb.
            
            Args:
                exp_dir: Experiment directory path
            '''
            plots_dir = os.path.join(exp_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Extract metrics by stage
            classifier_metrics = {k.replace("classifier/", ""): v 
                                for k, v in _metrics_history.items() 
                                if k.startswith("classifier/")}
            fine_tune_metrics = {k.replace("fine_tune/", ""): v 
                                for k, v in _metrics_history.items() 
                                if k.startswith("fine_tune/")}
            
            # 1. Training Curves: Accuracy
            if classifier_metrics.get("train_accuracy") and classifier_metrics.get("val_accuracy"):
                fig, ax = plt.subplots(figsize=(10, 6))
                cls_train = classifier_metrics["train_accuracy"]
                cls_val = classifier_metrics["val_accuracy"]
                ax.plot([m["epoch"] for m in cls_train], [m["value"] for m in cls_train], 
                       label="Classifier Train", marker='o', markersize=3)
                ax.plot([m["epoch"] for m in cls_val], [m["value"] for m in cls_val], 
                       label="Classifier Val", marker='s', markersize=3)
                
                if fine_tune_metrics.get("train_accuracy") and fine_tune_metrics.get("val_accuracy"):
                    ft_train = fine_tune_metrics["train_accuracy"]
                    ft_val = fine_tune_metrics["val_accuracy"]
                    ft_start_epoch = cls_train[-1]["epoch"] + 1 if cls_train else 1
                    ax.plot([m["epoch"] for m in ft_train], [m["value"] for m in ft_train], 
                           label="Fine-tune Train", marker='o', markersize=3)
                    ax.plot([m["epoch"] for m in ft_val], [m["value"] for m in ft_val], 
                           label="Fine-tune Val", marker='s', markersize=3)
                    # Add vertical line to separate stages
                    ax.axvline(x=ft_start_epoch, color='gray', linestyle='--', alpha=0.5, label='Fine-tuning Start')
                
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Accuracy")
                ax.set_title("Training and Validation Accuracy")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, "accuracy_curves.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.send_file("accuracy_curves", plot_path, "image")
            
            # 2. Training Curves: Loss
            if classifier_metrics.get("train_loss") and classifier_metrics.get("val_loss"):
                fig, ax = plt.subplots(figsize=(10, 6))
                cls_train_loss = classifier_metrics["train_loss"]
                cls_val_loss = classifier_metrics["val_loss"]
                ax.plot([m["epoch"] for m in cls_train_loss], [m["value"] for m in cls_train_loss], 
                       label="Classifier Train Loss", marker='o', markersize=3)
                ax.plot([m["epoch"] for m in cls_val_loss], [m["value"] for m in cls_val_loss], 
                       label="Classifier Val Loss", marker='s', markersize=3)
                
                if fine_tune_metrics.get("train_loss") and fine_tune_metrics.get("val_loss"):
                    ft_train_loss = fine_tune_metrics["train_loss"]
                    ft_val_loss = fine_tune_metrics["val_loss"]
                    ft_start_epoch = cls_train_loss[-1]["epoch"] + 1 if cls_train_loss else 1
                    ax.plot([m["epoch"] for m in ft_train_loss], [m["value"] for m in ft_train_loss], 
                           label="Fine-tune Train Loss", marker='o', markersize=3)
                    ax.plot([m["epoch"] for m in ft_val_loss], [m["value"] for m in ft_val_loss], 
                           label="Fine-tune Val Loss", marker='s', markersize=3)
                    ax.axvline(x=ft_start_epoch, color='gray', linestyle='--', alpha=0.5)
                
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training and Validation Loss")
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, "loss_curves.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.send_file("loss_curves", plot_path, "image")
            
            # 3. Loss Breakdown (Fine-tuning stage)
            if fine_tune_metrics.get("train_base_loss") and fine_tune_metrics.get("train_reg_loss"):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                ft_base = fine_tune_metrics["train_base_loss"]
                ft_reg = fine_tune_metrics["train_reg_loss"]
                ft_total = fine_tune_metrics.get("train_loss", [])
                
                epochs = [m["epoch"] for m in ft_base]
                base_vals = [m["value"] for m in ft_base]
                reg_vals = [m["value"] for m in ft_reg]
                
                # Stacked area plot
                ax1.fill_between(epochs, 0, base_vals, alpha=0.6, label="Base Loss")
                ax1.fill_between(epochs, base_vals, [b+r for b, r in zip(base_vals, reg_vals)], 
                               alpha=0.6, label="Reg Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.set_title("Loss Breakdown (Stacked)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Line plot
                ax2.plot(epochs, base_vals, label="Base Loss", marker='o', markersize=3)
                ax2.plot(epochs, reg_vals, label="Reg Loss", marker='s', markersize=3)
                if ft_total:
                    total_vals = [m["value"] for m in ft_total]
                    ax2.plot(epochs, total_vals, label="Total Loss", marker='^', markersize=3)
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Loss")
                ax2.set_title("Loss Breakdown (Lines)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, "loss_breakdown.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.send_file("loss_breakdown", plot_path, "image")
            
            # 4. Parameter Distance Tracking
            if fine_tune_metrics.get("parameter_distance"):
                fig, ax = plt.subplots(figsize=(10, 5))
                dist_data = fine_tune_metrics["parameter_distance"]
                ax.plot([m["epoch"] for m in dist_data], [m["value"] for m in dist_data], 
                       marker='o', markersize=3, color='purple')
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Parameter Distance")
                ax.set_title("Parameter Distance from Initialization (Fine-tuning)")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, "parameter_distance.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.send_file("parameter_distance", plot_path, "image")
            
            logger.info(f"Created experiment plots in {plots_dir}")
        
        # add methods to logger
        logger.log = log
        logger.send_file = send_file
        logger.finish = finish
        # Create wrapper functions that accept 'self' as first argument (for method binding)
        def log_metrics_wrapper(self, metrics, step=None, prefix=""):
            return log_metrics(metrics, step, prefix)
        
        def log_comparison_wrapper(self, metrics, title="Comparison"):
            return log_comparison(metrics, title)
        
        def track_metric_wrapper(self, prefix, metric_name, value, epoch):
            return track_metric(prefix, metric_name, value, epoch)
        
        def create_experiment_plots_wrapper(self, exp_dir):
            return create_experiment_plots(exp_dir)
        
        # Bind wrappers as methods
        logger.log_metrics = MethodType(log_metrics_wrapper, logger)
        logger.log_comparison = MethodType(log_comparison_wrapper, logger)
        logger.track_metric = MethodType(track_metric_wrapper, logger)
        logger.create_experiment_plots = MethodType(create_experiment_plots_wrapper, logger)
        logger._metrics_history = _metrics_history
        LoggerManager._logger = logger
        logger.info("===== Start log =====")
        if config:
            logger.info(f"Config: {config}")
        return logger
    
    @staticmethod
    def log_uncaught_exceptions(exctype, value, tb):
        """
        Logs uncaught exceptions to the error log.
        """
        if LoggerManager._logger is None:
            raise RuntimeError("Logger must be initialized before logging exceptions.")
        LoggerManager._logger.error("Unhandled exception", exc_info=(exctype, value, tb))


def wandb_login():
    """
    Logs into W&B using an API key from environment variable 'WANDB_API_KEY'.
    Handles errors if the key is missing or login fails.
    """
    api_key = os.getenv("WANDB_API_KEY")
    
    if api_key is None:
        raise ValueError("Environment variable 'WANDB_API_KEY' not found. Please set it first.")
    
    try:
        # login() returns True if successful, False if already logged in
        logged_in = wandb.login(key=api_key)
        if logged_in:
            print("Logged into W&B successfully.")
        else:
            print("Already logged into W&B.")
    except wandb.errors.CommError as e:
        print(f"Failed to log into W&B: {e}")
    except Exception as e:
        print(f"Unexpected error during W&B login: {e}")