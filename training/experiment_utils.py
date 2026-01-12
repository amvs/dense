"""
Shared utilities for experiment setup and logging.
Used by train_wph.py and train_wph_svm.py.
"""
import os
import platform
import sys
from datetime import datetime
import torch
from dense.helpers import LoggerManager


def log_model_parameters(model, classifier=None, logger=None):
    """
    Log trainable parameters for feature extractor and classifier.
    
    Args:
        model: Model with feature_extractors attribute
        classifier: Either nn.Module, sklearn pipeline, or int (for SVM param count)
        logger: Logger instance (uses LoggerManager if None)
    """
    if logger is None:
        logger = LoggerManager.get_logger()
    
    feature_extractor_params = 0
    for fe in model.feature_extractors:
        feature_extractor_params += sum(p.numel() for p in fe.parameters() if p.requires_grad)
    
    # Handle different classifier types
    if isinstance(classifier, int):
        classifier_params = classifier
    elif classifier is None:
        classifier_params = 0
    elif hasattr(classifier, 'parameters'):
        # PyTorch module
        classifier_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    elif hasattr(classifier, 'named_steps'):
        # sklearn pipeline (for SVM)
        classifier_params = count_svm_parameters(classifier)
    else:
        classifier_params = 0
    
    total_params = feature_extractor_params + classifier_params
    
    logger.log(f"Feature_Extractor_Params={feature_extractor_params}", data=True)
    logger.log(f"Classifier_Params={classifier_params}", data=True)
    logger.log(f"Total_Params={total_params}", data=True)


def count_svm_parameters(svm_pipeline):
    """Count the number of parameters in the trained SVM classifier."""
    svm = svm_pipeline.named_steps['svm']
    
    if hasattr(svm, 'coef_'):
        # LinearSVC and SGDClassifier have coef_ (weights) and intercept_ (bias)
        n_params = svm.coef_.size
        if hasattr(svm, 'intercept_'):
            n_params += svm.intercept_.size
    elif hasattr(svm, 'support_vectors_'):
        # SVC with kernel stores support vectors
        # Parameters include dual coefficients and support vectors
        n_params = svm.dual_coef_.size + svm.support_vectors_.size
        if hasattr(svm, 'intercept_'):
            n_params += svm.intercept_.size
    else:
        n_params = 0
    
    return n_params


def log_environment(logger, device):
    """
    Log Python version, PyTorch version, CUDA info.
    
    Args:
        logger: Logger instance
        device: torch.device
    """
    logger.log(f"Python_Version={platform.python_version()}")
    logger.log(f"PyTorch_Version={torch.__version__}")
    logger.log(f"CUDA_Available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log(f"CUDA_Version={torch.version.cuda}")
        logger.log(f"GPU={torch.cuda.get_device_name(0)}")
    logger.log(f"Using_Device={device}")


def setup_experiment(args, config, wandb_project):
    """
    Set up experiment directory, logger, and device.
    
    Args:
        args: Parsed command-line arguments
        config: Configuration dictionary
        wandb_project: Weights & Biases project name
    
    Returns:
        tuple: (exp_dir, logger, device, seed)
    """
    # Create output folder
    train_ratio = config["train_ratio"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # Optional GPU id suffix provided by sweep launcher to avoid log collisions
    gpu_id_env = os.getenv("SWEEP_GPU_ID")
    run_suffix = f"-gpu{gpu_id_env}" if gpu_id_env is not None else ""
    
    if args.sweep_dir is not None:
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, f"{train_ratio=}", f"run-{timestamp}{run_suffix}")
    else:
        exp_dir = os.path.join("experiments", f"{train_ratio=}", f"run-{timestamp}{run_suffix}")
    
    os.makedirs(exp_dir, exist_ok=True)
    
    # Initialize logger
    logger = LoggerManager.get_logger(
        log_dir=exp_dir,
        wandb_project=wandb_project,
        config=config,
        name=f"{train_ratio=}(run-{timestamp})"
    )
    sys.excepthook = LoggerManager.log_uncaught_exceptions
    logger.log("===== Start log =====")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Log environment
    log_environment(logger, device)
    
    # Extract seed
    seed = config.get("seed", None)
    
    return exp_dir, logger, device, seed
