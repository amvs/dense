import argparse
import os
from datetime import datetime
import torch
import sys
import numpy as np
import platform
import time
from dotenv import load_dotenv
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import warnings
load_dotenv()

from configs import load_config, save_config, apply_overrides, AutoConfig
from training.base_trainer import recompute_bn_running_stats
from training.experiment_utils import setup_experiment, log_model_parameters, count_svm_parameters
from training.data_utils import load_and_split_data
from wph.wph_model import WPHSvm
from wph.model_factory import create_wph_feature_extractor
from dense.helpers import LoggerManager
from train_wph import set_seed, worker_init_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train WPHSvm with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
    parser.add_argument("--override", nargs="*", default=[],
                        help="List of key=value pairs to override config")
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="If this is a sweep job, specify the sweep output dir")
    parser.add_argument("--wandb_project", type=str, default="WPHWavelet-SVM")
    parser.add_argument("--svm_type", type=str, default="auto", 
                        choices=["auto", "linear", "rbf", "sgd"],
                        help="SVM type: auto (chooses based on data size), linear (LinearSVC), rbf (SVC with RBF kernel), sgd (SGDClassifier)")
    parser.add_argument("--no_scale", action='store_true',
                        help="Skip feature scaling (not recommended for SVM)")
    parser.add_argument("--skip-finetuning", action='store_true',
                        help="If set, skip the fine-tuning feature extractor phase")
    return parser.parse_args()




def extract_all_features(model, loader, device, vmap_chunk_size=None):
    """
    Extract features from all samples in the loader.

    Args:
        model (WPHSvm): The model with feature extractors.
        loader (DataLoader): Data loader.
        device (torch.device): Device to use.
        vmap_chunk_size (int, optional): Chunk size for vmap operations.

    Returns:
        tuple: (features, labels) as numpy arrays.
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            features = model.extract_features(inputs, vmap_chunk_size=vmap_chunk_size)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.vstack(all_features), np.concatenate(all_labels)


def create_svm_classifier(n_samples, n_features, n_classes, svm_type="auto", scale_features=True):
    """
    Create an appropriate SVM classifier based on dataset size.

    Args:
        n_samples (int): Number of training samples.
        n_features (int): Number of features.
        n_classes (int): Number of classes.
        svm_type (str): Type of SVM to use ("auto", "linear", "rbf", "sgd").
        scale_features (bool): Whether to scale features.

    Returns:
        sklearn.pipeline.Pipeline: SVM classifier pipeline.
    """
    logger = LoggerManager.get_logger()
    
    # Determine SVM type
    if svm_type == "auto":
        if n_samples > 50000:
            chosen_type = "sgd"
        elif n_samples > 10000 or n_features > 5000:
            chosen_type = "linear"
        else:
            chosen_type = "rbf"
        logger.log(f"Auto-selected SVM type: {chosen_type} (n_samples={n_samples}, n_features={n_features})")
    else:
        chosen_type = svm_type
        logger.log(f"Using specified SVM type: {chosen_type}")
    
    # Create classifier
    if chosen_type == "sgd":
        # SGDClassifier for very large datasets
        classifier = SGDClassifier(
            loss='hinge',
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            n_jobs=-1
        )
    elif chosen_type == "linear":
        # LinearSVC for large datasets with many features
        classifier = LinearSVC(
            max_iter=2000,
            tol=1e-4,
            random_state=42,
            dual="auto"
        )
    elif chosen_type == "rbf":
        # Standard SVC with RBF kernel for smaller datasets
        classifier = SVC(
            kernel='rbf',
            gamma='scale',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown SVM type: {chosen_type}")
    
    # Create pipeline with optional scaling
    if scale_features:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', classifier)
        ])
        logger.log("Using StandardScaler before SVM")
    else:
        pipeline = Pipeline([
            ('svm', classifier)
        ])
        logger.log("Skipping feature scaling")
    
    return pipeline


def svm_predict_proba_torch(svm_pipeline, features_tensor, device):
    """
    Get class probabilities from SVM for PyTorch tensors (for computing cross-entropy loss).
    
    Args:
        svm_pipeline: Trained sklearn pipeline with SVM.
        features_tensor: PyTorch tensor of features.
        device: Device for computation.
        
    Returns:
        torch.Tensor: Log probabilities for each class (for use with NLLLoss or CrossEntropyLoss).
    """
    # Convert to numpy for sklearn
    features_np = features_tensor.detach().cpu().numpy()
    
    # Get decision function or predict_proba
    svm = svm_pipeline.named_steps['svm']
    if hasattr(svm_pipeline, 'decision_function'):
        # For LinearSVC and SGDClassifier, use decision function
        decision_values = svm_pipeline.decision_function(features_np)
        # Convert to log probabilities using softmax
        decision_tensor = torch.tensor(decision_values, dtype=torch.float32, device=device)
        log_probs = torch.nn.functional.log_softmax(decision_tensor, dim=1)
    else:
        # For SVC with probability=True
        probs = svm_pipeline.predict_proba(features_np)
        log_probs = torch.tensor(np.log(probs + 1e-10), dtype=torch.float32, device=device)
    
    return log_probs


def finetune_feature_extractor(model, svm_pipeline, train_loader, val_loader, device, 
                                epochs, lr, lambda_reg, logger, config, exp_dir, original_fe_params):
    """
    Fine-tune the feature extractor using the trained SVM as a frozen classifier.
    
    We can't backprop through sklearn, so we:
    1. Extract features with gradient tracking
    2. Get SVM predictions (no grad)
    3. Compute loss and backprop through feature extractor only
    
    Args:
        model: WPHSvm model with feature extractors
        svm_pipeline: Trained sklearn SVM pipeline
        train_loader: Training data loader
        val_loader: Validation data loader
        device: PyTorch device
        epochs: Number of fine-tuning epochs
        lr: Learning rate for feature extractor
        lambda_reg: Regularization strength
        logger: Logger instance
        config: Configuration dict
        exp_dir: Experiment directory
        original_fe_params: Original feature extractor parameters for regularization
    
    Returns:
        tuple: (best_val_acc, final_l2_norm)
    """
    
    # Set up optimizer for feature extractors only
    optimizer = torch.optim.Adam(
        [p for fe in model.feature_extractors for p in fe.parameters()],
        lr=lr
    )
    
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
    
    criterion = torch.nn.NLLLoss()
    best_val_acc = 0.0
    normalize_reg = config.get('normalize_reg', True)
    vmap_chunk_size = config.get('vmap_chunk_size', None)
    l2_norm = 0.0  # Initialize L2 norm
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        epoch_reg_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Extract features (with gradient)
            features = model.extract_features(inputs, vmap_chunk_size=vmap_chunk_size)
            
            # Get SVM predictions as log probabilities
            log_probs = svm_predict_proba_torch(svm_pipeline, features, device)
            
            # Compute base loss
            base_loss = criterion(log_probs, labels)
            
            # Compute regularization loss
            if lambda_reg > 0 and original_fe_params is not None:
                current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
                if normalize_reg:
                    # Normalize by number of parameters
                    reg_loss = lambda_reg * sum((p - o).pow(2).sum() for p, o in zip(current_fe_params, original_fe_params)) / len(current_fe_params)
                else:
                    reg_loss = lambda_reg * sum((p - o).pow(2).sum() for p, o in zip(current_fe_params, original_fe_params))
            else:
                reg_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            total_loss = base_loss + reg_loss
            
            # Backprop through feature extractor
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            epoch_base_loss += base_loss.item()
            epoch_reg_loss += reg_loss.item()
            
            _, predicted = torch.max(log_probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = correct / total
        avg_loss = epoch_loss / len(train_loader)
        avg_base_loss = epoch_base_loss / len(train_loader)
        avg_reg_loss = epoch_reg_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                features = model.extract_features(inputs, vmap_chunk_size=vmap_chunk_size)
                log_probs = svm_predict_proba_torch(svm_pipeline, features, device)
                loss = criterion(log_probs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(log_probs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(val_acc)
        
        # Log progress
        logger.log(f"Phase: feature_extractor Epoch: {epoch+1}")
        logger.log(f"Epoch={epoch+1} Train_Acc={train_acc:.4f} Val_Acc={val_acc:.4f} Base_Loss={avg_base_loss:.4e} Reg_Loss={avg_reg_loss:.4e} Total_Loss={avg_loss:.4e}", data=True)
        
        # Compute L2 norm distance
        current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
        l2_norm = sum((p - o).norm().item() for p, o in zip(current_fe_params, original_fe_params))
        logger.log(f"Epoch={epoch+1} L2_Norm_Distance={l2_norm:.4f}", data=True)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(exp_dir, "best_feature_extractor_model_state.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"Saved best feature_extractor model state_dict to {best_model_path}")
    
    # Save final model
    final_model_path = os.path.join(exp_dir, "final_feature_extractor_model_state.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.log(f"Saved final feature_extractor model state_dict to {final_model_path}")
    
    # Compute final L2 norm distance after all epochs
    current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
    l2_norm = sum((p - o).norm().item() for p, o in zip(current_fe_params, original_fe_params))
    logger.log(f"Final L2_Norm_Distance={l2_norm:.4f}", data=True)
    
    return best_val_acc, l2_norm


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
    from functools import partial
    worker_init_with_seed = partial(worker_init_fn, seed=seed)
    train_loader, val_loader, test_loader, nb_class, image_shape = load_and_split_data(
        config, worker_init_with_seed
    )

    # Initialize feature extractor
    feature_extractor, filters = create_wph_feature_extractor(config, image_shape, device)
    
    model = WPHSvm(
        feature_extractor,
        copies=int(config.get("copies", 1)),
        noise_std=float(config.get("noise_std", 0.01)),
    ).to(device)

    # Log model architecture
    logger.log("Model_Architecture:")
    logger.log(str(model))
    log_model_parameters(model, None, logger)

    # Clone original parameters for regularization during fine-tuning
    with torch.no_grad():
        original_fe_params = [p.clone().detach() for fe in model.feature_extractors for p in fe.parameters()]

    # Warm up BatchNorm running stats before feature extraction (if model uses BN)
    if hasattr(model.feature_extractors[0], 'batch_norm') or any(
        isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d)
        for fe in model.feature_extractors for m in fe.modules()
    ):
        bn_warmup_batches = int(config.get("bn_warmup_batches", 100))
        try:
            t0 = time.time()
            recompute_bn_running_stats(model, train_loader, device, max_batches=bn_warmup_batches, logger=logger, momentum=0.9)
            logger.log(f"BN warmup took {time.time()-t0:.2f}s", data=True)
        except Exception as e:
            logger.log(f"BN warmup failed: {e}", data=True)

    # Extract features
    logger.log("Extracting features from training set...")
    vmap_chunk_size = config.get('vmap_chunk_size', None)
    train_features, train_labels = extract_all_features(model, train_loader, device, vmap_chunk_size)
    logger.log(f"Training features shape: {train_features.shape}")
    
    logger.log("Extracting features from validation set...")
    val_features, val_labels = extract_all_features(model, val_loader, device, vmap_chunk_size)
    logger.log(f"Validation features shape: {val_features.shape}")
    
    logger.log("Extracting features from test set...")
    test_features, test_labels = extract_all_features(model, test_loader, device, vmap_chunk_size)
    logger.log(f"Test features shape: {test_features.shape}")

    # Train SVM
    logger.log("Training SVM classifier...")
    svm_pipeline = create_svm_classifier(
        n_samples=train_features.shape[0],
        n_features=train_features.shape[1],
        n_classes=nb_class,
        svm_type=args.svm_type,
        scale_features=not args.no_scale
    )
    
    # Suppress convergence warnings for large datasets
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        svm_pipeline.fit(train_features, train_labels)
    
    logger.log("SVM training completed")
    
    # Store SVM in model
    model.svm = svm_pipeline
    
    # Log classifier parameters
    log_model_parameters(model, svm_pipeline, logger)

    # Evaluate on validation set
    val_predictions = svm_pipeline.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.log(f"Validation_Accuracy={val_accuracy:.4f}", data=True)

    # Evaluate on test set
    test_predictions = svm_pipeline.predict(test_features)
    svm_test_accuracy = accuracy_score(test_labels, test_predictions)
    logger.log(f"SVM Test Accuracy: {svm_test_accuracy:.4f}")

    # Save model before fine-tuning (matching train_wph.py pattern)
    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Saved original model state_dict to {save_original}")
    
    svm_path = os.path.join(exp_dir, "svm_classifier.pkl")
    joblib.dump(svm_pipeline, svm_path)
    logger.log(f"Saved SVM classifier to {svm_path}")
    
    # Fine-tune feature extractor
    if not args.skip_finetuning:
        logger.log("Fine-tuning feature extractor...")
        # Config option to control whether the SVM classifier is retrained during finetuning.
        # Default: keep frozen (True) - SVM is expensive to retrain on every epoch
        freeze_classifier = config.get("freeze_classifier", True)
        if freeze_classifier:
            logger.log("Finetune: SVM classifier will remain frozen (features evaluated against original SVM)")
        else:
            logger.log("Finetune: SVM classifier will be retrained each epoch (expensive!)")
        
        lr_conv = float(config.get("lr_conv", config.get("lr", 1e-3) * 0.01))
        conv_epochs = config.get("conv_epochs", 10)
        lambda_reg = config.get("lambda_reg", 0.0)
        
        start_time = time.time()
        best_acc_feature_extractor, l2_norm = finetune_feature_extractor(
            model=model,
            svm_pipeline=svm_pipeline,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=conv_epochs,
            lr=lr_conv,
            lambda_reg=lambda_reg,
            logger=logger,
            config=config,
            exp_dir=exp_dir,
            original_fe_params=original_fe_params
        )
        elapsed_time = time.time() - start_time
        logger.log(f"Feature extractor fine-tuning completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping fine-tuning phase.")
        best_acc_feature_extractor = 0.0
        l2_norm = 0.0
    
    # Evaluate feature extractor phase (matching train_wph.py pattern)
    logger.log("Extracting features from test set with fine-tuned model...")
    test_features_finetuned, test_labels_finetuned = extract_all_features(model, test_loader, device, vmap_chunk_size)
    test_predictions_finetuned = svm_pipeline.predict(test_features_finetuned)
    feature_extractor_test_acc = accuracy_score(test_labels_finetuned, test_predictions_finetuned)
    logger.log(f"Feature Extractor Test Accuracy={feature_extractor_test_acc:.4f}", data=True)

    # Save final model after fine-tuning
    model_path = os.path.join(exp_dir, "wph_svm_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.log(f"Saved model state_dict to {model_path}")

    # Log final results (matching train_wph.py pattern)
    logger.log("===== Final Test Accuracies =====")
    logger.log(f"SVM_Test_Accuracy={svm_test_accuracy:.4f}", data=True)
    logger.log(f"Feature_Extractor_Test_Accuracy={feature_extractor_test_acc:.4f}", data=True)

    # Update config (matching train_wph.py keys where possible)
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["nb_moments"] = feature_extractor.nb_moments
    config["l2_norm_finetuning"] = l2_norm
    config["feature_extractor_test_acc"] = feature_extractor_test_acc
    config["svm_test_acc"] = svm_test_accuracy  # SVM acts as "classifier" phase
    config["classifier_last_acc"] = svm_test_accuracy
    config["classifier_best_acc"] = svm_test_accuracy  # SVM trained once, no epochs
    config["feature_extractor_last_acc"] = feature_extractor_test_acc
    config["feature_extractor_best_acc"] = best_acc_feature_extractor
    config["feature_extractor_params"] = sum(p.numel() for fe in model.feature_extractors for p in fe.parameters())
    config["classifier_params"] = count_svm_parameters(svm_pipeline)
    config["finetuning_gain"] = feature_extractor_test_acc - svm_test_accuracy
    config["val_acc"] = val_accuracy
    config["device"] = str(device)
    config["model_type"] = "wph_svm"
    config["svm_type"] = args.svm_type
    config["scale_features"] = not args.no_scale
    
    # Save updated config
    save_config(exp_dir, config)

    # Visualization (optional)
    try:
        from visualize import visualize_main
        logger.log("Visualizing filters and activations...")
        tuned_filename = 'best_feature_extractor_model_state.pt' if not args.skip_finetuning else 'wph_svm_model.pt'
        visualize_main(exp_dir, tuned_filename=tuned_filename, model_type='wph', filters=filters)
        activation_img_path = os.path.join(exp_dir, "activations.png")
        if os.path.exists(activation_img_path):
            logger.send_file("activations", activation_img_path, "image")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.log(f"Visualization failed: {e}")

    logger.finish()


if __name__ == "__main__":
    main()
