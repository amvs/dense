import argparse
import os
import time
import torch
import numpy as np
import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from configs import load_config, save_config, apply_overrides, AutoConfig
from training.base_trainer import recompute_bn_running_stats
from training.experiment_utils import setup_experiment, log_model_parameters
from training.data_utils import load_and_split_data
from wph.wph_model import WPHClassifier
from wph.classifiers import PCAClassifier
from wph.model_factory import create_wph_feature_extractor
from dense.helpers import LoggerManager
from train_wph import set_seed, worker_init_fn

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Train WPHPca with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
    parser.add_argument("--override", nargs="*", default=[],
                        help="List of key=value pairs to override config")
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="If this is a sweep job, specify the sweep output dir")
    parser.add_argument("--wandb_project", type=str, default="WPHWavelet-PCA")
    parser.add_argument("--pca_components", type=str, default=None,
                        help="Number of PCA components (int, float fraction, or 'auto')")
    parser.add_argument("--no_scale", action='store_true',
                        help="Skip per-class feature scaling before PCA")
    parser.add_argument("--skip-finetuning", action='store_true',
                        help="If set, skip the fine-tuning feature extractor phase")
    return parser.parse_args()


def extract_all_features(model, loader, device, vmap_chunk_size=None):
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


def resolve_n_components(arg_components, config_components, n_features):
    source = arg_components if arg_components is not None else config_components
    if source is None or str(source).lower() == "auto":
        return min(n_features, 50)
    try:
        if isinstance(source, str) and "." in source:
            frac = float(source)
            return max(1, int(frac * n_features))
        value = float(source)
        if 0 < value <= 1:
            return max(1, int(value * n_features))
        return int(value)
    except ValueError:
        return min(n_features, 50)


def finetune_feature_extractor(model, train_loader, val_loader, device,
                                epochs, lr, lambda_reg, logger, config, exp_dir,
                                original_fe_params):
    optimizer = torch.optim.Adam(
        [p for fe in model.feature_extractors for p in fe.parameters()],
        lr=lr
    )
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
    freeze_classifier = config.get("freeze_classifier", True)
    l2_norm = 0.0

    for epoch in range(epochs):
        if not freeze_classifier:
            model.eval()
            with torch.no_grad():
                train_feats, train_lbls = extract_all_features(model, train_loader, device, vmap_chunk_size)
            train_feats_np = train_feats.cpu().numpy() if isinstance(train_feats, torch.Tensor) else train_feats
            train_lbls_np = train_lbls.cpu().numpy() if isinstance(train_lbls, torch.Tensor) else train_lbls
            model.classifier.fit(train_feats_np, train_lbls_np)

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
            features = model.extract_features(inputs, vmap_chunk_size=vmap_chunk_size)
            log_probs = model.classifier(features)
            base_loss = criterion(log_probs, labels)
            if lambda_reg > 0 and original_fe_params is not None:
                current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
                if normalize_reg:
                    reg_loss = lambda_reg * sum((p - o).pow(2).sum() for p, o in zip(current_fe_params, original_fe_params)) / len(current_fe_params)
                else:
                    reg_loss = lambda_reg * sum((p - o).pow(2).sum() for p, o in zip(current_fe_params, original_fe_params))
            else:
                reg_loss = torch.tensor(0.0, device=device)
            total_loss = base_loss + reg_loss
            total_loss.backward()
            optimizer.step()
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

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                features = model.extract_features(inputs, vmap_chunk_size=vmap_chunk_size)
                log_probs = model.classifier(features)
                loss = criterion(log_probs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(log_probs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        logger.log(f"Phase: feature_extractor Epoch: {epoch+1}")
        logger.log(f"Epoch={epoch+1} Train_Acc={train_acc:.4f} Val_Acc={val_acc:.4f} Base_Loss={avg_base_loss:.4e} Reg_Loss={avg_reg_loss:.4e} Total_Loss={avg_loss:.4e}", data=True)

        current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
        l2_norm = sum((p - o).norm().item() for p, o in zip(current_fe_params, original_fe_params))
        logger.log(f"Epoch={epoch+1} L2_Norm_Distance={l2_norm:.4f}", data=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(exp_dir, "best_feature_extractor_model_state.pt")
            torch.save(model.state_dict(), best_model_path)
            logger.log(f"Saved best feature_extractor model state_dict to {best_model_path}")

    final_model_path = os.path.join(exp_dir, "final_feature_extractor_model_state.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.log(f"Saved final feature_extractor model state_dict to {final_model_path}")

    current_fe_params = [p for fe in model.feature_extractors for p in fe.parameters()]
    l2_norm = sum((p - o).norm().item() for p, o in zip(current_fe_params, original_fe_params))
    logger.log(f"Final L2_Norm_Distance={l2_norm:.4f}", data=True)
    return best_val_acc, l2_norm


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    config = AutoConfig(config)

    exp_dir, logger, device, seed = setup_experiment(args, config, args.wandb_project)
    set_seed(seed)

    from functools import partial
    worker_init_with_seed = partial(worker_init_fn, seed=seed)
    train_loader, val_loader, test_loader, nb_class, image_shape = load_and_split_data(
        config, worker_init_with_seed
    )

    feature_extractor, filters = create_wph_feature_extractor(config, image_shape, device)
    
    # Create PCA classifier wrapper
    n_components = resolve_n_components(args.pca_components, config.get("pca_components", None), 1000)  # Will be updated after feature extraction
    scale_features = not args.no_scale
    whiten = bool(config.get("pca_whiten", False))
    
    nb_moments_int = int(feature_extractor.nb_moments)
    pca_classifier = PCAClassifier(
        input_dim=nb_moments_int,
        n_components=n_components,
        scale_features=scale_features,
        whiten=whiten
    )
    
    model = WPHClassifier(
        feature_extractor=feature_extractor,
        classifier=pca_classifier,
        copies=int(config.get("copies", 1)),
        noise_std=float(config.get("noise_std", 0.01)),
    ).to(device)

    logger.log("Model_Architecture:")
    logger.log(str(model))
    log_model_parameters(model, None, logger)

    with torch.no_grad():
        original_fe_params = [p.clone().detach() for fe in model.feature_extractors for p in fe.parameters()]

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

    n_components = resolve_n_components(args.pca_components, config.get("pca_components", None), train_features.shape[1])
    scale_features = not args.no_scale
    whiten = bool(config.get("pca_whiten", False))

    logger.log("Training PCA classifier...")
    train_features_np = train_features.cpu().numpy() if isinstance(train_features, torch.Tensor) else train_features
    train_labels_np = train_labels.cpu().numpy() if isinstance(train_labels, torch.Tensor) else train_labels
    model.classifier.fit(train_features_np, train_labels_np)
    logger.log("PCA training completed")

    log_model_parameters(model, model.classifier.count_parameters(), logger)

    val_predictions = model.classifier.predict(torch.tensor(val_features, dtype=torch.float32))
    val_accuracy = accuracy_score(val_labels, val_predictions)
    logger.log(f"Validation_Accuracy={val_accuracy:.4f}", data=True)

    test_predictions = model.classifier.predict(torch.tensor(test_features, dtype=torch.float32))
    pca_test_accuracy = accuracy_score(test_labels, test_predictions)
    logger.log(f"PCA Test Accuracy: {pca_test_accuracy:.4f}")

    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Saved original model state_dict to {save_original}")

    classifier_path = os.path.join(exp_dir, "pca_classifier.pkl")
    joblib.dump(pca_classifier, classifier_path)
    logger.log(f"Saved PCA classifier to {classifier_path}")

    if not args.skip_finetuning:
        logger.log("Fine-tuning feature extractor...")
        lr_conv = float(config.get("lr_conv", config.get("lr", 1e-3) * 0.01))
        conv_epochs = config.get("conv_epochs", 10)
        lambda_reg = config.get("lambda_reg", 0.0)
        start_time = time.time()
        best_acc_feature_extractor, l2_norm = finetune_feature_extractor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=conv_epochs,
            lr=lr_conv,
            lambda_reg=lambda_reg,
            logger=logger,
            config=config,
            exp_dir=exp_dir,
            original_fe_params=original_fe_params,
        )
        elapsed_time = time.time() - start_time
        logger.log(f"Feature extractor fine-tuning completed in {elapsed_time:.2f} seconds.")
    else:
        logger.log("Skipping fine-tuning phase.")
        best_acc_feature_extractor = 0.0
        l2_norm = 0.0

    logger.log("Extracting features from test set with fine-tuned model...")
    test_features_finetuned, test_labels_finetuned = extract_all_features(model, test_loader, device, vmap_chunk_size)
    test_predictions_finetuned = model.classifier.predict(torch.tensor(test_features_finetuned, dtype=torch.float32))
    feature_extractor_test_acc = accuracy_score(test_labels_finetuned, test_predictions_finetuned)
    logger.log(f"Feature Extractor Test Accuracy={feature_extractor_test_acc:.4f}", data=True)

    model_path = os.path.join(exp_dir, "wph_pca_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.log(f"Saved model state_dict to {model_path}")

    logger.log("===== Final Test Accuracies =====")
    logger.log(f"PCA_Test_Accuracy={pca_test_accuracy:.4f}", data=True)
    logger.log(f"Feature_Extractor_Test_Accuracy={feature_extractor_test_acc:.4f}", data=True)

    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["nb_moments"] = feature_extractor.nb_moments
    config["l2_norm_finetuning"] = l2_norm
    config["feature_extractor_test_acc"] = feature_extractor_test_acc
    config["pca_test_acc"] = pca_test_accuracy
    config["classifier_last_acc"] = pca_test_accuracy
    config["classifier_best_acc"] = pca_test_accuracy
    config["feature_extractor_last_acc"] = feature_extractor_test_acc
    config["feature_extractor_best_acc"] = best_acc_feature_extractor
    config["feature_extractor_params"] = sum(p.numel() for fe in model.feature_extractors for p in fe.parameters())
    config["classifier_params"] = model.classifier.count_parameters()
    config["finetuning_gain"] = feature_extractor_test_acc - pca_test_accuracy
    config["val_acc"] = val_accuracy
    config["device"] = str(device)
    config["model_type"] = "wph_pca"
    config["pca_components"] = n_components
    config["scale_features"] = scale_features

    save_config(exp_dir, config)

    try:
        from visualize import visualize_main
        logger.log("Visualizing filters and activations...")
        tuned_filename = 'best_feature_extractor_model_state.pt' if not args.skip_finetuning else 'wph_pca_model.pt'
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
