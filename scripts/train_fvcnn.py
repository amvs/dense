import argparse
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
import gc

from fv_cnn.feature_extractors import MultiScaleCNNExtractor
from fv_cnn.encoders import FisherVectorEncoder
from fv_cnn import FCFeatureExtractor
from training.data_utils import load_and_split_data
from dense.helpers import LoggerManager
from training.experiment_utils import log_model_parameters
from configs import save_config

# Silence Lightning SLURM warning when running without srun
os.environ.setdefault("PL_DISABLE_SLURM_WARNING", "1")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_descriptors(train_loader: DataLoader, extractor: MultiScaleCNNExtractor) -> np.ndarray:
    collected: List[np.ndarray] = []
    extractor.eval()
    # Resolve target device from extractor; fallback to CPU
    if hasattr(extractor, "device"):
        device = extractor.device
    else:
        try:
            device = next(extractor.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            for i in range(images.size(0)):
                img = images[i]
                feats = extractor.extract(img)
                for d in feats["descriptors"]:
                    arr = d.detach().cpu().numpy()
                    collected.append(arr)
    if len(collected) == 0:
        raise RuntimeError("No descriptors collected; check dataset/extractor.")
    return np.concatenate(collected, axis=0)


def encode_split(loader: DataLoader, extractor, encoder=None) -> Tuple[np.ndarray, np.ndarray]:
    codes: List[np.ndarray] = []
    labels: List[int] = []
    extractor.eval()
    # Resolve target device from extractor; fallback to CPU
    if hasattr(extractor, "device"):
        device = extractor.device
    else:
        try:
            device = next(extractor.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for idx, (images, ys) in enumerate(loader):
            images = images.to(device)
            ys = ys.to(device)
            bsz = images.size(0)
            for i in range(bsz):
                img = images[i]
                if encoder is None:
                    # FC pathway
                    feat = extractor.extract(img)["descriptor"].detach().cpu().numpy()
                    codes.append(feat)
                else:
                    feats = extractor.extract(img)
                    desc_list = [d.detach().cpu().numpy() for d in feats["descriptors"]]
                    desc_concat = np.concatenate(desc_list, axis=0)
                    code = encoder.encode(desc_concat)
                    codes.append(code)
                labels.append(int(ys[i]))
    return np.stack(codes, axis=0), np.array(labels)


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    def parse_value(v: str) -> Any:
        if v.lower() in {"true", "false"}:
            return v.lower() == "true"
        if v.lower() == "none":
            return None
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        parts = key.split(".")
        tgt = cfg
        for p in parts[:-1]:
            if p not in tgt or not isinstance(tgt[p], dict):
                tgt[p] = {}
            tgt = tgt[p]
        tgt[parts[-1]] = parse_value(value)
    return cfg


def prepare_experiment(cfg: Dict[str, Any], sweep_dir: str = None, wandb_project: str = None) -> tuple[str, LoggerManager]:
    """Prepare experiment directory and logger, compatible with sweep integration."""
    train_ratio = cfg.get("train_ratio", 1.0)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Optional GPU id suffix provided by sweep launcher to avoid log collisions
    gpu_id_env = os.getenv("SWEEP_GPU_ID")
    run_suffix = f"-gpu{gpu_id_env}" if gpu_id_env is not None else ""
    
    if sweep_dir is not None:
        # Sweep mode: organize by train_ratio inside sweep_dir
        if not os.path.exists(sweep_dir):
            raise ValueError(f"Sweep dir {sweep_dir} does not exist!")
        exp_dir = os.path.join(sweep_dir, f"train_ratio={train_ratio}", f"run-{timestamp}{run_suffix}")
    else:
        # Standalone mode: use exp_root
        exp_root = cfg.get("exp_root", "experiments_fvcnn")
        os.makedirs(exp_root, exist_ok=True)
        exp_name = cfg.get("exp_name")
        if exp_name is None:
            exp_name = f"{cfg.get('dataset','run')}-{cfg.get('backbone','net')}-{timestamp}"
        exp_dir = os.path.join(exp_root, exp_name)
    
    os.makedirs(exp_dir, exist_ok=True)
    cfg["exp_dir"] = exp_dir
    
    logger = LoggerManager.get_logger(
        log_dir=exp_dir,
        wandb_project=wandb_project,
        name=f"train_fvcnn-{timestamp}",
        config=cfg
    )
    return exp_dir, logger


def save_artifacts(exp_dir: str, cfg: Dict[str, Any]) -> None:
    """Save unified config with all results."""
    save_config(exp_dir, cfg)

def worker_init_fn(worker_id, seed=None):
    """Ensure deterministic behavior in DataLoader workers."""
    if seed is not None:
        np.random.seed(seed + worker_id)


def train_and_eval(cfg: Dict[str, Any], logger) -> None:
    set_seed(cfg.get("seed", 42))
    framework = cfg.get("framework", "fvcnn")
    backbone = cfg.get("backbone", "vgg16")
    feature_layer = cfg.get("feature_layer", "conv5_3")

    logger.log(f"Framework={framework} backbone={backbone} feature_layer={feature_layer}")
    train_loader, val_loader, test_loader, nb_class, img_shape = load_and_split_data(cfg, worker_init_fn=worker_init_fn)
    logger.log(f"Dataset has {nb_class} classes")
    
    if framework == "fvcnn":
        extractor = MultiScaleCNNExtractor(
            backbone=backbone,
            feature_layer=feature_layer,
            scales=cfg.get("scales"),
            min_edge=cfg.get("min_edge", 30),
            max_sqrt_hw=cfg.get("max_sqrt_hw", 1024),
        )
        encoder_type = cfg.get("encoder_type", "fv")
        if encoder_type != "fv":
            raise ValueError(f"encoder_type '{encoder_type}' not supported; expected 'fv'")
        gmm_batch_size = cfg.get("gmm_batch_size", 1024)
        encoder = FisherVectorEncoder(
            num_components=cfg.get("gmm_components", 64),
            signed_sqrt_postprocess=cfg.get("signed_sqrt", True),
            l2_postprocess=cfg.get("l2_normalize", True),
            random_state=cfg.get("seed", 42),
            gmm_batch_size=gmm_batch_size,
        )
        logger.log("Collecting all descriptors for GMM fitting...")
        desc = collect_descriptors(train_loader, extractor)
        logger.log(f"Collected {desc.shape[0]} descriptors of dimension {desc.shape[1]}")
        logger.log(f"GMM mini-batch size: {gmm_batch_size}")
        encoder.fit(desc)
        logger.log("GMM fitted")
        train_codes, train_labels = encode_split(train_loader, extractor, encoder)
        logger.log(f"Encoded training set: {train_codes.shape}")
        
        # Apply PCA reduction if specified
        pca_model = None
        if cfg.get("pca_dim") is not None:
            pca_dim = cfg.get("pca_dim")
            logger.log(f"Fitting PCA to reduce {train_codes.shape[1]}-D Fisher Vectors to {pca_dim}-D...")
            pca_model = PCA(n_components=pca_dim, random_state=cfg.get("seed", 42), svd_solver='randomized')
            train_codes = pca_model.fit_transform(train_codes)
            logger.log(f"PCA fitted and applied. Reduced training set shape: {train_codes.shape}")
        
        nb_moments = train_codes.shape[1]
    elif framework == "fccnn":
        extractor = FCFeatureExtractor(backbone=backbone, fc_layer=feature_layer)
        encoder = None
        train_codes, train_labels = encode_split(train_loader, extractor, None)
        nb_moments = train_codes.shape[1]
    else:
        raise ValueError(f"Unsupported framework {framework}")

    # Log model information
    logger.log("Feature Extractor:")
    logger.log(str(extractor))
    if encoder is not None:
        logger.log("Fisher Vector Encoder:")
        logger.log(str(encoder))
    logger.log(f"Number of output features (nb_moments): {nb_moments}")
    
    # Calculate and log parameter counts
    extractor_params = sum(p.numel() for p in extractor.parameters())
    logger.log(f"Feature extractor parameters: {extractor_params}")
    
    encoder_params = 0
    if encoder is not None:
        # Count trainable parameters (if any)
        if hasattr(encoder, "parameters"):
            encoder_params += sum(p.numel() for p in encoder.parameters() if p is not None)
        # Count stored buffers (e.g., GMM weights/means/covariances, PCA stats)
        if hasattr(encoder, "buffers"):
            encoder_params += sum(b.numel() for b in encoder.buffers() if b is not None)
        logger.log(f"Encoder parameters: {encoder_params}")

    C = cfg.get("svm_C", 1.0)
    calibrate = bool(cfg.get("svm_calibration", False))
    if calibrate:
        clf = CalibratedClassifierCV(estimator=LinearSVC(C=C), method="sigmoid", cv=3)
    else:
        clf = LinearSVC(C=C)

    clf.fit(train_codes, train_labels)
    logger.log("Done fitting SVM classifier, computing scores")
    train_acc = clf.score(train_codes, train_labels)
    logger.log(f"Train_Acc={train_acc:.4f}", data=True)
    # Free training encodings
    del train_codes, train_labels
    gc.collect()

    # Validation encoding and eval
    val_codes, val_labels = encode_split(val_loader, extractor, encoder if framework == "fvcnn" else None)
    logger.log(f"Encoded validation set: {val_codes.shape}")
    if pca_model is not None:
        val_codes = pca_model.transform(val_codes)
    breakpoint()
    val_acc = clf.score(val_codes, val_labels)
    del val_codes, val_labels
    gc.collect()

    # Test encoding and eval
    test_codes, test_labels = encode_split(test_loader, extractor, encoder if framework == "fvcnn" else None)
    logger.log(f"Encoded test set: {test_codes.shape}")
    if pca_model is not None:
        test_codes = pca_model.transform(test_codes)
    test_acc = clf.score(test_codes, test_labels)
    del test_codes, test_labels
    gc.collect()

    logger.log(f"Train_Acc={train_acc:.4f} Val_Acc={val_acc:.4f} Test_Acc={test_acc:.4f}", data=True)
    
    # Update config with results
    cfg["nb_class"] = nb_class
    cfg["nb_moments"] = nb_moments
    cfg["framework"] = framework
    cfg["backbone"] = backbone
    cfg["feature_layer"] = feature_layer
    cfg["train_acc"] = float(train_acc)
    cfg["val_acc"] = float(val_acc)
    cfg["test_acc"] = float(test_acc)
    cfg["feature_extractor_params"] = extractor_params
    if encoder is not None:
        cfg["encoder_params"] = encoder_params
    cfg["svm_C"] = C
    
    # Save artifacts
    svm_path = os.path.join(cfg.get("exp_dir", "."), "svm.joblib")
    joblib.dump(clf, svm_path)
    if encoder is not None:
        encoder_path = os.path.join(cfg.get("exp_dir", "."), "encoder.pt")
        encoder.save(encoder_path)
    
    save_artifacts(cfg.get("exp_dir", "."), cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage FV/FC CNN training")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries with key=value (supports dotted keys)"
    )
    parser.add_argument(
        "--sweep_dir", type=str, default=None,
        help="If this is a sweep job, specify the sweep output dir"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Weights and Biases project name for logging"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override)
    exp_dir, logger = prepare_experiment(cfg, sweep_dir=args.sweep_dir, wandb_project=args.wandb_project)
    cfg["exp_dir"] = exp_dir
    train_and_eval(cfg, logger)
    logger.finish()


if __name__ == "__main__":
    main()
