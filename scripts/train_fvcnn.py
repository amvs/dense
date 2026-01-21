import argparse
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
from sklearn.svm import LinearSVC
import gc

from fv_cnn import FisherVectorEncoder, MultiScaleCNNExtractor, FCFeatureExtractor
from training.datasets.outex import OutexDataset
from training.datasets.kaggle import KaggleDataset
from training.datasets.kthtips2b import KHTTips2bDataset, CenterCropToSquare
from training.datasets.base import stratify_split
from torchvision.datasets import MNIST
from dense.helpers import LoggerManager
from training.experiment_utils import log_model_parameters
from configs import save_config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(resize: int, center_crop: bool = False) -> transforms.Compose:
    t: List[Any] = []
    if center_crop:
        t.append(CenterCropToSquare())
    t.append(transforms.Resize((resize, resize)))
    t.append(transforms.ToTensor())
    # no normalization here; extractor handles ImageNet normalization
    return transforms.Compose(t)


def build_outex_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    root_dir = cfg["root_dir"]
    resize = cfg.get("resize", 128)
    batch_size = cfg.get("batch_size", 32)
    problem_id = cfg.get("problem_id", "000")
    train_val_ratio = cfg.get("train_val_ratio", 2)
    train_ratio = cfg.get("train_ratio", 1.0)
    drop_last = cfg.get("drop_last", False)

    transform = build_transforms(resize)
    train_dataset = OutexDataset(root_dir, problem_id=problem_id, split="train", transform=transform)
    test_dataset = OutexDataset(root_dir, problem_id=problem_id, split="test", transform=transform)

    if train_ratio < 1.0:
        reduced_len = int(len(train_dataset) * train_ratio)
        train_dataset, _ = stratify_split(train_dataset, train_size=reduced_len, seed=42)

    total_test_len = len(test_dataset)
    val_len = total_test_len // (train_val_ratio + 1)
    test_len = total_test_len - val_len
    test_dataset, val_dataset = stratify_split(test_dataset, train_size=test_len, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    nb_class = len(test_dataset.classes)
    return train_loader, val_loader, test_loader, nb_class


def build_curet_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset_name = cfg["dataset"]
    deeper_path = cfg.get("deeper_path", "")
    resize = cfg.get("resize", 200)
    batch_size = cfg.get("batch_size", 32)
    train_ratio = cfg.get("train_ratio", 0.8)
    drop_last = cfg.get("drop_last", False)

    from training.datasets.kaggle import get_kaggle_dataset
    root = get_kaggle_dataset(dataset_name)
    transform = build_transforms(resize)
    base_dataset = KaggleDataset(root, deeper_path=deeper_path, transform=transform)
    total_len = len(base_dataset)
    train_len = int(total_len * train_ratio)
    test_len = total_len - train_len
    train_dataset, test_dataset = stratify_split(base_dataset, train_size=train_len, seed=42)
    # create val from test split to mimic other loaders
    val_len = max(1, test_len // 5)
    test_len_final = test_len - val_len
    test_dataset, val_dataset = stratify_split(test_dataset, train_size=test_len_final, seed=43)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    nb_class = len(base_dataset.classes)
    return train_loader, val_loader, test_loader, nb_class


def build_kth_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    root_dir = cfg["root_dir"]
    resize = cfg.get("resize", 200)
    batch_size = cfg.get("batch_size", 16)
    fold = cfg.get("fold", 0)
    train_ratio = cfg.get("train_ratio", 1.0)
    drop_last = cfg.get("drop_last", False)

    transform = build_transforms(resize, center_crop=True)
    # prepare datasets according to fold scheme
    samples = ['sample_a', 'sample_b', 'sample_c', 'sample_d']
    test_sample = samples[fold]
    val_sample = samples[(fold + 1) % 4]
    train_samples = [samples[(fold + 2) % 4], samples[(fold + 3) % 4]]

    train_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=train_samples)
    val_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=[val_sample])
    test_dataset = KHTTips2bDataset(root_dir, transform=transform, sample_filter=[test_sample])

    if train_ratio < 1.0:
        reduced_len = int(len(train_dataset) * train_ratio)
        train_dataset, _ = stratify_split(train_dataset, train_size=reduced_len, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    nb_class = len(train_dataset.classes)
    return train_loader, val_loader, test_loader, nb_class


def build_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = cfg["dataset"].lower()
    if "outex" in dataset:
        return build_outex_loaders(cfg)
    if "curet" in dataset:
        return build_curet_loaders(cfg)
    if "kth" in dataset:
        return build_kth_loaders(cfg)
    if "mnist" in dataset:
        return build_mnist_loaders(cfg)
    raise ValueError(f"Unsupported dataset {dataset}")


def build_mnist_loaders(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    root_dir = cfg["root_dir"]
    resize = cfg.get("resize", 28)
    batch_size = cfg.get("batch_size", 16)
    train_ratio = cfg.get("train_ratio", 0.8)
    drop_last = cfg.get("drop_last", False)

    transform = build_transforms(resize)
    train_dataset = MNIST(root_dir, train=True, download=True, transform=transform)
    test_dataset = MNIST(root_dir, train=False, download=True, transform=transform)

    if train_ratio < 1.0:
        reduced_len = int(len(train_dataset) * train_ratio)
        train_dataset, _ = stratify_split(train_dataset, train_size=reduced_len, seed=42)

    val_len = max(1, len(test_dataset) // 5)
    test_len = len(test_dataset) - val_len
    test_dataset, val_dataset = stratify_split(test_dataset, train_size=test_len, seed=42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    nb_class = 10  # MNIST has 10 classes
    return train_loader, val_loader, test_loader, nb_class



def collect_descriptors(train_loader: DataLoader, extractor: MultiScaleCNNExtractor, max_samples: int) -> np.ndarray:
    collected: List[np.ndarray] = []
    total = 0
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
        for images, _ in train_loader:
            images = images.to(device)
            for i in range(images.size(0)):
                img = images[i]
                feats = extractor.extract(img)
                for d in feats["descriptors"]:
                    arr = d.detach().cpu().numpy()
                    collected.append(arr)
                    total += arr.shape[0]
                    if total >= max_samples:
                        return np.concatenate(collected, axis=0)[:max_samples]
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


def train_and_eval(cfg: Dict[str, Any], logger) -> None:
    set_seed(cfg.get("seed", 42))
    framework = cfg.get("framework", "fvcnn")
    backbone = cfg.get("backbone", "vgg16")
    feature_layer = cfg.get("feature_layer", "conv5_3")

    logger.log(f"Framework={framework} backbone={backbone} feature_layer={feature_layer}")
    train_loader, val_loader, test_loader, nb_class = build_loaders(cfg)
    logger.log(f"Dataset has {nb_class} classes")

    if framework == "fvcnn":
        extractor = MultiScaleCNNExtractor(
            backbone=backbone,
            feature_layer=feature_layer,
            scales=cfg.get("scales"),
            min_edge=cfg.get("min_edge", 30),
            max_sqrt_hw=cfg.get("max_sqrt_hw", 1024),
        )
        encoder = FisherVectorEncoder(
            num_components=cfg.get("gmm_components", 64),
            pca_dim=cfg.get("pca_dim"),
            signed_sqrt_postprocess=cfg.get("signed_sqrt", True),
            l2_postprocess=cfg.get("l2_normalize", True),
            random_state=cfg.get("seed", 42),
        )
        max_samples = cfg.get("gmm_samples", 100000)
        logger.log(f"Collecting up to {max_samples} descriptors for GMM fitting...")
        desc = collect_descriptors(train_loader, extractor, max_samples=max_samples)
        encoder.fit(desc)
        logger.log("GMM fitted")
        train_codes, train_labels = encode_split(train_loader, extractor, encoder)
        logger.log(f"Encoded training set: {train_codes.shape}")
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
    clf = LinearSVC(C=C)
    clf.fit(train_codes, train_labels)
    logger.log("Done fitting SVM classifier, computing scores")
    train_acc = clf.score(train_codes, train_labels)
    # Free training encodings
    del train_codes, train_labels
    gc.collect()

    # Validation encoding and eval
    val_codes, val_labels = encode_split(val_loader, extractor, encoder if framework == "fvcnn" else None)
    logger.log(f"Encoded validation set: {val_codes.shape}")
    val_acc = clf.score(val_codes, val_labels)
    del val_codes, val_labels
    gc.collect()

    # Test encoding and eval
    test_codes, test_labels = encode_split(test_loader, extractor, encoder if framework == "fvcnn" else None)
    logger.log(f"Encoded test set: {test_codes.shape}")
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
        action="append",
        default=[],
        help="Override config entries with key=value (supports dotted keys)",
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
