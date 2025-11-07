import argparse
from configs import load_config, save_config, apply_overrides
import os
from datetime import datetime
import torch
import torch.nn as nn
from training.datasets import get_loaders, split_train_val
from training import train_one_epoch, evaluate
from dense import dense
from dense.helpers import LoggerManager
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
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
    val_ratio = config["val_ratio"]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.sweep_dir is not None: # if this is a sweep job, save to the sweep dir
        if not os.path.exists(args.sweep_dir):
            raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")
        exp_dir = os.path.join(args.sweep_dir, f"{val_ratio=}", f"run-{timestamp}")
    else:
        exp_dir = os.path.join("experiments", f"{val_ratio=}", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    logger.info("===== Start log =====")
    
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")    

    # get data loader
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    if dataset=="mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=dataset, 
                                                batch_size=batch_size, 
                                                train_ratio=train_ratio)
    else: # only kaggle dataset needs deeper path and resize
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=dataset, 
                                                resize=resize,
                                                deeper_path=deeper_path,
                                                batch_size=batch_size, 
                                                train_ratio=train_ratio)
    train_loader, val_loader = split_train_val(
                                train_loader.dataset,
                                val_ratio=val_ratio,
                                batch_size=batch_size)
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    efficient = config["efficient"]
    isShared = config.get("isShared", False)
    model = dense(max_scale, nb_orients, image_shape,
                wavelet=wavelet, nb_class=nb_class, efficient=efficient, isShared=isShared).to(device)
    
    # Train classifier
    lr = float(config["lr"])
    lambda_reg = float(config["lambda_reg"])
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam([
        {"params": model.linear.parameters(), "lr": lr},   # different lr
        {"params": model.sequential_conv.parameters(), "lr": lr * 0.01}   # 
    ])

    base_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.parameters()]
    #
    logger.info("Training linear classifier...") 
    model.train_classifier()
    for epoch in range(classifier_epochs):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, base_loss, device)
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    logger.info("Finish linear layer training task.")

    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.info(f"Save model to {save_original}")
    #
    logger.info("Fine tuning conv layers...")
    model.train_conv()
    for epoch in range(conv_epochs):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, base_loss, device, original_params, lambda_reg)
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
    test_loss, test_acc = evaluate(model, test_loader, base_loss, device)
    logger.info(f"Finish conv fine tuning task. Test Acc={test_acc:.4f}")
    #
    save_fine_tuned = os.path.join(exp_dir, "fine_tuned.pt")
    torch.save(model.state_dict(), save_fine_tuned)
    logger.info(f"Save model to {save_fine_tuned}")

    # back up config
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["last_train_acc"] = train_acc
    config["last_val_acc"] = val_acc
    config["test_acc"] = test_acc
    save_config(exp_dir, config)
    
if __name__ == "__main__":
    main()