import argparse
from configs import load_config, save_config
import os
from datetime import datetime
import torch
import torch.nn as nn
from training.datasets import get_loaders
from training import train_one_epoch, evaluate
from dense import dense
from dense.helpers import LoggerManager
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist.yaml)"
    )
    return parser.parse_args()

def main():
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("experiments", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    logger.info("Start log:")

    # Read config
    args = parse_args()
    config = load_config(args.config)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")    

    # get data loader
    train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=config["dataset"], 
                                            batch_size=config["batch_size"], 
                                            train_ratio=config["train_ratio"])
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    efficient = config["efficient"]
    model = dense(max_scale, nb_orients, image_shape,
                wavelet=wavelet, nb_class=nb_class, efficient=efficient).to(device)
    
    # Train classifier
    lr = float(config["lr"])
    weight_decay = float(config["weight_decay"])
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
        test_loss, test_acc = evaluate(model, test_loader, base_loss, device)
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    logger.info("Finish linear layer training task.")

    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.info(f"Save model to {save_original}")
    #
    logger.info("Fine tuning conv layers...")
    model.train_conv()
    for epoch in range(conv_epochs):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, base_loss, device, original_params)
        test_loss, test_acc = evaluate(model, test_loader, base_loss, device)
        logger.info(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    logger.info("Finish conv fine tuning task.")
    #
    save_fine_tuned = os.path.join(exp_dir, "fine_tuned.pt")
    torch.save(model.state_dict(), save_fine_tuned)
    logger.info(f"Save model to {save_fine_tuned}")

    # back up config
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    save_config(exp_dir, config)
    
if __name__ == "__main__":
    main()