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
    # Read config
    args = parse_args()
    config = load_config(args.config)

    # Create output folder and save config
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = os.path.join("experiments", f"run-{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    save_config(exp_dir, config)

    # init logger
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    logger.info("Start log:")
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
    lr = config["lr"]
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # 
    model.train_classifier()
    for epoch in range(classifier_epochs):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(f"Classifier Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    #
    model.train_conv()
    for epoch in range(conv_epochs):  # Change number of epochs as needed
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        logger.info(f"Conv Epoch {epoch+1}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

if __name__ == "__main__":
    main()