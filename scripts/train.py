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
from plot_before_and_after import plot_kernels
from visualize import visualize_main
from dotenv import load_dotenv

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
    config["lr_lambda"] = float(config["lr"]) * float(config["lambda_reg"])

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
    if dataset=="mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=dataset, 
                                                batch_size=batch_size, 
                                                train_ratio=1-test_ratio)
    else: # only kaggle dataset needs deeper path and resize
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=dataset, 
                                                resize=resize,
                                                deeper_path=deeper_path,
                                                batch_size=batch_size, 
                                                train_ratio=1-test_ratio)
    train_loader, val_loader = split_train_val(
                                train_loader.dataset,
                                train_ratio=train_ratio,
                                batch_size=batch_size,
                                train_val_ratio=train_val_ratio)
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    efficient = config["efficient"]
    share_channels = config.get("share_channels", False)
    model = dense(max_scale, nb_orients, image_shape,
                wavelet=wavelet, nb_class=nb_class, efficient=efficient, share_channels=share_channels).to(device)
    
    # Train classifier
    lr = float(config["lr"])
    lambda_reg = float(config["lambda_reg"])
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam([
        {"params": model.linear.parameters(), "lr": lr},   # different lr
        {"params": model.fine_tuned_params(), "lr": lr * 0.01}   # 
    ])

    base_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.fine_tuned_params()]
    #
    logger.log("Training linear classifier...") 
    model.train_classifier()
    for classifier_epoch in range(classifier_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, base_loss, device)
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
        logger.log(f"Epoch {classifier_epoch+1}: Train_Acc={train_metrics['accuracy']:.4f} Val_Acc={val_acc:.4f} Base_Loss={train_metrics['base_loss']:.4e} Reg_Loss={train_metrics['reg_loss']:.4e} Total_Loss={train_metrics['total_loss']:.4e}", data=True)
    logger.log("Finish linear layer training task.")
    ini_test_loss, ini_test_acc = evaluate(model, test_loader, base_loss, device)


    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Save model to {save_original}")
    #
    logger.log("Fine tuning conv layers...")
    model.train_conv()
    for conv_epoch in range(conv_epochs):  # Change number of epochs as needed
        train_metrics = train_one_epoch(model, train_loader, optimizer, base_loss, device, original_params, lambda_reg)
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
        logger.log(f"Epoch={conv_epoch + classifier_epoch + 1} Train_Acc={train_metrics['accuracy']:.4f} Base_Loss={train_metrics['base_loss']:.4e} Val_Acc={val_acc:.4f} Val_Loss={val_loss:.4f} Reg_Loss={train_metrics['reg_loss']:.4e} Total_Loss={train_metrics['total_loss']:.4e}", data=True)
    test_loss, test_acc = evaluate(model, test_loader, base_loss, device)
    logger.log(f"Finish conv fine tuning task.")
    logger.log(f"Test_Acc={test_acc:.4f} Ini_Test_Acc={ini_test_acc:.4f} Train_Ratio={train_ratio:.4f}"
    f" Lambda={lambda_reg:.4f} Out_dim={model.out_dim}", data=True)
    #
    save_fine_tuned = os.path.join(exp_dir, "trained.pt")
    torch.save(model.state_dict(), save_fine_tuned)
    logger.log(f"Save model to {save_fine_tuned}")

    # back up config
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["last_train_acc"] = train_metrics['accuracy']
    config["last_val_acc"] = val_acc
    config["test_acc"] = test_acc
    config["random"] = False
    save_config(exp_dir, config)

    # Plot kernels before and after training
    logger.log("Plotting kernels before and after training...")
    plot_kernels(exp_dir)

    # Visualize filters(sampled first few for each layer) and activations
    logger.log("Visualizing filters and activations...")
    visualize_main(exp_dir)

    logger.finish()
    
if __name__ == "__main__":
    main()