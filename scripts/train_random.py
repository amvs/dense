import argparse
from configs import load_config, save_config, apply_overrides
import os
from datetime import datetime
import torch
import torch.nn as nn
from training.datasets import get_loaders, split_train_val
from training import train_one_epoch, evaluate
from dense import dense, ScatterParams
from dense.helpers import LoggerManager
from plot_before_and_after import plot_kernels
from visualize import visualize_main
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
                            config=config, name=f"rand_{train_ratio=}(run-{timestamp})")
    
    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")    

    # get data loader
    dataset = config["dataset"]
    batch_size = config["batch_size"]
    test_ratio = config["test_ratio"]
    train_ratio = config["train_ratio"]
    train_val_ratio = config["train_val_ratio"]
    seed = config["seed"]
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
                                train_val_ratio=train_val_ratio,
                                seed=seed)
    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    out_size = config["out_size"]
    n_copies = config["n_copies"]
    pca_dim  = config["pca_dim"]
    depth = config["depth"]
    share_channels = config.get("share_channels", False)
    params = ScatterParams(
        n_scale=max_scale,
        n_orient=nb_orients,
        in_channels=image_shape[0],
        n_copies=n_copies,
        wavelet=wavelet,
        n_class=nb_class,
        share_channels=share_channels,
        in_size=image_shape[1],
        out_size=out_size,
        depth=depth,
        pca_dim=pca_dim,
        random=True
    )
    model = dense(params).to(device)
    # model = dense(max_scale, nb_orients, image_shape,
    #             wavelet=wavelet, nb_class=nb_class, 
    #             efficient=efficient, share_channels=share_channels,
    #             random=True).to(device)
    
    # Train classifier
    lr = float(config["lr"])
    weight_decays = float(config["weight_decays"])
    classifier_epochs = config["classifier_epochs"]
    conv_epochs = config["conv_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decays)

    base_loss = nn.CrossEntropyLoss()

    # Save initialized model
    save_original = os.path.join(exp_dir, "origin.pt")
    torch.save(model.state_dict(), save_original)
    logger.log(f"Save initialized model to {save_original}")
    with torch.no_grad():
        original_params = [p.clone().detach() for p in model.fine_tuned_params()]
    # Evaluate initial test accuracy before training
    ini_test_loss, ini_test_acc = evaluate(model, test_loader, base_loss, device)
    logger.log(f"Initial test accuracy (before training): {ini_test_acc:.4f}")
    
    #
    logger.log("Training a model from random initialization...") 
    ##
    # Configuration
    patience = config.get("conv_patience", 5)  # number of epochs to wait without improvement
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_state = None
    counter = 0

    n_tuned_params = model.n_tuned_params()
    logger.log(f"n_tuned_params={n_tuned_params} n_class_params={model.out_dim*model.n_class}", data=True)

    model.full_train()  # training mode

    for epoch in range(classifier_epochs + conv_epochs):
        # Train one epoch
        train_metrics = train_one_epoch(model, train_loader, optimizer, base_loss, device)
        train_loss = train_metrics['total_loss']
        train_acc = train_metrics['accuracy']

        # Evaluate on validation set
        val_loss, val_acc = evaluate(model, val_loader, base_loss, device)

        logger.log(f"Epoch={epoch+1} Train_Acc={train_acc:.4f} Train_Loss={train_loss:.4f} Val_Acc={val_acc:.4f} Val_Loss={val_loss:.4f}", data=True)
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_train_loss = train_loss
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_acc = train_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0  # reset patience counter
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    dist_sq = 0.0
    for p, p0 in zip(model.fine_tuned_params(), original_params):
        if p.requires_grad:
            diff = p - p0
            dist_sq += torch.sum(torch.abs(diff) ** 2)

    dist = torch.sqrt(dist_sq)
    ##
    # model.full_train()
    # for epoch in range(classifier_epochs + conv_epochs):  # Change number of epochs as needed
    #     train_metrics = train_one_epoch(model, train_loader, optimizer, base_loss, device)
    #     train_loss = train_metrics['total_loss']
    #     train_acc = train_metrics['accuracy']
    #     val_loss, val_acc = evaluate(model, val_loader, base_loss, device)
    #     logger.log(f"Epoch={epoch} Train_Acc={train_acc:.4f} Train_Loss={train_loss:.4f} Val_Acc={val_acc:.4f} Val_Loss={val_loss:.4f}", data=True)
    logger.log("Finish training task.")

    #
    test_loss, test_acc = evaluate(model, test_loader, base_loss, device)
    
    logger.log(f"Finish testing task.")
    logger.log(f"Test_Acc={test_acc:.4f} Ini_Test_Acc={ini_test_acc:.4f} Train_Ratio={train_ratio:.4f} "
    f"Test_Loss={test_loss:.4f} Best_Train_Loss={best_train_loss:.4f} Best_Val_Loss={best_val_loss:.4f} "
    f"weight_decays={weight_decays:.5f} Out_dim={model.out_dim} dist={dist}", data=True)
    #
    save_fine_tuned = os.path.join(exp_dir, "trained.pt")
    torch.save(model.state_dict(), save_fine_tuned)
    logger.log(f"Save trained model to {save_fine_tuned}")

    # back up config
    config["nb_class"] = nb_class
    config["image_shape"] = list(image_shape)
    config["best_train_acc"] = best_train_acc
    config["best_val_acc"] = best_val_acc
    config["test_acc"] = test_acc
    config["test_loss"] = test_loss
    config["best_train_loss"] = best_train_loss
    config["best_val_loss"] = best_val_loss

    config["n_tuned_params"] = n_tuned_params
    config["n_linear_params"] = model.out_dim * model.n_class
    config["random"] = True
    save_config(exp_dir, config)

    # Plot kernels before and after training
    # logger.log("Plotting kernels before and after training...")
    # plot_kernels(exp_dir)

    # # Visualize filters(sampled first few for each layer) and activations
    # logger.log("Visualizing filters and activations...")
    # visualize_main(exp_dir)

    logger.finish()
    
if __name__ == "__main__":
    main()