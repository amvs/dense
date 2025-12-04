# Visualization
# Read an experiment result model and original scattering model
# 1. visualize the convolutional filters before and after training
# sampled the first few filters for each layer and plot their real and imaginary parts
# 2. visualize the intermediate activations before and after training
# sampled a random image from the training set, run through both models, and plot the activ
# https://chatgpt.com/share/68bd8d75-ac88-8008-bd56-8b7030de4981
import os
from configs import load_config
from datetime import datetime
from dense.helpers import LoggerManager
from dense import dense
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T
from training.datasets import get_loaders



def visualize_main(exp_dir):
    # create outpt folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    analyze_dir = os.path.join(exp_dir, f"analyze-{timestamp}")
    os.makedirs(analyze_dir, exist_ok=True)

    # init logger
    logger = LoggerManager.get_logger(log_dir=analyze_dir)

    # load config
    config_path = os.path.join(exp_dir, "config.yaml")
    config = load_config(config_path)

    # set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}") 

    # init model
    max_scale = config["max_scale"]
    nb_orients = config["nb_orients"]
    wavelet = config["wavelet"]
    efficient = config["efficient"]
    nb_class = config["nb_class"]
    image_shape = config["image_shape"]
    random = config["random"]
    share_channels = config["share_channels"]
    origin_path = os.path.join(exp_dir, "origin.pt")
    tuned_path = os.path.join(exp_dir, "trained.pt")
    if not os.path.exists(origin_path) or not os.path.exists(tuned_path):
            raise FileNotFoundError(f"At least one model not found.")
    origin_model = dense(max_scale, nb_orients, image_shape,
                wavelet=wavelet, nb_class=nb_class, efficient=efficient, random=random, share_channels=share_channels).to(device)
    tuned_model = dense(max_scale, nb_orients, image_shape,
                wavelet=wavelet, nb_class=nb_class, efficient=efficient, random=random, share_channels=share_channels).to(device)

    # load model weights
    logger.log("Loading model weights...")
    origin_state = torch.load(origin_path, map_location=device, weights_only=True)
    tuned_state = torch.load(tuned_path, map_location=device, weights_only=True)

    origin_model.load_state_dict(origin_state)
    tuned_model.load_state_dict(tuned_state)

    origin_model.eval()
    tuned_model.eval()
    logger.log("Both origin and tuned models loaded and set to eval mode.")
    

    # 
    logger.log("Comparing convolution filters...")
    def compare_filters(origin_model, tuned_model, max_nb_filters=2):
        diffs = []
        save_dir = os.path.join(analyze_dir, "filters")
        os.makedirs(save_dir, exist_ok=True)

        for idx, (layer_o, layer_t) in enumerate(
            zip(origin_model.sequential_conv, tuned_model.sequential_conv)
        ):
            if not isinstance(layer_o, torch.nn.Conv2d):
                continue

            w_o = layer_o.weight.detach().cpu()  # complex tensor [out,in,h,w]
            w_t = layer_t.weight.detach().cpu()

            # flatten to vector for metrics
            v_o = w_o.flatten()
            v_t = w_t.flatten()

            diff = torch.norm(v_t - v_o, p=1).item() # l1 norm
            # complex cosine similarity: <a,b>/||a|| ||b||
            cos = torch.real(torch.vdot(v_o, v_t) / (torch.norm(v_o)*torch.norm(v_t) + 1e-12)).item()
            diffs.append((idx, diff, cos))

            # visualize magnitude and phase of a few filters
            num_show = min(max_nb_filters, w_o.shape[0])
            fig, axes = plt.subplots(2, num_show*2, figsize=(3*num_show*2, 6))
            for i in range(num_show):
                f_o = w_o[i,0].numpy()
                f_t = w_t[i,0].numpy()
                # magnitude
                axes[0, 2*i].imshow(f_o.real, cmap="viridis"); axes[0, 2*i].set_title(f"I Real f{i}")
                axes[0, 2*i+1].imshow(f_o.imag, cmap="viridis"); axes[0, 2*i+1].set_title(f"I Imag f{i}")
                axes[1, 2*i].imshow(f_t.real, cmap="viridis"); axes[1, 2*i].set_title(f"T Real f{i}")
                axes[1, 2*i+1].imshow(f_t.imag, cmap="viridis"); axes[1, 2*i+1].set_title(f"T Imag f{i}")
                for ax in axes[:, 2*i:2*i+2]:
                    for a in ax: a.axis("off")
            plt.tight_layout()
            fig_path = os.path.join(save_dir, f"conv{idx}_complex_filters.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            logger.log(f"Saved complex filter comparison for conv{idx} -> {fig_path}")
            logger.send_file(f"Sample filters for Initialized(I), Trained(T) model at layer{idx:02d}", fig_path, "image")

        return diffs

    filter_diffs = compare_filters(origin_model, tuned_model)
    # logger.log("Filter differences summary:")
    # for idx, diff, cos in filter_diffs:
    #     logger.log(f"Conv{idx}: L2 diff={diff:.4f}, CosSim={cos:.4f}")

    #
    logger.log("Comparing intermediate activations...")
    # load test image
    logger.log("Sampling a random image from train_loader...")
    # train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=config["dataset"], 
    #                                             batch_size=config["batch_size"], 
    #                                             train_ratio=config["train_ratio"])
    if config["dataset"]=="mnist":
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=config["dataset"], 
                                                batch_size=config["batch_size"], 
                                                train_ratio=1-config["test_ratio"])
    else: # only kaggle dataset needs deeper path and resize
        resize = config["resize"]
        deeper_path = config["deeper_path"]
        train_loader, test_loader, nb_class, image_shape = get_loaders(dataset=config["dataset"], 
                                                resize=resize,
                                                deeper_path=deeper_path,
                                                batch_size=config["batch_size"], 
                                                train_ratio=1-config["test_ratio"])
    images, labels = next(iter(train_loader))
    # pick first image
    img_tensor = images[0:1].to(device)   # keep batch dimension [1,C,H,W]

    to_pil = T.ToPILImage()
    img_pil = to_pil(img_tensor.squeeze(0).cpu())
    png_path = os.path.join(analyze_dir, "test_img.png")
    img_pil.save(png_path)
    logger.log(f"Saved test image as PNG -> {png_path}")
    logger.send_file("test_image", png_path, "image")
    def run_modules(model, x):
        """Run input through module_list step by step and collect outputs."""
        inputs = [x]
        activations = []
        for idx, module in enumerate(model.module_list):
            result = module(*inputs)
            activations.append(result.detach().cpu())
            inputs.append(result)
            inputs = [model.pooling(inp) for inp in inputs]
        return activations

    # run both models
    acts_origin = run_modules(origin_model, img_tensor)
    acts_tuned = run_modules(tuned_model, img_tensor)

    # visualize a few activations per module
    def visualize_activations(acts_origin, acts_tuned, max_modules=3, max_maps=3):
        save_dir = os.path.join(analyze_dir, "activations")
        os.makedirs(save_dir, exist_ok=True)
        for m_idx, (a_o, a_t) in enumerate(zip(acts_origin, acts_tuned)):
            if m_idx >= max_modules:  
                break
            a_o = a_o[0]  # [C,H,W]
            a_t = a_t[0]
            num_maps = min(max_maps, a_o.shape[0])

            fig, axes = plt.subplots(2, num_maps, figsize=(3*num_maps, 6))
            for i in range(num_maps):
                f_o = a_o[i].numpy()
                f_t = a_t[i].numpy()
                axes[0, i].imshow(f_o, cmap="viridis"); axes[0, i].set_title(f"I F{i}")
                axes[1, i].imshow(f_t, cmap="viridis"); axes[1, i].set_title(f"T F{i}")
                for ax in axes[:, i:i+1]:
                    for a in ax: a.axis("on")
            plt.tight_layout()
            fig_path = os.path.join(save_dir, f"module{m_idx}_acts.png")
            plt.savefig(fig_path, dpi=150)
            plt.close()
            logger.log(f"Saved activation comparison for module{m_idx} -> {fig_path}")
            logger.send_file(f"Visualize Activation(conv+abs) at Layer{m_idx:02d} for Initialized(I), Trained(T) model", fig_path, "image")

    # save results
    visualize_activations(acts_origin, acts_tuned)

if __name__ == "__main__":
    exp_dir = input("Enter exp_dir to analyze: ")
    visualize_main(exp_dir)