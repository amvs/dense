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
from torchvision import transforms
from training.datasets import get_loaders
from colorsys import hls_to_rgb


def colorize(z):
    """Colorize a complex-valued 2D array as an RGB image."""
    z = np.asarray(z)
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)
    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0/(1.0 + np.abs(z[idx])**0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


def run_modules(model, x):
    """Run input through module_list step by step and collect outputs (for dense model)."""
    inputs = [x]
    activations = []
    for idx, module in enumerate(model.module_list):
        result = module(*inputs)
        activations.append(result.detach().cpu())
        inputs.append(result)
        inputs = [model.pooling(inp) for inp in inputs]
    return activations

def calc_wph_activations(model, x, flatten=False, vmap_chunk_size=None):
    """
    Collect intermediate activations from WPHClassifier/WPHModel.
    Returns a dict of activations for each submodule.
    """
    from wph.wph_model import WPHModel, WPHModelDownsample
    acts = {}
    # If model is WPHClassifier, get feature_extractor
    if hasattr(model, 'feature_extractor'):
        fe = model.feature_extractor
    else:
        fe = model
    # Wavelet convolution
    nb = x.shape[0]
    xpsi = fe.wave_conv(x)
    if type(fe) is WPHModel:
        acts['wave_conv'] = xpsi.detach().cpu()
        xrelu = fe.relu_center(xpsi)
        acts['relu_center'] = xrelu.detach().cpu()
        xcorr = fe.corr(xrelu.view(nb, fe.num_channels * fe.J * fe.L * fe.A, fe.M, fe.N), flatten=flatten, vmap_chunk_size=vmap_chunk_size)
        acts['corr'] = xcorr.detach().cpu()
    elif type(fe) is WPHModelDownsample and fe.share_scale_pairs:
        # xpsi, xrelu, xcorr are lists of lists of tensors; flatten into a single list
        acts['wave_conv'] = [t.detach().cpu() for t in xpsi]
        xrelu = fe.relu_center(xpsi)
        acts['relu_center'] = [t.detach().cpu() for t in xrelu]
        xcorr = fe.corr(xrelu, flatten=flatten, vmap_chunk_size=vmap_chunk_size)
        acts['corr'] = [t.detach().cpu() for t in xcorr]
    elif type(fe) is WPHModelDownsample and not fe.share_scale_pairs:
        # xpsi, xrelu, xcorr are lists of lists of tensors; flatten into a single list
        acts['wave_conv'] = [t.detach().cpu() for inner in xpsi for t in inner]
        xrelu = fe.relu_center(xpsi)
        acts['relu_center'] = [t.detach().cpu() for inner in xrelu for t in inner]
        xcorr = fe.corr(xrelu, flatten=flatten, vmap_chunk_size=vmap_chunk_size)
        acts['corr'] = [t.detach().cpu() for t in xcorr]
    
    # FFT
    hatx_c = torch.fft.fft2(x)
    # Lowpass
    xlow = fe.lowpass(hatx_c)
    acts['lowpass'] = xlow.detach().cpu()
    # Highpass
    xhigh = fe.highpass(hatx_c)
    acts['highpass'] = xhigh.detach().cpu()
    return acts


def compare_filters(origin_model, tuned_model, analyze_dir, max_nb_filters=2):
    logger = LoggerManager.get_logger()
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



# visualize a few activations per module

def visualize_activations(acts_origin, acts_tuned, analyze_dir, max_modules=3, max_maps=3):
    """Visualize activations for dense model (list format)."""
    logger = LoggerManager.get_logger()
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

def plot_wave_conv_color(a_o, a_t, key, save_dir, max_maps):
    logger = LoggerManager.get_logger()
    nc, J, L, A, M, N = a_o.shape
    max_J = min(J, max_maps)
    max_L = min(L, max_maps)
    max_A = min(A, max_maps)
    for j in range(max_J):
        fig, axes = plt.subplots(max_A*2, max_L*2, figsize=(3*max_L*2, 3*max_A*2))
        for a in range(max_A):
            for l in range(max_L):
                img_o = a_o[:, j, l, a].cpu().numpy()
                img_t = a_t[:, j, l, a].cpu().numpy()
                axes[2*a, 2*l].imshow(np.transpose(img_o.real, (1, 2, 0)))
                axes[2*a, 2*l].set_title(f"Orig Real J{j} L{l} A{a}")
                axes[2*a, 2*l+1].imshow(np.transpose(img_o.imag, (1, 2, 0)))
                axes[2*a, 2*l+1].set_title(f"Orig Imag J{j} L{l} A{a}")
                axes[2*a+1, 2*l].imshow(np.transpose(img_t.real, (1, 2, 0)))
                axes[2*a+1, 2*l].set_title(f"Train Real J{j} L{l} A{a}")
                axes[2*a+1, 2*l+1].imshow(np.transpose(img_t.imag, (1, 2, 0)))
                axes[2*a+1, 2*l+1].set_title(f"Train Imag J{j} L{l} A{a}")
                for ax in axes[2*a:2*a+2, 2*l:2*l+2].flatten():
                    ax.axis("off")
        plt.tight_layout()
        fname = f"{key}_J{j}_grid.png"
        fig_path = os.path.join(save_dir, fname)
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.log(f"Saved activation grid for {key} J{j} -> {fig_path}")
        logger.send_file(f"Visualize Activation {key} J{j} grid for Initialized(I), Trained(T) model", fig_path, "image")


def plot_wave_conv_grayscale(a_o, a_t, key, save_dir, max_maps):
    logger= LoggerManager.get_logger()
    nc, J, L, A, M, N = a_o.shape
    max_J = min(J, max_maps)
    max_L = min(L, max_maps)
    max_A = min(A, max_maps)
    nchan = min(nc, max_maps)
    for j in range(max_J):
        fig, axes = plt.subplots(max_A*2, max_L*nchan, figsize=(3*max_L*nchan, 3*max_A*2))
        for a in range(max_A):
            for l in range(max_L):
                for c in range(nchan):
                    img_o = a_o[c, j, l, a].cpu().numpy()
                    img_t = a_t[c, j, l, a].cpu().numpy()
                    if np.iscomplexobj(img_o):
                        img_o = colorize(img_o)
                    if np.iscomplexobj(img_t):
                        img_t = colorize(img_t)
                    axes[2*a, l*nchan+c].imshow(img_o, cmap=None if img_o.ndim == 3 else "viridis")
                    axes[2*a, l*nchan+c].set_title(f"Orig C{c} J{j} L{l} A{a}")
                    axes[2*a+1, l*nchan+c].imshow(img_t, cmap=None if img_t.ndim == 3 else "viridis")
                    axes[2*a+1, l*nchan+c].set_title(f"Train C{c} J{j} L{l} A{a}")
                    axes[2*a, l*nchan+c].axis("off")
                    axes[2*a+1, l*nchan+c].axis("off")
        plt.tight_layout()
        fname = f"{key}_J{j}_grid.png"
        fig_path = os.path.join(save_dir, fname)
        plt.savefig(fig_path, dpi=150)
        plt.close()
        logger.log(f"Saved activation grid for {key} J{j} -> {fig_path}")
        logger.send_file(f"Visualize Activation {key} J{j} grid for Initialized(I), Trained(T) model", fig_path, "image")

def plot_corr_activations(a_o, a_t, key, save_dir, max_maps):
    # a_o, a_t: [nb, npairs, M, N] (use first batch)
    logger= LoggerManager.get_logger()
    a_o = a_o[0]
    a_t = a_t[0]
    npairs = a_o.shape[0]
    nshow = min(npairs, max_maps)
    fig, axes = plt.subplots(2, nshow, figsize=(3*nshow, 6))
    for i in range(nshow):
        img_o = a_o[i].cpu().numpy()
        img_t = a_t[i].cpu().numpy()
        if np.iscomplexobj(img_o):
            img_o = colorize(img_o)
        if np.iscomplexobj(img_t):
            img_t = colorize(img_t)
        axes[0, i].imshow(img_o, cmap=None if img_o.ndim == 3 else "viridis")
        axes[0, i].set_title(f"Orig CorrPair {i}")
        axes[1, i].imshow(img_t, cmap=None if img_t.ndim == 3 else "viridis")
        axes[1, i].set_title(f"Train CorrPair {i}")
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    fname = f"{key}_corr_grid.png"
    fig_path = os.path.join(save_dir, fname)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.log(f"Saved corr activation grid for {key} -> {fig_path}")
    logger.send_file(f"Visualize Corr Activation {key} for Initialized(I), Trained(T) model", fig_path, "image")

def visualize_wph_activations(acts_origin, acts_tuned, analyze_dir, max_maps=3):
    """
    Visualize activations for WPH model (dict format).
    Plots each submodule's activations side by side for origin and tuned models.
    """
    save_dir = os.path.join(analyze_dir, "activations_wph")
    os.makedirs(save_dir, exist_ok=True)

    for key in acts_origin.keys():
        a_o = acts_origin[key]
        a_t = acts_tuned[key]
        if key == 'corr':
            if type(a_o) is list:
                for j in range(len(a_o)):
                    plot_corr_activations(a_o[j], a_t[j], f"{key}_level{j}", save_dir, max_maps*4)
            else:
                plot_corr_activations(a_o, a_t, key, save_dir, max_maps*4)
            continue
        if type(a_o) is list:
            if a_o[0].ndim < 5:
                continue
        else:
            if a_o.ndim < 5:
                continue
        a_o = a_o[0]
        a_t = a_t[0]
        nc = a_o.shape[0]
        if nc == 3:
            if type(a_o) is list:
                for j in range(len(a_o)):
                    plot_wave_conv_color(a_o[j], a_t[j], f"{key}_level{j}", save_dir, max_maps)
            else:
                plot_wave_conv_color(a_o, a_t, key, save_dir, max_maps)
        else:
            if type(a_o) is list:
                for j in range(len(a_o)):
                    plot_wave_conv_grayscale(a_o[j], a_t[j], f"{key}_level{j}", save_dir, max_maps)
            else:
                plot_wave_conv_grayscale(a_o, a_t, key, save_dir, max_maps)


def visualize_main(exp_dir, model_type="dense", origin_filename="origin.pt", tuned_filename="trained.pt", filters=None):
    # create output folder
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


    origin_path = os.path.join(exp_dir, origin_filename)
    tuned_path = os.path.join(exp_dir, tuned_filename)
    if not os.path.exists(origin_path) or not os.path.exists(tuned_path):
        raise FileNotFoundError(f"At least one model not found.")

    logger.log(f"Using model_type: {model_type}")

    if model_type == "dense":
        from dense import dense as DenseModel
        max_scale = config["max_scale"]
        nb_orients = config["nb_orients"]
        wavelet = config["wavelet"]
        efficient = config["efficient"]
        nb_class = config["nb_class"]
        image_shape = config["image_shape"]
        random = config["random"]
        share_channels = config["share_channels"]
        origin_model = DenseModel(max_scale, nb_orients, image_shape,
                    wavelet=wavelet, nb_class=nb_class, efficient=efficient, random=random, share_channels=share_channels).to(device)
        tuned_model = DenseModel(max_scale, nb_orients, image_shape,
                    wavelet=wavelet, nb_class=nb_class, efficient=efficient, random=random, share_channels=share_channels).to(device)
        logger.log("Loading model weights...")
        origin_state = torch.load(origin_path, map_location=device)
        tuned_state = torch.load(tuned_path, map_location=device)
        origin_model.load_state_dict(origin_state)
        tuned_model.load_state_dict(tuned_state)
    elif model_type == "wph":
        max_scale = config["max_scale"]
        nb_orients = config["nb_orients"]
        num_phases = config["num_phases"]
        image_shape = config["image_shape"]
        if config['downsample']:
            from wph.wph_model import WPHModel, WPHClassifier, WPHModelDownsample
            T = filters['psi'].shape[-1]
            origin_fe = WPHModelDownsample(J=max_scale,
                L=nb_orients,
                A=num_phases,
                A_prime=config.get("num_phases_prime", 1),
                M=image_shape[1],
                N=image_shape[2],
                T=T,
                filters=filters,
                num_channels=image_shape[0],
                share_rotations=config["share_rotations"],
                share_phases=config["share_phases"],
                share_channels=config["share_channels"],
                share_scales=config['share_scales'],
                share_scale_pairs=config.get('share_scale_pairs', True),
                normalize_relu=config["normalize_relu"],
                delta_j=config.get("delta_j"),
                delta_l=config.get("delta_l"),
                shift_mode=config["shift_mode"],
                mask_angles=config["mask_angles"],
                mask_union_highpass=config["mask_union_highpass"],
            )
            tuned_fe = WPHModelDownsample(J=max_scale,
                L=nb_orients,
                A=num_phases,
                A_prime=config.get("num_phases_prime", 1),
                M=image_shape[1],
                N=image_shape[2],
                T=T,
                filters=filters,
                num_channels=image_shape[0],
                share_rotations=config["share_rotations"],
                share_phases=config["share_phases"],
                share_channels=config["share_channels"],
                share_scales=config['share_scales'],
                share_scale_pairs=config.get('share_scale_pairs', True),
                normalize_relu=config["normalize_relu"],
                delta_j=config.get("delta_j"),
                delta_l=config.get("delta_l"),
                shift_mode=config["shift_mode"],
                mask_angles=config["mask_angles"],
                mask_union_highpass=config["mask_union_highpass"],
            )
        else:
            origin_fe = WPHModel(
                J=max_scale,
                L=nb_orients,
                A=num_phases,
                A_prime=config.get("num_phases_prime", 1),
                M=image_shape[1],
                N=image_shape[2],
                filters=filters,
                num_channels=image_shape[0],
                share_rotations=config["share_rotations"],
                share_phases=config["share_phases"],
                share_channels=config["share_channels"],
                normalize_relu=config["normalize_relu"],
                delta_j=config.get("delta_j"),
                delta_l=config.get("delta_l"),
                shift_mode=config["shift_mode"],
                mask_union=config["mask_union"],
                mask_angles=config["mask_angles"],
                mask_union_highpass=config["mask_union_highpass"],
            )
            tuned_fe = WPHModel(
            J=max_scale,
            L=nb_orients,
            A=num_phases,
            A_prime=config.get("num_phases_prime", 1),
            M=image_shape[1],
            N=image_shape[2],
            filters=filters,
            num_channels=image_shape[0],
            share_rotations=config["share_rotations"],
            share_phases=config["share_phases"],
            share_channels=config["share_channels"],
            normalize_relu=config["normalize_relu"],
            delta_j=config.get("delta_j"),
            delta_l=config.get("delta_l"),
            shift_mode=config["shift_mode"],
            mask_union=config["mask_union"],
            mask_angles=config["mask_angles"],
            mask_union_highpass=config["mask_union_highpass"],
        )
        origin_model = WPHClassifier(feature_extractor=origin_fe,
                                     num_classes=config["num_classes"],
                                     use_batch_norm=config["use_batch_norm"]).to(device)
        origin_model.feature_extractor.wave_conv.get_full_filters()
        
        tuned_model = WPHClassifier(feature_extractor=tuned_fe,
                                    num_classes=config["num_classes"],
                                    use_batch_norm=config["use_batch_norm"]).to(device)
        tuned_model.feature_extractor.wave_conv.get_full_filters()
        logger.log("Loading model weights...")
        origin_state = torch.load(origin_path, map_location=device, weights_only=True)
        tuned_state = torch.load(tuned_path, map_location=device, weights_only=True)
        origin_model.load_state_dict(origin_state)
        tuned_model.load_state_dict(tuned_state)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    origin_model.eval()
    tuned_model.eval()
    logger.log("Both origin and tuned models loaded and set to eval mode.")

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
    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_tensor.squeeze(0).cpu())
    png_path = os.path.join(analyze_dir, "test_img.png")
    img_pil.save(png_path)
    logger.log(f"Saved test image as PNG -> {png_path}")
    logger.send_file("test_image", png_path, "image")


    # run both models
    if model_type == "dense":
        acts_origin = run_modules(origin_model, img_tensor)
        acts_tuned = run_modules(tuned_model, img_tensor)
        visualize_activations(acts_origin, acts_tuned, analyze_dir)
    elif model_type == "wph":
        acts_origin = calc_wph_activations(origin_model, img_tensor)
        acts_tuned = calc_wph_activations(tuned_model, img_tensor)
        visualize_wph_activations(acts_origin, acts_tuned, analyze_dir)

    # save results

if __name__ == "__main__":
    exp_dir = input("Enter exp_dir to analyze: ")
    visualize_main(exp_dir)