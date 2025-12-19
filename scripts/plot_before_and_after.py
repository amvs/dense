"""
Plotting scheme for the output of the training.
visualize the most changed kernels before and after training.
Put the "trained.pt , origin.pt, config.yaml" files in the same directory as this code
"""

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from dense.helpers import LoggerManager

def extract_kernels_dense(orig):
    def conv_index(k: str):
        # expects "sequential_conv.{j}.weight"
        m = re.search(r"sequential_conv\.(\d+)\.weight$", k)
        return int(m.group(1)) if m else 10**9


    conv_keys = sorted(
        [k for k in orig.keys() if k.startswith("sequential_conv.") and k.endswith(".weight")],
        key=conv_index
    )

    return conv_keys, conv_index

def extract_kernels_wph(orig):
    def conv_index(k: str):
        # expects "feature_extractor.base_filters.{j}.weight"
        m = re.search(r"feature_extractor\.base_filters", k)
        return int(m.group(1)) if m else 10**9
    
    conv_keys = sorted(
        [k for k in orig.keys() if 'base_filters' in k],
        key=conv_index
    )
    return conv_keys, conv_index

def plot_kernels_wph_base_filters(exp_dir, trained_filename='trained.pt', origin_filename='origin.pt', base_filters_key:list[str]=['feature_extractor.wave_conv.base_filters']):
    origin_path = os.path.join(exp_dir, origin_filename)
    trained_path = os.path.join(exp_dir, trained_filename)
    orig = torch.load(origin_path, map_location="cpu", weights_only=True)
    tune = torch.load(trained_path, map_location="cpu", weights_only=True)

    # Get base_filters from state dict
    if len(base_filters_key) != 1:
        W_o = torch.complex(orig[base_filters_key[0]], orig[base_filters_key[1]]).detach().cpu().numpy()
        W_t = torch.complex(tune[base_filters_key[0]], tune[base_filters_key[1]]).detach().cpu().numpy()
    else:
        W_o = orig[base_filters_key[0]].detach().cpu().numpy()
        W_t = tune[base_filters_key[0]].detach().cpu().numpy()

    # Determine dimensions
    shape = W_o.shape
    dims = len(shape)
    assert dims in (5, 6), f"Unexpected base_filters shape: {shape}"

    # Get indices to loop over (all except last two dims)
    loop_dims = shape[:-2]
    out_root = os.path.join(exp_dir, "kernel_plots_wph")
    os.makedirs(out_root, exist_ok=True)
    filenames = []



    # Precompute global vmin/vmax for before/after plots (real and imag separately)
    all_real = []
    all_imag = []
    for idx in np.ndindex(*loop_dims):
        o = W_o[idx]
        t = W_t[idx]
        all_real.extend([o.real.flatten(), t.real.flatten()])
        all_imag.extend([o.imag.flatten(), t.imag.flatten()])
    all_real = np.concatenate(all_real)
    all_imag = np.concatenate(all_imag)
    vmax_r = float(np.max(np.abs(all_real)))
    vmax_i = float(np.max(np.abs(all_imag)))

    # Iterate and plot with shared scale for before/after, separate scale for delta
    for idx in np.ndindex(*loop_dims):
        o = W_o[idx]  # shape (M, N)
        t = W_t[idx]
        d = t - o

        # L1 norm of difference
        l1 = float(np.sum(np.abs(d)))

        # Plot real/imag before/after/delta
        o_r, t_r, d_r = o.real, t.real, d.real
        o_i, t_i, d_i = o.imag, t.imag, d.imag

        # Delta scales
        vmax_dr = float(np.max(np.abs(d_r)))
        vmax_di = float(np.max(np.abs(d_i)))

        fig, axes = plt.subplots(2, 3, figsize=(9, 6))
        im00 = axes[0,0].imshow(o_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,0].set_title("Real — Before"); axes[0,0].axis("off")
        im01 = axes[0,1].imshow(t_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,1].set_title("Real — After");  axes[0,1].axis("off")
        im02 = axes[0,2].imshow(d_r, vmin=-vmax_dr, vmax=+vmax_dr, interpolation="nearest"); axes[0,2].set_title("Real Δ");        axes[0,2].axis("off")

        im10 = axes[1,0].imshow(o_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,0].set_title("Imag — Before"); axes[1,0].axis("off")
        im11 = axes[1,1].imshow(t_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,1].set_title("Imag — After");  axes[1,1].axis("off")
        im12 = axes[1,2].imshow(d_i, vmin=-vmax_di, vmax=+vmax_di, interpolation="nearest"); axes[1,2].set_title("Imag Δ");        axes[1,2].axis("off")

        # Add colorbars with labels as legends
        for ax, im, label in zip(
            axes.flatten(),
            [im00, im01, im02, im10, im11, im12],
            ["Real — Before", "Real — After", "Real Δ", "Imag — Before", "Imag — After", "Imag Δ"]
        ):
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel(label, rotation=270, labelpad=15)

        idx_str = "_".join([f"{i}" for i in idx])
        fig.suptitle(f"base_filters idx={idx_str} | L1(Δ)={l1:.3e}")
        fig.tight_layout()
        fname = f"base_filters_{idx_str}.png"
        plot_path = os.path.join(out_root, fname)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        filenames.append(fname)

    print("[done] WPH base_filters plots saved under", out_root)
    return(filenames)

def plot_kernels(exp_dir, trained_filename='trained.pt', origin_filename='origin.pt', model_type='dense',):
    logger = LoggerManager.get_logger(log_dir=exp_dir)
    origin_path = os.path.join(exp_dir, origin_filename)
    trained_path = os.path.join(exp_dir, trained_filename)
    orig = torch.load(origin_path, map_location="cpu", weights_only=True)
    tune = torch.load(trained_path, map_location="cpu", weights_only=True)
    conv_keys, conv_index = extract_kernels_dense(orig)
    
    out_root = os.path.join(exp_dir, "kernel_plots")
    os.makedirs(out_root, exist_ok=True)

    for k in conv_keys:
        if k not in tune:
            print(f"[skip] {k} not in tuned file"); continue

        W_o = orig[k].detach().cpu().numpy()
        W_t = tune[k].detach().cpu().numpy()
        if W_o.shape != W_t.shape or W_o.ndim != 4:
            print(f"[skip] shape mismatch: {k}: {W_o.shape} vs {W_t.shape}"); continue

        O, I, KH, KW = W_o.shape
        layer_id = conv_index(k)
        save_dir = os.path.join(out_root, f"layer_{layer_id:02d}")
        os.makedirs(save_dir, exist_ok=True)

        # ---- track the most-changed kernel (by L1 = sum |Δ|) ----
        best_l1 = -1.0
        best = dict(oc=None, ic=None, o=None, t=None, d=None)

        for oc in range(O):
            for ic in range(I):
                o = W_o[oc, ic]         # (KH, KW) complex
                t = W_t[oc, ic]
                d = t - o

                # L1 over complex delta magnitude
                l1 = float(np.sum(np.abs(d)))
                if l1 > best_l1:
                    best_l1 = l1
                    best.update(oc=oc, ic=ic, o=o, t=t, d=d)

                # ----- your per-kernel 2×3 plot (real/imag) -----
                # o_r, t_r, d_r = o.real, t.real, d.real
                # o_i, t_i, d_i = o.imag, t.imag, d.imag

                # vmax_r = float(np.max(np.abs([o_r, t_r, d_r])))
                # vmax_i = float(np.max(np.abs([o_i, t_i, d_i])))

                # fig, axes = plt.subplots(2, 3, figsize=(9, 6))
                # im00 = axes[0,0].imshow(o_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,0].set_title("Real — Before"); axes[0,0].axis("off")
                # im01 = axes[0,1].imshow(t_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,1].set_title("Real — After");  axes[0,1].axis("off")
                # im02 = axes[0,2].imshow(d_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,2].set_title("Real Δ");        axes[0,2].axis("off")

                # im10 = axes[1,0].imshow(o_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,0].set_title("Imag — Before"); axes[1,0].axis("off")
                # im11 = axes[1,1].imshow(t_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,1].set_title("Imag — After");  axes[1,1].axis("off")
                # im12 = axes[1,2].imshow(d_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,2].set_title("Imag Δ");        axes[1,2].axis("off")

                # for ax, im in zip(axes.flatten(), [im00, im01, im02, im10, im11, im12]):
                #     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # fig.suptitle(f"{k}  |  oc={oc}, ic={ic}")
                # fig.tight_layout()
                # fig.savefig(os.path.join(save_dir, f"{k.replace('.', '_')}_oc{oc}_ic{ic}.png"), dpi=150)
                # plt.close(fig)

        # ---- after scanning all (oc, ic), save the MOST-CHANGED kernel for this layer ----
        if best["o"] is not None:
            oc, ic = best["oc"], best["ic"]
            o, t, d = best["o"], best["t"], best["d"]

            # real/imag rows again, with row-consistent symmetric scales
            o_r, t_r, d_r = o.real, t.real, d.real
            o_i, t_i, d_i = o.imag, t.imag, d.imag
            vmax_r = float(np.max(np.abs([o_r, t_r, d_r])))
            vmax_i = float(np.max(np.abs([o_i, t_i, d_i])))

            fig, axes = plt.subplots(2, 3, figsize=(9, 6))
            im00 = axes[0,0].imshow(o_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,0].set_title("Real — Before"); axes[0,0].axis("off")
            im01 = axes[0,1].imshow(t_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,1].set_title("Real — After");  axes[0,1].axis("off")
            im02 = axes[0,2].imshow(d_r, vmin=-vmax_r, vmax=+vmax_r, interpolation="nearest"); axes[0,2].set_title("Real Δ");        axes[0,2].axis("off")

            im10 = axes[1,0].imshow(o_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,0].set_title("Imag — Before"); axes[1,0].axis("off")
            im11 = axes[1,1].imshow(t_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,1].set_title("Imag — After");  axes[1,1].axis("off")
            im12 = axes[1,2].imshow(d_i, vmin=-vmax_i, vmax=+vmax_i, interpolation="nearest"); axes[1,2].set_title("Imag Δ");        axes[1,2].axis("off")

            for ax, im in zip(axes.flatten(), [im00, im01, im02, im10, im11, im12]):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            fig.suptitle(f"Most changed kernel — {layer_id:02d} | oc={oc}, ic={ic} | L1(Δ)={best_l1:.3e}")
            fig.tight_layout()
            plot_path = os.path.join(save_dir, f"layer_{layer_id:02d}_MOST_CHANGED_oc{oc}_ic{ic}.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            logger.send_file(f"layer_{layer_id:02d}_most_changed_kernel", plot_path, "image")

            # Also drop a tiny text file for quick lookup
            with open(os.path.join(save_dir, "most_changed.txt"), "w", encoding="utf-8") as f:
                f.write(f"layer={layer_id} key={k}\noc={oc} ic={ic}\nL1_delta={best_l1:.6e}\n")

            print(f"[layer {layer_id:02d}] most changed: oc={oc}, ic={ic}, L1(Δ)={best_l1:.3e} -> saved")

    print("[done] Plots saved under", out_root)

if __name__ == "__main__":
    exp_dir = input("Enter exp_dir to analyze: ")
    plot_kernels(exp_dir)