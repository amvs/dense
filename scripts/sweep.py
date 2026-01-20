import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config, expand_param
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist_sweep.yaml)"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Weights and Biases project name for logging")
    parser.add_argument(
        "--random", action="store_true",
        help="If set, use random filters"
    )
    parser.add_argument('--model-type', type=str, choices=['wph', 'scat'], default='scat',
                        help='Type of model to train (default: scat (dense))')
    parser.add_argument('--name', type=str, default='',
                        help='Optional short name for the sweep folder')
    parser.add_argument('--metric', type=str, default='last_val_acc',
                        help='Metric to evaluate top models (default: last_val_acc)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3"). If not specified, uses all available GPUs.')
    parser.add_argument('--max_parallel', type=int, default=None,
                        help='Maximum number of parallel jobs. Defaults to number of GPUs if not specified.')
    return parser.parse_args()

# Parse arguments
args = parse_args()

# Read config
sweep = load_config(args.config)

# Load base configuration
base_config = load_config(sweep["base_config"])

# Extract dataset name after loading base_config
dataset_name = base_config["dataset"].split("/")[-1] if "dataset" in base_config else "unknown"

# Create output folder
short_name = f"{args.name}-" if args.name else ""
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sweep_dir = os.path.join("experiments", dataset_name, f"{short_name}sweeps-{timestamp}")
os.makedirs(sweep_dir, exist_ok=True)

params = sweep["sweep"]  # dict of lists

# random_flag = args.random
# if random_flag:
#     script = "scripts/train_random.py"
# else:
#     script = "scripts/train.py"


# init logger
wandb_project = args.wandb_project
sweep_name = os.path.splitext(os.path.basename(args.config))[0]
logger = LoggerManager.get_logger(log_dir=sweep_dir, wandb_project=wandb_project, name=f"{sweep_name}-{timestamp}")


# ===== GPU DETECTION AND CONFIGURATION =====
def get_available_gpus():
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Will run on CPU.")
        return []
    return list(range(torch.cuda.device_count()))

def parse_gpu_list(gpu_str):
    """Parse comma-separated GPU string into list of integers."""
    if gpu_str is None:
        return None
    return [int(x.strip()) for x in gpu_str.split(",")]

# Determine which GPUs to use
if args.gpus is not None:
    requested_gpus = parse_gpu_list(args.gpus)
    available_gpus = get_available_gpus()
    gpu_list = [gpu for gpu in requested_gpus if gpu in available_gpus]
    if len(gpu_list) != len(requested_gpus):
        missing = set(requested_gpus) - set(gpu_list)
        logger.warning(f"Requested GPUs {missing} not available. Using {gpu_list}")
else:
    gpu_list = get_available_gpus()

if len(gpu_list) == 0:
    logger.warning("No GPUs available. Running sequentially on CPU.")
    max_parallel = 1
    gpu_list = [None]  # Use None to indicate CPU
else:
    logger.info(f"Using GPUs: {gpu_list}")
    max_parallel = args.max_parallel if args.max_parallel is not None else len(gpu_list)

logger.info(f"Maximum parallel jobs: {max_parallel}")

# ===== PARALLEL EXECUTION FUNCTION =====
def run_training_job(job_info):
    """
    Run a single training job on a specific GPU.
    
    Args:
        job_info: Tuple of (run_idx, values, keys, base_config, sweep_dir, 
                            file, wandb_project, gpu_id)
    
    Returns:
        Tuple of (run_idx, success, error_message)
    """
    run_idx, values, keys, base_config, sweep_dir, file, wandb_project, gpu_id = job_info
    
    try:
        # Set CUDA_VISIBLE_DEVICES if GPU is specified
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Create merged config
        overrides = {k: v for k, v in zip(keys, values)}
        merged_config = {**base_config, **overrides}
        
        # Save merged config to a temporary file
        temp_config_path = os.path.join(sweep_dir, f"temp_config_{run_idx}.yaml")
        with open(temp_config_path, "w") as f:
            yaml.dump(merged_config, f)
        
        # Build command
        cmd = [
            "python", file,
            "--config", temp_config_path,
            "--sweep_dir", sweep_dir
        ]
        if wandb_project is not None:
            cmd.extend(["--wandb_project", wandb_project])
        
        # Run training
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        
        if result.returncode == 0:
            return (run_idx, True, None)
        else:
            error_msg = f"Return code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr: {result.stderr[:500]}"  # Limit error message length
            return (run_idx, False, error_msg)
    
    except Exception as e:
        return (run_idx, False, str(e))

# ===== PREPARE ALL JOBS =====
# All parameter names
expanded_values = [expand_param(v) for v in params.values()]
keys = list(params.keys())
all_combinations = list(itertools.product(*expanded_values))
total_runs = len(all_combinations)

# Determine training script
if args.model_type == 'scat':
    if args.random:
        file = "scripts/train_random.py"
    else:
        file = "scripts/train.py"
elif args.model_type == 'wph':
    file = "scripts/train_wph_classifier.py"

# Prepare all job info
job_queue = []
for run_idx, values in enumerate(all_combinations, 1):
    # Assign GPU in round-robin fashion
    gpu_id = gpu_list[(run_idx - 1) % len(gpu_list)] if len(gpu_list) > 0 else None
    
    job_info = (run_idx, values, keys, base_config, sweep_dir, file, wandb_project, gpu_id)
    job_queue.append(job_info)
    
    # Log job assignment
    overrides = {k: v for k, v in zip(keys, values)}
    gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    logger.info(f"Queued Run {run_idx}/{total_runs} on {gpu_str} — overrides: {overrides}")

# ===== EXECUTE JOBS IN PARALLEL =====
logger.info(f"Starting {total_runs} runs across {max_parallel} parallel workers...")
start_time = time.time()

completed_runs = 0
failed_runs = []

if max_parallel == 1:
    # Sequential execution (for debugging or CPU-only)
    for job_info in job_queue:
        run_idx, success, error_msg = run_training_job(job_info)
        completed_runs += 1
        if success:
            logger.info(f"[{completed_runs}/{total_runs}] Finished run {run_idx}.")
        else:
            logger.error(f"[{completed_runs}/{total_runs}] Run {run_idx} failed: {error_msg}")
            failed_runs.append((run_idx, error_msg))
else:
    # Parallel execution using ThreadPoolExecutor
    # Using threads instead of processes to avoid CUDA initialization issues
    # Each thread launches a subprocess with its own CUDA_VISIBLE_DEVICES
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all jobs
        future_to_job = {executor.submit(run_training_job, job_info): job_info 
                         for job_info in job_queue}
        
        # Process completed jobs
        for future in as_completed(future_to_job):
            run_idx, success, error_msg = future.result()
            completed_runs += 1
            
            job_info = future_to_job[future]
            gpu_id = job_info[7]
            gpu_str = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
            
            if success:
                logger.info(f"[{completed_runs}/{total_runs}] Finished run {run_idx} on {gpu_str}.")
            else:
                logger.error(f"[{completed_runs}/{total_runs}] Run {run_idx} on {gpu_str} failed: {error_msg}")
                failed_runs.append((run_idx, error_msg))

elapsed_time = time.time() - start_time
logger.info(f"All runs finished in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
if failed_runs:
    logger.warning(f"{len(failed_runs)} runs failed:")
    for run_idx, error_msg in failed_runs:
        logger.warning(f"  Run {run_idx}: {error_msg}")
else:
    logger.info("All runs completed successfully!")

logger.info("Analyzing results...")
rows = []
for ratio_folder in os.listdir(sweep_dir):
    ratio_path = os.path.join(sweep_dir, ratio_folder)
    if not os.path.isdir(ratio_path) or not ratio_folder.startswith("train_ratio="):
        continue

    train_ratio = float(ratio_folder.split("=")[1])  # extract the number
    for run_folder in os.listdir(ratio_path):
        run_path = os.path.join(ratio_path, run_folder)
        config_path = os.path.join(run_path, "config.yaml")
        if not os.path.isfile(config_path):
            continue

        config = load_config(config_path)
        config["run"] = run_folder
        rows.append(config)

results_path = os.path.join(sweep_dir, "results")
os.makedirs(results_path, exist_ok=True)

df = pd.DataFrame(rows)
logger.info(f"df head:\n {df.head()}")

# Extract classifier and fine-tuned test accuracies for low-data regime analysis
# These should be saved in config.yaml during training
if "FineTuning_Comparison/classifier_test_acc" in df.columns:
    df["classifier_test_acc"] = df["FineTuning_Comparison/classifier_test_acc"]
if "FineTuning_Comparison/fine_tuned_test_acc" in df.columns:
    df["fine_tuned_test_acc"] = df["FineTuning_Comparison/fine_tuned_test_acc"]
elif "test_acc" in df.columns:
    df["fine_tuned_test_acc"] = df["test_acc"]
if "FineTuning_Comparison/improvement" in df.columns:
    df["improvement"] = df["FineTuning_Comparison/improvement"]
elif "classifier_test_acc" in df.columns and "fine_tuned_test_acc" in df.columns:
    df["improvement"] = df["fine_tuned_test_acc"] - df["classifier_test_acc"]

df.to_csv(os.path.join(results_path, "all_runs.csv"), index=False)
logger.info(f"Saved all runs to {os.path.join(results_path, 'all_runs.csv')}")
logger.send_file("all_runs_data", os.path.join(results_path, "all_runs.csv"), "table")

# ===== LOW-DATA REGIME ANALYSIS PLOTS =====
if "classifier_test_acc" in df.columns and "fine_tuned_test_acc" in df.columns:
    # 1. Improvement vs Train Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    if "fine_tune_mode" in df.columns:
        for mode in df["fine_tune_mode"].unique():
            mode_df = df[df["fine_tune_mode"] == mode]
            if "improvement" in mode_df.columns:
                ax.scatter(mode_df["train_ratio"], mode_df["improvement"], 
                          label=f"Fine-tune mode: {mode}", alpha=0.6, s=50)
    else:
        if "improvement" in df.columns:
            ax.scatter(df["train_ratio"], df["improvement"], alpha=0.6, s=50)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No improvement')
    ax.set_xlabel("Train Ratio")
    ax.set_ylabel("Improvement (Fine-tuned - Classifier)")
    ax.set_title("Fine-tuning Improvement vs Train Ratio\n(Low-Data Regime Analysis)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(results_path, "improvement_vs_train_ratio.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.send_file("improvement_vs_train_ratio", plot_path, "image")
    
    # 2. Performance Comparison by Train Ratio
    fig, ax = plt.subplots(figsize=(10, 6))
    train_ratios = sorted(df["train_ratio"].unique())
    classifier_means = []
    fine_tuned_means = []
    classifier_stds = []
    fine_tuned_stds = []
    
    for tr in train_ratios:
        tr_df = df[df["train_ratio"] == tr]
        classifier_means.append(tr_df["classifier_test_acc"].mean())
        fine_tuned_means.append(tr_df["fine_tuned_test_acc"].mean())
        classifier_stds.append(tr_df["classifier_test_acc"].std() if len(tr_df) > 1 else 0)
        fine_tuned_stds.append(tr_df["fine_tuned_test_acc"].std() if len(tr_df) > 1 else 0)
    
    x = np.arange(len(train_ratios))
    width = 0.35
    ax.bar(x - width/2, classifier_means, width, yerr=classifier_stds, 
           label="Classifier", alpha=0.7, capsize=5)
    ax.bar(x + width/2, fine_tuned_means, width, yerr=fine_tuned_stds, 
           label="Fine-tuned", alpha=0.7, capsize=5)
    ax.set_xlabel("Train Ratio")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Performance Comparison: Classifier vs Fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{tr:.2f}" for tr in train_ratios])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plot_path = os.path.join(results_path, "performance_comparison_by_train_ratio.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.send_file("performance_comparison_by_train_ratio", plot_path, "image")
    
    # 3. Relative Improvement (%)
    if "improvement" in df.columns and "classifier_test_acc" in df.columns:
        df["relative_improvement"] = (df["improvement"] / df["classifier_test_acc"]) * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        if "fine_tune_mode" in df.columns:
            for mode in df["fine_tune_mode"].unique():
                mode_df = df[df["fine_tune_mode"] == mode]
                ax.scatter(mode_df["train_ratio"], mode_df["relative_improvement"], 
                          label=f"Fine-tune mode: {mode}", alpha=0.6, s=50)
        else:
            ax.scatter(df["train_ratio"], df["relative_improvement"], alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel("Train Ratio")
        ax.set_ylabel("Relative Improvement (%)")
        ax.set_title("Relative Improvement from Fine-tuning vs Train Ratio\n(Percentage increase)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(results_path, "relative_improvement_vs_train_ratio.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.send_file("relative_improvement_vs_train_ratio", plot_path, "image")
    
    # 4. Summary Statistics by Train Ratio
    if "improvement" in df.columns:
        summary_stats = df.groupby("train_ratio").agg({
            "improvement": ["mean", "std", "count"],
            "classifier_test_acc": "mean",
            "fine_tuned_test_acc": "mean"
        }).round(4)
        summary_stats.columns = ["improvement_mean", "improvement_std", "n_runs", 
                                 "classifier_acc_mean", "fine_tuned_acc_mean"]
        summary_path = os.path.join(results_path, "low_data_regime_summary.csv")
        summary_stats.to_csv(summary_path)
        logger.info(f"Saved low-data regime summary to {summary_path}")
        logger.send_file("low_data_regime_summary", summary_path, "table")

# ===== HYPERPARAMETER EFFECT ANALYSIS =====
# Identify hyperparameters to analyze (exclude non-hyperparameter columns)
exclude_cols = {"run", "train_ratio", "val_ratio", "test_ratio", "nb_class", "image_shape", 
                "best_train_acc", "best_val_acc", "test_acc", "test_loss", "best_train_loss", 
                "best_val_loss", "last_val_acc", "n_tuned_params", "n_linear_params",
                "classifier_test_acc", "fine_tuned_test_acc", "improvement", "relative_improvement",
                "FineTuning_Comparison/classifier_test_acc", "FineTuning_Comparison/fine_tuned_test_acc",
                "FineTuning_Comparison/improvement", "random", "device", "filters", "dataset",
                "deeper_path", "kth_root_dir", "fold", "resize", "batch_size"}

# Common hyperparameters to analyze
hyperparams_to_analyze = ["lr", "linear_lr", "lambda_reg", "weight_decays", "nb_orients", 
                          "max_scale", "depth", "classifier_type", "fine_tune_mode", 
                          "hypernet_hidden_dim", "attention_d_model", "attention_num_heads",
                          "attention_num_layers", "n_copies", "wavelet", "classifier_epochs", 
                          "conv_epochs"]

# Filter to only hyperparameters that exist in the dataframe and have multiple values
available_hyperparams = [hp for hp in hyperparams_to_analyze 
                         if hp in df.columns and df[hp].nunique() > 1]

if available_hyperparams and "fine_tuned_test_acc" in df.columns:
    logger.info(f"Analyzing hyperparameter effects for: {available_hyperparams}")
    
    # 1. Hyperparameter Effect on Test Accuracy
    n_hps = len(available_hyperparams)
    n_cols = min(3, n_hps)
    n_rows = (n_hps + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_hps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, hp in enumerate(available_hyperparams):
        ax = axes[idx]
        hp_data = df.groupby(hp).agg({
            "fine_tuned_test_acc": ["mean", "std"],
            "classifier_test_acc": ["mean", "std"] if "classifier_test_acc" in df.columns else None
        })
        
        if hp_data.index.dtype in [np.int64, np.float64]:
            # Numeric hyperparameter - line plot
            x_vals = sorted(hp_data.index)
            if "classifier_test_acc" in df.columns:
                ax.errorbar(x_vals, [hp_data.loc[x, ("classifier_test_acc", "mean")] for x in x_vals],
                           yerr=[hp_data.loc[x, ("classifier_test_acc", "std")] for x in x_vals],
                           label="Classifier", marker='o', capsize=3, alpha=0.7)
            ax.errorbar(x_vals, [hp_data.loc[x, ("fine_tuned_test_acc", "mean")] for x in x_vals],
                       yerr=[hp_data.loc[x, ("fine_tuned_test_acc", "std")] for x in x_vals],
                       label="Fine-tuned", marker='s', capsize=3, alpha=0.7)
            ax.set_xlabel(hp)
        else:
            # Categorical hyperparameter - bar plot
            x_pos = np.arange(len(hp_data.index))
            width = 0.35
            if "classifier_test_acc" in df.columns:
                ax.bar(x_pos - width/2, [hp_data.loc[x, ("classifier_test_acc", "mean")] for x in hp_data.index],
                      width, yerr=[hp_data.loc[x, ("classifier_test_acc", "std")] for x in hp_data.index],
                      label="Classifier", alpha=0.7, capsize=3)
            ax.bar(x_pos + width/2, [hp_data.loc[x, ("fine_tuned_test_acc", "mean")] for x in hp_data.index],
                  width, yerr=[hp_data.loc[x, ("fine_tuned_test_acc", "std")] for x in hp_data.index],
                  label="Fine-tuned", alpha=0.7, capsize=3)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(hp_data.index, rotation=45, ha='right')
            ax.set_xlabel(hp)
        
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Effect of {hp}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Hide unused subplots
    for idx in range(n_hps, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(results_path, "hyperparameter_effects_on_accuracy.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.send_file("hyperparameter_effects_accuracy", plot_path, "image")
    
    # 2. Hyperparameter Effect on Improvement
    if "improvement" in df.columns:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_hps == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, hp in enumerate(available_hyperparams):
            ax = axes[idx]
            hp_data = df.groupby(hp).agg({
                "improvement": ["mean", "std"]
            })
            
            if hp_data.index.dtype in [np.int64, np.float64]:
                # Numeric hyperparameter - line plot
                x_vals = sorted(hp_data.index)
                ax.errorbar(x_vals, [hp_data.loc[x, ("improvement", "mean")] for x in x_vals],
                           yerr=[hp_data.loc[x, ("improvement", "std")] for x in x_vals],
                           marker='o', capsize=3, alpha=0.7, color='green')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel(hp)
            else:
                # Categorical hyperparameter - bar plot
                x_pos = np.arange(len(hp_data.index))
                ax.bar(x_pos, [hp_data.loc[x, ("improvement", "mean")] for x in hp_data.index],
                      yerr=[hp_data.loc[x, ("improvement", "std")] for x in hp_data.index],
                      alpha=0.7, capsize=3, color='green')
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(hp_data.index, rotation=45, ha='right')
                ax.set_xlabel(hp)
            
            ax.set_ylabel("Improvement")
            ax.set_title(f"Effect of {hp} on Improvement")
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(n_hps, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(results_path, "hyperparameter_effects_on_improvement.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.send_file("hyperparameter_effects_improvement", plot_path, "image")
    
    # 3. Interaction Effects: Hyperparameter vs Train Ratio
    # Focus on key hyperparameters that might interact with train_ratio
    key_hyperparams = [hp for hp in ["lambda_reg", "lr", "fine_tune_mode", "classifier_type"] 
                      if hp in available_hyperparams]
    
    if key_hyperparams and "train_ratio" in df.columns:
        for hp in key_hyperparams[:4]:  # Limit to 4 to avoid too many plots
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left plot: Fine-tuned accuracy
            ax1 = axes[0]
            train_ratios = sorted(df["train_ratio"].unique())
            hp_values = sorted(df[hp].unique())
            
            for hp_val in hp_values:
                hp_df = df[df[hp] == hp_val]
                means = []
                stds = []
                for tr in train_ratios:
                    tr_hp_df = hp_df[hp_df["train_ratio"] == tr]
                    if len(tr_hp_df) > 0:
                        means.append(tr_hp_df["fine_tuned_test_acc"].mean())
                        stds.append(tr_hp_df["fine_tuned_test_acc"].std() if len(tr_hp_df) > 1 else 0)
                    else:
                        means.append(np.nan)
                        stds.append(0)
                ax1.errorbar(train_ratios, means, yerr=stds, label=f"{hp}={hp_val}", 
                           marker='o', capsize=3, alpha=0.7)
            
            ax1.set_xlabel("Train Ratio")
            ax1.set_ylabel("Fine-tuned Test Accuracy")
            ax1.set_title(f"Fine-tuned Accuracy: {hp} vs Train Ratio")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Right plot: Improvement
            if "improvement" in df.columns:
                ax2 = axes[1]
                for hp_val in hp_values:
                    hp_df = df[df[hp] == hp_val]
                    means = []
                    stds = []
                    for tr in train_ratios:
                        tr_hp_df = hp_df[hp_df["train_ratio"] == tr]
                        if len(tr_hp_df) > 0:
                            means.append(tr_hp_df["improvement"].mean())
                            stds.append(tr_hp_df["improvement"].std() if len(tr_hp_df) > 1 else 0)
                        else:
                            means.append(np.nan)
                            stds.append(0)
                    ax2.errorbar(train_ratios, means, yerr=stds, label=f"{hp}={hp_val}", 
                               marker='o', capsize=3, alpha=0.7)
                ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                ax2.set_xlabel("Train Ratio")
                ax2.set_ylabel("Improvement")
                ax2.set_title(f"Improvement: {hp} vs Train Ratio")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                axes[1].axis('off')
            
            plt.tight_layout()
            plot_path = os.path.join(results_path, f"interaction_{hp}_vs_train_ratio.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.send_file(f"interaction_{hp}", plot_path, "image")
    
    # 4. Best Hyperparameter Values Summary
    best_hp_summary = []
    for hp in available_hyperparams:
        hp_summary = df.groupby(hp).agg({
            "fine_tuned_test_acc": "mean",
            "improvement": "mean" if "improvement" in df.columns else None
        }).reset_index()
        hp_summary = hp_summary.sort_values("fine_tuned_test_acc", ascending=False)
        best_val = hp_summary.iloc[0][hp]
        best_acc = hp_summary.iloc[0]["fine_tuned_test_acc"]
        best_hp_summary.append({
            "hyperparameter": hp,
            "best_value": best_val,
            "best_accuracy": best_acc,
            "n_values": df[hp].nunique()
        })
    
    best_hp_df = pd.DataFrame(best_hp_summary)
    best_hp_path = os.path.join(results_path, "best_hyperparameter_values.csv")
    best_hp_df.to_csv(best_hp_path, index=False)
    logger.info(f"Saved best hyperparameter values to {best_hp_path}")
    logger.send_file("best_hyperparameters", best_hp_path, "table")
    
    # 5. Hyperparameter Importance (correlation with performance)
    if "fine_tuned_test_acc" in df.columns:
        numeric_hps = [hp for hp in available_hyperparams 
                      if df[hp].dtype in [np.int64, np.float64]]
        
        if numeric_hps:
            correlations = []
            for hp in numeric_hps:
                corr = df[hp].corr(df["fine_tuned_test_acc"])
                correlations.append({"hyperparameter": hp, "correlation": corr})
            
            corr_df = pd.DataFrame(correlations).sort_values("correlation", key=abs, ascending=False)
            
            # Plot correlation bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['green' if c > 0 else 'red' for c in corr_df["correlation"]]
            ax.barh(corr_df["hyperparameter"], corr_df["correlation"], color=colors, alpha=0.7)
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax.set_xlabel("Correlation with Fine-tuned Test Accuracy")
            ax.set_title("Hyperparameter Importance\n(Correlation with Performance)")
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plot_path = os.path.join(results_path, "hyperparameter_importance.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.send_file("hyperparameter_importance", plot_path, "image")
            
            # Save correlations
            corr_path = os.path.join(results_path, "hyperparameter_correlations.csv")
            corr_df.to_csv(corr_path, index=False)
            logger.info(f"Saved hyperparameter correlations to {corr_path}")
    
    logger.info("Completed hyperparameter effect analysis")

# Step 1: Plot validation accuracy per run per train_ratio, sorted decreasingly
for train_ratio, group in df.groupby("train_ratio"):
    # Sort runs by last_val_acc descending
    group_sorted = group.sort_values("last_val_acc", ascending=False)
    
    plt.figure(figsize=(8, 5))
    plt.title(f"Validation Error per Run (train_ratio={train_ratio})")
    plt.bar(group_sorted["run"], 1.0 - group_sorted["last_val_acc"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Validation Error(1-val_acc)")
    plt.tight_layout()
    plot_path = os.path.join(results_path, f"val_err_train_ratio_{train_ratio}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.send_file("val_err_plot", plot_path, "image")
logger.info(f"Saved results summary plot to {results_path}")

topN = 3
# Step 2: take top-N runs per val_ratio by validation accuracy
# Sort by val_ratio and the specified metric
metric = args.metric
df = df.sort_values(["val_ratio", metric], ascending=[True, False])

# Group by val_ratio and keep top N runs
top_runs = df.groupby("val_ratio").head(topN)

topN_csv_path = os.path.join(results_path, f"top{topN}_runs_per_val_ratio.csv")
top_runs.to_csv(topN_csv_path, index=False)
logger.info(f"Saved top-{topN} runs per val_ratio to {topN_csv_path}")

# Create a label for x-axis: "val_ratio-rank" only
labels = []
heights = []
for val_ratio, group in top_runs.groupby("val_ratio"):
    # sort by val_acc descending within this val_ratio
    group_sorted = group.sort_values("last_val_acc", ascending=False).reset_index()
    for rank, row in enumerate(group_sorted.itertuples(), 1):
        labels.append(f"{train_ratio:.2f}--{rank}")
        heights.append(1.0 - row.test_acc)  # or row.test_acc if that's the column name

# Plot
plt.figure(figsize=(max(10, len(labels)*0.6), 6))
plt.bar(range(len(heights)), heights, color="skyblue")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("train_ratio | rank")
plt.ylabel("Test Error(1 - test_acc)")
plt.title(f"Test Error of Top {topN} Runs per train_ratio")
plt.tight_layout()
plot_path = os.path.join(results_path, f"top{topN}_test_err_all_train_ratios.png")
plt.savefig(plot_path)
plt.close()
logger.send_file("topN_test_err_plot", plot_path, "image")
logger.info(f"Saved top-{topN} test accuracy plot across all train_ratios to {results_path}")
logger.finish()