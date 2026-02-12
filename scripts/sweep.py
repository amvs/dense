import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config, expand_param
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import time

dotenv.load_dotenv()

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
    parser.add_argument('--model-type', type=str, choices=['wph', 'scat', 'wph_pca', 'wph_hypernetwork'], default='scat',
                        help='Type of model to train (default: scat (dense))')
    parser.add_argument('--name', type=str, default='',
                        help='Optional short name for the sweep folder')
    parser.add_argument('--metric', type=str, default='best_feature_extractor_acc',
                        help='Metric to evaluate top models (default: last_val_acc)')
    parser.add_argument('--test-metric', type=str, default='feature_extractor_test_acc',)
    parser.add_argument(
        "--sweep-dir", type=str, default=None,
        help="Specify output sweep directory (skips timestamp generation)")
    parser.add_argument(
        "--fold-filter", type=int, default=None,
        help="Only run jobs with this fold value (for SLURM array jobs)")
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
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
if args.sweep_dir:
    sweep_dir = args.sweep_dir
    os.makedirs(sweep_dir, exist_ok=True)
else:
    short_name = f"{args.name}-" if args.name else ""
    sweep_dir = os.path.join("experiments", dataset_name, f"{short_name}sweeps-{timestamp}")
    os.makedirs(sweep_dir, exist_ok=True)

# Save a copy of the sweep and base configs for resume_sweep.py
sweep_copy_path = os.path.join(sweep_dir, "sweep_config.yaml")
base_copy_path = os.path.join(sweep_dir, "base_config.yaml")
with open(sweep_copy_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(sweep, f, sort_keys=False)
with open(base_copy_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(base_config, f, sort_keys=False)

params = sweep["sweep"]  # dict of lists

random_flag = args.random
if random_flag:
    script = "scripts/train_random.py"
else:
    script = "scripts/train.py"


# init logger
wandb_project = args.wandb_project
sweep_name = os.path.splitext(os.path.basename(args.config))[0]
logger = LoggerManager.get_logger(log_dir=sweep_dir, wandb_project=wandb_project, name=f"{sweep_name}-{timestamp}")


logger = LoggerManager.get_logger(log_dir=sweep_dir, wandb_project=wandb_project, name=f"{sweep_name}-{timestamp}")

# Detect available GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    gpu_ids = list(range(num_gpus))
    logger.info(f"Found {num_gpus} GPUs available for parallel execution")
else:
    num_gpus = 1
    gpu_ids = [None]  # Will run on CPU
    logger.info("No GPUs found, running on CPU")


def run_single_trial(run_idx, total_runs, values, keys, base_config, sweep_dir, wandb_project, args, gpu_id):
    """Run a single trial with a specific GPU assignment."""
    overrides = {k: v for k, v in zip(keys, values)}
    merged_config = {**base_config, **overrides}
    
    run_name = "_".join([f"{k}{v}" for k, v in overrides.items()])
    log_msg = f"Run {run_idx}/{total_runs} (GPU {gpu_id if gpu_id is not None else 'CPU'}) — overrides: {overrides}"
    logger.info(log_msg)
    
    if args.model_type == 'scat':
        file = "scripts/train.py"
    elif args.model_type in ['wph', 'wph_hypernetwork']:
        file = "scripts/train_wph.py"
    elif args.model_type == 'wph_pca':
        file = "scripts/train_wph_pca.py"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Save merged config to a temporary file in the main sweep dir
    temp_config_path = os.path.join(sweep_dir, f"temp_config_{run_idx}.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(merged_config, f)

    cmd = [
        "python", file,
        "--config", temp_config_path,
        "--sweep_dir", sweep_dir
    ]
    if wandb_project is not None:
        cmd.extend(["--wandb_project", wandb_project])
    
    # Set up environment with GPU assignment and export GPU id for run suffix
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["SWEEP_GPU_ID"] = str(gpu_id)
    else:
        env["SWEEP_GPU_ID"] = "cpu"
    
    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            logger.info(f"Run {run_idx} (GPU {gpu_id if gpu_id is not None else 'CPU'}) finished successfully")
            return {"run_idx": run_idx, "success": True, "gpu_id": gpu_id}
        else:
            logger.error(f"Run {run_idx} (GPU {gpu_id if gpu_id is not None else 'CPU'}) failed with return code {result.returncode}")
            return {"run_idx": run_idx, "success": False, "gpu_id": gpu_id, "returncode": result.returncode}
    except Exception as e:
        logger.error(f"Run {run_idx} (GPU {gpu_id if gpu_id is not None else 'CPU'}) failed with exception: {e}")
        return {"run_idx": run_idx, "success": False, "gpu_id": gpu_id, "exception": str(e)}


# All parameter names
expanded_values = [expand_param(v) for v in params.values()]
keys = list(params.keys())
all_combinations = list(itertools.product(*expanded_values))

# Filter combinations by fold if specified
if args.fold_filter is not None and 'fold' in keys:
    fold_idx = keys.index('fold')
    all_combinations = [combo for combo in all_combinations if combo[fold_idx] == args.fold_filter]
    logger.info(f"Filtered to fold={args.fold_filter}: {len(all_combinations)} combinations")

total_runs = len(all_combinations)

logger.info(f"Starting {total_runs} runs in parallel across {num_gpus} GPU(s)")

# Run trials in parallel
results = []
with ThreadPoolExecutor(max_workers=num_gpus) as executor:
    # Submit all jobs
    futures = []
    for run_idx, values in enumerate(all_combinations, 1):
        # Assign GPU in round-robin fashion
        gpu_id = gpu_ids[(run_idx - 1) % num_gpus]
        future = executor.submit(
            run_single_trial,
            run_idx, total_runs, values, keys, base_config,
            sweep_dir, wandb_project, args, gpu_id
        )
        futures.append(future)
        # Sleep briefly to ensure unique timestamps for each trial
        time.sleep(1.5 * gpu_id if gpu_id is not None else 0.1)
    
    # Wait for all jobs to complete
    for future in as_completed(futures):
        result = future.result()
        results.append(result)

# Summary of results
successful_runs = sum(1 for r in results if r["success"])
failed_runs = len(results) - successful_runs
logger.info(f"All runs finished. Successful: {successful_runs}/{total_runs}, Failed: {failed_runs}/{total_runs}")
if failed_runs > 0:
    failed_indices = [r["run_idx"] for r in results if not r["success"]]
    logger.warning(f"Failed run indices: {failed_indices}")

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
df.to_csv(os.path.join(results_path, "all_runs.csv"), index=False)
logger.info(f"Saved all runs to {os.path.join(results_path, 'all_runs.csv')}")
logger.send_file("all_runs_data", os.path.join(results_path, "all_runs.csv"), "table")

# Step 1: Plot validation accuracy per run per train_ratio, sorted decreasingly
for train_ratio, group in df.groupby("train_ratio"):
    # Sort runs by metric descending
    group_sorted = group.sort_values(args.metric, ascending=False)
    
    plt.figure(figsize=(8, 5))
    plt.title(f"Validation Error per Run (train_ratio={train_ratio})")
    plt.bar(group_sorted["run"], 1.0 - group_sorted[args.metric], color="skyblue")
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
df = df.sort_values(["train_ratio", metric], ascending=[True, False])

# Group by val_ratio and keep top N runs
top_runs = df.groupby("train_ratio").head(topN)

topN_csv_path = os.path.join(results_path, f"top{topN}_runs_per_train_ratio.csv")
top_runs.to_csv(topN_csv_path, index=False)
logger.info(f"Saved top-{topN} runs per train_ratio to {topN_csv_path}")

# Create a label for x-axis: "val_ratio-rank" only
labels = []
heights = []
for train_ratio, group in top_runs.groupby("train_ratio"):
    # sort by val_acc descending within this val_ratio
    group_sorted = group.sort_values(args.metric, ascending=False).reset_index()
    for rank, row in enumerate(group_sorted.itertuples(), 1):
        labels.append(f"{train_ratio:.2f}--{rank}")
        heights.append(1.0 - getattr(row, args.test_metric))  # or row.test_acc if that's the column name

# Plot
plt.figure(figsize=(max(10, len(labels)*0.6), 6))
plt.bar(range(len(heights)), heights, color="skyblue")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("train_ratio | rank")
plt.ylabel(f"Test Error(1 - {args.test_metric})")
plt.title(f"Test Error of Top {topN} Runs per train_ratio")
plt.tight_layout()
plot_path = os.path.join(results_path, f"top{topN}_test_err_all_train_ratios.png")
plt.savefig(plot_path)
plt.close()
logger.send_file("topN_test_err_plot", plot_path, "image")
logger.info(f"Saved top-{topN} test error plot across all train_ratios to {results_path}")
logger.finish()