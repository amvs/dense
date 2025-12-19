import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config, expand_param
import pandas as pd
import matplotlib.pyplot as plt
import dotenv

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
    parser.add_argument('--model-type', type=str, choices=['wph', 'scat'], default='scat',
                        help='Type of model to train (default: scat (dense))')
    parser.add_argument('--name', type=str, default='',
                        help='Optional short name for the sweep folder')
    parser.add_argument('--metric', type=str, default='last_val_acc',
                        help='Metric to evaluate top models (default: last_val_acc)')
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

random_flag = args.random
if random_flag:
    script = "scripts/train_random.py"
else:
    script = "scripts/train.py"


# init logger
wandb_project = args.wandb_project
sweep_name = os.path.splitext(os.path.basename(args.config))[0]
logger = LoggerManager.get_logger(log_dir=sweep_dir, wandb_project=wandb_project, name=f"{sweep_name}-{timestamp}")


# All parameter names
expanded_values = [expand_param(v) for v in params.values()]
keys = list(params.keys())
all_combinations = list(itertools.product(*expanded_values))
total_runs = len(all_combinations)

# All combinations of values
for run_idx, values in enumerate(all_combinations, 1):
    overrides = {k: v for k, v in zip(keys, values)}
    merged_config = {**base_config, **overrides}  # Merge base config with overrides

    # Debug: Log merged_config to verify content
    logger.info(f"Merged config for run {run_idx}: {merged_config}")

    # Optional: create a run name from param values
    run_name = "_".join([f"{k}{v}" for k, v in overrides.items()])
    logger.info(f"Run {run_idx}/{total_runs} — overrides: {overrides}")

    if args.model_type == 'scat':
        file = "scripts/train.py"
    elif args.model_type == 'wph':
        file = "scripts/train_wph.py"

    # Save merged config to a temporary file
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
    result = subprocess.run(cmd)
    if result.returncode == 0:
        logger.info(f"Finished run.")
    else:
        logger.error(f"Run failed with return code {result.returncode}")
logger.info("All runs finished.")

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
    group_sorted = group.sort_values(args.metric, ascending=False).reset_index()
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