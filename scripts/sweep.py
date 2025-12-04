import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config, expand_param
import pandas as pd
import matplotlib.pyplot as plt
import wandb
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist_sweep.yaml)"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="DenseWavelet",
        help="Weights and Biases project name for logging")
    parser.add_argument(
        "--random", action="store_true",
        help="If set, use random filters"
    )
    return parser.parse_args()

# Create output folder
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sweep_dir = os.path.join("experiments", f"sweeps-{timestamp}")
os.makedirs(sweep_dir, exist_ok=True)

# Read config
args = parse_args()
sweep = load_config(args.config)
base = sweep["base_config"]
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
    overrides = [f"{k}={v}" for k, v in zip(keys, values)]
    logger.log(f"Run {run_idx}/{total_runs} — overrides: {overrides}")
    result = subprocess.run(
        ["python", script, 
            "--config", base, 
            "--wandb_project", wandb_project, 
            "--sweep_dir", sweep_dir, 
            "--override"
        ] + overrides)
    if result.returncode == 0:
        logger.log(f"Finished run.")
    else:
        logger.error(f"Run failed with return code {result.returncode}")
logger.log("All runs finished.")

logger.log("Analyzing results...")
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
logger.log(f"df head:\n {df.head()}")
df.to_csv(os.path.join(results_path, "all_runs.csv"), index=False)
logger.log(f"Saved all runs to {os.path.join(results_path, 'all_runs.csv')}")
logger.send_file("all_runs_data", os.path.join(results_path, "all_runs.csv"), "table")

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
logger.log(f"Saved results summary plot to {results_path}")

topN = 3
# Step 2: take top-N runs per train_ratio by validation accuracy
topN_runs = df.sort_values(["train_ratio", "last_val_acc"], ascending=[True, False])\
               .groupby("train_ratio").head(topN)
topN_csv_path = os.path.join(results_path, f"top{topN}_runs_per_train_ratio.csv")
topN_runs.to_csv(topN_csv_path, index=False)
logger.log(f"Saved top-{topN} runs per train_ratio to {topN_csv_path}")
logger.send_file(f"top{topN}_runs_data", topN_csv_path, "table")
# Create a label for x-axis: "train_ratio-rank" only
labels = []
heights = []
for train_ratio, group in topN_runs.groupby("train_ratio"):
    # sort by val_acc descending within this train_ratio
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
logger.log(f"Saved top-{topN} test accuracy plot across all train_ratios to {results_path}")
logger.finish()