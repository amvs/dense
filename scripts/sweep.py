import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with config")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/mnist_sweep.yaml)"
    )
    return parser.parse_args()

# Create output folder
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
sweep_dir = os.path.join("experiments", f"sweeps-{timestamp}")
os.makedirs(sweep_dir, exist_ok=True)

# init logger
logger = LoggerManager.get_logger(log_dir=sweep_dir)
logger.info("Start log:")

# Read config
args = parse_args()
sweep = load_config(args.config)


base = sweep["base_config"]
params = sweep["sweep"]  # dict of lists

# All parameter names
keys = list(params.keys())
all_combinations = list(itertools.product(*params.values()))
total_runs = len(all_combinations)
# All combinations of values
for run_idx, values in enumerate(all_combinations, 1):
    overrides = [f"{k}={v}" for k, v in zip(keys, values)]
    # Optional: create a run name from param values
    run_name = "_".join([f"{k}{v}" for k, v in zip(keys, values)])
    logger.info(f"Run {run_idx}/{total_runs} — overrides: {overrides}")
    result = subprocess.run(["python", "scripts/train.py", "--config", base, "--sweep_dir", sweep_dir, "--override"] + overrides)
    if result.returncode == 0:
        logger.info(f"Finished run.")
    else:
        logger.error(f"Run failed with return code {result.returncode}")
logger.info("All runs finished.")

logger.info("Analyzing results...")
rows = []
for ratio_folder in os.listdir(sweep_dir):
    ratio_path = os.path.join(sweep_dir, ratio_folder)
    if not os.path.isdir(ratio_path) or not ratio_folder.startswith("val_ratio="):
        continue

    val_ratio = float(ratio_folder.split("=")[1])  # extract the number
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
logger.info("df head:\n%s", df.head())
df.to_csv(os.path.join(results_path, "all_runs.csv"), index=False)
logger.info(f"Saved all runs to {os.path.join(results_path, 'all_runs.csv')}")


# Step 1: Plot validation accuracy per run per val_ratio, sorted decreasingly
for val_ratio, group in df.groupby("val_ratio"):
    # Sort runs by last_val_acc descending
    group_sorted = group.sort_values("last_val_acc", ascending=False)
    
    plt.figure(figsize=(8, 5))
    plt.title(f"Validation Accuracy per Run (val_ratio={val_ratio})")
    plt.bar(group_sorted["run"], group_sorted["last_val_acc"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"val_acc_val_ratio_{val_ratio}.png"))
    plt.close()
logger.info(f"Saved results summary plot to {results_path}")

topN = 3
# Step 2: take top-N runs per val_ratio by validation accuracy
topN_runs = df.sort_values(["val_ratio", "last_val_acc"], ascending=[True, False])\
               .groupby("val_ratio").head(topN)
topN_csv_path = os.path.join(results_path, f"top{topN}_runs_per_val_ratio.csv")
topN_runs.to_csv(topN_csv_path, index=False)
logger.info(f"Saved top-{topN} runs per val_ratio to {topN_csv_path}")

# Create a label for x-axis: "val_ratio-rank" only
labels = []
heights = []
for val_ratio, group in topN_runs.groupby("val_ratio"):
    # sort by val_acc descending within this val_ratio
    group_sorted = group.sort_values("last_val_acc", ascending=False).reset_index()
    for rank, row in enumerate(group_sorted.itertuples(), 1):
        labels.append(f"{val_ratio:.2f}--{rank}")
        heights.append(row.test_acc)  # or row.test_acc if that's the column name

# Plot
plt.figure(figsize=(max(10, len(labels)*0.6), 6))
plt.bar(range(len(heights)), heights, color="skyblue")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.xlabel("val_ratio | rank")
plt.ylabel("Test Accuracy")
plt.title(f"Test Accuracy of Top {topN} Runs per val_ratio")
plt.tight_layout()
plt.savefig(os.path.join(results_path, f"top{topN}_test_acc_all_val_ratios.png"))
plt.close()

logger.info(f"Saved top-{topN} test accuracy plot across all val_ratios to {results_path}")