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
# All combinations of values
for values in itertools.product(*params.values()):
    overrides = [f"{k}={v}" for k, v in zip(keys, values)]
    # Optional: create a run name from param values
    run_name = "_".join([f"{k}{v}" for k, v in zip(keys, values)])
    logger.info(f"Starting run with overrides: {overrides}...")
    subprocess.run(["python", "scripts/train.py", "--config", base, "--sweep_dir", sweep_dir, "--override"] + overrides)
    logger.info(f"Finished run.")

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
        config["train_ratio"] = train_ratio
        config["run"] = run_folder
        rows.append(config)

df = pd.DataFrame(rows)
print(df.head())

for tr, group in df.groupby("train_ratio"):
    plt.figure(figsize=(8, 5))
    plt.title(f"Validation Accuracy per Run (train_ratio={tr})")
    plt.bar(group["run"], group["last_val_acc"], color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Validation Accuracy")
    plt.tight_layout()
    plt.show()

evaluate_config = sweep["evaluate_config"]
