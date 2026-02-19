import os
import argparse
import pandas as pd
from configs import load_config
from dense.helpers import LoggerManager

def parse_args():
    parser = argparse.ArgumentParser(description="Clean up sweep directory by keeping top models.")
    parser.add_argument(
        "--sweep_dir", type=str, required=True,
        help="Path to the sweep directory (e.g. experiments/mnist-sweeps-20251120-123456)"
    )
    parser.add_argument('--dataset-dir', action='store_true', help="If set, loop through all sweep directories in the specified dataset directory instead of a single sweep directory.")
    parser.add_argument(
        "--top_n", type=int, default=3,
        help="Number of top models to keep per val_ratio (default: 3)"
    )
    parser.add_argument(
        "--metric", type=str, default="last_val_acc",
        help="Metric to evaluate top models (default: last_val_acc)"
    )
    parser.add_argument('--val-ratio', action='store_true',
                        help='If set, group by val_ratio to determine top models, otherwise use train_ratio')
    return parser.parse_args()

def cleanup_sweep_dir(args, sweep_dir, col='train_ratio'):
    top_n = args.top_n
    use_val_ratio = args.val_ratio
    # Initialize logger
    logger = LoggerManager.get_logger(log_dir=sweep_dir)
    logger.info("Starting cleanup of sweep directory.")

    # Collect all results
    rows = []
    for ratio_folder in os.listdir(sweep_dir):
        ratio_path = os.path.join(sweep_dir, ratio_folder)
        if not os.path.isdir(ratio_path) or not ratio_folder.startswith(f"{col}="):
            continue

        ratio_value = float(ratio_folder.split("=")[1])  # Extract the number
        for run_folder in os.listdir(ratio_path):
            run_path = os.path.join(ratio_path, run_folder)
            config_path = os.path.join(run_path, "config.yaml")
            if not os.path.isfile(config_path):
                continue

            config = load_config(config_path)
            config["run"] = run_folder
            config[col] = ratio_value
            rows.append(config)

    # Create a DataFrame of results
    df = pd.DataFrame(rows)

    # Sort by val_ratio and the specified metric
    metric = args.metric
    df = df.sort_values([col, metric], ascending=[True, False])

    # Group by val_ratio and keep top N runs
    top_runs = df.groupby(col).head(top_n)

    # Identify runs to keep
    runs_to_keep = set(top_runs["run"])

    # Delete models not in the top N
    for ratio_folder in os.listdir(sweep_dir):
        ratio_path = os.path.join(sweep_dir, ratio_folder)
        if not os.path.isdir(ratio_path) or not ratio_folder.startswith(f"{col}="):
            continue

        for run_folder in os.listdir(ratio_path):
            run_path = os.path.join(ratio_path, run_folder)
            if run_folder not in runs_to_keep:
                # Delete all model files in this run folder
                for file_name in os.listdir(run_path):
                    if file_name.endswith(".pt"):
                        file_path = os.path.join(run_path, file_name)
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")

    logger.info("Cleanup complete. Top models retained.")

def main():
    args = parse_args()
    if args.dataset_dir:
        # Loop through all sweep directories in the specified dataset directory
        dataset_dir = args.sweep_dir
        for sweep_dir in os.listdir(dataset_dir):
            sweep_path = os.path.join(dataset_dir, sweep_dir)
            if os.path.isdir(sweep_path):
                print(f"Cleaning sweep dir: {sweep_path}")
                try:
                    cleanup_sweep_dir(args, sweep_path, 'train_ratio')
                except KeyError as e:
                    print(f"KeyError for {sweep_path}: {e}. Trying with 'val_ratio' instead.")
                    cleanup_sweep_dir(args, sweep_path, 'val_ratio')
    else:
        # Clean up a single specified sweep directory
        cleanup_sweep_dir(args, sweep_dir = args.sweep_dir, col = 'val_ratio' if args.val_ratio else 'train_ratio')


if __name__ == "__main__":
    main()


"""
To run this for every sweep directory in a folder,
run in terminal:

for d in experiments/curet-dataset/*; do
  echo "Cleaning sweep dir: $d"
  PYTHONPATH=. python scripts/cleanup_checkpoints.py --sweep_dir "$d" --top_n 3 --metric feature_extractor_last_acc
done
"""