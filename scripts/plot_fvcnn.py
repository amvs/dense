import os
import argparse
from configs import load_config
from dense.helpers.logger import LoggerManager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_sweep_dir(sweep_dir: str) -> pd.DataFrame:
    all_dfs = []
    
    # First, look for all_runs.csv files
    for root, _, files in os.walk(sweep_dir):
        if "all_runs.csv" in files:
            file_path = os.path.join(root, "all_runs.csv")
            print(f"Found all_runs.csv: {file_path}")
            df = pd.read_csv(file_path)
            all_dfs.append(df)
    
    # Only search config.yaml files if no all_runs.csv was found
    if not all_dfs:
        print(f"No all_runs.csv found in {sweep_dir}, reading config.yaml files...")
        config_data = []
        for root, _, files in os.walk(sweep_dir):
            parts = root.split('/')
            if len(parts) >= 2 and parts[-2].startswith('train_ratio') and parts[-1].startswith('run'):
                if 'config.yaml' in files:
                    file_path = os.path.join(root, 'config.yaml')
                    print(f"Reading config: {file_path}")
                    exp = load_config(file_path)
                    config_data.append(exp)
        if config_data:
            df = pd.DataFrame.from_records(config_data)
            all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()
    
def read_exp_dir(exp_dir):
    datasets = os.listdir(exp_dir)
    all_dfs = []
    for d in datasets:
        if not os.path.isdir(os.path.join(exp_dir, d)):
            continue
        sweep_dirs = os.listdir(os.path.join(exp_dir, d))
        sweep_dirs = [sd for sd in sweep_dirs if '-sweeps-' in sd]
        for sd in sweep_dirs:
            sweep_path = os.path.join(exp_dir, d, sd)
            df = read_sweep_dir(sweep_path)
            if not df.empty:
                df['dataset'] = d
                all_dfs.append(df)
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    return(df)

def make_tables(df):
    # average across folds
    val_acc = df.pivot(index = ['dataset', 'train_ratio'], columns = ['framework', 'backbone'], values = 'val_acc')
    val_acc.to_markdown(os.path.join(args.dir, "val_acc_table.md"))
    test_acc = df.pivot(index = ['dataset', 'train_ratio'], columns = ['framework', 'backbone'], values = 'test_acc')
    test_acc.to_markdown(os.path.join(args.dir, "test_acc_table.md"))

def plot_dataset(df):
    df['generalization_error'] = df['train_acc'] - df['test_acc']
    fix, ax = plt.subplots(nrows = 3, figsize=(8,18))
    sns.scatterplot(data=df, x='train_ratio', y='val_acc', style='framework', hue='backbone', ax=ax[0])
    ax[0].set_title('Validation Accuracy')
    sns.scatterplot(data=df, x='train_ratio', y='test_acc', style='framework', hue='backbone', ax=ax[1])
    ax[1].set_title('Test Accuracy')
    sns.scatterplot(data=df, x='train_ratio', y='generalization_error', style='framework', hue='backbone', ax=ax[2])
    ax[2].set_title('Generalization Error')
    plt.savefig(os.path.join(args.dir, f"val_test_acc_{df['dataset'].iloc[0]}.png"))
    plt.close()

    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Plot FVCNN results")
    arg_parser.add_argument("--dir", type=str, required=True, help="Directory containing results",
                            default="/projects/standard/lermang/vonse006/wph_collab/dense/experiments_fvcnn")
    args = arg_parser.parse_args()

    df = read_exp_dir(args.dir)
    df.to_csv(os.path.join(args.dir, "combined_results.csv"), index=False)
    # average results across folds - needed for kthtips and outex12 datasets
    mean_df = df.groupby(['dataset','train_ratio', 'framework', 'backbone'], as_index=False)[['val_acc', 'train_acc', 'test_acc']].mean()
    mean_df.to_csv(os.path.join(args.dir, "mean_results.csv"), index=False)
    make_tables(mean_df)

    for dataset in df['dataset'].unique():
        plot_dataset(df[df['dataset'] == dataset].reset_index(drop=True))
