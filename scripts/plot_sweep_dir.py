import os
import argparse
from configs import load_config
from dense.helpers.logger import LoggerManager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def combine_dfs(dir_name, pattern="*.csv"):
    dfs = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            if file.endswith(pattern):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    df['source_file'] = file_path
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Failed to read {file_path}: {e}")
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_path = os.path.join(root, f"combined_{dir_name.split('.')[0]}.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined CSV saved to {combined_path}")

def df_from_logs(args):
    sweep_dir = args.sweep_dir
    rows = []
    logger = LoggerManager.get_logger()
    for ratio_folder in os.listdir(sweep_dir):
        ratio_path = os.path.join(sweep_dir, ratio_folder)
        if not os.path.isdir(ratio_path) or not ratio_folder.startswith("train_ratio="):
            continue

        train_ratio = float(ratio_folder.split("=")[1])  # extract the number
        logger.info(f"Processing train_ratio={train_ratio}...")
        for run_folder in os.listdir(ratio_path):
            run_path = os.path.join(ratio_path, run_folder)
            config_path = os.path.join(run_path, "config.yaml")
            if not os.path.isfile(config_path):
                continue

            config = load_config(config_path)
            config["run"] = run_folder
            config["run_path"] = run_path
            rows.append(config)

    results_path = os.path.join(sweep_dir, "results")
    os.makedirs(results_path, exist_ok=True)

    df = pd.DataFrame(rows)
    if ('model_type' in df.columns) and (df.model_type.unique()[0] == "wph_pca"):
        df["classifier_test_acc"] = df["pca_test_acc"]
    df["finetuning_gain"] = df["feature_extractor_test_acc"] - df["classifier_test_acc"]
    df.to_csv(os.path.join(results_path, "sweep_results.csv"), index=False)
    return df


def add_intermediate_l2_norms(df, sweep_dir):
    """
    Read intermediate l2 norms from accuracy.csv files in each run folder
    and augment the dataframe with rows for each intermediate checkpoint.
    
    Args:
        df: DataFrame with run results
        sweep_dir: Path to sweep directory
        
    Returns:
        Augmented DataFrame with intermediate l2 norm rows
    """
    # Store original rows and new intermediate rows
    new_rows = []
    
    for idx, row in df.iterrows():
        run_path = row.get("run_path")
        if not run_path or not os.path.isdir(run_path):
            continue
        
        accuracy_path = os.path.join(run_path, "accuracy.csv")
        if not os.path.isfile(accuracy_path):
            continue
        
        try:
            acc_df = pd.read_csv(accuracy_path)
            # Filter for intermediate phase: l2_norm > 0
            intermediate_rows = acc_df[acc_df["l2_norm"] > 0].copy()
            
            if intermediate_rows.empty:
                continue
            
            # For each intermediate checkpoint, create a row with updated metrics
            for _, acc_row in intermediate_rows.iterrows():
                new_row = row.copy()
                # Use the validation and test accuracies from this epoch
                new_row["feature_extractor_test_acc"] = acc_row["test_acc"]
                new_row["feature_extractor_best_acc"] = acc_row["val_acc"]
                new_row["l2_norm_finetuning"] = acc_row["l2_norm"]
                new_row["_intermediate"] = True
                new_rows.append(new_row)
        except Exception as e:
            print(f"Warning: Failed to read accuracy.csv from {run_path}: {e}")
            continue
    
    # Combine original and intermediate rows
    if new_rows:
        intermediate_df = pd.DataFrame(new_rows)
        df_combined = pd.concat([df, intermediate_df], ignore_index=True)
        return df_combined
    return df


def boxplot_metric_vs_ratio(
    df,
    metric,
    group_cols=["train_ratio", "random_filters"],
    sweep_dir=None,
    fname_suffix="",
):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    df.boxplot(column=metric, by=group_cols, ax=ax)
    plt.title(f"Boxplot of {metric} vs Train Ratio")
    plt.suptitle("")
    plt.ylabel(metric)
    plt.grid(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    results_path = os.path.join(sweep_dir, "results")
    plot_path = os.path.join(
        results_path, f"boxplot_{metric}_vs_train_ratio_{fname_suffix}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    print(f"Boxplot saved to {plot_path}")


def plot_all_boxplots(
    df, group_cols=["train_ratio", "random_filters"], sweep_dir=None, fname_suffix=""
):
    metrics = ["feature_extractor_test_acc", "classifier_test_acc", "finetuning_gain"]
    for metric in metrics:
        try:
            boxplot_metric_vs_ratio(df, metric, group_cols, sweep_dir, fname_suffix)
        except KeyError as e:
            print(f"Error plotting {metric}: {e}")


def pair_boxplots(
    df,
    metric,
    pair_col="random_filters",
    group_cols=["lambda_reg", "random_filters", "train_ratio"],
    sweep_dir=None,
    fname_suffix="",
    logy=False
):
    """
    Creates paired boxplots comparing two conditions:
    downsampled vs fullsize images.
    """
    pair_true = df[df[pair_col] == True].reset_index(drop=True)
    pair_false = df[df[pair_col] == False].reset_index(drop=True)
    if pair_true.empty or pair_false.empty:
        print(f"Skipping pair_boxplots for {metric} as one condition is empty.")
        return
    fig, axs = plt.subplots(figsize=(10, 12), nrows=2, ncols=1)
    pair_true.boxplot(column=metric, by=group_cols, ax=axs[0])
    axs[0].set_title(f"Boxplot of {metric} ({pair_col}=True)")
    axs[0].set_ylabel(metric)
    axs[0].tick_params(axis="x", rotation=90)
    pair_false.boxplot(column=metric, by=group_cols, ax=axs[1])
    axs[1].set_title(f"Boxplot of {metric} ({pair_col}=False)")
    axs[1].set_ylabel(metric)
    axs[1].tick_params(axis="x", rotation=90)
    if logy:
        axs[0].set_yscale("log")
        axs[1].set_yscale("log")
    plt.suptitle("")
    plt.tight_layout()
    results_path = os.path.join(sweep_dir, "results")
    plot_path = os.path.join(results_path, f"pair_boxplot_{metric}_{fname_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Paired boxplot saved to {plot_path}")

def side_by_side_boxplots(
    df,
    metrics,
    group_cols=["lambda_reg", "train_ratio", "random_filters"],
    facet_col='train_ratio',
    sweep_dir=None,
    fname_suffix="",
):
    assert facet_col in group_cols, f"facet_col must be in group_cols but {facet_col} not in {group_cols}"
    df_melted = df.melt(id_vars=group_cols, value_vars=metrics, var_name="metric", value_name="value")
    set_cols = set(group_cols)
    set_cols.remove(facet_col)
    df_melted['params'] = df_melted[list(set_cols)].astype(str).agg(','.join, axis=1)
    g = sns.catplot(data = df_melted, x = 'params',y = 'value', hue='metric', col =facet_col, kind = 'strip', col_wrap=3, sharey=True, sharex=False)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'results', f'side_by_side_boxplots_{fname_suffix}.png'))
    plt.close()
    print(f"Side by side boxplots saved to {os.path.join(sweep_dir, 'results', f'side_by_side_boxplots_{fname_suffix}.png')}")


def slope_plot(df, group_cols, metric, sweep_dir, fname_suffix=""):
    calc_slope = lambda df: np.polyfit(df["train_ratio"], df[metric], 1)[0]
    # calc_intercept = lambda df: np.polyfit(df["train_ratio"], df[metric], 1)[1]
    df_grouped = (
        df.groupby(group_cols, as_index=True)
        .apply(calc_slope)
        .reset_index(name="slope")
    )
    df_grouped.plot.scatter(x="lambda_reg", y="slope")
    sns.scatterplot(
        df_grouped, x="lambda_reg", y="slope", hue="random_filters", alpha=0.7
    )
    plt.xscale("log")
    plt.title(f"Slope of {metric} vs Lambda_reg")
    plt.xlabel("Lambda_reg (log scale)")
    plt.ylabel(f"Slope of {metric} vs Train Ratio")
    plt.grid(True)
    results_path = os.path.join(sweep_dir, "results")
    plot_path = os.path.join(results_path, f"slope_plot_{metric}_{fname_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Slope plot saved to {plot_path}")


def plot_generalization_error(df, sweep_dir, fname_suffix="", use_best=True):
    """
    Plot generalization error vs sample size and L2 norm.
    
    Generalization error = validation accuracy - test accuracy
    
    Args:
        df: DataFrame with experiment results
        sweep_dir: Path to sweep directory
        fname_suffix: Suffix for output filename
        use_best: If True, use best_acc; if False, use last_acc
    """
    results_path = os.path.join(sweep_dir, "results")
    
    # Calculate generalization errors
    acc_suffix = "best_acc" if use_best else "last_acc"
    
    # Classifier generalization error
    if f"classifier_{acc_suffix}" in df.columns and "classifier_test_acc" in df.columns:
        df["classifier_gen_error"] = df[f"classifier_{acc_suffix}"] - df["classifier_test_acc"]
    
    # Feature extractor generalization error
    if f"feature_extractor_{acc_suffix}" in df.columns and "feature_extractor_test_acc" in df.columns:
        df["feature_extractor_gen_error"] = df[f"feature_extractor_{acc_suffix}"] - df["feature_extractor_test_acc"]
    
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Classifier generalization error vs train_ratio
    if "classifier_gen_error" in df.columns:
        ax = axes[0]
        for key, grp in df.groupby("random_filters"):
            label = f"random_filters={key}"
            ax.scatter(grp["train_ratio"], grp["classifier_gen_error"], label=label, alpha=0.6)
        ax.set_xlabel("Train Ratio")
        ax.set_ylabel("Generalization Error")
        ax.set_title(f"Classifier Generalization Error vs Sample Size ({acc_suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Feature extractor generalization error vs train_ratio
    if "feature_extractor_gen_error" in df.columns:
        ax = axes[1]
        for key, grp in df.groupby("random_filters"):
            label = f"random_filters={key}"
            ax.scatter(grp["train_ratio"], grp["feature_extractor_gen_error"], label=label, alpha=0.6)
        ax.set_xlabel("Train Ratio")
        ax.set_ylabel("Generalization Error")
        ax.set_title(f"Feature Extractor Generalization Error vs Sample Size ({acc_suffix})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(results_path, f"generalization_error_{acc_suffix}_{fname_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Generalization error plot saved to {plot_path}")


def plot_generalization_error_vs_l2(df, sweep_dir, fname_suffix="", use_best=True):
    """
    Plot feature extractor generalization error vs L2 norm, separated by train_ratio.
    
    Args:
        df: DataFrame with experiment results
        sweep_dir: Path to sweep directory
        fname_suffix: Suffix for output filename
        use_best: If True, use best_acc; if False, use last_acc
    """
    results_path = os.path.join(sweep_dir, "results")
    
    # Calculate generalization errors
    acc_suffix = "best_acc" if use_best else "last_acc"
    
    # Feature extractor generalization error
    if f"feature_extractor_{acc_suffix}" in df.columns and "feature_extractor_test_acc" in df.columns:
        df["feature_extractor_gen_error"] = df[f"feature_extractor_{acc_suffix}"] - df["feature_extractor_test_acc"]
    
    if "feature_extractor_gen_error" not in df.columns or "l2_norm_finetuning" not in df.columns:
        return
    
    df_with_l2 = df[df["l2_norm_finetuning"] > 0].copy()
    if df_with_l2.empty:
        return
    
    # Define markers for random_filters and colors for train_ratio
    random_filters_values = sorted(df_with_l2["random_filters"].unique())
    markers = {val: marker for val, marker in zip(random_filters_values, ['o', 's'][:len(random_filters_values)])}
    
    train_ratio_values = sorted(df_with_l2["train_ratio"].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(train_ratio_values)))
    color_map = {val: colors[i] for i, val in enumerate(train_ratio_values)}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for train_ratio_val, train_ratio_grp in df_with_l2.groupby("train_ratio"):
        for rf_val, rf_grp in train_ratio_grp.groupby("random_filters"):
            label = f"train_ratio={train_ratio_val}, random_filters={rf_val}"
            ax.scatter(
                rf_grp["l2_norm_finetuning"],
                rf_grp["feature_extractor_gen_error"],
                label=label,
                alpha=0.6,
                marker=markers[rf_val],
                color=color_map[train_ratio_val],
                s=80
            )
    
    ax.set_xlabel("L2 Norm (Fine-tuning)")
    ax.set_ylabel("Generalization Error")
    ax.set_title(f"Feature Extractor Generalization Error vs L2 Norm ({acc_suffix})")
    ax.set_xscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(results_path, f"generalization_error_vs_l2_{acc_suffix}_{fname_suffix}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Generalization error vs L2 plot saved to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep results.")
    parser.add_argument(
        "--sweep_dir",
        type=str,
        required=True,
        help="Path to the sweep directory (e.g. experiments/mnist-sweeps-20251120-123456)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="feature_extractor_best_acc",
        help="Metric to select best run (default: feature_extractor_best_acc)",
    )
    parser.add_argument(
        "--split_by",
        type=str,
        default=None,
        help="Column name to split dataset and generate separate plots for each unique value (e.g. 'S', 'max_scale'). If not provided, generates plots for entire dataset.",
    )
    args = parser.parse_args()
    df = df_from_logs(args)
    # df['wavelet_params'] = df['wavelet_params'].apply(literal_eval)
    df = df.join(pd.json_normalize(df['wavelet_params']))
    
    # Add intermediate l2 norms from accuracy.csv files
    df = add_intermediate_l2_norms(df, args.sweep_dir)
    
    df_lr1 = df.loc[df.lambda_reg == 1].reset_index(drop=True)
    df_lr1_downsample = df_lr1.loc[df_lr1.downsample].reset_index(drop=True)
    
    # Split dataset by specified column and generate separate plots
    split_values = [None]  # Default: no split, process entire dataset
    if args.split_by and args.split_by in df.columns:
        split_values = sorted(df[args.split_by].dropna().unique())
        print(f"Splitting plots by '{args.split_by}' with values: {split_values}")
    
    for split_val in split_values:
        if split_val is None:
            df_split = df
            suffix = "all"
        else:
            df_split = df.loc[df[args.split_by] == split_val].reset_index(drop=True)
            if df_split.empty:
                continue
            suffix = f"{args.split_by}={split_val}"
        
        plot_all_boxplots(
            df_split,
            group_cols=['max_scale', 'train_ratio', 'random_filters'],
            sweep_dir=args.sweep_dir,
            fname_suffix=suffix,
        )
    # df_lr1_fullsize = df_lr1.loc[~df_lr1.downsample].reset_index(drop=True)
    plot_all_boxplots(df, group_cols = [args.split_by, 'train_ratio','lambda_reg', 'random_filters'],
                    sweep_dir=args.sweep_dir,
                    fname_suffix = 'all')
    # plot_all_boxplots(df_lr1_fullsize, group_cols = ['max_scale', 'train_ratio', 'random_filters'],
    #                   sweep_dir=args.sweep_dir,
    #                   fname_suffix = 'lreg=1_fullsize')
    df_downsample = df.loc[df.downsample].reset_index(drop=True)
    # df_fullsize = df.loc[~df.downsample].reset_index(drop=True)
    pair_boxplots(
        df,
        metric="feature_extractor_test_acc",
        pair_col="random_filters",
        group_cols=["lambda_reg", "random_filters", "train_ratio"],
        sweep_dir=args.sweep_dir,
        fname_suffix="downsample",
    )
    pair_boxplots(
        df,
        metric="finetuning_gain",
        pair_col="random_filters",
        group_cols=["lambda_reg", "random_filters", "train_ratio"],
        sweep_dir=args.sweep_dir,
        fname_suffix="downsample",
        logy=True
    )
    # pair_boxplots(
    #     df_fullsize,
    #     metric="feature_extractor_test_acc",
    #     pair_col="random_filters",
    #     group_cols=["lambda_reg", "random_filters", "train_ratio"],
    #     sweep_dir=args.sweep_dir,
    #     fname_suffix="fullsize",
    # )

    # slope_plot(
    #     df_fullsize,
    #     group_cols=slope_group_cols,
    #     metric="feature_extractor_test_acc",
    #     sweep_dir=args.sweep_dir,
    #     fname_suffix="fullsize",
    # )
    side_by_side_boxplots(
        df,
        metrics=['feature_extractor_test_acc', 'classifier_test_acc'],
        group_cols=['lambda_reg', 'train_ratio', 'random_filters'],
        facet_col='train_ratio',
        sweep_dir=args.sweep_dir,
        fname_suffix='downsample',
    )

    # Plot generalization errors
    plot_generalization_error(df, args.sweep_dir, fname_suffix='all', use_best=True)
    plot_generalization_error(df, args.sweep_dir, fname_suffix='all', use_best=False)
    plot_generalization_error_vs_l2(df, args.sweep_dir, fname_suffix='all', use_best=True)
    plot_generalization_error_vs_l2(df, args.sweep_dir, fname_suffix='all', use_best=False)
    
    # Side-by-side plots for each split value
    for split_val in split_values:
        if split_val is None:
            continue  # Already plotted above for full dataset
        df_split = df.loc[df[args.split_by] == split_val].reset_index(drop=True)
        if df_split.empty:
            continue
        suffix = f"{args.split_by}={split_val}"
        side_by_side_boxplots(
            df_split,
            metrics=['feature_extractor_test_acc', 'classifier_test_acc'],
            group_cols=['lambda_reg', 'train_ratio', 'random_filters'],
            facet_col='train_ratio',
            sweep_dir=args.sweep_dir,
            fname_suffix=suffix,
        )
        plot_generalization_error(df_split, args.sweep_dir, fname_suffix=suffix, use_best=True)
        plot_generalization_error(df_split, args.sweep_dir, fname_suffix=suffix, use_best=False)
        plot_generalization_error_vs_l2(df_split, args.sweep_dir, fname_suffix=suffix, use_best=True)
        plot_generalization_error_vs_l2(df_split, args.sweep_dir, fname_suffix=suffix, use_best=False)

    slope_group_cols = [
        "share_rotations",
        "share_phases",
        "share_scales",
        "max_scale",
        "num_phases",
        "use_batch_norm",
        "downsample",
        "random_filters",
        "lambda_reg",
    ]
    slope_plot(
        df_downsample,
        group_cols=slope_group_cols,
        metric="feature_extractor_test_acc",
        sweep_dir=args.sweep_dir,
        fname_suffix="downsample",
    )

    # Print best-performing run folder according to requested metric
    metric = args.metric
    if metric in df.columns:
        try:
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]
            best_path = best_row.get('run_path', None)
            if best_path:
                print(f"Best run by {metric}: {best_path}")
            else:
                print(f"Best run by {metric}: run folder {best_row.get('run', '<unknown>')} (no run_path available)")
        except Exception as e:
            print(f"Failed to determine best run for metric '{metric}': {e}")
    else:
        print(f"Metric '{metric}' not found in results columns. Available columns: {list(df.columns)}")