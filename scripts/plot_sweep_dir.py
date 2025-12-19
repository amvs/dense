import yaml, itertools, subprocess
from dense.helpers import LoggerManager
from datetime import datetime
import os
import argparse
from configs import load_config, expand_param
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def df_from_logs(args):
    sweep_dir = args.sweep_dir
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
    df["finetuning_gain"] = df["feature_extractor_test_acc"] - df["classifier_test_acc"]
    df.to_csv(os.path.join(results_path, "sweep_results.csv"), index=False)
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
        boxplot_metric_vs_ratio(df, metric, group_cols, sweep_dir, fname_suffix)


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

    fix, axs = plt.subplots(figsize=(10, 12), nrows=2, ncols=1)
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
    assert facet_col in group_cols, "facet_col must be in group_cols"
    df_melted = df.melt(id_vars=group_cols, value_vars=metrics, var_name="metric", value_name="value")
    set_cols = set(group_cols)
    set_cols.remove(facet_col)
    df_melted['params'] = df_melted[list(set_cols)].astype(str).agg(','.join, axis=1)
    g = sns.catplot(data = df_melted, x = 'params',y = 'value', hue='metric', col ='train_ratio', kind = 'strip', col_wrap=3, sharey=False, sharex=False)
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_dir, 'results', f'side_by_side_boxplots_{fname_suffix}.png'))
    plt.close()
    print(f"Side by side boxplots saved to {os.path.join(sweep_dir, 'results', f'side_by_side_boxplots_{fname_suffix}.png')}")


def slope_plot(df, group_cols, metric, sweep_dir, fname_suffix=""):
    calc_slope = lambda df: np.polyfit(df["train_ratio"], df[metric], 1)[0]
    calc_intercept = lambda df: np.polyfit(df["train_ratio"], df[metric], 1)[1]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sweep results.")
    parser.add_argument(
        "--sweep_dir",
        type=str,
        required=True,
        help="Path to the sweep directory (e.g. experiments/mnist-sweeps-20251120-123456)",
    )
    args = parser.parse_args()
    df = df_from_logs(args)
    df_lr1 = df.loc[df.lambda_reg == 1].reset_index(drop=True)
    df_lr1_downsample = df_lr1.loc[df_lr1.downsample].reset_index(drop=True)
    # df_lr1_fullsize = df_lr1.loc[~df_lr1.downsample].reset_index(drop=True)
    plot_all_boxplots(df_lr1_downsample, group_cols = ['max_scale', 'train_ratio', 'random_filters'],
                      sweep_dir=args.sweep_dir,
                      fname_suffix = 'lreg=1_downsample')
    # plot_all_boxplots(df_lr1_fullsize, group_cols = ['max_scale', 'train_ratio', 'random_filters'],
    #                   sweep_dir=args.sweep_dir,
    #                   fname_suffix = 'lreg=1_fullsize')
    df_downsample = df.loc[df.downsample].reset_index(drop=True)
    # df_fullsize = df.loc[~df.downsample].reset_index(drop=True)
    pair_boxplots(
        df_downsample,
        metric="feature_extractor_test_acc",
        pair_col="random_filters",
        group_cols=["lambda_reg", "random_filters", "train_ratio"],
        sweep_dir=args.sweep_dir,
        fname_suffix="downsample",
    )
    pair_boxplots(
        df_downsample,
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
    # slope_plot(
    #     df_fullsize,
    #     group_cols=slope_group_cols,
    #     metric="feature_extractor_test_acc",
    #     sweep_dir=args.sweep_dir,
    #     fname_suffix="fullsize",
    # )
    side_by_side_boxplots(df_downsample, metrics = ['feature_extractor_test_acc', 'classifier_test_acc'],
                          group_cols = ['lambda_reg', 'train_ratio', 'random_filters'],
                          facet_col='train_ratio',
                          sweep_dir=args.sweep_dir,
                          fname_suffix='downsample')
