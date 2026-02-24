import os
import argparse
from configs import load_config
from dense.helpers.logger import LoggerManager
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def margin_loss(data, label, margin):
    loss = 1
    if np.argmax(data) == label:
        correct_prob = data[label]
        other_probs = np.delete(data, label)
        if (correct_prob - np.max(other_probs)) > margin:
            loss = 0
    return loss

def plot_margin_loss(margins: list[float] = [0.05, 0.2], num_classes: int = 5):
    # create figure to demonstrate margin loss
    classes = list(range(num_classes))
    correct_class = np.random.randint(num_classes)
    random_logits = np.random.rand(num_classes)
    random_logits = random_logits/random_logits.sum()

    correct_pred = np.random.rand(num_classes)
    correct_pred[correct_class] += 1
    correct_pred = correct_pred/correct_pred.sum()

    incorrect_pred = np.random.rand(num_classes)
    incorrect_class = np.random.randint(num_classes)
    while incorrect_class == correct_class:
        incorrect_class = np.random.randint(num_classes)
    incorrect_pred[incorrect_class] += 1
    incorrect_pred = incorrect_pred/incorrect_pred.sum()

    margins_sorted = sorted(margins)
    if len(margins_sorted) > 1:
        small_margin_delta = 0.5 * (margins_sorted[0] + margins_sorted[-1])
    else:
        small_margin_delta = margins_sorted[0] * 0.5
    small_margin_delta = min(max(small_margin_delta, 1e-3), 0.9)

    second_prob = 0.3
    correct_prob = second_prob + small_margin_delta
    if correct_prob >= 0.7:
        correct_prob = 0.6
        second_prob = correct_prob - small_margin_delta
    remaining_total = 1.0 - (correct_prob + second_prob)
    if remaining_total < 0:
        correct_prob = (1.0 + small_margin_delta) / 2.0 - 1e-3
        second_prob = correct_prob - small_margin_delta
        remaining_total = 1.0 - (correct_prob + second_prob)

    small_margin_pred = np.full(num_classes, remaining_total / max(num_classes - 2, 1))
    small_margin_pred[correct_class] = correct_prob
    second_best = (correct_class + 1) % num_classes
    small_margin_pred[second_best] = second_prob

    scenarios = [
        ("Random Predictions", random_logits),
        ("Small-Margin Correct", small_margin_pred),
        ("Correct Predictions", correct_pred),
        ("Incorrect Predictions", incorrect_pred),
    ]

    fig, axs = plt.subplots(len(scenarios), len(margins), figsize=(5 * len(margins), 4.5 * len(scenarios)))
    fig.suptitle(f"Margin Loss Demo: Correct Class = {correct_class}")

    bar_colors = ["#b0b0b0"] * num_classes
    bar_colors[correct_class] = "#2ca02c"

    for i, margin in enumerate(margins):
        axs[0, i].set_title(f"Margin = {margin}")
        for j, (scenario_label, scenario_data) in enumerate(scenarios):
            ax = axs[j, i]
            is_correct = np.argmax(scenario_data) == correct_class
            loss = margin_loss(data=scenario_data, label=correct_class, margin=margin)
            correct_prob = scenario_data[correct_class]
            max_other = np.max(scenario_data)
            ax.bar(x=classes, height=scenario_data, color=bar_colors)
            ax.set_ylim(0, 1)
            ax.hlines(
                max_other - margin,
                xmin=-0.5,
                xmax=num_classes - 0.5,
                colors="red",
                linestyles="dashed",
                label="Margin threshold",
            )
            ax.set_ylabel(scenario_label if i == 0 else "")
            status = "pass" if (loss == 0 and is_correct) else "fail"
            ax.set_xlabel(f"Correct={is_correct}, margin={status}")
            if i == len(margins) - 1:
                ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("margin_loss_demo.png")
    plt.close()

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
                new_row['base_loss'] = acc_row.get('base_loss', np.nan)
                new_row['reg_loss'] = acc_row.get('reg_loss', np.nan)
                new_row['total_loss'] = acc_row.get('total_loss', np.nan)
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


def plot_generalization_error_vs_l2(df, sweep_dir, fname_suffix="", metric_name="feature_extractor_best_acc", split_by=None):
    """
    Plot feature extractor generalization error vs L2 norm and L2 norm vs test accuracy, separated by train_ratio.
    
    Args:
        df: DataFrame with experiment results
        sweep_dir: Path to sweep directory
        fname_suffix: Suffix for output filename
        metric_name: Column name for the metric to use (e.g. 'feature_extractor_best_acc' or 'feature_extractor_last_acc')
        split_by: Column name to use for different markers (optional)
    """
    results_path = os.path.join(sweep_dir, "results")
    
    # Feature extractor generalization error
    if metric_name in df.columns and "feature_extractor_test_acc" in df.columns:
        df["feature_extractor_gen_error"] = df[metric_name] - df["feature_extractor_test_acc"]
    
    if "feature_extractor_gen_error" not in df.columns or "l2_norm_finetuning" not in df.columns:
        return
    
    df_with_l2 = df[df["l2_norm_finetuning"] > 0].copy()
    if df_with_l2.empty:
        return
    
    # Color by random_filters
    random_filters_values = sorted(df_with_l2["random_filters"].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(random_filters_values)))
    color_map = {val: colors[i] for i, val in enumerate(random_filters_values)}

    # Create marker map if split_by is specified
    marker_map = {}
    if split_by is not None and split_by in df_with_l2.columns:
        split_values = sorted(df_with_l2[split_by].unique())
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'x', '+']
        marker_map = {val: markers[i % len(markers)] for i, val in enumerate(split_values)}

    train_ratio_values = sorted(df_with_l2["train_ratio"].unique())
    nrows = len(train_ratio_values)
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.5 * nrows), squeeze=False)

    for idx, train_ratio_val in enumerate(train_ratio_values):
        train_ratio_grp = df_with_l2[df_with_l2["train_ratio"] == train_ratio_val]
        
        # Left plot: Generalization error vs L2 norm
        ax_left = axes[idx, 0]
        if split_by is not None and split_by in train_ratio_grp.columns:
            # Group by both random_filters and split_by
            for rf_val, rf_grp in train_ratio_grp.groupby("random_filters"):
                for split_val, split_grp in rf_grp.groupby(split_by):
                    label = f"random_filters={rf_val}, {split_by}={split_val}"
                    marker = marker_map.get(split_val, 'o')
                    ax_left.scatter(
                        split_grp["l2_norm_finetuning"],
                        split_grp["feature_extractor_gen_error"],
                        label=label,
                        alpha=0.6,
                        color=color_map[rf_val],
                        marker=marker,
                        s=80,
                    )
        else:
            # Original behavior without split_by
            for rf_val, rf_grp in train_ratio_grp.groupby("random_filters"):
                label = f"random_filters={rf_val}"
                ax_left.scatter(
                    rf_grp["l2_norm_finetuning"],
                    rf_grp["feature_extractor_gen_error"],
                    label=label,
                    alpha=0.6,
                    color=color_map[rf_val],
                    s=80,
                )
        
        ax_left.set_title(f"Gen. Error vs L2 (train_ratio={train_ratio_val})")
        ax_left.set_xlabel("L2 Norm (Fine-tuning)")
        ax_left.set_ylabel("Generalization Error")
        ax_left.set_xscale("log")
        ax_left.grid(True, alpha=0.3)
        ax_left.legend()
        
        # Right plot: L2 norm vs test accuracy
        ax_right = axes[idx, 1]
        if split_by is not None and split_by in train_ratio_grp.columns:
            # Group by both random_filters and split_by
            for rf_val, rf_grp in train_ratio_grp.groupby("random_filters"):
                for split_val, split_grp in rf_grp.groupby(split_by):
                    label = f"random_filters={rf_val}, {split_by}={split_val}"
                    marker = marker_map.get(split_val, 'o')
                    ax_right.scatter(
                        split_grp["l2_norm_finetuning"],
                        split_grp["feature_extractor_test_acc"],
                        label=label,
                        alpha=0.6,
                        color=color_map[rf_val],
                        marker=marker,
                        s=80,
                    )
        else:
            # Original behavior without split_by
            for rf_val, rf_grp in train_ratio_grp.groupby("random_filters"):
                label = f"random_filters={rf_val}"
                ax_right.scatter(
                    rf_grp["l2_norm_finetuning"],
                    rf_grp["feature_extractor_test_acc"],
                    label=label,
                    alpha=0.6,
                    color=color_map[rf_val],
                    s=80,
                )
        
        ax_right.set_title(f"Test Acc vs L2 (train_ratio={train_ratio_val})")
        ax_right.set_xlabel("L2 Norm (Fine-tuning)")
        ax_right.set_ylabel("Test Accuracy")
        ax_right.set_xscale("log")
        ax_right.grid(True, alpha=0.3)
        ax_right.legend()

    fig.suptitle(f"Generalization Error and Test Accuracy vs L2 Norm ({metric_name})")
    plt.tight_layout()
    
    plot_path = os.path.join(results_path, f"generalization_error_vs_l2_{metric_name}_{fname_suffix}.png")
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
    parser.add_argument('--split-plots-individual', action='store_true', help="If set, generates separate plots for each split value instead of combined plots.")
    parser.add_argument('--hybrid', action='store_true', help="If set, generates slope plot using rows downsample=hybrid instead of downsample=True.")
    args = parser.parse_args()
    df = df_from_logs(args)
    # df['wavelet_params'] = df['wavelet_params'].apply(literal_eval)
    df = df.join(pd.json_normalize(df['wavelet_params']))
    
    # Add intermediate l2 norms from accuracy.csv files
    df_l2 = add_intermediate_l2_norms(df, args.sweep_dir)
    
    df_lr1 = df.loc[df.lambda_reg == 1].reset_index(drop=True)
    df_lr1_downsample = df_lr1.loc[df_lr1.downsample].reset_index(drop=True)
    
    # Split dataset by specified column and generate separate plots
    split_values = [None]  # Default: no split, process entire dataset
    if args.split_by and args.split_by in df.columns:
        split_values = sorted(df[args.split_by].dropna().unique())
        print(f"Splitting plots by '{args.split_by}' with values: {split_values}")
    
    if args.split_plots_individual:
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
    group_cols = ['max_scale', 'train_ratio', 'random_filters']
    if args.split_by and args.split_by in df.columns:
        group_cols = [args.split_by] + group_cols
    plot_all_boxplots(df, group_cols = group_cols,
                    sweep_dir=args.sweep_dir,
                    fname_suffix = 'all')
    # plot_all_boxplots(df_lr1_fullsize, group_cols = ['max_scale', 'train_ratio', 'random_filters'],
    #                   sweep_dir=args.sweep_dir,
    #                   fname_suffix = 'lreg=1_fullsize')
    if args.hybrid:
        df_downsample = df.loc[df.downsample == 'hybrid'].reset_index(drop=True)
    else:
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
    # plot_generalization_error(df, args.sweep_dir, fname_suffix='all', use_best=True)
    # plot_generalization_error(df, args.sweep_dir, fname_suffix='all', use_best=False)
    plot_generalization_error_vs_l2(df_l2, args.sweep_dir, fname_suffix='intermediate_all', metric_name='feature_extractor_last_acc', split_by=args.split_by)
    plot_generalization_error_vs_l2(df, args.sweep_dir, fname_suffix='last_all', metric_name='feature_extractor_last_acc', split_by=args.split_by)
    plot_generalization_error_vs_l2(df, args.sweep_dir, fname_suffix='best_all', metric_name='feature_extractor_best_acc', split_by=args.split_by)

    if args.split_plots_individual:
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
            plot_generalization_error_vs_l2(df_split, args.sweep_dir, fname_suffix=suffix, metric_name='feature_extractor_best_acc', split_by=args.split_by)
            plot_generalization_error_vs_l2(df_split, args.sweep_dir, fname_suffix=suffix, metric_name='feature_extractor_last_acc', split_by=args.split_by)

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