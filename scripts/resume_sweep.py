import argparse
import itertools
import os
import subprocess
import time
from datetime import datetime
import glob

import torch

from configs import load_config, expand_param
from dense.helpers import LoggerManager


def parse_args():
    parser = argparse.ArgumentParser(description="Resume a sweep by running missing jobs")
    parser.add_argument(
        "--sweep-dir",
        type=str,
        required=True,
        help="Sweep directory created by sweep.py",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Sweep config YAML (fallback if sweep_dir has no saved configs)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Weights and Biases project name for logging",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="If set, use random filters",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["wph", "scat", "wph_pca", "wph_hypernetwork"],
        default="scat",
        help="Type of model to train (default: scat)",
    )
    parser.add_argument(
        "--fold-filter",
        type=int,
        default=None,
        help="Only run jobs with this fold value",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max number of parallel workers (defaults to GPU count)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing runs but do not execute",
    )
    return parser.parse_args()


def normalize_value(value):
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, dict):
        return tuple((k, normalize_value(v)) for k, v in sorted(value.items()))
    if isinstance(value, list):
        return tuple(normalize_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(normalize_value(v) for v in value)
    if isinstance(value, float):
        return round(value, 10)
    return value


def config_signature(config, keys):
    subset = {k: config.get(k) for k in keys}
    return normalize_value(subset)


def load_expected_configs_from_saved(sweep_dir, fold_filter=None):
    sweep_config_path = os.path.join(sweep_dir, "sweep_config.yaml")
    base_config_path = os.path.join(sweep_dir, "base_config.yaml")
    if not (os.path.isfile(sweep_config_path) and os.path.isfile(base_config_path)):
        return []
    sweep = load_config(sweep_config_path)
    base_config = load_config(base_config_path)
    params = sweep["sweep"]
    expanded_values = [expand_param(v) for v in params.values()]
    keys = list(params.keys())
    all_combinations = list(itertools.product(*expanded_values))

    if fold_filter is not None and "fold" in keys:
        fold_idx = keys.index("fold")
        all_combinations = [c for c in all_combinations if c[fold_idx] == fold_filter]

    expected = []
    for run_idx, values in enumerate(all_combinations, 1):
        overrides = {k: v for k, v in zip(keys, values)}
        merged_config = {**base_config, **overrides}
        expected.append({"run_idx": run_idx, "config": merged_config, "path": None})
    return expected


def load_expected_configs_from_temp(sweep_dir):
    temp_files = sorted(glob.glob(os.path.join(sweep_dir, "temp_config_*.yaml")))
    expected = []
    for path in temp_files:
        base = os.path.basename(path)
        run_idx_str = base.replace("temp_config_", "").replace(".yaml", "")
        try:
            run_idx = int(run_idx_str)
        except ValueError:
            run_idx = None
        expected.append({"run_idx": run_idx, "config": load_config(path), "path": path})
    return expected


def load_expected_configs_from_sweep(sweep_config_path, fold_filter=None):
    sweep = load_config(sweep_config_path)
    base_config = load_config(sweep["base_config"])
    params = sweep["sweep"]
    expanded_values = [expand_param(v) for v in params.values()]
    keys = list(params.keys())
    all_combinations = list(itertools.product(*expanded_values))

    if fold_filter is not None and "fold" in keys:
        fold_idx = keys.index("fold")
        all_combinations = [c for c in all_combinations if c[fold_idx] == fold_filter]

    expected = []
    for run_idx, values in enumerate(all_combinations, 1):
        overrides = {k: v for k, v in zip(keys, values)}
        merged_config = {**base_config, **overrides}
        expected.append({"run_idx": run_idx, "config": merged_config, "path": None})
    return expected


def load_completed_configs(sweep_dir):
    completed = []
    for ratio_folder in os.listdir(sweep_dir):
        ratio_path = os.path.join(sweep_dir, ratio_folder)
        if not os.path.isdir(ratio_path) or not ratio_folder.startswith("train_ratio="):
            continue
        for run_folder in os.listdir(ratio_path):
            run_path = os.path.join(ratio_path, run_folder)
            config_path = os.path.join(run_path, "config.yaml")
            if os.path.isfile(config_path):
                completed.append(load_config(config_path))
    return completed


def resolve_script(args):
    if args.random:
        return "scripts/train_random.py"
    if args.model_type == "scat":
        return "scripts/train.py"
    if args.model_type in ["wph", "wph_hypernetwork"]:
        return "scripts/train_wph.py"
    if args.model_type == "wph_pca":
        return "scripts/train_wph_pca.py"
    raise ValueError(f"Unknown model type: {args.model_type}")


def run_single_trial(run_idx, total_runs, config, sweep_dir, wandb_project, args, gpu_id):
    log_msg = f"Run {run_idx}/{total_runs} (GPU {gpu_id if gpu_id is not None else 'CPU'})"
    logger.info(log_msg)

    if args.model_type == "scat":
        file = "scripts/train.py"
    elif args.model_type in ["wph", "wph_hypernetwork"]:
        file = "scripts/train_wph.py"
    elif args.model_type == "wph_pca":
        file = "scripts/train_wph_pca.py"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    temp_config_path = os.path.join(sweep_dir, f"temp_config_{run_idx}.yaml")
    with open(temp_config_path, "w", encoding="utf-8") as f:
        import yaml

        yaml.safe_dump(config, f, sort_keys=False)

    cmd = ["python", file, "--config", temp_config_path, "--sweep_dir", sweep_dir]
    if wandb_project is not None:
        cmd.extend(["--wandb_project", wandb_project])

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["SWEEP_GPU_ID"] = str(gpu_id)
    else:
        env["SWEEP_GPU_ID"] = "cpu"

    try:
        result = subprocess.run(cmd, env=env)
        if result.returncode == 0:
            logger.info(f"Run {run_idx} finished successfully")
            return {"run_idx": run_idx, "success": True, "gpu_id": gpu_id}
        logger.error(f"Run {run_idx} failed with return code {result.returncode}")
        return {
            "run_idx": run_idx,
            "success": False,
            "gpu_id": gpu_id,
            "returncode": result.returncode,
        }
    except Exception as exc:
        logger.error(f"Run {run_idx} failed with exception: {exc}")
        return {"run_idx": run_idx, "success": False, "gpu_id": gpu_id, "exception": str(exc)}


args = parse_args()

if not os.path.isdir(args.sweep_dir):
    raise ValueError(f"Sweep dir {args.sweep_dir} does not exist!")

resume_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logger = LoggerManager.get_logger(
    log_dir=args.sweep_dir,
    wandb_project=args.wandb_project,
    name=f"resume-sweep-{resume_timestamp}",
)

expected = load_expected_configs_from_saved(args.sweep_dir, fold_filter=args.fold_filter)
if not expected:
    expected = load_expected_configs_from_temp(args.sweep_dir)
if not expected:
    if args.config is None:
        raise ValueError(
            "No saved sweep configs found. Provide --config to regenerate."
        )
    expected = load_expected_configs_from_sweep(args.config, fold_filter=args.fold_filter)

if args.fold_filter is not None:
    expected = [
        e for e in expected if e["config"].get("fold") == args.fold_filter
    ]

completed_configs = load_completed_configs(args.sweep_dir)
if not completed_configs:
    logger.info("No completed runs detected.")

expected_keys = sorted(expected[0]["config"].keys())
completed_signatures = {
    config_signature(cfg, expected_keys) for cfg in completed_configs
}

pending = []
for item in expected:
    signature = config_signature(item["config"], expected_keys)
    if signature not in completed_signatures:
        pending.append(item)

logger.info(
    f"Found {len(completed_signatures)} completed configs; {len(pending)} pending runs"
)

if args.dry_run:
    logger.info("Dry run enabled; no jobs will be submitted.")
else:
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
        logger.info(f"Found {num_gpus} GPUs available for parallel execution")
    else:
        num_gpus = 1
        gpu_ids = [None]
        logger.info("No GPUs found, running on CPU")

    if len(pending) == 0:
        logger.info("No pending jobs to run.")
    else:
        max_workers = args.max_workers or num_gpus
        max_workers = max(1, min(max_workers, len(pending)))

        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            total_runs = len(expected)
            for idx, item in enumerate(pending, 1):
                gpu_id = gpu_ids[(idx - 1) % num_gpus]
                futures.append(
                    executor.submit(
                        run_single_trial,
                        item["run_idx"] or idx,
                        total_runs,
                        item["config"],
                        args.sweep_dir,
                        args.wandb_project,
                        args,
                        gpu_id,
                    )
                )
                time.sleep(1.5 * gpu_id if gpu_id is not None else 0.1)

            for future in as_completed(futures):
                results.append(future.result())

        successful_runs = sum(1 for r in results if r["success"])
        failed_runs = len(results) - successful_runs
        logger.info(
            f"Pending runs finished. Successful: {successful_runs}/{len(results)}, Failed: {failed_runs}/{len(results)}"
        )
        if failed_runs > 0:
            failed_indices = [r["run_idx"] for r in results if not r["success"]]
            logger.warning(f"Failed run indices: {failed_indices}")

logger.finish()
