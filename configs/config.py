import os
import yaml
from dense.helpers import LoggerManager

def float_range(start, stop, step):
    """Generate a list of floats from start to stop (exclusive)"""
    vals = []
    v = start
    while v < stop:
        vals.append(round(v, 10))  # avoid floating-point accumulation errors
        v += step
    return vals

def expand_param(val):
    """
    If val is a dict with start/stop/step -> create a list via range
    Else assume it's already a list
    """
    if isinstance(val, dict) and {'start','stop','step'}.issubset(val.keys()):
        return float_range(val['start'], val['stop'], val['step'])
    elif isinstance(val, list):
        return val
    else:
        # Single value -> wrap in list
        return [val]

def load_config(filename: str):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Config file not found: {filename}")

    try:
        with open(filename, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error in {filename}: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error when reading {filename}: {e}")

    if config is None:
        raise ValueError(f"Config file {filename} is empty.")

    if not isinstance(config, dict):
        raise ValueError(f"Config file {filename} must define a dictionary at top-level.")
    # logger = LoggerManager.get_logger()
    # logger.info(f"Loading config file {filename}")
    return config

def apply_overrides(config, overrides):
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override '{ov}' not in key=value format")
        k, v = ov.split("=", 1)
        # try to cast to number or bool
        if v.lower() in ["true","false"]:
            v = v.lower() == "true"
        else:
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass  # leave as string
        config[k] = v
    return config

def save_config(folder:str, config):
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "config.yaml")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception as e:
        raise OSError(f"Failed to save config to {file_path}: {e}")
    logger = LoggerManager.get_logger()
    logger.log(f"Saving config file {file_path}")