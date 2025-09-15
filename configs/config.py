import os
import yaml
from dense.helpers import LoggerManager
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
    # logger = LoggerManager.get_logger()
    # logger.info(f"Overiding config for {overrides}")
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
    logger.info(f"Saving config file {file_path}")