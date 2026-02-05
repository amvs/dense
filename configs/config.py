import os
import yaml
from dense.helpers import LoggerManager
import numpy as np

def expand_param(val):
    """
    If val is a dict with start/stop/step -> create a list via range
    Else assume it's already a list
    """
    if isinstance(val, dict) and {'start', 'stop', 'step'}.issubset(val.keys()):
        return np.round(np.arange(val['start'], val['stop'], val['step']), 10).tolist()
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
        v_str = v.strip()

        # Helper to cast a single scalar token
        def _cast_scalar(token: str):
            tl = token.strip().lower()
            if tl in ("true", "false"):
                return tl == "true"
            if tl in ("none", "null"):
                return None
            try:
                return int(token)
            except ValueError:
                try:
                    return float(token)
                except ValueError:
                    return token

        parsed = None
        try:
            # YAML-style list/dict, e.g. [1, 2, 3] or {a: 1, b: 2}
            if v_str.startswith("[") or v_str.startswith("{"):
                parsed = yaml.safe_load(v_str)
            # Comma-separated list, e.g. a,b,c or 1,2,3
            elif "," in v_str:
                parsed = [_cast_scalar(tok) for tok in v_str.split(",")]
            else:
                parsed = _cast_scalar(v_str)
        except Exception:
            # Fallback: leave as raw string if parsing fails
            parsed = v_str

        config[k] = parsed
    return config

def save_config(folder:str, config):
    # Ensure config is a mapping (AutoConfig subclasses dict are accepted)
    from collections.abc import Mapping
    if not isinstance(config, Mapping):
        raise ValueError("config must be a dictionary/mapping")

    # Prefer a conversion method on config if provided (e.g., AutoConfig.to_plain)
    if hasattr(config, "to_plain") and callable(getattr(config, "to_plain")):
        plain_config = config.to_plain()
    else:
        plain_config = config
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "config.yaml")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(plain_config, f, sort_keys=False)
    except Exception as e:
        raise OSError(f"Failed to save config to {file_path}: {e}")
    logger = LoggerManager.get_logger()
    logger.log(f"Saving config file {file_path}")



class AutoConfig(dict):
    """Dictionary subclass that records default values used via `get()`.

    Any call to `config.get(key, default)` where `key` is missing will
    insert `key: default` into the dictionary and return `default`.
    This ensures defaults become part of the saved config.
    """
    def get(self, key, default=None):
        if key in self:
            return super().get(key)
        # Record the default into the config so it is saved later
        super().__setitem__(key, default)
        return default

    def to_plain(self):
        """Return a plain Python dict/list/scalar representation of this config.

        This recursively converts nested mappings (including other AutoConfig
        instances), lists/tuples, and NumPy scalar types to native Python
        types so the result can be safely dumped to YAML.
        """
        from collections.abc import Mapping

        def _make_plain(obj):
            if isinstance(obj, Mapping):
                return {k: _make_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_plain(v) for v in obj]
            try:
                import numpy as _np
                if isinstance(obj, _np.generic):
                    return obj.item()
            except Exception:
                pass
            return obj

        return _make_plain(self)
