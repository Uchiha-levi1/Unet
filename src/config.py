from pathlib import Path

import yaml


def _merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path):
    """Load a YAML config; a top-level `base:` key names a config (relative path) to inherit from."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop("base", None)
    if base is not None:
        cfg = _merge(load_config(path.parent / base), cfg)
    return cfg
