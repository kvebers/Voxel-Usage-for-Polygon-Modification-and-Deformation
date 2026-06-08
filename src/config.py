import json
from pathlib import Path
from types import SimpleNamespace
import re
from src.constants import DEFAULTS


def transfer_to_namespace(data):
    for k, v in data.items():
        if isinstance(v, dict):
            data[k] = transfer_to_namespace(v)
        elif isinstance(v, list):
            data[k] = [transfer_to_namespace(item) if isinstance(item, dict) else item for item in v]
    return SimpleNamespace(**data)


def deep_merge(base, override):
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def remove_comments(text):
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def load_config(path="scene.jsonc"):
    """
    Checks correct file tipe loads defaults
    """
    path = Path(path)
    if not path.exists() and path.suffix == ".jsonc":
        print("Wrong file type")
        exit()
    if path.exists():
        with open(path) as f:
            user_config = json.loads(remove_comments(f.read()))
        merged = deep_merge(DEFAULTS, user_config)
    else:
        merged = DEFAULTS.copy()
    return transfer_to_namespace(merged)
