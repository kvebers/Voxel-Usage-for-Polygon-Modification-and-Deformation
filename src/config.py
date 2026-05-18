from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Union
import re

from src.constants import DEFAULTS


def _to_namespace(d: dict) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _to_namespace(v)
        elif isinstance(v, list):
            d[k] = [
                _to_namespace(item) if isinstance(item, dict) else item for item in v
            ]
    return SimpleNamespace(**d)


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


def load_config(path: Union[str, Path] = "scene.jsonc") -> SimpleNamespace:
    path = Path(path)
    if not path.exists() and path.suffix == ".jsonc":
        print("Wrong file type")
        exit()
    if path.exists():
        with open(path) as f:
            user = json.loads(_strip_comments(f.read()))
        merged = _deep_merge(DEFAULTS, user)
    else:
        merged = DEFAULTS.copy()
    return _to_namespace(merged)
