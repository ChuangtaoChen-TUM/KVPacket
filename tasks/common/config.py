"""Config loading with ``_default.json`` inheritance.

A config directory may hold a ``_default.json`` whose keys are merged (non-destructively,
deep) into every sibling config. Ported and slimmed from the reference ``utils/config.py``.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

__all__ = ["broadcast_dict", "load_config", "load_configs"]


def broadcast_dict(base: dict, target: dict, overwrite: bool = False) -> dict:
    """Deep-merge ``base`` into ``target`` (target wins unless ``overwrite``)."""
    for key, value in base.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            broadcast_dict(value, target[key], overwrite)
        elif overwrite:
            target[key] = copy.deepcopy(value)
    return target


def load_config(path: str, default_name: str | None = "_default.json") -> dict[str, Any]:
    """Load one JSON config, merging the sibling ``_default.json`` if present."""
    folder = os.path.dirname(path)
    default: dict = {}
    if default_name:
        dp = os.path.join(folder, default_name)
        if os.path.isfile(dp):
            with open(dp) as f:
                default = json.load(f)
    with open(path) as f:
        cfg = json.load(f)
    broadcast_dict(default, cfg, overwrite=False)
    return cfg


def load_configs(
    path: str,
    default_name: str | None = "_default.json",
    skip: str = "_default.json",
) -> list[dict[str, Any]]:
    """Load a single config file or every ``*.json`` (bar the default) in a directory."""
    if os.path.isfile(path):
        return [load_config(path, default_name)]
    if not os.path.isdir(path):
        raise ValueError(f"{path} is neither a file nor a directory")
    out: list[dict[str, Any]] = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json") or fname == skip:
            continue
        out.append(load_config(os.path.join(path, fname), default_name))
    return out
