"""Internal persistence helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(value: object, path: str | os.PathLike[str]) -> None:
    """Write a torch checkpoint atomically using a same-directory replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
