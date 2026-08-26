"""Small manifest and checksum helpers for reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path | str, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path | str, payload: Any) -> Path:
    """Atomically write indented JSON, converting NumPy scalar values."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    def default(value: Any):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot serialize {type(value).__name__}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=default, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def array_record(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    values = np.load(target, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(target),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "bytes": int(target.stat().st_size),
        "sha256": sha256_file(target),
    }


def resolve_external_path(value: str, *, config_dir: Path) -> Path:
    """Resolve an external asset from an explicit or configuration-relative path."""

    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded if expanded.is_absolute() else (config_dir / expanded).resolve()
