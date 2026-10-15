"""Resolve explicitly configured data storage without an internal-disk fallback."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import ContractError


def resolve_data_root(value: str | None = None) -> Path:
    configured = value if value is not None else os.environ.get("SEMFLOW_DATA_ROOT")
    if not configured:
        raise ContractError("set SEMFLOW_DATA_ROOT to an existing directory on a mounted data volume")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ContractError("SEMFLOW_DATA_ROOT must be absolute")
    if not path.is_dir():
        raise ContractError(f"data root is unavailable: {path}; no fallback will be created")
    path = path.resolve()
    # A disappeared removable volume leaves an ordinary directory behind. Require
    # a mount boundary other than the system root instead of trusting that path.
    mount = next((parent for parent in (path, *path.parents) if parent.is_mount()), None)
    if mount is None or mount == Path(mount.anchor):
        raise ContractError(f"{path} is not on a mounted data volume distinct from the system root")
    if os.statvfs(path).f_flag & os.ST_RDONLY:
        raise ContractError(f"data volume is read-only: {mount}; no fallback will be created")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise ContractError(f"data root is not readable and writable: {path}")
    return path


def storage_status(value: str | None = None) -> dict[str, str | int]:
    path = resolve_data_root(value)
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_bytes": usage.total, "free_bytes": usage.free}
