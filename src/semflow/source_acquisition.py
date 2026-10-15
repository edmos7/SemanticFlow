"""Acquire only pinned pilot sources, without a Hub cache or suite archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import urllib.request

from .config import ContractError, validate_task_manifest
from .data import _digest
from .storage import resolve_data_root


DATASET = "yifengzhu-hf/LIBERO-datasets"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DIRECTORY = "datasets/libero90-source-v1"


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def validate_source_lock(lock: dict, tasks: dict) -> list[dict]:
    validate_task_manifest(tasks)
    if (lock.get("schema_version") != 1 or lock.get("repository") != DATASET
            or lock.get("libero_revision") != LIBERO_REVISION or lock.get("suite") != "libero_90"
            or not re.fullmatch(r"[0-9a-f]{40}", str(lock.get("revision", "")))):
        raise ContractError("source lock must pin the approved LIBERO mirror and revisions")
    expected = {s for target in tasks["targets"] for s in target["source_tasks"]}
    files = lock.get("files")
    if not isinstance(files, list) or len(files) != 6 or any(not isinstance(f, dict) for f in files):
        raise ContractError("source lock must contain exactly six files")
    if {f.get("task") for f in files} != expected:
        raise ContractError("source lock contains a target, unknown, duplicate, or missing task")
    for entry in files:
        if (not re.fullmatch(r"[A-Za-z0-9_]+", entry["task"])
                or type(entry.get("bytes")) is not int or entry["bytes"] <= 0
                or not _digest(entry.get("sha256")) or not _digest(entry.get("bddl_sha256"))):
            raise ContractError("invalid source file pin")
    return sorted(files, key=lambda f: f["task"])


def source_url(lock: dict, entry: dict) -> str:
    return (f"https://huggingface.co/datasets/{DATASET}/resolve/{lock['revision']}/"
            f"libero_90/{entry['task']}_demo.hdf5")


def external_path(root: Path, relative: str) -> Path:
    """Reject symlinks, nested mounts, traversal, and storage inside the repo."""
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ContractError("source path must stay beneath SEMFLOW_DATA_ROOT")
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ContractError("source data cannot be stored inside the repository")
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ContractError("source storage cannot contain symlinks")
        if part.exists() and part.stat().st_dev != root.stat().st_dev:
            raise ContractError("source storage cannot cross to another volume")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict) -> None:
    if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
        raise ContractError(f"source file size/hash mismatch: {path.name}")


def download_file(root: Path, lock: dict, entry: dict) -> Path:
    resolve_data_root(str(root))
    relative = f"{DIRECTORY}/raw/{entry['task']}_demo.hdf5"
    destination = external_path(root, relative)
    if destination.exists():
        verify_file(destination, entry)
        return destination
    partial = external_path(root, relative + ".partial")
    if partial.exists():
        # Never overwrite an interrupted download. A completely received file
        # can be finalized without making a second network request.
        verify_file(partial, entry)
        partial.rename(destination)
        return destination
    if shutil.disk_usage(root).free < entry["bytes"] + 256 * 1024**2:
        raise ContractError("insufficient external-volume space for the source file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    digest, count = hashlib.sha256(), 0
    request = urllib.request.Request(source_url(lock, entry), headers={"Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
        if response.status != 200 or not response.url.startswith("https://"):
            raise ContractError("source download requires a complete HTTPS response")
        while block := response.read(4 * 1024 * 1024):
            if time.monotonic() - started > 1800:
                raise TimeoutError("source file download exceeded 1800 seconds")
            resolve_data_root(str(root))
            external_path(root, relative + ".partial")
            count += len(block)
            if count > entry["bytes"]:
                raise ContractError("source response exceeds its pinned size")
            output.write(block)
            digest.update(block)
        output.flush()
        os.fsync(output.fileno())
    if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ContractError("source download failed size/hash verification; partial preserved")
    verify_file(partial, entry)
    partial.rename(destination)
    return destination


def acquire_sources(lock: dict, tasks: dict) -> list[str]:
    entries = validate_source_lock(lock, tasks)  # reject targets before any I/O
    root = resolve_data_root()
    return [str(download_file(root, lock, entry).relative_to(root)) for entry in entries]
