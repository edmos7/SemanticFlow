"""Data contracts that separate human RGB supervision from robot actions."""

from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Sequence

from .config import ContractError, positive_number, validate_task_manifest


def _keys(record: dict[str, Any], required: set[str], optional: set[str] = frozenset()) -> None:
    missing = required - record.keys()
    unknown = record.keys() - required - optional
    if missing or unknown:
        raise ContractError(f"schema fields missing={sorted(missing)}, unknown={sorted(unknown)}")


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _relative_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not PurePosixPath(value).is_absolute() and ".." not in PurePosixPath(value).parts and "\\" not in value


def validate_timestamps(timestamps: Sequence[float]) -> None:
    if not timestamps:
        raise ContractError("frame timestamps cannot be empty")
    previous = -1.0
    for timestamp in timestamps:
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ContractError("frame timestamps must be finite, nonnegative, and strictly increasing")
        previous = timestamp


@dataclass(frozen=True)
class ForecastWindow:
    """Indices for one causal input and strictly subsequent supervision."""

    observation_indices: tuple[int, ...]
    target_indices: tuple[int, ...]
    horizons_seconds: tuple[float, ...]


def forecast_window(timestamps: Sequence[float], anchor_index: int, horizons: Sequence[float] = (0.5, 1.0, 2.0), history_frames: int = 1) -> ForecastWindow:
    """Use the first frame at/after each horizon; never clamp to the last frame.

    Missing future coverage raises an error. Decoders should provide timestamps
    for every decoded frame, preserving variable-frame-rate timing.
    """
    validate_timestamps(timestamps)
    if type(anchor_index) is not int or not 0 <= anchor_index < len(timestamps):
        raise ContractError("anchor_index is outside the clip")
    if type(history_frames) is not int or history_frames < 1 or history_frames > anchor_index + 1:
        raise ContractError("insufficient causal history")
    if not horizons or any(not positive_number(value) for value in horizons) or list(horizons) != sorted(set(horizons)):
        raise ContractError("horizons must be positive, unique, and increasing")
    targets = tuple(bisect.bisect_left(timestamps, timestamps[anchor_index] + horizon) for horizon in horizons)
    if any(index >= len(timestamps) for index in targets):
        raise ContractError("clip does not cover every requested future horizon")
    if len(set(targets)) != len(targets):
        raise ContractError("frame sampling is too sparse to distinguish forecast horizons")
    observations = tuple(range(anchor_index - history_frames + 1, anchor_index + 1))
    return ForecastWindow(observations, targets, tuple(horizons))


def deterministic_split(episode_sha256: str, seed: int = 42, validation_fraction: float = 0.2) -> str:
    """Assign by whole-episode content, so copied episodes keep their split."""
    if not _digest(episode_sha256):
        raise ContractError("episode_sha256 must be a lowercase SHA-256 digest")
    if type(seed) is not int or not positive_number(validation_fraction) or validation_fraction >= 1:
        raise ContractError("invalid split seed or validation fraction")
    digest = hashlib.sha256(f"{seed}:{episode_sha256}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / 2**64
    return "validation" if score < validation_fraction else "train"


def validate_episode_manifest(manifest: dict[str, Any], tasks: dict[str, Any], *, require_complete: bool = False, split_seed: int = 42, validation_fraction: float = 0.2) -> dict[str, int]:
    validate_task_manifest(tasks)
    _keys(manifest, {"schema_version", "episodes", "human_clips"})
    if manifest["schema_version"] != 1:
        raise ContractError("unsupported episode manifest schema_version")
    if not isinstance(manifest["episodes"], list) or not isinstance(manifest["human_clips"], list):
        raise ContractError("episodes and human_clips must be lists")
    source_names = {source for target in tasks["targets"] for source in target["source_tasks"]}
    target_ids = {target["id"] for target in tasks["targets"]}
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    covered_sources: set[tuple[str, str]] = set()
    counts = {"train": 0, "validation": 0, "human_clips": 0}
    for episode in manifest["episodes"]:
        if not isinstance(episode, dict):
            raise ContractError("robot episode must be an object")
        _keys(episode, {"id", "suite", "task", "split", "sha256", "path", "episode_key"})
        if not isinstance(episode["id"], str) or not episode["id"] or episode["id"] in seen_ids:
            raise ContractError("robot episode ids must be nonempty and unique")
        if episode["suite"] != tasks["source_suite"] or not isinstance(episode["task"], str) or episode["task"] not in source_names:
            raise ContractError("robot training manifest contains a held-out or unknown task")
        if not _digest(episode["sha256"]) or episode["sha256"] in seen_content:
            raise ContractError("robot episode has invalid or duplicate content hash")
        if not _relative_path(episode["path"]) or not isinstance(episode["episode_key"], str) or not episode["episode_key"]:
            raise ContractError("robot episode needs a data-root-relative path and episode_key")
        expected = deterministic_split(episode["sha256"], split_seed, validation_fraction)
        if episode["split"] != expected:
            raise ContractError(f"episode {episode['id']}: split must be {expected}")
        seen_ids.add(episode["id"])
        seen_content.add(episode["sha256"])
        covered_sources.add((episode["task"], episode["split"]))
        counts[episode["split"]] += 1
    seen_targets: set[str] = set()
    for clip in manifest["human_clips"]:
        if not isinstance(clip, dict):
            raise ContractError("human clip must be an object")
        # In particular, actions/proprioception/manual semantic annotations are
        # not accepted fields; absence is not represented by fabricated zeros.
        _keys(clip, {"id", "target_id", "path", "sha256", "instruction", "viewpoint", "timestamps_seconds"})
        if not isinstance(clip["id"], str) or not clip["id"] or clip["id"] in seen_ids:
            raise ContractError("human clip ids must be nonempty and unique")
        if not isinstance(clip["target_id"], str) or clip["target_id"] not in target_ids or clip["target_id"] in seen_targets:
            raise ContractError("require at most one egocentric clip per known target")
        if clip["viewpoint"] != "egocentric":
            raise ContractError("human supervision must be egocentric")
        if not _relative_path(clip["path"]) or not _digest(clip["sha256"]) or clip["sha256"] in seen_content:
            raise ContractError("invalid clip path or invalid/duplicate content hash")
        if not isinstance(clip["instruction"], str) or not clip["instruction"].strip():
            raise ContractError("human clip requires its ordinary task instruction")
        if not isinstance(clip["timestamps_seconds"], list):
            raise ContractError("timestamps_seconds must be a list")
        validate_timestamps(clip["timestamps_seconds"])
        if clip["timestamps_seconds"][-1] - clip["timestamps_seconds"][0] < 2:
            raise ContractError("human clip must cover the longest forecast horizon")
        seen_ids.add(clip["id"])
        seen_content.add(clip["sha256"])
        seen_targets.add(clip["target_id"])
        counts["human_clips"] += 1
    if require_complete:
        expected_sources = {(task, split) for task in source_names for split in ("train", "validation")}
        if covered_sources != expected_sources or seen_targets != target_ids:
            raise ContractError("incomplete manifest: require train/validation episodes for every source and one clip per target")
    return counts
