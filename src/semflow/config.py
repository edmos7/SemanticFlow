"""Versioned, explicit experiment configuration without training dependencies."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class ContractError(ValueError):
    """An input violates a research or data contract."""


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def validate_task_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ContractError("unsupported task manifest schema_version")
    if manifest.get("source_suite") != "libero_90" or manifest.get("target_suite") != "libero_10":
        raise ContractError("pilot requires LIBERO-90 sources and LIBERO-10 targets")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or len(targets) != 3:
        raise ContractError("pilot requires exactly three target tasks")
    ids, target_names, source_names = set(), set(), set()
    for target in targets:
        if not isinstance(target, dict):
            raise ContractError("target must be an object")
        for field in ("id", "task", "instruction", "human_analogue"):
            if not isinstance(target.get(field), str) or not target[field].strip():
                raise ContractError(f"target requires nonempty {field}")
        if target["id"] in ids or target["task"] in target_names:
            raise ContractError("duplicate target id or task")
        ids.add(target["id"])
        target_names.add(target["task"])
        sources = target.get("source_tasks")
        if not isinstance(sources, list) or len(sources) != 2:
            raise ContractError("each composition requires two source component tasks")
        if any(not isinstance(item, str) or not item for item in sources):
            raise ContractError("source task names must be nonempty strings")
        source_names.update(sources)
    if len(source_names) != 6:
        raise ContractError("pilot requires six distinct source tasks")
    if source_names & target_names:
        raise ContractError("target task appears in source training tasks")
    return manifest


def validate_config(config: dict[str, Any], *, training_ready: bool = False) -> dict[str, Any]:
    if not isinstance(config.get("experiment"), str) or not config["experiment"].strip():
        raise ContractError("experiment name is required")
    locked = {
        "schema_version": 1,
        "backbone": "EgoSteer/EgoSteer-3B-Base",
        "scene_queries": 16,
        "horizons_seconds": [0.5, 1.0, 2.0],
        "action_dimension": 7,
        "action_chunk_size": 32,
        "execute_actions": 8,
        "lora_rank": 8,
        "conditions": ["frozen", "standard", "semflow"],
        "human_clips_per_target": 1,
        "target_robot_demonstrations": False,
        "target_checkpoint_selection": False,
    }
    for field, expected in locked.items():
        if field not in config or config[field] != expected or isinstance(config[field], bool) != isinstance(expected, bool):
            raise ContractError(f"pilot {field} must equal {expected!r}")
    fraction = config.get("source_validation_fraction")
    if not positive_number(fraction) or not fraction < 1:
        raise ContractError("source_validation_fraction must be between 0 and 1")
    seeds = config.get("adaptation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 3:
        raise ContractError("adaptation_seeds must contain three distinct integers")
    for field in ("split_seed", "pilot_initializations", "evaluation_initializations"):
        if type(config.get(field)) is not int or config[field] < (0 if field == "split_seed" else 1):
            raise ContractError(f"invalid {field}")
    gate = config.get("source_success_gate", {})
    if gate != {"macro": 0.6, "minimum_task": 0.3}:
        raise ContractError("source success gate must be macro=0.6 and minimum_task=0.3")
    if not isinstance(config.get("task_manifest"), str) or not config["task_manifest"]:
        raise ContractError("task_manifest path is required")
    revision = config.get("checkpoint_revision")
    if revision is not None and (not isinstance(revision, str) or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision)):
        raise ContractError("checkpoint_revision must be a pinned 40-character commit hash or null")
    for field in ("adaptation_steps", "learning_rate", "semantic_loss_weight"):
        value = config.get(field)
        if value is not None and not positive_number(value):
            raise ContractError(f"{field} must be positive or null")
    if config.get("adaptation_steps") is not None and type(config["adaptation_steps"]) is not int:
        raise ContractError("adaptation_steps must be an integer")
    if training_ready:
        missing = [field for field in ("checkpoint_revision", "adaptation_steps", "learning_rate", "semantic_loss_weight") if config.get(field) is None]
        if missing:
            raise ContractError("not training-ready; source development must lock: " + ", ".join(missing))
    return config


def load_config(path: str | Path, *, training_ready: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = Path(path)
    config = validate_config(load_json(config_path), training_ready=training_ready)
    tasks = validate_task_manifest(load_json(config_path.parent / config["task_manifest"]))
    return config, tasks
