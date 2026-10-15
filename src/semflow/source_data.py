"""Audited LIBERO source ingestion and train-only Panda normalization.

HDF5 is optional. Simulator states/rewards are hashed for provenance only;
the observation reader exposes RGB, measured Panda state, and causal actions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

from .config import ContractError
from .data import deterministic_split, validate_episode_manifest
from .source_acquisition import (DIRECTORY, external_path, file_sha256, json_bytes,
                                 validate_source_lock, verify_file)
from .storage import resolve_data_root


HASH_VERSION = "libero-decoded-numeric-v1"
STATE_FIELDS = ["eef_x", "eef_y", "eef_z", "quat_x", "quat_y", "quat_z", "quat_w",
                "gripper_left_qpos", "gripper_right_qpos"]
ACTION_FIELDS = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
OBS_FIELDS = {"agentview_rgb", "eye_in_hand_rgb", "ee_states", "ee_pos", "ee_ori",
              "gripper_states", "joint_states"}
EPISODE_FIELDS = {"actions", "states", "robot_states", "rewards", "dones", "obs"}


def dependencies():
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise ContractError("source ingestion requires the optional numpy and h5py dependencies") from error
    return h5py, np


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise ContractError("expected textual LIBERO metadata")
    return value


def _hard_children(group) -> None:
    h5py, _ = dependencies()
    for name in group:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ContractError("HDF5 external/soft links are forbidden")


def validate_metadata(handle, entry: dict) -> dict:
    h5py, _ = dependencies()
    if set(handle) != {"data"}:
        raise ContractError("source HDF5 must contain only the data group")
    _hard_children(handle)
    data = handle["data"]
    if not isinstance(data, h5py.Group):
        raise ContractError("source data must be a group")
    _hard_children(data)
    attrs = data.attrs
    bddl = _text(attrs["bddl_file_content"]) if "bddl_file_content" in attrs else None
    embedded_bddl_hash = hashlib.sha256(bddl.encode()).hexdigest() if bddl is not None else None
    if (PurePosixPath(_text(attrs["bddl_file_name"])).name != entry["task"] + ".bddl"
            or (embedded_bddl_hash is not None and embedded_bddl_hash != entry["bddl_sha256"])):
        raise ContractError("embedded task identity/BDDL mismatch; target or mislabeled source rejected")
    info = json.loads(_text(attrs["problem_info"]))
    env = json.loads(_text(attrs["env_args"]))
    settings = env["env_kwargs"]
    controller = settings["controller_configs"]
    if (settings["robots"] not in (["Panda"], "Panda") or settings["control_freq"] != 20
            or controller["type"] != "OSC_POSE" or controller.get("control_delta") is not True
            or controller.get("input_min") != -1 or controller.get("input_max") != 1):
        raise ContractError("source must use the 20 Hz Panda delta OSC_POSE controller with [-1,1] inputs")
    for name in (env["bddl_file"], settings["bddl_file_name"]):
        if PurePosixPath(name).name != entry["task"] + ".bddl":
            raise ContractError("environment task identity mismatch")
    instruction = info["language_instruction"]
    expected_instruction = re.sub(r"^.*?_SCENE[0-9]+_", "", entry["task"]).replace("_", " ")
    if not isinstance(instruction, str) or instruction != expected_instruction:
        raise ContractError("source instruction does not match the approved component task")
    convention = _text(attrs["macros_image_convention"])
    if convention not in ("opengl", "opencv"):
        raise ContractError("unknown RGB image convention")
    if len(data) != int(attrs["num_demos"]) or not len(data):
        raise ContractError("source episode count disagrees with metadata")
    if any(not re.fullmatch(r"demo_[0-9]+", name) for name in data):
        raise ContractError("unexpected source episode key")
    return {"instruction": instruction, "bddl_sha256": entry["bddl_sha256"],
            "embedded_bddl_sha256": embedded_bddl_hash,
            "task_identity_basis": "pinned file SHA256, three task filenames, exact component instruction",
            "control_hz": 20, "robot": "Panda", "controller": controller,
            "image_convention": convention, "num_demos": len(data)}


def episode_content_hash(episode) -> tuple[str, dict]:
    """Hash decoded numeric arrays, independent of ID, compression, and paths.

    Include every stored dataset (including provenance-only simulator fields).
    Canonical little-endian float64/int64/uint64 encodings ignore storage width.
    Attributes, task labels and demo names cannot change content identity.
    """
    h5py, np = dependencies()
    _hard_children(episode)
    if set(episode) != EPISODE_FIELDS or not isinstance(episode["obs"], h5py.Group):
        raise ContractError("unexpected LIBERO episode fields")
    obs = episode["obs"]
    _hard_children(obs)
    if not OBS_FIELDS <= set(obs) or set(obs) - OBS_FIELDS - {"agentview_depth", "eye_in_hand_depth"}:
        raise ContractError("missing or unexpected LIBERO observation fields")
    arrays = {key: episode[key] for key in sorted(EPISODE_FIELDS - {"obs"})}
    arrays.update({f"obs/{key}": obs[key] for key in sorted(obs)})
    n = episode["actions"].shape[0]
    if n < 2 or int(episode.attrs["num_samples"]) != n:
        raise ContractError("episode too short or sample count mismatch")
    dimensions = {"actions": 7, "obs/ee_states": 6, "obs/ee_pos": 3,
                  "obs/ee_ori": 3, "obs/gripper_states": 2, "obs/joint_states": 7}
    digest = hashlib.sha256(HASH_VERSION.encode() + b"\0")
    inventory = {}
    for name, dataset in sorted(arrays.items()):
        if (not isinstance(dataset, h5py.Dataset) or dataset.is_virtual or dataset.external
                or not dataset.shape or dataset.shape[0] != n or dataset.dtype.kind not in "fiub"):
            raise ContractError(f"invalid, external, or unaligned dataset: {name}")
        if name in dimensions and dataset.shape != (n, dimensions[name]):
            raise ContractError(f"invalid Panda shape: {name}")
        if name.endswith("_rgb") and (dataset.shape != (n, 128, 128, 3) or dataset.dtype != np.uint8):
            raise ContractError(f"invalid source RGB shape/dtype: {name}")
        dtype = {"f": "<f8", "i": "<i8", "u": "<u8", "b": "<u8"}[dataset.dtype.kind]
        descriptor = {"name": name, "shape": list(dataset.shape), "canonical_dtype": dtype}
        digest.update(json_bytes(descriptor))
        field_hash = hashlib.sha256()
        for start in range(0, n, 16):
            block = np.asarray(dataset[start:start + 16], dtype=dtype, order="C")
            if not np.isfinite(block).all():
                raise ContractError(f"nonfinite source data: {name}")
            raw = block.tobytes(order="C")
            field_hash.update(raw)
            digest.update(raw)
        inventory[name] = {**descriptor, "sha256": field_hash.hexdigest()}
    return digest.hexdigest(), inventory


def aligned_panda(episode):
    """obs[i] was rendered after actions[i]; its next command is actions[i+1]."""
    _, np = dependencies()
    obs = episode["obs"]
    ee = np.asarray(obs["ee_states"], dtype=np.float64)
    if not np.array_equal(ee[:, :3], obs["ee_pos"][()]) or not np.array_equal(ee[:, 3:], obs["ee_ori"][()]):
        raise ContractError("redundant EEF pose fields disagree")
    angle = np.linalg.norm(ee[:, 3:], axis=-1, keepdims=True)
    # sin(angle/2)/angle, including its continuous value at zero.
    xyz = ee[:, 3:] * (0.5 * np.sinc(angle / (2 * np.pi)))
    quaternion = np.concatenate((xyz, np.cos(angle / 2)), axis=-1)
    # Fixed hemisphere resolves q/-q without inventing a human orientation.
    quaternion = np.where(quaternion[:, 3:4] < 0, -quaternion, quaternion)
    state = np.concatenate((ee[:, :3], quaternion, obs["gripper_states"][()]), axis=-1)[:-1]
    action = np.asarray(episode["actions"][1:], dtype=np.float64)
    if not np.isfinite(state).all() or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-6):
        raise ContractError("nonfinite Panda values or controller action outside [-1,1]")
    return state, action


class Moments:
    """Frame-weighted population moments merged stably in deterministic order."""

    def __init__(self, width: int):
        _, np = dependencies()
        self.count = 0
        self.mean, self.m2 = np.zeros(width), np.zeros(width)
        self.minimum, self.maximum = np.full(width, np.inf), np.full(width, -np.inf)

    def add(self, values) -> None:
        _, np = dependencies()
        n = len(values)
        if not n or values.shape != (n, len(self.mean)) or not np.isfinite(values).all():
            raise ContractError("invalid normalization samples")
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + n
        self.m2 += ((values - mean)**2).sum(axis=0) + delta**2 * (self.count * n / total)
        self.mean += delta * (n / total)
        self.count = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self) -> dict:
        _, np = dependencies()
        if not self.count:
            raise ContractError("normalization requires source training frames")
        std = np.sqrt(self.m2 / self.count)
        return {"count": self.count, "mean": self.mean.tolist(), "std": std.tolist(),
                "scale": np.where(std < 1e-6, 1.0, std).tolist(),
                "minimum": self.minimum.tolist(), "maximum": self.maximum.tolist()}


def normalize(values, statistics: dict, *, inverse: bool = False):
    """Reversible z-score; no clipping. This is not a safe controller command."""
    _, np = dependencies()
    values = np.asarray(values, dtype=np.float64)
    mean, scale = np.asarray(statistics["mean"]), np.asarray(statistics["scale"])
    if (values.ndim < 1 or mean.ndim != 1 or scale.shape != mean.shape or values.shape[-1] != len(mean)
            or not np.isfinite(values).all() or not np.isfinite(mean).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ContractError("invalid normalization input/statistics")
    return values * scale + mean if inverse else (values - mean) / scale


def build_index(lock: dict, tasks: dict, config: dict, open_source) -> dict:
    """Ingest verified files; injected opener allows fully in-memory tests."""
    entries = validate_source_lock(lock, tasks)
    episodes, provenance, sources = [], {}, []
    train_hashes, seen = [], set()
    state_moments, action_moments = Moments(9), Moments(7)
    coverage = {entry["task"]: {"train": 0, "validation": 0} for entry in entries}
    for entry in entries:
        relative = f"{DIRECTORY}/raw/{entry['task']}_demo.hdf5"
        with open_source(entry, relative) as handle:
            metadata = validate_metadata(handle, entry)
            sources.append({**entry, **metadata, "path": relative})
            total = 0
            for key in sorted(handle["data"], key=lambda key: int(key.split("_")[-1])):
                episode = handle["data"][key]
                digest, inventory = episode_content_hash(episode)
                if digest in seen:
                    raise ContractError("duplicate decoded episode content, including across task labels")
                seen.add(digest)
                split = deterministic_split(digest, config["split_seed"], config["source_validation_fraction"])
                state, action = aligned_panda(episode)
                record = {"id": f"{entry['task']}/{key}", "suite": "libero_90", "task": entry["task"],
                          "split": split, "sha256": digest, "path": relative, "episode_key": f"data/{key}"}
                episodes.append(record)
                provenance[digest] = {"file_sha256": entry["sha256"], "datasets": inventory,
                                      "raw_frames": len(state) + 1, "aligned_frames": len(state)}
                total += len(state) + 1
                coverage[entry["task"]][split] += 1
                if split == "train":
                    train_hashes.append(digest)
                    state_moments.add(state)
                    action_moments.add(action)
            if total != int(handle["data"].attrs["total"]):
                raise ContractError("source frame count disagrees with metadata")
    if any(not counts[split] for counts in coverage.values() for split in ("train", "validation")):
        raise ContractError("fixed split lacks train/validation coverage for a source; do not reseed")
    manifest = {"schema_version": 1, "episodes": episodes, "human_clips": []}
    counts = validate_episode_manifest(manifest, tasks, split_seed=config["split_seed"],
                                      validation_fraction=config["source_validation_fraction"])
    return {"schema_version": 1, "source_lock": lock, "hash_version": HASH_VERSION,
            "split": {"seed": config["split_seed"], "validation_fraction": config["source_validation_fraction"]},
            "sources": sources, "manifest": manifest, "provenance": provenance,
            "counts": counts, "coverage": coverage,
            "alignment": {"observation_slice": "[:-1]", "action_slice": "[1:]", "control_hz": 20,
                          "reason": "upstream create_dataset.py records post-action observations"},
            "normalization": {"fit_split": "train", "fit_episode_sha256": sorted(train_hashes),
                              "method": "population_zscore", "constant_std_threshold": 1e-6,
                              "constant_scale": 1.0, "state_fields": STATE_FIELDS,
                              "action_fields": ACTION_FIELDS, "state": state_moments.result(),
                              "action": action_moments.result()},
            "boundary": {"target_robot_demonstrations": False, "human_modalities": "absent",
                         "simulator_states": "provenance hash only; never exposed by observation reader"}}


def ingest_sources(lock: dict, tasks: dict, config: dict, *, verify_only: bool = False) -> dict:
    entries = validate_source_lock(lock, tasks)
    root = resolve_data_root()
    h5py, _ = dependencies()
    raw = external_path(root, f"{DIRECTORY}/raw")
    expected = {entry["task"] + "_demo.hdf5" for entry in entries}
    if not raw.is_dir() or {p.name for p in raw.iterdir()} != expected:
        raise ContractError("raw source directory must contain exactly the six approved HDF5 files")

    def open_source(entry, relative):
        path = external_path(root, relative)
        verify_file(path, entry)  # no labels trusted without the pinned bytes
        return h5py.File(path, "r")

    result = build_index(lock, tasks, config, open_source)
    payload = json_bytes(result)
    destination = external_path(root, f"{DIRECTORY}/index.json")
    if verify_only or destination.exists():
        if not destination.is_file() or destination.read_bytes() != payload:
            raise ContractError("stored ingestion index differs from recomputed source evidence")
    else:
        resolve_data_root(str(root))
        temporary = external_path(root, f"{DIRECTORY}/index.json.partial")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(destination)
    if file_sha256(destination) != hashlib.sha256(payload).hexdigest():
        raise ContractError("ingestion index readback failed")
    return {"index": str(destination), "sha256": hashlib.sha256(payload).hexdigest(),
            "counts": result["counts"], "coverage": result["coverage"], "verified": True}


def read_episode(root: Path, index: dict, episode_id: str) -> dict:
    """Read an indexed robot episode only; RGB is returned upright and unscaled.

    The index must come from a completed `ingest_sources(..., verify_only=True)`
    audit. Check decoded content again before exposing this episode to training.
    """
    h5py, _ = dependencies()
    root = resolve_data_root(str(root))
    matches = [e for e in index["manifest"]["episodes"] if e["id"] == episode_id]
    if len(matches) != 1:
        raise ContractError("episode ID not uniquely present in source manifest")
    record = matches[0]
    _record_source(index, record)
    with h5py.File(external_path(root, record["path"]), "r") as handle:
        return decode_episode(handle, index, record)


def decode_episode(handle, index: dict, record: dict) -> dict:
    """Decode a verified file handle, including a cloud RAM-only HDF5 handle."""
    _, np = dependencies()
    source = _record_source(index, record)
    metadata = validate_metadata(handle, source)
    episode = handle[record["episode_key"]]
    if episode_content_hash(episode)[0] != record["sha256"]:
        raise ContractError("episode content changed since ingestion")
    state, action = aligned_panda(episode)
    rgb = {name: episode[f"obs/{name}"][:-1] for name in ("agentview_rgb", "eye_in_hand_rgb")}
    if metadata["image_convention"] == "opengl":
        rgb = {name: values[:, ::-1].copy() for name, values in rgb.items()}
    stats = index["normalization"]
    return {"rgb": rgb, "panda_state": normalize(state, stats["state"]),
            "actions": normalize(action, stats["action"]), "timestamps_seconds": np.arange(len(state)) / 20,
            "instruction": metadata["instruction"], "split": record["split"], "sha256": record["sha256"]}


def _record_source(index, record):
    sources = [s for s in index["sources"] if s["task"] == record["task"]]
    if (len(sources) != 1 or record["suite"] != "libero_90" or record["path"] != sources[0]["path"]
            or record["split"] != deterministic_split(record["sha256"], index["split"]["seed"],
                                                       index["split"]["validation_fraction"])):
        raise ContractError("episode does not match the verified source index")
    return sources[0]
