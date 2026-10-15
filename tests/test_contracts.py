"""Tests for contamination, causal alignment, and fail-loud storage contracts."""

import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from semflow.config import ContractError, load_config, validate_config, validate_task_manifest
from semflow.data import deterministic_split, forecast_window, validate_episode_manifest
from semflow.storage import resolve_data_root


ROOT = Path(__file__).resolve().parents[1]


class ResearchContracts(unittest.TestCase):
    def setUp(self):
        self.config, self.tasks = load_config(ROOT / "configs/pilot.json")
        self.digest = hashlib.sha256(b"robot episode").hexdigest()
        self.robot = {
            "id": "source/demo_0",
            "suite": "libero_90",
            "task": self.tasks["targets"][0]["source_tasks"][0],
            "split": deterministic_split(self.digest),
            "sha256": self.digest,
            "path": "datasets/source.hdf5",
            "episode_key": "data/demo_0",
        }
        self.human = {
            "id": "human/basket",
            "target_id": self.tasks["targets"][0]["id"],
            "path": "recordings/basket.mp4",
            "sha256": hashlib.sha256(b"human video").hexdigest(),
            "instruction": "Put both food containers in the basket.",
            "viewpoint": "egocentric",
            "timestamps_seconds": [0, 0.5, 1.0, 2.0, 3.0],
        }

    def manifest(self):
        return {"schema_version": 1, "episodes": [self.robot], "human_clips": [self.human]}

    def test_actual_task_contract_and_unset_training_choices(self):
        self.assertEqual(len(self.tasks["targets"]), 3)
        with self.assertRaisesRegex(ContractError, "not training-ready"):
            validate_config(self.config, training_ready=True)

    def test_no_robot_target_demonstrations(self):
        self.robot["task"] = self.tasks["targets"][0]["task"]
        with self.assertRaisesRegex(ContractError, "held-out"):
            validate_episode_manifest(self.manifest(), self.tasks)

    def test_target_cannot_appear_as_source(self):
        self.tasks["targets"][0]["source_tasks"][0] = self.tasks["targets"][1]["task"]
        with self.assertRaisesRegex(ContractError, "target task appears"):
            validate_task_manifest(self.tasks)

    def test_episode_content_cannot_cross_splits_under_new_id(self):
        manifest = self.manifest()
        duplicate = {**self.robot, "id": "copied", "split": "validation" if self.robot["split"] == "train" else "train"}
        manifest["episodes"].append(duplicate)
        with self.assertRaisesRegex(ContractError, "duplicate content"):
            validate_episode_manifest(manifest, self.tasks)

    def test_whole_episode_split_reproducible_and_enforced(self):
        self.assertEqual(deterministic_split(self.digest), deterministic_split(self.digest))
        self.robot["split"] = "validation" if self.robot["split"] == "train" else "train"
        with self.assertRaisesRegex(ContractError, "split must"):
            validate_episode_manifest(self.manifest(), self.tasks)

    def test_rgb_only_human_clip_accepted_and_robot_modalities_rejected(self):
        self.assertEqual(validate_episode_manifest(self.manifest(), self.tasks)["human_clips"], 1)
        for field in ("actions", "proprioception", "hand_trajectory", "boxes"):
            clip = {**self.human, field: []}
            with self.subTest(field=field), self.assertRaisesRegex(ContractError, "unknown"):
                validate_episode_manifest({"schema_version": 1, "episodes": [], "human_clips": [clip]}, self.tasks)

    def test_one_egocentric_clip_per_target(self):
        self.human["viewpoint"] = "third_person"
        with self.assertRaisesRegex(ContractError, "egocentric"):
            validate_episode_manifest(self.manifest(), self.tasks)
        self.human["viewpoint"] = "egocentric"
        manifest = self.manifest()
        manifest["human_clips"].append({**self.human, "id": "another"})
        with self.assertRaisesRegex(ContractError, "at most one"):
            validate_episode_manifest(manifest, self.tasks)

    def test_missing_data_does_not_count_as_complete(self):
        empty = {"schema_version": 1, "episodes": [], "human_clips": []}
        self.assertEqual(validate_episode_manifest(empty, self.tasks)["train"], 0)
        with self.assertRaisesRegex(ContractError, "incomplete"):
            validate_episode_manifest(empty, self.tasks, require_complete=True)

    def test_paths_cannot_escape_data_root(self):
        for value in ("../../video.mp4", "/tmp/video.mp4", "..\\video.mp4"):
            self.human["path"] = value
            with self.subTest(path=value), self.assertRaisesRegex(ContractError, "clip path"):
                validate_episode_manifest(self.manifest(), self.tasks)

    def test_causal_history_excludes_supervised_futures(self):
        timestamps = [0.0, 0.1, 0.25, 0.7, 0.8, 1.3, 2.3, 2.8]
        window = forecast_window(timestamps, 2, history_frames=3)
        self.assertEqual(window.observation_indices, (0, 1, 2))
        self.assertEqual(window.target_indices, (4, 5, 6))
        self.assertLess(max(window.observation_indices), min(window.target_indices))
        for index, horizon in zip(window.target_indices, window.horizons_seconds):
            self.assertGreaterEqual(timestamps[index] - timestamps[2], horizon)

    def test_future_window_does_not_repeat_last_frame_at_clip_end(self):
        with self.assertRaisesRegex(ContractError, "does not cover"):
            forecast_window([0, 0.5, 1, 2], 1)

    def test_invalid_and_sparse_timestamps_rejected(self):
        for timestamps in ([0, 1, 1, 3], [0, float("nan"), 3], [0, -1, 3]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ContractError):
                forecast_window(timestamps, 0)
        with self.assertRaisesRegex(ContractError, "too sparse"):
            forecast_window([0, 3], 0)

    def test_locked_objective_contract_rejects_target_selection(self):
        config = copy.deepcopy(self.config)
        config["target_checkpoint_selection"] = True
        with self.assertRaisesRegex(ContractError, "target_checkpoint_selection"):
            validate_config(config)


class StorageContracts(unittest.TestCase):
    def test_no_implicit_data_root(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ContractError, "set SEMFLOW_DATA_ROOT"):
            resolve_data_root()

    def test_missing_volume_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "unmounted" / "data"
            with self.assertRaisesRegex(ContractError, "unavailable"):
                resolve_data_root(str(missing))
            self.assertFalse(missing.exists())

    def test_leftover_mountpoint_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Path, "is_mount", lambda path: path == Path("/")):
                with self.assertRaisesRegex(ContractError, "system root"):
                    resolve_data_root(directory)

    def test_existing_mounted_data_root_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.object(Path, "is_mount", lambda path: path in (root, Path("/"))):
                self.assertEqual(resolve_data_root(directory), root)


if __name__ == "__main__":
    unittest.main()
