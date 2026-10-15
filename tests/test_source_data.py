"""In-memory HDF5 tests: no robot dataset is written to the system disk."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from semflow.config import ContractError, load_config, load_json
from semflow.data import deterministic_split
from semflow.source_acquisition import (acquire_sources, download_file, source_url,
                                        validate_source_lock)
from semflow.storage import resolve_data_root

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = np = None

ROOT = Path(__file__).resolve().parents[1]


class SourceAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.config, self.tasks = load_config(ROOT / "configs/pilot.json")
        self.lock = load_json(ROOT / "configs/source_data.json")

    def test_six_pins_and_no_target_requests(self):
        entries = validate_source_lock(self.lock, self.tasks)
        self.assertEqual(len(entries), 6)
        for entry in entries:
            self.assertIn(f"/{self.lock['revision']}/libero_90/", source_url(self.lock, entry))
        self.assertEqual(sum(entry["bytes"] for entry in entries), 3550136643)

    def test_target_lock_rejected_before_storage_or_network(self):
        self.lock["files"][0]["task"] = self.tasks["targets"][0]["task"]
        with patch("semflow.source_acquisition.resolve_data_root") as storage:
            with patch("semflow.source_acquisition.urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ContractError, "target"):
                    acquire_sources(self.lock, self.tasks)
                storage.assert_not_called()
                network.assert_not_called()

    def test_unavailable_volume_rejected_before_network(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("semflow.source_acquisition.urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ContractError, "SEMFLOW_DATA_ROOT"):
                    acquire_sources(self.lock, self.tasks)
                network.assert_not_called()

    def test_readonly_mount_rejected(self):
        root = Path("/tmp")
        with patch.object(Path, "is_mount", lambda p: p in (root, Path("/"))):
            with patch("semflow.storage.os.statvfs", return_value=MagicMock(f_flag=os.ST_RDONLY)):
                with self.assertRaisesRegex(ContractError, "read-only"):
                    resolve_data_root(str(root))

    def test_bad_revision_duplicate_and_wrong_suite_rejected(self):
        for mutation in (lambda x: x.update(revision="main"),
                         lambda x: x.update(suite="libero_10"),
                         lambda x: x["files"].__setitem__(0, x["files"][1])):
            candidate = copy.deepcopy(self.lock)
            mutation(candidate)
            with self.assertRaises(ContractError):
                validate_source_lock(candidate, self.tasks)

    def test_download_size_hash_and_interruption_do_not_publish(self):
        # A fake filesystem exercises download failure without persisting data.
        destination, partial, root = MagicMock(), MagicMock(), MagicMock()
        destination.exists.return_value = partial.exists.return_value = False
        entry = {**self.lock["files"][0], "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
        for payload, message in ((b"abcd", "exceeds"), (b"abd", "size/hash"), (b"ab", "size/hash")):
            response = MagicMock(status=200, url="https://example.org/file")
            response.read.side_effect = [payload, b""]
            with patch("semflow.source_acquisition.resolve_data_root", return_value=root), \
                    patch("semflow.source_acquisition.external_path", side_effect=lambda r, p: partial if p.endswith(".partial") else destination), \
                    patch("semflow.source_acquisition.shutil.disk_usage", return_value=MagicMock(free=10**9)), \
                    patch("semflow.source_acquisition.os.fsync"), \
                    patch("semflow.source_acquisition.urllib.request.urlopen") as network:
                network.return_value.__enter__.return_value = response
                with self.assertRaisesRegex(ContractError, message):
                    download_file(root, self.lock, entry)
                partial.rename.assert_not_called()

    def test_download_publishes_only_after_readback_verification(self):
        destination, partial, root = MagicMock(), MagicMock(), MagicMock()
        destination.exists.return_value = partial.exists.return_value = False
        entry = {**self.lock["files"][0], "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
        response = MagicMock(status=200, url="https://example.org/file")
        response.read.side_effect = [b"ab", b"c", b""]
        with patch("semflow.source_acquisition.resolve_data_root", return_value=root), \
                patch("semflow.source_acquisition.external_path", side_effect=lambda r, p: partial if p.endswith(".partial") else destination), \
                patch("semflow.source_acquisition.shutil.disk_usage", return_value=MagicMock(free=10**9)), \
                patch("semflow.source_acquisition.os.fsync"), \
                patch("semflow.source_acquisition.urllib.request.urlopen") as network, \
                patch("semflow.source_acquisition.verify_file") as verify:
            network.return_value.__enter__.return_value = response
            self.assertEqual(download_file(root, self.lock, entry), destination)
            verify.assert_called_once_with(partial, entry)
            partial.rename.assert_called_once_with(destination)
            self.assertEqual(network.call_args.args[0].full_url, source_url(self.lock, entry))

    def test_cached_file_must_match_pin_and_never_downloads(self):
        root, destination = MagicMock(), MagicMock()
        destination.exists.return_value = True
        with patch("semflow.source_acquisition.resolve_data_root", return_value=root), \
                patch("semflow.source_acquisition.external_path", return_value=destination), \
                patch("semflow.source_acquisition.verify_file", side_effect=ContractError("hash mismatch")), \
                patch("semflow.source_acquisition.urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ContractError, "hash mismatch"):
                download_file(root, self.lock, self.lock["files"][0])
            network.assert_not_called()


@unittest.skipUnless(h5py is not None, "optional h5py/numpy unavailable")
class SourceIngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from semflow.source_data import episode_content_hash

        cls.config, cls.tasks = load_config(ROOT / "configs/pilot.json")
        cls.lock = load_json(ROOT / "configs/source_data.json")
        cls.images = {}
        # Keep two episodes in each content-derived split for every task.
        for task_index, entry in enumerate(cls.lock["files"]):
            with h5py.File.in_memory() as f:
                data = f.create_group("data")
                text = "source fixture " + entry["task"]
                entry["bddl_sha256"] = hashlib.sha256(text.encode()).hexdigest()
                data.attrs["bddl_file_content"] = text
                data.attrs["bddl_file_name"] = f"libero_90/{entry['task']}.bddl"
                instruction = re.sub(r"^.*?_SCENE[0-9]+_", "", entry["task"]).replace("_", " ")
                data.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
                data.attrs["macros_image_convention"] = "opengl"
                data.attrs["env_args"] = json.dumps({
                    "bddl_file": data.attrs["bddl_file_name"],
                    "env_kwargs": {"bddl_file_name": data.attrs["bddl_file_name"], "robots": ["Panda"],
                                   "control_freq": 20, "controller_configs": {"type": "OSC_POSE",
                                   "control_delta": True, "input_min": -1, "input_max": 1}}})
                counts, attempt = {"train": 0, "validation": 0}, 0
                while min(counts.values()) < 2:
                    key = f"demo_{sum(counts.values())}"
                    episode = data.create_group(key)
                    cls.populate(episode, task_index * 1000 + attempt)
                    split = deterministic_split(episode_content_hash(episode)[0])
                    if counts[split] < 2:
                        counts[split] += 1
                    else:
                        del data[key]
                    attempt += 1
                data.attrs["num_demos"] = 4
                data.attrs["total"] = 16
                f.flush()
                cls.images[entry["task"]] = f.id.get_file_image()

    @staticmethod
    def populate(episode, value):
        obs = episode.create_group("obs")
        ee = np.zeros((4, 6))
        ee[:, 0] = value + np.arange(4) / 10
        ee[:, 5] = np.pi / 2
        for key, values in {"ee_states": ee, "ee_pos": ee[:, :3], "ee_ori": ee[:, 3:],
                            "gripper_states": np.full((4, 2), 0.02), "joint_states": np.zeros((4, 7))}.items():
            obs.create_dataset(key, data=values)
        rgb = np.zeros((4, 128, 128, 3), dtype=np.uint8)
        rgb[:, 0] = 31
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            obs.create_dataset(key, data=rgb, compression="gzip")
        episode.create_dataset("actions", data=np.repeat(np.arange(4)[:, None] / 4, 7, axis=1))
        episode.create_dataset("states", data=np.zeros((4, 10)))
        episode.create_dataset("robot_states", data=np.zeros((4, 9)))
        episode.create_dataset("rewards", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        episode.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        episode.attrs["num_samples"] = 4

    def opener(self, entry, relative):
        return h5py.File.in_memory(file_image=self.images[entry["task"]])

    def index(self):
        from semflow.source_data import build_index
        return build_index(self.lock, self.tasks, self.config, self.opener)

    def test_complete_sources_deterministic_and_human_modalities_absent(self):
        first = self.index()
        self.assertEqual(first, self.index())
        self.assertEqual(first["counts"], {"train": 12, "validation": 12, "human_clips": 0})
        self.assertEqual(first["manifest"]["human_clips"], [])
        self.assertEqual(first["normalization"]["state"]["count"], 36)
        self.assertEqual(len(first["provenance"]), 24)

    def test_normalization_uses_only_aligned_train_frames(self):
        from semflow.source_data import aligned_panda, normalize
        index = self.index()
        states, actions = [], []
        for record in index["manifest"]["episodes"]:
            if record["split"] == "train":
                with self.opener({"task": record["task"]}, "") as f:
                    state, action = aligned_panda(f[record["episode_key"]])
                    states.append(state)
                    actions.append(action)
        stats = index["normalization"]
        for kind, values in (("state", np.concatenate(states)), ("action", np.concatenate(actions))):
            np.testing.assert_allclose(stats[kind]["mean"], values.mean(0))
            np.testing.assert_allclose(stats[kind]["std"], values.std(0), atol=1e-10)
            np.testing.assert_allclose(normalize(normalize(values, stats[kind]), stats[kind], inverse=True), values)
        self.assertEqual(stats["fit_episode_sha256"], sorted(e["sha256"] for e in index["manifest"]["episodes"] if e["split"] == "train"))

    def test_validation_outlier_cannot_change_normalization(self):
        from semflow.source_data import build_index, episode_content_hash
        original = self.index()
        record = next(e for e in original["manifest"]["episodes"] if e["split"] == "validation")
        with self.opener({"task": record["task"]}, "") as f:
            ep = f[record["episode_key"]]
            for attempt in range(100):
                pose = ep["obs/ee_states"][()]
                pose[:, 0] = 1e9 + attempt
                ep["obs/ee_states"][...] = pose
                ep["obs/ee_pos"][...] = pose[:, :3]
                if deterministic_split(episode_content_hash(ep)[0]) == "validation":
                    break
            else:
                self.fail("no deterministic validation fixture found")
            f.flush()
            changed = f.id.get_file_image()
        def opener(entry, relative):
            return h5py.File.in_memory(file_image=changed if entry["task"] == record["task"] else self.images[entry["task"]])
        revised = build_index(self.lock, self.tasks, self.config, opener)
        self.assertEqual(original["normalization"], revised["normalization"])

    def test_content_hash_ignores_episode_name_but_detects_values(self):
        from semflow.source_data import episode_content_hash
        with self.opener(self.lock["files"][0], "") as f:
            before = episode_content_hash(f["data/demo_0"])[0]
            f["data"].move("demo_0", "demo_900")
            self.assertEqual(before, episode_content_hash(f["data/demo_900"])[0])
            f["data/demo_900/actions"][0, 0] = -0.5
            self.assertNotEqual(before, episode_content_hash(f["data/demo_900"])[0])

    def test_copied_episode_with_new_id_rejected(self):
        from semflow.source_data import build_index
        def opener(entry, relative):
            f = self.opener(entry, relative)
            del f["data/demo_1"]
            f.copy("data/demo_0", "data/demo_1")
            return f
        with self.assertRaisesRegex(ContractError, "duplicate decoded"):
            build_index(self.lock, self.tasks, self.config, opener)

    def test_target_metadata_cannot_hide_behind_source_filename(self):
        from semflow.source_data import validate_metadata
        with self.opener(self.lock["files"][0], "") as f:
            f["data"].attrs["bddl_file_name"] = self.tasks["targets"][0]["task"] + ".bddl"
            with self.assertRaisesRegex(ContractError, "task identity"):
                validate_metadata(f, self.lock["files"][0])

    def test_embedded_target_content_rejected_even_with_source_name(self):
        from semflow.source_data import validate_metadata
        with self.opener(self.lock["files"][0], "") as f:
            f["data"].attrs["bddl_file_content"] = "target composition"
            with self.assertRaisesRegex(ContractError, "BDDL mismatch"):
                validate_metadata(f, self.lock["files"][0])

    def test_missing_embedded_bddl_is_recorded_as_absent(self):
        from semflow.source_data import validate_metadata
        with self.opener(self.lock["files"][0], "") as f:
            del f["data"].attrs["bddl_file_content"]
            result = validate_metadata(f, self.lock["files"][0])
            self.assertIsNone(result["embedded_bddl_sha256"])
            self.assertEqual(result["bddl_sha256"], self.lock["files"][0]["bddl_sha256"])

    def test_target_instruction_rejected_without_embedded_bddl(self):
        from semflow.source_data import validate_metadata
        with self.opener(self.lock["files"][0], "") as f:
            del f["data"].attrs["bddl_file_content"]
            f["data"].attrs["problem_info"] = json.dumps({"language_instruction": self.tasks["targets"][0]["instruction"]})
            with self.assertRaisesRegex(ContractError, "instruction"):
                validate_metadata(f, self.lock["files"][0])

    def test_alignment_and_quaternion_order(self):
        from semflow.source_data import aligned_panda
        with self.opener(self.lock["files"][0], "") as f:
            state, action = aligned_panda(f["data/demo_0"])
            self.assertEqual(state.shape, (3, 9))
            np.testing.assert_allclose(state[:, 3:7], np.tile([0, 0, 2**-0.5, 2**-0.5], (3, 1)))
            np.testing.assert_allclose(action[:, 0], [0.25, 0.5, 0.75])

    def test_invalid_values_and_links_rejected(self):
        from semflow.source_data import aligned_panda, episode_content_hash
        with self.opener(self.lock["files"][0], "") as f:
            ep = f["data/demo_0"]
            ep["actions"][1, 0] = 1.1
            with self.assertRaisesRegex(ContractError, "outside"):
                aligned_panda(ep)
            ep["actions"][1, 0] = np.nan
            with self.assertRaisesRegex(ContractError, "nonfinite"):
                episode_content_hash(ep)
            del ep["actions"]
            ep["actions"] = h5py.ExternalLink("target.hdf5", "data/demo_0/actions")
            with self.assertRaisesRegex(ContractError, "links"):
                episode_content_hash(ep)

    def test_reader_excludes_simulator_fields_and_flips_rgb(self):
        from semflow.source_data import read_episode
        index = self.index()
        record = index["manifest"]["episodes"][0]
        handle = self.opener({"task": record["task"]}, "")
        with patch("semflow.source_data.resolve_data_root", return_value=Path("/data")), \
                patch("semflow.source_data.external_path", return_value=Path("/data/source.hdf5")), \
                patch("h5py.File", return_value=handle):
            result = read_episode(Path("/data"), index, record["id"])
        self.assertEqual(set(result), {"rgb", "panda_state", "actions", "timestamps_seconds", "instruction", "split", "sha256"})
        self.assertTrue((result["rgb"]["agentview_rgb"][:, -1] == 31).all())
        self.assertTrue((result["rgb"]["agentview_rgb"][:, 0] == 0).all())
        np.testing.assert_allclose(result["timestamps_seconds"], [0, 0.05, 0.1])

    def test_raw_directory_rejects_target_file_before_opening(self):
        from semflow.source_data import ingest_sources
        raw = MagicMock()
        raw.is_dir.return_value = True
        raw.iterdir.return_value = [Path(entry["task"] + "_demo.hdf5") for entry in self.lock["files"]] + [Path("target_demo.hdf5")]
        with patch("semflow.source_data.resolve_data_root", return_value=Path("/data")), \
                patch("semflow.source_data.external_path", return_value=raw), \
                patch("h5py.File") as opener:
            with self.assertRaisesRegex(ContractError, "exactly the six"):
                ingest_sources(self.lock, self.tasks, self.config)
            opener.assert_not_called()

    def test_verify_recomputes_index_and_rejects_changed_statistics(self):
        from semflow.source_acquisition import json_bytes
        from semflow.source_data import ingest_sources
        index = self.index()
        raw, destination = MagicMock(), MagicMock()
        raw.is_dir.return_value = destination.is_file.return_value = True
        raw.iterdir.return_value = [Path(entry["task"] + "_demo.hdf5") for entry in self.lock["files"]]
        payload = json_bytes(index)
        destination.read_bytes.return_value = payload
        memory_open = h5py.File.in_memory
        def path_for(root, relative):
            if relative.endswith("/raw"):
                return raw
            if relative.endswith("index.json"):
                return destination
            return Path(relative)
        def open_file(path, mode):
            return memory_open(file_image=self.images[path.name.removesuffix("_demo.hdf5")])
        with patch("semflow.source_data.resolve_data_root", return_value=Path("/data")), \
                patch("semflow.source_data.external_path", side_effect=path_for), \
                patch("semflow.source_data.verify_file") as verify, \
                patch("semflow.source_data.file_sha256", return_value=hashlib.sha256(payload).hexdigest()), \
                patch("h5py.File", side_effect=open_file):
            self.assertTrue(ingest_sources(self.lock, self.tasks, self.config, verify_only=True)["verified"])
            self.assertEqual(verify.call_count, 6)
            index["normalization"]["state"]["mean"][0] += 1
            destination.read_bytes.return_value = json_bytes(index)
            with self.assertRaisesRegex(ContractError, "recomputed"):
                ingest_sources(self.lock, self.tasks, self.config, verify_only=True)
            destination.open.assert_not_called()


if __name__ == "__main__":
    unittest.main()
