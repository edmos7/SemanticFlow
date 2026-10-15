"""Balanced exposure, deterministic panel, source separation and resume contracts."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_balanced_source as balanced

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from torch import nn


def fixture():
    records, provenance, decoded = [], {}, {}
    tasks = ["task%d" % i for i in range(6)]
    for task in tasks:
        for split in ("train", "validation"):
            for i in range(3):
                key = hashlib.sha256((task + split + str(i)).encode()).hexdigest()
                record = dict(sha256=key, task=task, split=split)
                records.append(record)
                provenance[key] = {"aligned_frames": 90}
                decoded[key] = {"sha256": key, "split": split, "instruction": task,
                    "actions": np.full((90, 7), i + 1.), "panda_state": np.zeros((90, 9)),
                    "rgb": {c: np.zeros((90, 2, 2, 3), dtype=np.uint8)
                            for c in ("agentview_rgb", "eye_in_hand_rgb")}}
    return SimpleNamespace(split="train", tasks=tasks, episode=lambda r: decoded[r["sha256"]],
        episodes={t: [r for r in records if r["task"] == t and r["split"] == "train"] for t in tasks},
        index={"manifest": {"episodes": records}, "provenance": provenance,
               "normalization": {"action": {"mean": [0] * 7, "scale": [1] * 7}}})


class BalancedScheduleTests(unittest.TestCase):
    def test_every_block_is_balanced_episodes_cycle_and_windows_vary(self):
        sampler = balanced.BalancedSchedule(fixture())
        seen = {t: [] for t in sampler.tasks}
        for _ in range(600):
            record, anchor = sampler.next_record()
            w = sampler.sample()[0]
            self.assertEqual(w["provenance"]["split"], "train")
            self.assertEqual(w["provenance"]["anchor"], anchor)
            seen[record["task"]].append((record["sha256"], anchor))
        for visits in seen.values():
            self.assertEqual(len(visits), 100)
            self.assertEqual(len({v[0] for v in visits[:3]}), 3)
            self.assertGreater(len({v[1] for v in visits}), 1)

    def test_resume_reproduces_next_windows_and_rejects_changed_state(self):
        source = fixture()
        original = balanced.BalancedSchedule(source)
        for _ in range(37):
            original.sample()
        clone = balanced.BalancedSchedule(source)
        clone.load_state_dict(original.state_dict())
        for _ in range(50):
            self.assertEqual(original.next_record(), clone.next_record())
            np.testing.assert_array_equal(original.sample()[0]["loss_targets"]["actions"],
                                          clone.sample()[0]["loss_targets"]["actions"])
        for changes in ({"updates": -1}, {"updates": True}, {"updates": 1801}, {"seed": 1},
                        {"episode_order_sha256": "bad"}, {"kind": "random"}):
            with self.assertRaises(ValueError):
                clone.load_state_dict(dict(original.state_dict(), **changes))
        clone.load_state_dict(dict(original.state_dict(), updates=1800))
        with self.assertRaisesRegex(ValueError, "budget"):
            clone.sample()

    def test_validation_cannot_enter_optimizer_schedule(self):
        source = fixture()
        source.split = "validation"
        with self.assertRaises(ValueError):
            balanced.BalancedSchedule(source)
        source.split = "train"
        source.episodes[source.tasks[0]][0]["split"] = "validation"
        with self.assertRaises(ValueError):
            balanced.BalancedSchedule(source)

    def test_panel_selection_is_order_independent_and_requires_disjoint_coverage(self):
        source = fixture()
        selected = balanced.panel_records(source)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({i["record"]["sha256"] for i in selected}), 24)
        source.index["manifest"]["episodes"].reverse()
        self.assertEqual(selected, balanced.panel_records(source))
        source.index["manifest"]["episodes"] = [r for r in source.index["manifest"]["episodes"] if r["split"] == "train"]
        with self.assertRaisesRegex(ValueError, "two distinct"):
            balanced.panel_records(source)


@unittest.skipUnless(TORCH, "PyTorch is an optional cloud training dependency")
class BalancedPanelTests(unittest.TestCase):
    def test_panel_pairs_noise_is_readonly_and_reports_complete_baselines(self):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.value = nn.Parameter(torch.tensor(.2))
                self.observations = []
            def encode_observations(self, **kwargs):
                observation = object()
                self.observations.append(observation)
                return observation
            def sample_actions(self, observation, noise, *, num_steps):
                self_test.assertIs(observation, self.observations[-1])
                self_test.assertEqual(num_steps, 10)
                return noise * self.value
        self_test = self
        policy = Policy()
        reports = [{}, {}]
        rng = torch.get_rng_state().clone()
        for r in reports:
            balanced.run_panel(policy, lambda windows, device: {}, fixture(), lambda: None, r, device="cpu")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(policy.observations), 48)
        self.assertEqual(len(reports[0]["panel_summary"]["tasks"]), 6)
        self.assertEqual(len({s["initial_noise_sha256"] for w in reports[0]["panel"] for s in w["samples"]}), 4)
        for a, b in zip(reports[0]["panel"], reports[1]["panel"]):
            self.assertEqual(a["window_sha256"], b["window_sha256"])
            self.assertEqual([s["normalized_prediction"] for s in a["samples"]],
                             [s["normalized_prediction"] for s in b["samples"]])
        incomplete = copy.deepcopy(reports[0]["panel"])
        incomplete[0]["samples"].pop()
        with self.assertRaisesRegex(ValueError, "coverage"):
            balanced.summarize_panel(incomplete)
