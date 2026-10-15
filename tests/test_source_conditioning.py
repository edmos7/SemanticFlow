"""Crossover causality, paired-noise and frozen-state contracts without a GPU."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_source_conditioning as probe
from test_balanced_source import fixture
from semflow.source_batching import source_window

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from torch import nn
    from semflow.models.native_policy import EncodedObservation


def windows():
    source = fixture()
    return [source_window(source.episode(r), 20) for r in source.episodes[source.tasks[0]][:2]]


class WindowTests(unittest.TestCase):
    def test_pinned_selection_rejects_target_changes_and_nontraining_records(self):
        source = fixture()
        records = source.episodes[source.tasks[0]][:2]
        declared = tuple((r["sha256"], 20, probe.window_digest(w)) for r, w in zip(records, windows()))
        rows = [{"task": source.tasks[0], **w["provenance"], "window_sha256": digest,
                 "normalized_target": w["loss_targets"]["actions"].tolist(),
                 "samples": [{"seed": s, "integration_steps": 10} for s in probe.SEEDS]}
                for w, (_, _, digest) in zip(windows(), declared)]
        with patch.object(probe, "WINDOWS", declared), patch.object(probe, "TASK", source.tasks[0]):
            selected, references = probe.selected_windows(source, {"panel": rows})
            self.assertEqual(len(selected), 2)
            changed = copy.deepcopy(rows)
            changed[0]["normalized_target"][0][0] += 1
            with self.assertRaisesRegex(ValueError, "target"):
                probe.selected_windows(source, {"panel": changed})
            records[0]["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "train episode"):
                probe.selected_windows(source, {"panel": rows})


@unittest.skipUnless(TORCH, "PyTorch is an optional cloud training dependency")
class CrossoverTests(unittest.TestCase):
    def policy(self, mutate_noise=False, mutate_buffer=False):
        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
                self.register_buffer("counter", torch.tensor(0.))
                self.calls = []
                self.encoded = []

            def encode_observations(self, **kwargs):
                i = len(self.encoded)
                observation = EncodedObservation({"identity": i}, SimpleNamespace(prefix_cache=i),
                    torch.zeros(1, 2, 3, 1, 1), torch.full((1, 3, 1, 1), float(i)),
                    torch.full((1, 1, 1), float(i)), torch.ones(1, 2, dtype=torch.bool),
                    torch.zeros(1, 1, 1), torch.ones(1, 1, dtype=torch.bool))
                self.encoded.append(observation)
                return observation

            def sample_actions(self, observation, noise, *, num_steps):
                if mutate_noise:
                    noise.add_(1.)
                if mutate_buffer:
                    self.counter.add_(1.)
                prefix = observation.backbone.prefix_cache
                scene = observation.current_scene.item()
                self.calls.append((prefix, scene, num_steps, noise.clone()))
                return noise * .1 + prefix * 2 + scene * 3
        return Policy().requires_grad_(False).eval()

    def run_probe(self, policy):
        def processor(items, device):
            self.assertNotIn("loss_targets", items[0])
            return {}
        report = {}
        probe.run_crossover(policy, processor, windows(), fixture().index["normalization"], report,
                            lambda: None, device="cpu")
        return report

    def test_crossing_keeps_prefix_and_positions_together(self):
        policy = self.policy()
        a, b = policy.encode_observations(), policy.encode_observations()
        mixed = probe.crossed_observation(a, b)
        for name in ("batch", "backbone", "task_tokens", "task_valid", "raw_forecasts"):
            self.assertIs(getattr(mixed, name), getattr(a, name))
        self.assertIs(mixed.current_scene, b.current_scene)
        self.assertIs(mixed.scene_forecasts, b.scene_forecasts)
        self.assertIsNot(mixed.current_scene, a.current_scene)

    def test_four_cells_sixteen_samples_and_known_factor_effects(self):
        policy = self.policy()
        report = self.run_probe(policy)
        self.assertEqual(len(policy.calls), 16)
        self.assertEqual(report["sampled_chunks"], 16)
        self.assertTrue(report["parameters_buffers_windows_rng_unchanged"])
        for i in range(4):
            for j in range(4):
                self.assertTrue(torch.equal(policy.calls[i][3], policy.calls[4*j+i][3]))
        for summary in report["summary"].values():
            np.testing.assert_allclose(summary["prefix_swap_action_mse_by_scene"], [4., 4.], atol=1e-6)
            np.testing.assert_allclose(summary["scene_swap_action_mse_by_prefix"], [9., 9.], atol=1e-6)
            self.assertLess(summary["interaction_action_mse"], 1e-12)
        cells = copy.deepcopy(report["cells"])
        cells[0]["samples"].reverse()
        with self.assertRaisesRegex(ValueError, "seed pairing"):
            probe.summarize(cells, [w["loss_targets"]["actions"] for w in windows()])

    def test_rejects_noise_or_buffer_mutation_and_unfrozen_policy(self):
        with self.assertRaisesRegex(RuntimeError, "paired noise"):
            self.run_probe(self.policy(mutate_noise=True))
        with self.assertRaisesRegex(RuntimeError, "mutated weights"):
            self.run_probe(self.policy(mutate_buffer=True))
        with self.assertRaisesRegex(ValueError, "frozen eval"):
            self.run_probe(self.policy().requires_grad_(True))

    def test_rejects_validation_before_encoding(self):
        policy = self.policy()
        selected = windows()
        selected[0]["provenance"]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training windows"):
            probe.run_crossover(policy, lambda *a: {}, selected, {}, {}, lambda: None, device="cpu")
        self.assertEqual(len(policy.encoded), 0)
