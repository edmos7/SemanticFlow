"""Probe contracts: fixed provenance, controller metrics, paired sampling and optimizer scope."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_source_actions as probe

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from torch import nn


def fixed_fixture():
    episode = {"rgb": {c: np.zeros((80, 2, 2, 3), dtype=np.uint8)
                       for c in ("agentview_rgb", "eye_in_hand_rgb")},
               "panda_state": np.zeros((80, 9)), "actions": np.zeros((80, 7)),
               "instruction": "close the drawer", "sha256": probe.EPISODE, "split": "train"}
    record = {"sha256": probe.EPISODE, "task": probe.TASK, "split": "train"}
    sampler = SimpleNamespace(index={"manifest": {"episodes": [record]}}, episode=lambda _: episode)
    return sampler, episode


class ProbeContractTests(unittest.TestCase):
    def test_fixed_window_and_checkpoint_reject_provenance_or_content_changes(self):
        sampler, episode = fixed_fixture()
        fixed = probe.FixedWindow(sampler)
        fixed.load_state_dict(fixed.state_dict())
        for key, value in (("anchor", 21), ("split", "validation"), ("window_sha256", "bad")):
            with self.assertRaises(ValueError):
                fixed.load_state_dict(dict(fixed.state_dict(), **{key: value}))
        self.assertIs(fixed.sample()[0], fixed.sample()[0])
        fixed.window["loss_targets"]["actions"][0, 0] = 9
        with self.assertRaisesRegex(ValueError, "contents"):
            fixed.sample()
        for field, value in (("task", "target"), ("split", "validation"), ("sha256", "bad")):
            altered = copy.deepcopy(sampler.index)
            altered["manifest"]["episodes"][0][field] = value
            with self.assertRaisesRegex(ValueError, "uniquely"):
                probe.FixedWindow(SimpleNamespace(index=altered, episode=lambda _: episode))

    def test_gripper_sign_uses_denormalized_actions_and_prefix_clipping(self):
        stats = {"action": {"mean": [0] * 6 + [-.5], "scale": [1] * 7}}
        target = np.zeros((32, 7))
        prediction = target.copy()
        prediction[:, 6] = .25  # positive normalized, negative controller command
        prediction[8:, 0] = 2
        result = probe.metrics(prediction, target, stats)
        self.assertEqual(result["full"]["gripper_sign_accuracy"], 1)
        self.assertEqual(result["full"]["gripper_mean"], -.25)
        self.assertEqual(result["clipped_values"], 24)
        self.assertEqual(result["first_eight"]["clipped_values"], 0)
        self.assertEqual(result["full"]["per_dimension_mse"][0], 3)
        self.assertEqual(result["first_eight"]["per_dimension_mse"][0], 0)
        for bad in (np.full((32, 7), np.inf), np.zeros((8, 7))):
            with self.assertRaises(ValueError):
                probe.metrics(bad, target, stats)


@unittest.skipUnless(TORCH, "PyTorch is an optional cloud training dependency")
class ProbeTorchTests(unittest.TestCase):
    def test_audit_encodes_once_pairs_noise_and_never_calls_optimizer(self):
        fixed = probe.FixedWindow(fixed_fixture()[0])
        observation, calls = object(), []
        class Policy:
            def eval(self):
                return self
            def encode_observations(self, **kwargs):
                calls.append("encode")
                return observation
            def sample_actions(self, encoded, noise, *, num_steps):
                self_test.assertIs(encoded, observation)
                calls.append((num_steps, noise.clone()))
                return noise / num_steps
        self_test = self
        result = {}
        probe.audit(Policy(), lambda windows, device: {}, fixed,
                    {"action": {"mean": [0] * 7, "scale": [1] * 7}}, lambda: None, result, device="cpu")
        self.assertEqual(calls.count("encode"), 1)
        self.assertEqual(len(result["samples"]), 16)
        for i in range(4):
            group = result["samples"][i * 4:(i + 1) * 4]
            self.assertEqual([s["integration_steps"] for s in group], [10, 20, 50, 100])
            self.assertEqual(len({s["initial_noise_sha256"] for s in group}), 1)
        self.assertEqual(len({s["initial_noise_sha256"] for s in result["samples"]}), 4)

    def test_optimizer_rejects_reset_counters_hyperparameters_scope_and_order(self):
        policy = nn.Module()
        policy.native = nn.Module()
        policy.native.lora_A = nn.Parameter(torch.ones(2, 2))
        policy.native.base = nn.Parameter(torch.ones(2), requires_grad=False)
        policy.panda_action_in = nn.Linear(2, 2)
        reconstruction = nn.Linear(2, 2)
        config = {"learning_rate": .0003, "adapter_learning_rate": .00005, "weight_decay": .01}
        optimizer = torch.optim.AdamW([
            {"params": list(policy.panda_action_in.parameters()) + list(reconstruction.parameters()), "lr": .0003},
            {"params": [policy.native.lora_A], "lr": .00005}], weight_decay=.01, foreach=False)
        with patch("semflow.models.adapters.named_shared_adapter_parameters", return_value=[("lora_A", policy.native.lora_A)]):
            probe.optimizer_contract(policy, reconstruction, optimizer, config, 0)
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, -1)
            with self.assertRaisesRegex(ValueError, "state/counter"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 600)
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad = torch.ones_like(p)
            optimizer.step()
            with self.assertRaisesRegex(ValueError, "fresh optimizer"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 0)
            probe.optimizer_contract(policy, reconstruction, optimizer, config, 1)
            with self.assertRaisesRegex(ValueError, "state/counter"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 600)
            optimizer.param_groups[0]["lr"] = .5
            with self.assertRaisesRegex(ValueError, "hyperparameters"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 1)
            optimizer.param_groups[0]["lr"] = .0003
            optimizer.param_groups[0]["params"].reverse()
            with self.assertRaisesRegex(ValueError, "order/scope"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 1)
            optimizer.param_groups[0]["params"].reverse()
            policy.native.base.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, "scope"):
                probe.optimizer_contract(policy, reconstruction, optimizer, config, 1)
