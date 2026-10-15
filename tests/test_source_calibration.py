"""Causal source windows, controller bounds, non-collapse and restart contracts."""

import copy
import importlib.util
import io
import unittest

import numpy as np

from semflow.config import ContractError
from semflow.source_batching import (controller_actions, panda_observation,
                                    source_success_gate, source_window)

TORCH = importlib.util.find_spec("torch") is not None
if TORCH:
    import torch
    from torch import nn
    from semflow.models.source_calibration import (SpatialReconstruction, checkpoint_payload,
        restore_checkpoint, noncollapse_loss)


class SourceBatchingTests(unittest.TestCase):
    def setUp(self):
        self.episode = {"rgb": {key: np.broadcast_to(np.arange(90, dtype=np.uint8)[:, None, None, None], (90, 2, 2, 3))
                                for key in ("agentview_rgb", "eye_in_hand_rgb")},
                        "panda_state": np.repeat(np.arange(90)[:, None], 9, axis=1),
                        "actions": np.repeat(np.arange(90)[:, None], 7, axis=1),
                        "instruction": "close the drawer", "sha256": "a" * 64, "split": "train"}
        self.stats = {"state": {"mean": [0.] * 9, "scale": [1.] * 9},
                      "action": {"mean": [0.] * 7, "scale": [2.] * 7}}

    def test_causal_windows_and_separate_future_labels(self):
        window = source_window(self.episode, 30)
        np.testing.assert_array_equal(window["history_rgb"][0, :, 0, 0, 0], [10, 14, 18, 22, 26, 30])
        np.testing.assert_array_equal(window["loss_targets"]["future_rgb"][0, :, 0, 0, 0], [40, 50, 70])
        np.testing.assert_array_equal(window["loss_targets"]["actions"][:, 0], np.arange(30, 62))
        self.assertEqual(set(window["loss_targets"]), {"actions", "future_rgb"})
        self.assertLessEqual(window["observation_times"].max(), window["forecast_origin"])

    def test_missing_future_is_not_padded(self):
        for anchor in (19, 50):
            with self.assertRaises(ContractError):
                source_window(self.episode, anchor)

    def test_controller_denormalization_and_safety_bounds(self):
        normalized = np.full((8, 7), .75)
        actions, clipped = controller_actions(normalized, self.stats, -np.ones(7), np.ones(7))
        np.testing.assert_array_equal(actions, np.ones((8, 7)))
        self.assertEqual(clipped, 56)
        for bad in (np.full((8, 7), np.nan), np.ones((8, 6))):
            with self.assertRaises(ContractError):
                controller_actions(bad, self.stats, -np.ones(7), np.ones(7))

    def test_measured_panda_state_and_quaternion_hemisphere(self):
        obs = {"robot0_eef_pos": np.array([.1, .2, .3]), "robot0_eef_quat": np.array([0, 0, 0, -1]),
               "robot0_gripper_qpos": np.array([.02, -.02]),
               "agentview_image": np.zeros((128, 128, 3), dtype=np.uint8),
               "robot0_eye_in_hand_image": np.zeros((128, 128, 3), dtype=np.uint8)}
        converted = panda_observation(obs, self.stats, image_convention="opengl")
        np.testing.assert_array_equal(converted["state"][3:7], [0, 0, 0, 1])
        del obs["robot0_gripper_qpos"]
        with self.assertRaises(KeyError):
            panda_observation(obs, self.stats, image_convention="opengl")

    def test_source_gate_checks_every_component_and_macro(self):
        tasks = [f"source_{i}" for i in range(6)]
        counts = dict.fromkeys(tasks, 10)
        self.assertTrue(source_success_gate(dict.fromkeys(tasks, 6), counts, tasks)["passed"])
        successes = dict.fromkeys(tasks, 8)
        successes[tasks[0]] = 2
        self.assertFalse(source_success_gate(successes, counts, tasks)["passed"])
        self.assertFalse(source_success_gate(dict.fromkeys(tasks, 3), counts, tasks)["passed"])
        with self.assertRaises(ContractError):
            source_success_gate(dict.fromkeys(tasks, 3), dict.fromkeys(tasks, 5), tasks)


@unittest.skipUnless(TORCH, "PyTorch is an optional cloud training dependency")
class SourceObjectiveTests(unittest.TestCase):
    def test_constant_observations_cannot_cheat_with_distinct_query_ids(self):
        tokens = torch.randn(1, 16, 8).repeat(4, 1, 1).requires_grad_()
        self.assertGreater(noncollapse_loss(tokens).item(), .98)

    def test_reconstruction_routes_gradients_through_scene_bottleneck(self):
        head = SpatialReconstruction(8, 4, patches=4)
        scene = torch.randn(3, 16, 8, requires_grad=True)
        target = torch.randn(3, 8, 4)
        loss = (head(scene) - target).square().mean()
        loss.backward()
        self.assertGreater(scene.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is not None for p in head.parameters()))

    def test_full_delta_optimizer_rng_and_sampler_restore_next_step(self):
        class Sampler:
            def __init__(self):
                self.rng = np.random.default_rng(42)
            def state_dict(self):
                return copy.deepcopy(self.rng.bit_generator.state)
            def load_state_dict(self, state):
                self.rng.bit_generator.state = copy.deepcopy(state)
        torch.manual_seed(23)
        model, head = nn.Linear(3, 4), nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=.01)
        sampler = Sampler()
        contract = {"base": "pinned-base", "source_index": "a" * 64}
        def step(model, head, optimizer, sampler):
            x = torch.randn(2, 3) + sampler.rng.random()
            loss = head(model(x)).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            return x, loss.detach(), copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict())
        step(model, head, optimizer, sampler)
        payload = checkpoint_payload(model, head, optimizer, sampler, step=1, contract=contract)
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=True)
        expected = step(model, head, optimizer, sampler)
        clone, clone_head = nn.Linear(3, 4), nn.Linear(4, 2)
        clone_optimizer = torch.optim.AdamW(list(clone.parameters()) + list(clone_head.parameters()), lr=.01)
        clone_sampler = Sampler()
        self.assertEqual(restore_checkpoint(restored, clone, clone_head, clone_optimizer, clone_sampler, contract=contract), 1)
        actual = step(clone, clone_head, clone_optimizer, clone_sampler)
        for a, b in zip(expected[:2], actual[:2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for before, after in zip(expected[2:], actual[2:]):
            for name in before:
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "contract"):
            restore_checkpoint(restored, clone, clone_head, clone_optimizer, clone_sampler, contract={"base": "other"})


if __name__ == "__main__":
    unittest.main()
