"""CPU wiring/guard tests; these do not establish native RealMan compatibility."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_realman as probe
import probe_egosteer as native_probe

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import numpy as np
    import torch
    from test_native_policy import FakeNativePolicy, TinyTeacher
    from semflow.models.native_policy import NativeSemFlowPolicy

    class Teacher(TinyTeacher):
        def forward(self, images):
            features = self.projection(images.float().mean(dim=(2, 3)) / 255.)
            return features[:, :, None].expand(-1, -1, 16, -1)


class RealManPinTests(unittest.TestCase):
    def test_candidate_does_not_replace_original_base_pin(self):
        self.assertEqual(native_probe.ARTIFACTS["egosteer"][0], "EgoSteer/EgoSteer-3B-Base")
        self.assertEqual(probe.ARTIFACTS["egosteer"][1], "c41ea92b7197855dc7ab2a95ed543ddad94f7596")
        for key in ("qwen", "teacher"):
            self.assertEqual(probe.ARTIFACTS[key], native_probe.ARTIFACTS[key])

    def test_wrong_config_rejected_before_weight_read(self):
        with patch.object(probe, "file_sha", return_value="wrong") as digest:
            with self.assertRaisesRegex(ValueError, "pinned release"):
                probe.verify_files({"config.yaml": "fixture"})
        digest.assert_called_once_with("fixture")

    def test_wrong_source_split_rejected_before_episode_access(self):
        from probe_source_conditioning import TASK, WINDOWS
        source = SimpleNamespace(index={"manifest": {"episodes": [
            {"sha256": WINDOWS[0][0], "task": TASK, "split": "validation"}]}})
        with self.assertRaisesRegex(ValueError, "source train"):
            probe.select_window(source)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud dependency")
class RealManAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        native = FakeNativePolicy()
        native.frozen_teacher = Teacher()
        self.policy = NativeSemFlowPolicy(native, scene_dim=8).requires_grad_(False).eval()
        # The tiny fixture uses this observed-prefix parameter instead of real LoRA.
        native.token_embedding.weight.requires_grad_(True)
        self.parameters = [native.token_embedding.weight]
        self.window = dict(history_rgb=np.zeros((2, 6, 4, 4, 3), dtype=np.uint8),
            current_rgb=np.full((2, 4, 4, 3), 100, dtype=np.uint8),
            panda_states=np.ones((6, 9), dtype=np.float32),
            observation_times=np.arange(-5, 1, dtype=np.float32) / 5,
            forecast_origin=0., instruction="source fixture",
            provenance={"split": "train", "episode_sha256": "fixture", "anchor": 20},
            loss_targets={"future_rgb": np.full((2, 3, 4, 4, 3), 120, dtype=np.uint8),
                          "actions": np.ones((32, 7), dtype=np.float32)})

    def processor(self, windows, device):
        self.assertNotIn("loss_targets", windows[0])
        ids = torch.tensor([[1, 90, 90, 90, 90, 90, 90, 2, 3]])
        return dict(batch={"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "answer_start_idx": torch.tensor([9]), "pixel_values": torch.ones((1, 6, 3)),
            "mm_token_type_ids": torch.zeros_like(ids)},
            current_rgb=torch.from_numpy(self.window["current_rgb"][None]),
            view_valid=torch.ones((1, 2), dtype=torch.bool),
            observation_times=torch.from_numpy(self.window["observation_times"][None]),
            forecast_origin=torch.zeros(1),
            panda_states=torch.from_numpy(self.window["panda_states"][None]))

    def audit(self):
        return probe.audit(self.policy, self.processor, self.window, self.parameters, device="cpu")

    def test_backward_and_sampling_without_optimizer_or_mutation(self):
        with patch.object(torch.optim, "AdamW", side_effect=AssertionError("optimizer forbidden")):
            result = self.audit()
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["rollouts"], 0)
        self.assertEqual(result["sampled_chunks"], 3)
        self.assertEqual(result["flow_steps"], 10)
        self.assertTrue(result["repeat_bitwise_equal"])
        self.assertTrue(result["parameters_buffers_inputs_torch_rng_unchanged"])
        self.assertTrue(all(v > 0 for v in result["gradients"].values()))

    def test_extra_trainable_readout_rejected(self):
        self.policy.scene.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "trainable set"):
            self.audit()

    def test_frozen_prediction_gradient_break_is_detected(self):
        encode = self.policy.encode_observations
        from dataclasses import replace
        def detached(**kwargs):
            encoded = encode(**kwargs)
            return replace(encoded, raw_forecasts=encoded.raw_forecasts.detach())
        with patch.object(self.policy, "encode_observations", side_effect=detached):
            with self.assertRaises(RuntimeError):
                self.audit()

    def test_input_mutation_is_detected(self):
        sample = self.policy.sample_actions
        def mutate(*args, **kwargs):
            self.window["panda_states"][0, 0] += 1
            return sample(*args, **kwargs)
        with patch.object(self.policy, "sample_actions", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "mutated"):
                self.audit()


if __name__ == "__main__":
    unittest.main()
