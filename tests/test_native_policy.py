"""Runtime action/causality contracts with a tiny native-shaped test fixture.

These tests exercise the real controller interfaces; they do not load EgoSteer
or establish Panda action normalization, source calibration, or task success.
"""

import importlib.util
from types import SimpleNamespace
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from semflow.models.native_policy import NativeSemFlowPolicy
    from test_native_forecast import TinyPolicy

    class TinyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(3, 4)
            self.calls = []

        def forward(self, images):
            self.calls.append(images.detach().clone())
            features = self.projection(images.float().mean(dim=(2, 3)) / 255.0)
            return features[:, :, None].expand(-1, -1, 4, -1)

    class TinyTimeEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(1, 8)

        def forward(self, times):
            return self.projection(times[:, None])

    class TinyFlowExpert(nn.Module):
        detach_prefix_kv = False

        def __init__(self):
            super().__init__()
            self.context = nn.Linear(4, 8)
            self.output = nn.Linear(8, 8)
            self.calls = 0

        def forward(self, *, suffix_embeds, prefix_cache, suffix_position_ids,
                    cond, suffix_mask, num_parallel_chunks):
            self.calls += 1
            batch, count, _ = suffix_embeds.shape
            assert suffix_position_ids.shape == (4, batch, count)
            assert suffix_mask.shape == (batch, count) and suffix_mask.dtype == torch.bool
            assert num_parallel_chunks == 1
            observed = self.context(prefix_cache.mean(dim=1, keepdim=True))
            return self.output(torch.tanh(suffix_embeds + observed + cond)).masked_fill(~suffix_mask[..., None], 0)

    class FakeNativePolicy(TinyPolicy):
        action_horizon = 32
        num_state_tokens = 6
        vlm_hidden_size = 4
        action_hidden_size = 8
        action_dim = 48
        state_dim = 48
        action_token_index = 91
        state_token_index = 90

        def __init__(self):
            super().__init__()
            self.world_model_expert.detach_prefix_kv = False
            self.flow_expert = TinyFlowExpert()
            self.frozen_teacher = TinyTeacher()
            self.token_embedding = nn.Embedding(100, 4)
            self.pixel_projection = nn.Linear(3, 4)
            self.state_encoder = nn.Linear(48, 4)
            self.action_encoder = nn.Linear(48, 8)
            self.action_decoder = nn.Linear(8, 48)
            self.time_embedding = TinyTimeEmbedding()
            self.slot_calls = 0
            self.backbone_calls = 0
            self.last_batch = None
            self.last_slots = None

        def build_slot_embeddings(self, batch):
            self.slot_calls += 1
            if "states" in batch or "actions" in batch:
                raise AssertionError("runtime must not fabricate native human modalities")
            return {"state": None, "action": None, "camera": None}

        def forward_backbone_stream(self, batch, slots):
            self.backbone_calls += 1
            self.last_batch = batch
            self.last_slots = slots
            hidden = self.token_embedding(batch["input_ids"])
            hidden = hidden + self.pixel_projection(batch["pixel_values"].mean(dim=1))[:, None]
            if slots["state"] is not None:
                mask = batch["input_ids"] == self.state_token_index
                indices = (mask.long().cumsum(-1) - 1).clamp(min=0, max=5)
                states = slots["state"].gather(1, indices[..., None].expand(-1, -1, 4))
                hidden = hidden + states * mask[..., None]
            hidden = hidden * batch["attention_mask"][..., None]
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None].expand(hidden.shape[0], -1)
            return SimpleNamespace(last_hidden_states=hidden, prefix_cache=hidden, position_ids=positions)

        @staticmethod
        def build_action_position_ids(batch, backbone_output, action_ref):
            batch_size, count = action_ref.shape[:2]
            return torch.arange(count, device=action_ref.device)[None, None].expand(4, batch_size, -1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud training dependency")
class NativePolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(67)
        self.native = FakeNativePolicy()
        self.model = NativeSemFlowPolicy(self.native, scene_dim=8).eval()
        # Distinct synthetic horizons make permutation interventions observable
        # through the real pooling/conditioning modules without learned data.
        with torch.no_grad():
            self.model.forecast.horizon_embedding.mul_(25)
        self.batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
            "answer_start_idx": torch.tensor([4, 3]),
            "pixel_values": torch.randn(2, 6, 3, requires_grad=True),
            "image_grid_thw": None, "pixel_values_videos": None, "video_grid_thw": None,
            "mm_token_type_ids": torch.zeros(2, 4, dtype=torch.long),
        }
        self.current = torch.randint(0, 256, (2, 2, 4, 4, 3), dtype=torch.uint8)
        self.views = torch.tensor([[True, True], [True, False]])
        self.times = torch.tensor([[-2.5, -2., -1.5, -1., -0.5, 0.]]).expand(2, -1).clone()
        self.origin = torch.zeros(2)
        self.executions = []
        self.hooks = [module.register_forward_pre_hook(lambda module, _args: self.executions.append(type(module).__name__))
                      for module in self.model.modules()]

    def tearDown(self):
        for hook in self.hooks:
            hook.remove()

    def encode(self, **overrides):
        arguments = dict(batch=self.batch, current_rgb=self.current, view_valid=self.views,
                         observation_times=self.times, forecast_origin=self.origin)
        arguments.update(overrides)
        return self.model.encode_observations(**arguments)

    def assert_no_model_execution(self):
        self.assertEqual(self.executions, [])
        self.assertEqual(self.native.slot_calls, 0)
        self.assertEqual(self.native.backbone_calls, 0)
        self.assertEqual(self.native.flow_expert.calls, 0)
        self.assertEqual(self.native.world_model_expert.calls, [])
        self.assertEqual(self.native.frozen_teacher.calls, [])

    def panda_batch(self):
        ids = torch.tensor([[1, 90, 90, 90, 90, 90, 90, 2, 3]]).expand(2, -1).clone()
        return {**self.batch, "input_ids": ids, "attention_mask": torch.ones_like(ids),
                "answer_start_idx": torch.full((2,), 9), "mm_token_type_ids": torch.zeros_like(ids)}

    def test_privileged_fields_rejected_before_any_model_execution(self):
        for name in ("future_frames", "teacher_features", "actions", "states", "labels", "future_head_motion"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "observed inputs only"):
                self.encode(batch={**self.batch, name: object()})
        self.assert_no_model_execution()

    def test_answers_internal_padding_and_action_slots_rejected_before_execution(self):
        with self.assertRaisesRegex(ValueError, "complete prompt only"):
            self.encode(batch={**self.batch, "answer_start_idx": torch.tensor([3, 3])})
        gaps = self.batch["attention_mask"].clone()
        gaps[0, 1] = 0
        with self.assertRaisesRegex(ValueError, "complete prompt only"):
            self.encode(batch={**self.batch, "attention_mask": gaps, "answer_start_idx": gaps.sum(-1)})
        actions = self.batch["input_ids"].clone()
        actions[0, 1] = self.native.action_token_index
        with self.assertRaisesRegex(ValueError, "action target slots"):
            self.encode(batch={**self.batch, "input_ids": actions})
        self.assert_no_model_execution()

    def test_future_unordered_and_wrong_origin_timestamps_rejected_before_execution(self):
        future = self.times.clone()
        future[0, -1] = 0.5
        unordered = self.times.clone()
        unordered[0, 1] = -3
        stale = self.times.clone()
        stale[0, -1] = -0.1
        for timestamps in (future, unordered, stale, self.times[:, :-1]):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.encode(observation_times=timestamps)
        self.assert_no_model_execution()

    def test_missing_human_state_and_malformed_panda_state_fail_before_execution(self):
        with self.assertRaisesRegex(ValueError, "missing human state stays absent"):
            self.encode(batch=self.panda_batch())
        with self.assertRaisesRegex(ValueError, "six state slots"):
            self.encode(panda_states=torch.randn(2, 6, 9))
        with self.assertRaisesRegex(ValueError, "Panda states"):
            self.encode(batch=self.panda_batch(), panda_states=torch.randn(2, 6, 8))
        self.assert_no_model_execution()

    def test_human_rgb_path_keeps_state_absent_and_does_not_mutate_callers(self):
        before = {name: value.detach().clone() if torch.is_tensor(value) else value for name, value in self.batch.items()}
        current, views, timestamps = self.current.clone(), self.views.clone(), self.times.clone()
        observation = self.encode()
        noise = torch.randn(2, 32, 7)
        noise_before = noise.clone()
        self.model.sample_actions(observation, noise, num_steps=2)
        self.assertIsNone(self.native.last_slots["state"])
        self.assertNotIn("states", self.native.last_batch)
        self.assertNotIn("actions", self.native.last_batch)
        self.assertIsNot(observation.batch, self.batch)
        self.assertEqual(self.batch.keys(), before.keys())
        for name, expected in before.items():
            if torch.is_tensor(expected):
                torch.testing.assert_close(self.batch[name], expected, rtol=0, atol=0)
            else:
                self.assertIs(self.batch[name], expected)
        for actual, expected in ((self.current, current), (self.views, views), (self.times, timestamps), (noise, noise_before)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_paired_velocity_and_sampled_actions_depend_on_runtime_forecasts(self):
        observation = self.encode()
        noise = torch.randn(2, 32, 7)
        flow_time = torch.full((2, 32), 0.4)
        baseline_velocity = self.model.predict_velocity(observation, noise, flow_time)
        baseline_actions = self.model.sample_actions(observation, noise, num_steps=3)
        self.assertEqual(tuple(baseline_velocity.shape), (2, 32, 7))
        self.assertEqual(tuple(baseline_actions.shape), (2, 32, 7))
        torch.testing.assert_close(self.model.sample_actions(observation, noise, num_steps=3),
                                   baseline_actions, rtol=0, atol=0)
        for intervention in ("zero", "static", "shuffle"):
            with self.subTest(intervention=intervention):
                velocity = self.model.predict_velocity(observation, noise, flow_time, forecast_intervention=intervention)
                actions = self.model.sample_actions(observation, noise, num_steps=3, forecast_intervention=intervention)
                self.assertTrue(torch.isfinite(velocity).all().item())
                self.assertTrue(torch.isfinite(actions).all().item())
                self.assertGreater((velocity - baseline_velocity).abs().max().item(), 0)
                self.assertGreater((actions - baseline_actions).abs().max().item(), 0)

    def test_frozen_controller_preserves_forecast_and_observation_gradients(self):
        self.model.requires_grad_(False).eval()
        observation = self.encode()
        observation.raw_forecasts.retain_grad()
        observation.scene_forecasts.retain_grad()
        noise, time = torch.randn(2, 32, 7), torch.full((2, 32), 0.5)
        loss = self.model.predict_velocity(observation, noise, time).square().mean()
        loss.backward()
        for gradient in (observation.raw_forecasts.grad, observation.scene_forecasts.grad, self.batch["pixel_values"].grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        # Replacing the future must remove this particular causal path, while
        # observed-prefix/current-scene paths remain available independently.
        replaced = self.encode()
        zero_loss = self.model.predict_velocity(replaced, noise, time, forecast_intervention="zero").square().mean()
        gradient = torch.autograd.grad(zero_loss, replaced.raw_forecasts, allow_unused=True)[0]
        self.assertTrue(gradient is None or gradient.abs().sum().item() == 0)

    def test_six_real_panda_states_use_9d_adapter_and_keep_input_gradients(self):
        self.model.requires_grad_(False).eval()
        states = torch.randn(2, 6, 9, requires_grad=True)
        observed_adapter_inputs = []
        hook = self.model.panda_state.register_forward_pre_hook(lambda _module, inputs: observed_adapter_inputs.append(inputs[0]))
        try:
            observation = self.encode(batch=self.panda_batch(), panda_states=states)
        finally:
            hook.remove()
        self.assertEqual(len(observed_adapter_inputs), 1)
        self.assertIs(observed_adapter_inputs[0], states)
        expected_state = torch.nn.functional.linear(states, self.model.panda_state.weight, self.model.panda_state.bias)
        expected_slots = torch.nn.functional.linear(expected_state, self.native.state_encoder.weight, self.native.state_encoder.bias)
        torch.testing.assert_close(self.native.last_slots["state"], expected_slots)
        velocity = self.model.predict_velocity(observation, torch.randn(2, 32, 7), torch.zeros(2, 32))
        velocity.square().mean().backward()
        self.assertIsNotNone(states.grad)
        self.assertTrue(torch.isfinite(states.grad).all().item())
        self.assertGreater(states.grad.norm().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_training_mode_keeps_teacher_frozen_and_evaluating(self):
        self.model.train()
        self.assertTrue(self.model.training)
        self.assertFalse(self.native.frozen_teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.native.frozen_teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
