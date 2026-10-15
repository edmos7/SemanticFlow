"""Native-layout and observation-only contracts using a tiny frozen fixture."""

import importlib.util
from types import SimpleNamespace
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
    from torch import nn

    from semflow.models.native_forecast import NativeForecastReadout

    class ForbiddenModule(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("teacher/action/motion modules are forbidden during prediction")

    class TinyExpert(nn.Module):
        hidden_size = 4

        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 4)
            self.calls = []

        def forward(self, *, suffix_embeds, prefix_cache, suffix_position_ids, suffix_mask):
            self.calls.append((suffix_embeds, suffix_position_ids.clone(), suffix_mask.clone()))
            visible = suffix_embeds.masked_fill(~suffix_mask[..., None], 0)
            context = visible.sum(1, keepdim=True) / suffix_mask.sum(1)[:, None, None]
            positions = suffix_position_ids[1:].sum(0)[..., None].to(suffix_embeds.dtype) * 0.01
            return self.projection(suffix_embeds + context + prefix_cache.mean(1, keepdim=True) + positions)

    class TinyPolicy(nn.Module):
        wm_num_future_frames = 1
        wm_grid_h = 2
        wm_grid_w = 2
        wm_upsample_factor = 2

        def __init__(self):
            super().__init__()
            self.world_model_config = SimpleNamespace(action_conditioning=False, motion_conditioning=False)
            self.wm_head = nn.Module()
            self.wm_head.query_embed = nn.Parameter(torch.randn(4, 4))
            self.wm_head.view_embed = nn.Parameter(torch.randn(2, 4))
            self.wm_head.output_proj = nn.Linear(4, 16)
            self.world_model_expert = TinyExpert()
            self.frozen_teacher = ForbiddenModule()
            self.wm_action_encoder = ForbiddenModule()
            self.wm_motion_encoder = ForbiddenModule()

        def forward_world_model_stream(self, *args, **kwargs):
            raise AssertionError("training world stream would invoke future teachers")

        @staticmethod
        def _build_vision_mrope_position_ids(*, text_start, rope_start, num_frames, grid_h, grid_w, device):
            assert num_frames == 1
            count = grid_h * grid_w
            text = torch.arange(text_start, text_start + count, device=device)
            temporal = torch.full((count,), rope_start, device=device)
            rows = rope_start + torch.arange(grid_h, device=device).repeat_interleave(grid_w)
            columns = rope_start + torch.arange(grid_w, device=device).repeat(grid_h)
            return torch.stack([text, temporal, rows, columns]), text_start + count, rope_start + max(grid_h, grid_w)

        @staticmethod
        def build_suffix_position_ids_from_relative(batch, backbone_output, relative, *, device):
            positions = relative[:, None, :].expand(-1, batch["input_ids"].shape[0], -1).clone()
            positions[0] += batch["answer_start_idx"][:, None]
            positions[1:] += backbone_output.position_ids.amax(dim=-1)[None, :, None] + 1
            return positions


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud training dependency")
class NativeForecastTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.policy = TinyPolicy().requires_grad_(False).eval()
        self.model = NativeForecastReadout(4)
        self.batch = {"input_ids": torch.tensor([[3, 4, 5], [6, 7, 8]]),
                      "answer_start_idx": torch.tensor([3, 3]),
                      "attention_mask": torch.ones(2, 3, dtype=torch.bool)}
        self.prefix = torch.randn(2, 3, 4, requires_grad=True)
        self.backbone = SimpleNamespace(last_hidden_states=self.prefix, prefix_cache=self.prefix,
                                        position_ids=torch.tensor([[0, 1, 4], [0, 2, 3]]))
        self.views = torch.tensor([[True, True], [True, False]])

    def run_readout(self, **overrides):
        arguments = dict(policy=self.policy, batch=self.batch, backbone_output=self.backbone, view_valid=self.views)
        arguments.update(overrides)
        return self.model(**arguments)

    def native_reference(self):
        # Reference the native two-view suffix and its independent subpatch
        # indexing, without calling the implementation under test.
        head = self.policy.wm_head
        base = head.query_embed[None].expand(2, -1, -1)
        suffix = torch.cat([base + head.view_embed[0], base + head.view_embed[1]], dim=1)
        relative = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7],
                                 [0, 0, 0, 0, 2, 2, 2, 2],
                                 [0, 0, 1, 1, 2, 2, 3, 3],
                                 [0, 1, 0, 1, 2, 3, 2, 3]])
        positions = self.policy.build_suffix_position_ids_from_relative(self.batch, self.backbone, relative, device=self.prefix.device)
        mask = self.views.repeat_interleave(4, dim=1)
        hidden = self.policy.world_model_expert(suffix_embeds=suffix, prefix_cache=self.prefix,
                                                suffix_position_ids=positions, suffix_mask=mask)
        projected = head.output_proj(hidden.reshape(2, 2, 4, 4))
        patches = torch.empty(2, 2, 16, 4)
        for row in range(4):
            for column in range(4):
                query = (row // 2) * 2 + column // 2
                subpatch = (row % 2) * 2 + column % 2
                patches[:, :, row * 4 + column] = projected[:, :, query, subpatch * 4:(subpatch + 1) * 4]
        return patches.masked_fill(~self.views[:, :, None, None], 0), suffix, positions, mask

    def test_native_one_second_layout_and_observed_horizon_masks(self):
        expected, suffix, positions, mask = self.native_reference()
        self.policy.world_model_expert.calls.clear()
        result = self.run_readout()
        self.assertEqual(tuple(result["pred"].shape), (2, 2, 3, 16, 4))
        self.assertTrue(torch.equal(result["pred"][:, :, 1], expected))
        self.assertTrue(torch.equal(result["valid"], self.views[:, :, None].expand(-1, -1, 3)))
        self.assertEqual(len(self.policy.world_model_expert.calls), 3)
        native_call = self.policy.world_model_expert.calls[1]
        for actual, reference in zip(native_call, (suffix, positions, mask)):
            self.assertTrue(torch.equal(actual, reference))
        self.assertEqual(result["pred"][1, 1].abs().sum().item(), 0)
        self.assertGreater((result["pred"][:, :, 0] - result["pred"][:, :, 2]).abs().sum().item(), 0)

    def test_frozen_native_heads_preserve_prefix_and_horizon_gradients(self):
        output = self.run_readout()["pred"]
        output.square().mean().backward()
        for gradient in (self.prefix.grad, self.model.horizon_embedding.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.policy.parameters()))
        self.assertEqual(list(self.model.state_dict()), ["horizon_embedding"])
        self.assertEqual(list(dict(self.model.named_parameters())), ["horizon_embedding"])

    def test_horizon_learning_cannot_change_native_one_second_queries(self):
        before = self.run_readout()["pred"].detach()
        with torch.no_grad():
            self.model.horizon_embedding.add_(1)
        after = self.run_readout()["pred"]
        self.assertTrue(torch.equal(before[:, :, 1], after[:, :, 1]))
        self.assertFalse(torch.equal(before[:, :, 0], after[:, :, 0]))

    def test_missing_view_queries_cannot_affect_observed_view(self):
        self.views = torch.tensor([[True, False], [True, False]])
        before = self.run_readout()["pred"].detach()
        with torch.no_grad():
            self.policy.wm_head.view_embed[1].add_(1000)
        self.assertTrue(torch.equal(before, self.run_readout()["pred"]))

    def test_runtime_batch_and_prefix_are_not_mutated(self):
        original = {name: value.clone() for name, value in self.batch.items()}
        prefix_before = self.prefix.detach().clone()
        views_before = self.views.clone()
        self.run_readout()
        self.assertEqual(self.batch.keys(), original.keys())
        for name, value in original.items():
            self.assertTrue(torch.equal(self.batch[name], value))
        self.assertTrue(torch.equal(self.prefix, prefix_before))
        self.assertTrue(torch.equal(self.views, views_before))

    def test_privileged_keys_rejected_before_native_execution(self):
        for name in ("future_frames", "chest_future_frames", "future_head_motion", "n_future_frames",
                     "actions", "actions_valid_mask", "target_features", "teacher_tokens", "labels"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "privileged"):
                self.run_readout(batch={**self.batch, name: object()})
        self.assertEqual(self.policy.world_model_expert.calls, [])

    def test_invalid_horizons_views_and_native_shapes_fail_loudly(self):
        for horizons in ((1.0, 2.0), (0.5, 2.0, 1.0), (0.5, 1.0, float("nan")), (0.25, 1.0, 2.0)):
            with self.subTest(horizons=horizons), self.assertRaisesRegex(ValueError, "horizons"):
                NativeForecastReadout(4, horizons=horizons)
        for mask in (torch.ones(2, 2), torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 2, dtype=torch.bool)):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "view_valid"):
                self.run_readout(view_valid=mask)
        self.policy.wm_num_future_frames = 2
        with self.assertRaisesRegex(ValueError, "one-frame"):
            self.run_readout()
        self.policy.wm_num_future_frames = 1
        self.policy.wm_grid_h = 3
        with self.assertRaisesRegex(ValueError, "embedding shapes"):
            self.run_readout()

    def test_conditioning_and_missing_prefix_fail_loudly(self):
        for attribute in ("action_conditioning", "motion_conditioning"):
            setattr(self.policy.world_model_config, attribute, True)
            with self.subTest(attribute=attribute), self.assertRaisesRegex(ValueError, "conditioning disabled"):
                self.run_readout()
            setattr(self.policy.world_model_config, attribute, False)
        self.backbone.prefix_cache = None
        with self.assertRaisesRegex(ValueError, "prefix cache"):
            self.run_readout()


if __name__ == "__main__":
    unittest.main()
