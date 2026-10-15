"""Scientific/gradient contracts; skipped locally when optional torch is absent."""

import importlib.util
import inspect
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from semflow.models import (
        CausalForecastReadout,
        ForecastActionConditioner,
        SceneResampler,
        SourceFeatureStatistics,
        raw_future_feature_loss,
        semantic_flow_loss,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud training dependency")
class SemanticLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.stats = SourceFeatureStatistics(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0]), "source-manifest-test")

    def test_source_calibration_excludes_padding_and_rejects_target_split(self):
        features = torch.tensor([[[1.0, 3.0]], [[3.0, 7.0]], [[900.0, 900.0]]], requires_grad=True)
        valid = torch.tensor([True, True, False])
        stats = SourceFeatureStatistics.fit(features, valid, source_manifest="source-split-hash")
        torch.testing.assert_close(stats.mean, torch.tensor([2.0, 5.0]))
        torch.testing.assert_close(stats.scale, torch.tensor([1.0, 2.0]))
        self.assertFalse(stats.mean.requires_grad)
        self.assertFalse(stats.scale.requires_grad)
        with self.assertRaisesRegex(ValueError, "source_train"):
            SourceFeatureStatistics.fit(features, valid, source_manifest="target", split="target")

    def test_teacher_detached_prediction_gradient_and_source_normalization(self):
        target = torch.zeros(1, 3, 1, 2, requires_grad=True)
        prediction = torch.tensor([[[[2.0, 4.0]], [[4.0, 8.0]], [[6.0, 12.0]]]], requires_grad=True)
        result = semantic_flow_loss(prediction, target, torch.ones(1, 3, dtype=torch.bool), self.stats)
        # Normalized residuals are [1,1], [2,2], [3,3]; delta residuals are [1,1].
        torch.testing.assert_close(result.state, torch.tensor(14.0 / 3))
        torch.testing.assert_close(result.transition, torch.tensor(1.0))
        self.assertEqual(result.valid_states, 3)
        self.assertEqual(result.valid_transitions, 2)
        result.total.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertIsNone(self.stats.mean.grad)

    def test_invalid_horizons_do_not_change_loss_or_gradients(self):
        target = torch.randn(2, 3, 2, 2)
        prediction = torch.randn(2, 3, 2, 2, requires_grad=True)
        valid = torch.tensor([[True, True, False], [True, True, True]])
        original = semantic_flow_loss(prediction, target, valid, self.stats)
        original.total.backward()
        changed = prediction.detach().clone()
        changed[0, 2] = 1e30
        changed.requires_grad_()
        altered_target = target.clone()
        altered_target[0, 2] = -1e30
        result = semantic_flow_loss(changed, altered_target, valid, self.stats)
        result.total.backward()
        torch.testing.assert_close(result.total, original.total)
        torch.testing.assert_close(changed.grad, prediction.grad)
        self.assertEqual(changed.grad[0, 2].abs().sum().item(), 0)

    def test_temporal_shuffle_changes_ordered_flow(self):
        target = torch.tensor([[[[0.0, 0.0]], [[2.0, 4.0]], [[8.0, 12.0]]]])
        valid = torch.ones(1, 3, dtype=torch.bool)
        ordered = semantic_flow_loss(target.clone(), target, valid, self.stats)
        shuffled = semantic_flow_loss(target.clone(), target.flip(1), valid, self.stats)
        self.assertEqual(ordered.total.item(), 0)
        self.assertGreater(shuffled.state.item(), 0)
        self.assertGreater(shuffled.transition.item(), 0)

    def test_missing_supervision_and_missing_transition_fail_loudly(self):
        values = torch.zeros(1, 3, 1, 2)
        with self.assertRaisesRegex(ValueError, "at least one valid"):
            semantic_flow_loss(values, values, torch.zeros(1, 3, dtype=torch.bool), self.stats)
        valid = torch.tensor([[True, False, True]])
        with self.assertRaisesRegex(ValueError, "adjacent valid"):
            semantic_flow_loss(values, values, valid, self.stats)
        state_only = semantic_flow_loss(values, values, valid, self.stats, transition_weight=0)
        self.assertEqual(state_only.valid_transitions, 0)

    def test_raw_feature_loss_preserves_scale_and_detaches_target(self):
        prediction = torch.ones(1, 3, 2, 2, requires_grad=True)
        target = torch.zeros_like(prediction, requires_grad=True)
        valid = torch.ones(1, 3, dtype=torch.bool)
        baseline = raw_future_feature_loss(prediction, target, valid)
        scaled = raw_future_feature_loss(prediction * 10, target, valid)
        torch.testing.assert_close(scaled, baseline * 100)
        baseline.backward()
        self.assertIsNone(target.grad)
        self.assertGreater(prediction.grad.abs().sum().item(), 0)

    def test_raw_masked_extreme_padding_has_zero_gradient(self):
        prediction = torch.tensor([[[[1.0]], [[2.0]], [[1e30]]]], requires_grad=True)
        target = torch.tensor([[[[0.0]], [[0.0]], [[-1e30]]]])
        result = raw_future_feature_loss(prediction, target, torch.tensor([[True, True, False]]))
        torch.testing.assert_close(result, torch.tensor(2.5))
        result.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all().item())
        self.assertEqual(prediction.grad[0, 2, 0, 0].item(), 0)

    def test_nan_padding_and_invalid_masks_are_rejected(self):
        values = torch.zeros(1, 3, 1, 2)
        values[0, 2, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN"):
            raw_future_feature_loss(values, torch.zeros_like(values), torch.tensor([[True, True, False]]))
        with self.assertRaisesRegex(ValueError, "boolean"):
            raw_future_feature_loss(torch.zeros_like(values), torch.zeros_like(values), torch.ones(1, 3))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional cloud training dependency")
class SemanticInterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_resampler_whole_scene_shape_and_padding_invariance(self):
        model = SceneResampler(8, 6, dim=16, num_queries=16, num_heads=2).eval()
        patches = torch.randn(2, 5, 8)
        task = torch.randn(2, 3, 6)
        patch_valid = torch.tensor([[True, True, True, True, False], [True, True, True, True, True]])
        task_valid = torch.tensor([[True, True, False], [True, True, True]])
        expected = model(patches, task, patch_valid=patch_valid, task_valid=task_valid)
        patches[0, 4] = 1e30
        task[0, 2] = -1e30
        actual = model(patches, task, patch_valid=patch_valid, task_valid=task_valid)
        self.assertEqual(tuple(actual.shape), (2, 16, 16))
        torch.testing.assert_close(actual, expected)

    def test_frozen_scene_teacher_allows_prediction_gradient(self):
        teacher = SceneResampler(8, 6, dim=16, num_queries=4, num_heads=2).eval().requires_grad_(False)
        patches = torch.randn(2, 5, 8, requires_grad=True)
        scene = teacher(patches, torch.randn(2, 3, 6), patch_valid=torch.ones(2, 5, dtype=torch.bool), task_valid=torch.ones(2, 3, dtype=torch.bool))
        # A channel-specific objective avoids the constant sum/variance of LayerNorm.
        scene[..., 0].sum().backward()
        self.assertGreater(patches.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_empty_scene_or_task_attention_fails(self):
        teacher = SceneResampler(8, 6, dim=16, num_heads=2)
        with self.assertRaisesRegex(ValueError, "patch_valid"):
            teacher(torch.zeros(1, 5, 8), torch.zeros(1, 2, 6), patch_valid=torch.zeros(1, 5, dtype=torch.bool), task_valid=torch.ones(1, 2, dtype=torch.bool))

    def test_readout_rejects_future_hidden_state_timestamp(self):
        readout = CausalForecastReadout(8, 4, dim=16, num_heads=2)
        with self.assertRaisesRegex(ValueError, "future observations"):
            readout(torch.randn(1, 3, 8), observation_times=torch.tensor([[0.0, 1.0, 2.0]]), forecast_origin=torch.tensor([1.0]), history_valid=torch.ones(1, 3, dtype=torch.bool))
        self.assertNotIn("future_features", inspect.signature(readout.forward).parameters)
        self.assertNotIn("teacher", inspect.signature(readout.forward).parameters)

    def test_readout_rejects_time_reordering_across_padding(self):
        readout = CausalForecastReadout(8, 4, dim=16, num_heads=2)
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            readout(torch.randn(1, 3, 8), observation_times=torch.tensor([[1.0, 9.0, 0.0]]), forecast_origin=torch.tensor([1.0]), history_valid=torch.tensor([[True, False, True]]))

    def test_frozen_readout_gradients_and_padding_invariance(self):
        readout = CausalForecastReadout(8, 2, num_tokens=4, dim=16, num_heads=2).eval().requires_grad_(False)
        history = torch.randn(1, 3, 8, requires_grad=True)
        valid = torch.tensor([[True, True, False]])
        times = torch.tensor([[0.0, 1.0, 100.0]])
        forecast = readout(history, observation_times=times, forecast_origin=torch.tensor([1.0]), history_valid=valid)
        self.assertEqual(tuple(forecast.shape), (1, 3, 4, 2))
        changed = history.detach().clone()
        changed[0, 2] = 1e30
        result = readout(changed, observation_times=times, forecast_origin=torch.tensor([1.0]), history_valid=valid)
        torch.testing.assert_close(result, forecast)
        statistics = SourceFeatureStatistics(torch.zeros(2), torch.ones(2), "source")
        loss = semantic_flow_loss(forecast, torch.zeros_like(forecast), torch.ones(1, 3, dtype=torch.bool), statistics)
        loss.total.backward()
        self.assertGreater(history.grad[:, :2].abs().sum().item(), 0)
        self.assertEqual(history.grad[:, 2].abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in readout.parameters()))

    def _action_inputs(self):
        return (torch.randn(1, 4, 8), torch.randn(1, 5, 6), torch.randn(1, 3, 4, 2),
                torch.ones(1, 5, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))

    def test_action_tokens_use_predicted_future_and_temporal_order(self):
        model = ForecastActionConditioner(8, 6, 2, dim=16, num_heads=2).eval()
        action, current, future, current_valid, future_valid = self._action_inputs()
        future.requires_grad_()
        result = model(action, current, future, current_valid=current_valid, future_valid=future_valid)
        shuffled = model(action, current, future.flip(1), current_valid=current_valid, future_valid=future_valid)
        changed = model(action, current, future + 2, current_valid=current_valid, future_valid=future_valid)
        self.assertEqual(result.shape, action.shape)
        self.assertFalse(torch.allclose(result, shuffled))
        self.assertFalse(torch.allclose(result, changed))
        result.square().sum().backward()
        self.assertGreater(future.grad.abs().sum().item(), 0)

    def test_action_residual_preserves_original_tokens_and_inputs(self):
        model = ForecastActionConditioner(8, 6, 2, dim=16, num_heads=2).eval()
        action, current, future, current_valid, future_valid = self._action_inputs()
        action_copy, current_copy, future_copy = action.clone(), current.clone(), future.clone()
        action_valid = torch.tensor([[True, True, True, False]])
        result = model(action, current, future, current_valid=current_valid, future_valid=future_valid, action_valid=action_valid)
        torch.testing.assert_close(result[:, 3], action[:, 3])
        torch.testing.assert_close(action, action_copy)
        torch.testing.assert_close(current, current_copy)
        torch.testing.assert_close(future, future_copy)
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.zero_()
        unchanged = model(action, current, future, current_valid=current_valid, future_valid=future_valid)
        torch.testing.assert_close(unchanged, action)

    def test_action_padding_invariance_and_missing_forecast_rejection(self):
        model = ForecastActionConditioner(8, 6, 2, dim=16, num_heads=2).eval()
        action, current, future, current_valid, future_valid = self._action_inputs()
        current_valid[0, 4] = False
        future_valid[0, 2] = False
        result = model(action, current, future, current_valid=current_valid, future_valid=future_valid)
        current[0, 4] = 1e30
        future[0, 2] = -1e30
        changed = model(action, current, future, current_valid=current_valid, future_valid=future_valid)
        torch.testing.assert_close(result, changed)
        with self.assertRaisesRegex(ValueError, "future_valid"):
            model(action, current, future, current_valid=current_valid, future_valid=torch.zeros_like(future_valid))


if __name__ == "__main__":
    unittest.main()
