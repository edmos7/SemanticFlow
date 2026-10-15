"""Human input boundaries; tiny native fixtures do not establish transfer."""

import importlib.util
from types import SimpleNamespace
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'torch', 'PIL'))
if AVAILABLE:
    import numpy as np
    import torch
    from semflow.human_batching import (HumanBatchProcessor, MeasuredHumanVisionInputs,
        human_indices, letterbox, timestamp_metadata)
    from semflow.models.native_policy import NativeSemFlowPolicy
    from test_native_policy import FakeNativePolicy

    class FakeProcessor(MeasuredHumanVisionInputs):
        def encode_messages(self, messages, samples, **kwargs):
            self.messages, self.samples = messages, samples
            self.vision = self.build_vision_inputs(samples)
            count = len(samples)
            return {'input_ids': torch.tensor([[1, 2, 3, 4]]*count),
                    'attention_mask': torch.ones(count, 4, dtype=torch.long),
                    'pixel_values': torch.stack([s['chest_images'].float().mean((1, 2))/255 for s in samples]),
                    'mm_token_type_ids': torch.zeros(count, 4, dtype=torch.long)}

        def find_answer_start_idx(self, ids):
            return torch.full((len(ids),), ids.shape[1])


@unittest.skipUnless(AVAILABLE, 'numpy, torch and Pillow required')
class HumanBatchingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.native = FakeNativePolicy()
        self.processor = FakeProcessor()
        self.batch = HumanBatchProcessor(self.native, SimpleNamespace(data=SimpleNamespace(target_image_size=(8, 8))),
                                        processor=self.processor)
        self.observed = {'history_rgb': np.arange(6*8*8*3, dtype=np.uint8).reshape(6, 8, 8, 3),
                         'observation_times': np.array([-1.013, -.83, -.63, -.43, -.23, 0.]),
                         'instruction': 'Put the cup on the plate.'}

    def test_irregular_history_and_future_use_opposite_lookup_directions(self):
        times = [0., .2, .38, .6, .78, 1.01, 1.2, 1.51, 2.01, 3.01, 3.1]
        history, future = human_indices(times, 5)
        self.assertEqual(history, (0, 1, 2, 3, 4, 5))
        self.assertEqual(future, (7, 8, 9))
        for index, age in zip(history, (1., .8, .6, .4, .2, 0.)):
            self.assertLessEqual(times[index], times[5]-age)
        for anchor in (0, 3, 9):
            with self.assertRaises(ValueError): human_indices(times, anchor)

    def test_timestamp_metadata_preserves_gap_and_disables_sampling(self):
        result = self.batch([self.observed], 'cpu')
        vision = self.processor.vision
        metadata = vision['video_metadata'][0]
        actual = np.array(metadata['frames_indices'])/metadata['fps']
        expected = self.observed['observation_times']-self.observed['observation_times'][0]
        np.testing.assert_allclose(actual, expected, atol=.5e-6, rtol=0)
        self.assertFalse(vision['do_sample_frames'])
        self.assertEqual(len(vision['videos']), 1)
        self.assertEqual(self.processor.samples[0]['active_views'], ['chest'])
        self.assertNotIn('images', self.processor.samples[0])
        self.assertNotIn('panda_states', result)
        self.assertEqual(result['view_valid'].tolist(), [[False, True]])
        self.assertEqual(result['current_rgb'][:, 0].count_nonzero(), 0)

    def test_privileged_or_future_fields_fail_before_processor(self):
        for key in ('loss_targets', 'future_rgb', 'panda_states', 'actions', 'camera_motion'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'observed RGB'):
                self.batch([{**self.observed, key: object()}], 'cpu')
        self.assertFalse(hasattr(self.processor, 'messages'))

    def test_control_tokens_and_invalid_times_fail(self):
        for text in ('', 'Do <state> now', '<|im_start|>'):
            with self.assertRaises(ValueError): self.batch([{**self.observed, 'instruction': text}], 'cpu')
        for times in ([0]*6, [-1, -.8, -.6, -.4, -.2, .1], [-1, -.8, -.6, -.4, -.2, float('nan')]):
            with self.assertRaises(ValueError): self.batch([{**self.observed, 'observation_times': np.array(times)}], 'cpu')

    def test_letterbox_preserves_portrait_edges_without_stretching(self):
        image = np.full((8, 4, 3), 200, dtype=np.uint8)
        output = letterbox(image, (8, 8))
        np.testing.assert_array_equal(output[:, 2:6], image)
        self.assertFalse(output[:, :2].any())
        self.assertFalse(output[:, 6:].any())

    def test_masked_missing_view_has_no_forecast_or_scene_influence(self):
        model = NativeSemFlowPolicy(self.native, scene_dim=8).eval()
        inputs = self.batch([self.observed], 'cpu')
        first = model.encode_observations(**inputs)
        changed = {**inputs, 'current_rgb': inputs['current_rgb'].clone()}
        changed['current_rgb'][:, 0] = 255
        second = model.encode_observations(**changed)
        for a,b in ((first.raw_forecasts, second.raw_forecasts),
                    (first.current_scene, second.current_scene), (first.scene_forecasts, second.scene_forecasts)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(first.raw_forecasts[:, 0].count_nonzero(), 0)
        self.assertIsNone(self.native.last_slots['state'])

    def test_native_output_state_token_rejected(self):
        original = self.processor.encode_messages
        def injected(*args, **kwargs):
            result = original(*args, **kwargs)
            result['input_ids'][0, 0] = self.native.state_token_index
            return result
        self.processor.encode_messages = injected
        with self.assertRaisesRegex(ValueError, 'state/action slots'):
            self.batch([self.observed], 'cpu')


if __name__ == '__main__':
    unittest.main()
