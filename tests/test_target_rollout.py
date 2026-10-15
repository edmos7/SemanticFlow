"""Target boundaries, causal observations, timing and exported video coverage."""
import base64
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import subprocess
import sys
import tarfile
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np


def tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = tool('rollout_human_targets')
worker = tool('source_simulator_worker')


class TargetRolloutTests(unittest.TestCase):
    def test_target_reset_rejects_replay_source_and_alternative_state(self):
        request = dict(operation='reset', task='target', seed=10000, initial_state_id=0)
        worker.validate_reset_request(request, {'target'}, target=True)
        for changed in (dict(request, operation='reset_demo'), dict(request, task='source'),
                        dict(request, initial_state_id=1), dict(request, simulator_state=[999])):
            with self.assertRaises(ValueError):
                worker.validate_reset_request(changed, {'target'}, target=True)
        with self.assertRaises(ValueError):
            worker.validate_reset_request(request, {'target'}, target=True, diagnostic=True)
        with self.assertRaises(ValueError):
            worker.validate_reset_request(request, {'source'})

    def test_target_policy_cannot_receive_privileged_fields(self):
        rgb = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        rgb[:, 0] = 255
        frame = dict(step=0, rgb=base64.b64encode(rgb.tobytes()).decode(), robot0_eef_pos=[0, 0, 0],
                     robot0_eef_quat=[0, 0, 0, 1], robot0_gripper_qpos=[0, 0], simulator_state=[999], future_rgb='forbidden')
        norm = {'state': {'mean': [0] * 9, 'scale': [1] * 9}}
        window, frames = runner.observed_window({'observations': [frame], 'initial_state_sha256': 'audit-only'}, {}, 'target instruction', norm, 'opengl')
        self.assertEqual(set(window), {'history_rgb', 'current_rgb', 'panda_states', 'observation_times', 'instruction'})
        self.assertEqual(window['instruction'], 'target instruction')
        self.assertEqual(window['history_rgb'].shape, (2, 6, 128, 128, 3))
        np.testing.assert_array_equal(window['current_rgb'][:, -1], 255)
        np.testing.assert_array_equal(window['observation_times'], 0)
        self.assertEqual(frames[0][0], 0)

    def test_transition_requires_all_frames_and_exact_twenty_hz(self):
        good = {'step': 8, 'simulated_seconds': .4, 'observations': [{'step': i} for i in range(1, 9)]}
        runner.validate_transition(good, 0, 8)
        for bad in (dict(good, observations=[{'step': 8}]), dict(good, simulated_seconds=.8), dict(good, step=401)):
            with self.assertRaises(ValueError):
                runner.validate_transition(bad, 0, 8)
        with self.assertRaises(ValueError):
            runner.validate_transition(good, 0, 7)

    def test_benchmark_state_export_uses_pinned_file_without_simulator_torch(self):
        import torch
        from run_libero_cloud import LIBERO_SHA
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'libero/libero/init_files/libero_10/target.pruned_init'
            source.parent.mkdir(parents=True)
            torch.save([np.array([1., 2., 3.])], source)
            with patch.object(runner.subprocess, 'check_output', side_effect=[LIBERO_SHA, source.read_bytes()]):
                pins = runner.prepare_initial_states(root, [{'id': 'test', 'task': 'target'}], root / 'export')
            np.testing.assert_array_equal(np.load(root / 'export/target.npy', allow_pickle=False), [1, 2, 3])
            self.assertEqual(pins['test']['initial_state_id'], 0)
            with patch.object(runner.subprocess, 'check_output', return_value='wrong revision'):
                with self.assertRaisesRegex(ValueError, 'pinned LIBERO'):
                    runner.prepare_initial_states(root, [], root / 'bad')
            with patch.object(runner.subprocess, 'check_output', side_effect=[LIBERO_SHA, b'tampered']):
                with self.assertRaisesRegex(ValueError, 'differs from pinned'):
                    runner.prepare_initial_states(root, [{'id': 'test', 'task': 'target'}], root / 'tampered')

    def test_frozen_comparison_requires_verified_matching_adapted_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            report = root / 'after.json'
            receipt = root / 'receipt.json'
            request, spec = {'checkpoint_sha256': 'base'}, {'artifacts': {'semflow': 'pinned'}}
            after = dict(status='completed', condition='semflow', source_request=request, artifact_pins=spec['artifacts'],
                         config=runner.CONFIG, planned_targets=['target'], sampler_clock='historical indexed',
                         rollouts=[dict(target_id='target', status='completed', parameters_buffers_unchanged=True,
                                        seed=10000, initial_state_id=0)])
            def write():
                report.write_text(json.dumps(after))
                receipt.write_text(json.dumps(dict(stage='target-rollouts', comparison_report_sha256=runner.sha(report))))
            write()
            args = SimpleNamespace(condition='frozen', after_report=report, after_receipt=receipt)
            self.assertEqual(set(runner.validate_after_reference(args, request, spec, ['target'])), {'target'})
            after['rollouts'][0]['seed'] = 1
            write()
            with self.assertRaisesRegex(ValueError, 'differently seeded'):
                runner.validate_after_reference(args, request, spec, ['target'])
            report.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'receipt differs'):
                runner.validate_after_reference(args, request, spec, ['target'])
            args.after_report = None
            with self.assertRaisesRegex(ValueError, 'requires exported'):
                runner.validate_after_reference(args, request, spec, ['target'])

    def test_video_exports_require_explicit_opt_in(self):
        script = Path(__file__).resolve().parents[1] / 'tools/export_cloud_results.py'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / 'results'
            root.mkdir()
            (root / 'report.json').write_text('{}')
            (root / 'partial.mp4').write_bytes(b'partial video fixture')
            for include in (False, True):
                output = Path(d) / ('with.tar.gz' if include else 'without.tar.gz')
                command = [sys.executable, str(script), '--root', str(root), '--output', str(output)]
                if include:
                    command.append('--include-videos')
                subprocess.run(command, check=True, capture_output=True)
                with tarfile.open(output) as archive:
                    self.assertEqual('partial.mp4' in archive.getnames(), include)
                    manifest = json.load(archive.extractfile('ARTIFACT_MANIFEST.json'))
                    self.assertEqual('partial.mp4' in manifest['files'], include)

    def test_video_roundtrip_retains_partial_frames_and_timing(self):
        import av
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'partial.mp4'
            video = runner.Video(path)
            frames = [(i, np.full((2, 128, 128, 3), i * 40, dtype=np.uint8)) for i in range(3)]
            video.add(frames)
            video.close()
            with av.open(str(path)) as container:
                decoded = list(container.decode(video=0))
                self.assertEqual(len(decoded), 3)
                self.assertEqual([float(f.pts * f.time_base) for f in decoded], [0, .05, .1])
                self.assertEqual(decoded[0].to_ndarray(format='rgb24').shape, (128, 256, 3))
            self.assertEqual(video.count, 3)


if __name__ == '__main__':
    unittest.main()
