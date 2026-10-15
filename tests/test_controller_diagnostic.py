"""Diagnostic action metrics and strict separation of privileged replay state."""

import base64
import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


diagnostic = load_tool('diagnose_source_controller')
worker = load_tool('source_simulator_worker')


class DiagnosticTests(unittest.TestCase):
    def test_demo_assets_relocate_without_torch_or_libero_utilities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite, assets = root / 'robosuite', root / 'assets'
            suite.mkdir()
            (assets / 'textures').mkdir(parents=True)
            (suite / 'hand.stl').touch()
            (assets / 'textures/wood.png').touch()
            xml = '<mujoco><asset><mesh file="/old/robosuite/robosuite/hand.stl"/><texture file="/old/assets/scenes/../textures/wood.png"/><texture builtin="flat"/></asset><worldbody><camera name="agentview" pos="1 2 3" quat="1 0 0 0"/></worldbody></mujoco>'
            original_import = builtins.__import__
            def restricted_import(name, *args, **kwargs):
                if name == 'torch' or name.startswith('libero'):
                    raise ModuleNotFoundError(name)
                return original_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=restricted_import):
                result = ET.fromstring(worker.relocate_demo_assets(xml, suite, assets))
            self.assertEqual(result.find('./asset/mesh').get('file'), str(suite / 'hand.stl'))
            self.assertEqual(result.find('./asset/texture').get('file'), str(assets / 'textures/wood.png'))
            self.assertEqual(result.find('./worldbody/camera').attrib, ET.fromstring(xml).find('./worldbody/camera').attrib)

    def test_demo_assets_reject_missing_files_and_root_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'assets'
            base.mkdir()
            (root / 'outside.stl').touch()
            for filename in ('/old/assets/missing.stl', '/old/assets/../outside.stl', '/unknown/mesh.stl'):
                with self.assertRaises(ValueError):
                    worker.relocate_demo_assets('<mujoco><asset><mesh file="' + filename + '"/></asset></mujoco>', base, base)

    def test_sampled_error_reports_executed_prefix_separately(self):
        target = np.ones((32, 7))
        prediction = target.copy()
        prediction[8:] = 3
        result = diagnostic.action_comparison(prediction, target)
        self.assertEqual(result['first_eight_mse'], 0)
        self.assertEqual(result['mse'], 3)
        self.assertEqual(result['train_mean_baseline_mse'], 1)
        with self.assertRaises(ValueError):
            diagnostic.action_comparison(np.full((32, 7), np.nan), target)

    def test_privileged_reset_requires_opt_in_and_source_task(self):
        request = {'operation': 'reset_demo', 'task': 'source', 'seed': 42,
                   'initial_state': [0], 'model_xml': 'fixture'}
        with self.assertRaisesRegex(ValueError, 'diagnostic-only'):
            worker.validate_reset_request(request, {'source'})
        worker.validate_reset_request(request, {'source'}, diagnostic=True)
        with self.assertRaises(ValueError):
            worker.validate_reset_request(dict(request, task='target'), {'source'}, diagnostic=True)
        with self.assertRaises(ValueError):
            worker.validate_reset_request(dict(request, operation='reset'), {'source'}, diagnostic=True)

    def test_observation_window_cannot_forward_simulator_state_or_targets(self):
        image = np.zeros((2, 128, 128, 3), dtype=np.uint8)
        frame = {'step': 0, 'rgb': base64.b64encode(image.tobytes()).decode(),
                 'robot0_eef_pos': [0, 0, 0], 'robot0_eef_quat': [0, 0, 0, 1],
                 'robot0_gripper_qpos': [0, 0], 'simulator_state': [999], 'future_rgb': 'forbidden'}
        index = {'normalization': {'state': {'mean': [0] * 9, 'scale': [1] * 9}},
                 'sources': [{'task': diagnostic.TASK, 'instruction': 'close the drawer'}]}
        window = diagnostic.observed_window({'observations': [frame]}, {}, 0, 'opencv', index)
        self.assertEqual(set(window), {'history_rgb', 'current_rgb', 'panda_states', 'observation_times', 'instruction'})
        self.assertEqual(window['history_rgb'].shape, (2, 6, 128, 128, 3))
        np.testing.assert_array_equal(window['observation_times'], np.zeros(6))
