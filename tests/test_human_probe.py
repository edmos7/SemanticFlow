"""Reject changed preflight inputs before allocating or loading a model."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import probe_human_inputs as probe


class HumanInputGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.args = SimpleNamespace(**{k:root/(k+'.json') for k in
            ('request', 'manifest', 'index', 'checkpoint', 'checkpoint_receipt', 'pilot')})
        self.args.manifest.write_text(json.dumps({'episodes': [], 'human_clips': [
            {'target_id': x} for x in ('a','b','c')]}))
        self.args.index.write_text(json.dumps({'manifest': {'episodes': [], 'human_clips': []}}))
        self.args.checkpoint.write_bytes(b'fixture checkpoint')
        self.request = {'manifest_sha256': probe.sha(self.args.manifest),
                        'source_index_sha256': probe.sha(self.args.index),
                        'checkpoint_sha256': probe.sha(self.args.checkpoint), 'windows': {'a':1,'b':2,'c':3}}
        self.args.request.write_text(json.dumps(self.request))
        self.args.checkpoint_receipt.write_text(json.dumps({'stage':'train',
            'checkpoint_sha256':self.request['checkpoint_sha256']}))

    def validate(self):
        with patch('semflow.config.load_config', return_value=({}, {})), \
             patch('semflow.data.validate_episode_manifest', return_value={'human_clips':3}):
            return probe.validate_inputs(self.args)

    def test_matching_inputs_pass_before_model_code(self):
        self.assertEqual(self.validate()[0], self.request)

    def test_changed_clip_manifest_or_checkpoint_rejected(self):
        for field in ('manifest', 'checkpoint', 'index'):
            path = getattr(self.args, field)
            saved = path.read_bytes()
            path.write_bytes(saved+b' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                self.validate()
            path.write_bytes(saved)

    def test_wrong_checkpoint_receipt_rejected(self):
        self.args.checkpoint_receipt.write_text(json.dumps({'stage':'train','checkpoint_sha256':'0'*64}))
        with self.assertRaisesRegex(ValueError, 'training receipt'):
            self.validate()

    def test_missing_clip_anchor_rejected(self):
        del self.request['windows']['c']
        self.args.request.write_text(json.dumps(self.request))
        with self.assertRaisesRegex(ValueError, 'fixed anchor'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
