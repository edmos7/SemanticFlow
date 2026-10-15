"""Guard against damaged or mismatched published numerical evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from reproduce_results import verify_manifest, validate_actions, TARGETS


class SubmissionResultsTests(unittest.TestCase):
    def test_manifest_tamper_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'sample.csv').write_bytes(b'original')
            manifest = {'files': {'sample.csv': hashlib.sha256(b'original').hexdigest()}}
            (root / 'MANIFEST.json').write_text(json.dumps(manifest))
            verify_manifest(root)
            (root / 'sample.csv').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_manifest(root)
            (root / 'MANIFEST.json').write_text(json.dumps({'files': {'../outside': 'x'}}))
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                verify_manifest(root)

    def test_unpaired_action_noise_rejected(self):
        data = []
        models = [('frozen', 'none'), ('retention', 'none')] + [(a, t) for a in ('standard', 'semflow') for t in TARGETS]
        for model, target in models:
            for episode in range(12):
                for steps in ('5', '10', '20'):
                    for seed in ('4242', '4243', '4244', '4245'):
                        data.append(dict(model=model, human_task=target, source_task=str(episode // 2), episode_sha256=str(episode), anchor='0', steps=steps, noise_seed=seed, noise_sha256=seed, split='validation'))
        validate_actions(data)
        data[-1]['noise_sha256'] = 'unpaired'
        with self.assertRaisesRegex(ValueError, 'not paired'):
            validate_actions(data)


if __name__ == '__main__':
    unittest.main()
