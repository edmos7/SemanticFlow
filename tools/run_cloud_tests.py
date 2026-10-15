#!/usr/bin/env python3
"""Extract the reviewed source bundle and run tests on the cloud host."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile


def main() -> None:
    root = Path('/content/semflow')
    root.mkdir(exist_ok=True)
    bundle = Path('/content/semflow-source.tar.gz')
    with tarfile.open(bundle, 'r:gz') as archive:
        archive.extractall(root, filter='data')
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source hash mismatch: {name}')
    import os
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                         cwd=root, env=env, text=True, capture_output=True, timeout=120)
    output = Path('/content/semflow-results')
    output.mkdir(exist_ok=True)
    (output / 'unit-tests.txt').write_text(run.stdout + run.stderr)
    print(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(f'Cloud test suite failed: {run.returncode}')
    if 'skipped=' in run.stderr or '(skipped' in run.stderr:
        raise RuntimeError('Cloud test suite skipped tests; inspect the dependency gate')
    report = {'schema_version': 1, 'stage': 'component_tests', 'returncode': run.returncode,
              'bundle_sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
              'limitations': ['Component tests only; no integrated pretrained EgoSteer policy evaluated.']}
    (output / 'component-tests.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
