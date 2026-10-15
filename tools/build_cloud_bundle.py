#!/usr/bin/env python3
"""Build a deterministic source-only archive from an explicit allowlist."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT_FILES = ('pyproject.toml', 'README.md', 'configs/requirements-simulator.txt',
              'licenses/egosteer/LICENSE', 'licenses/egosteer/NOTICE')
TREE_SUFFIXES = {'src': {'.py'}, 'tests': {'.py'}, 'configs': {'.json'}}
PROBE_FILES = ('build_cloud_bundle.py', 'probe_environment.py', 'probe_egosteer.py',
               'probe_libero.py', 'probe_semantic_gpu.py', 'run_libero_cloud.py',
               'probe_native_dependencies.py', 'run_egosteer_cloud.py',
               'probe_native_integration.py',
               'verify_source_ingestion.py', 'probe_source_tests.py',
               'source_simulator_worker.py', 'train_source_calibration.py', 'run_source_calibration_cloud.py',
               'run_cloud_tests.py', 'export_cloud_results.py', 'diagnose_source_controller.py',
               'probe_source_actions.py', 'train_balanced_source.py', 'probe_source_conditioning.py',
               'probe_realman.py', 'compare_source_initializations.py', 'continue_realman_source.py',
               'probe_human_inputs.py', 'rehearse_adaptation.py', 'probe_adapted_actions.py', 'adapt_human.py', 'probe_human_actions.py', 'decision_diagnostics.py', 'probe_sampler_clock.py', 'reproduce_results.py', 'rollout_human_targets.py')


def selected_files(root: Path) -> list[Path]:
    root = root.resolve()
    paths = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory, suffixes in TREE_SUFFIXES.items():
        folder = root / directory
        if folder.is_symlink():
            raise ValueError(f'Symlinked source directory is forbidden: {directory}')
        if folder.exists():
            paths.extend(path for path in folder.rglob('*') if path.suffix in suffixes and path.is_file())
    paths.extend(root / 'tools' / name for name in PROBE_FILES if (root / 'tools' / name).is_file())
    for path in paths:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != root and root in p.parents):
            raise ValueError(f'Symlinked source file is forbidden: {path.relative_to(root)}')
        if not path.resolve().is_relative_to(root):
            raise ValueError('Source file escapes project root')
    return sorted(paths, key=lambda path: str(path.relative_to(root)))


def build_bundle(root: Path, output: Path) -> dict:
    root = root.resolve()
    files = selected_files(root)
    if not files:
        raise ValueError('No source files selected')
    manifest = {
        'schema_version': 1,
        'files': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        'excluded': ['agent guidance', 'context', 'credentials', 'recordings', 'datasets', 'model weights', 'run artifacts'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as file, gzip.GzipFile(fileobj=file, mode='wb', mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            entries = [(str(path.relative_to(root)), path.read_bytes()) for path in files]
            entries.append(('SOURCE_MANIFEST.json', (json.dumps(manifest, indent=2) + '\n').encode()))
            for name, data in entries:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    return {'bundle': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'bytes': output.stat().st_size, 'files': list(manifest['files'])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
