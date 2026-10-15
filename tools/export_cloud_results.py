#!/usr/bin/env python3
"""Archive bounded probe outputs, with hashes, for direct durable download."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/content/semflow-results'))
    parser.add_argument('--output', type=Path, default=Path('/content/semflow-results.tar.gz'))
    parser.add_argument("--include-videos", action="store_true", help="Explicitly export qualitative rollout MP4s")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    # These dedicated directories contain only outputs from our bounded probes.
    # No home, credential cache, model cache, or arbitrary /content traversal.
    suffixes = {'.json', '.txt', '.log', '.png', '.pt'} | ({'.mp4'} if args.include_videos else set())
    files = sorted(path for path in root.rglob('*')
                   if path.is_file() and path.suffix in suffixes
                   and not any(part in {'numba_cache', 'libero_config'} for part in path.relative_to(root).parts))
    if not files:
        raise RuntimeError('No probe artifacts found')
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Artifact escapes output directory: {path.name}')
    manifest = {'schema_version': 1, 'files': {
        str(path.relative_to(root)): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                     'bytes': path.stat().st_size} for path in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, 'w:gz') as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        content = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('ARTIFACT_MANIFEST.json')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    print(json.dumps({'archive': str(args.output), 'bytes': args.output.stat().st_size,
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      'files': list(manifest['files'])}))


if __name__ == '__main__':
    main()
