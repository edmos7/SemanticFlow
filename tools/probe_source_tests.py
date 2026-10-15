#!/usr/bin/env python3
"""Kernel-safe, bounded cloud unit tests; no datasets or model downloads."""

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import traceback


def main():
    root = Path("/content/semflow")
    output = Path("/content/semflow-results")
    root.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    bundle = Path("/content/semflow-source.tar.gz")
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(root, filter="data")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("source bundle member hash mismatch")
    report = {"status": "running", "python": platform.python_version(),
              "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
              "no_dataset_or_model_downloads": True}
    try:
        installation = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "h5py==3.15.1"],
            capture_output=True, text=True, timeout=90)
        (output / "dependency-install.txt").write_text(installation.stdout + installation.stderr)
        installation.check_returncode()
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        metadata = subprocess.run([sys.executable, "-c",
            "import json,torch,numpy,h5py; print(json.dumps({'torch':torch.__version__,'numpy':numpy.__version__,"
            "'h5py':h5py.__version__,'cuda_available':torch.cuda.is_available(),"
            "'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"],
            env=env, capture_output=True, text=True, check=True, timeout=30)
        report["environment"] = json.loads(metadata.stdout)
        tests = subprocess.run([sys.executable, str(root / "tools/run_cloud_tests.py")],
                               env=env, capture_output=True, text=True, timeout=150)
        (output / "test-wrapper.txt").write_text(tests.stdout + tests.stderr)
        tests.check_returncode()
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
    finally:
        (output / "source-tests.json").write_text(json.dumps(report, indent=2) + "\n")
        exported = subprocess.run([sys.executable, str(root / "tools/export_cloud_results.py")],
                                  capture_output=True, text=True, timeout=30, check=True)
        print(exported.stdout, flush=True)
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise RuntimeError("cloud source test gate failed; inspect exported report")


if __name__ == "__main__":
    main()
