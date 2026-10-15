#!/usr/bin/env python3
"""Bounded isolated native dependency gate; no weights, credentials, or training.

Uses managed Python 3.12 and official binary Torch/FlashAttention packages without
replacing the notebook stack. Fresh directories and a <=600 second budget apply.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

from probe_egosteer import UPSTREAM_REVISION
from probe_native_dependencies import EXPECTED_PACKAGES


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_URL = "https://github.com/egosteer/egosteer.git"
PYTORCH_INDEX = "https://download.pytorch.org/whl/cu130"
PYPI_INDEX = "https://pypi.org/simple"
UV_BOOTSTRAP = "uv==0.8.22"
WHEEL_NAME = "flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/" + WHEEL_NAME.replace("+", "%2B")
# https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.8.3
WHEEL_SHA256 = "910d8db9def162de5b7c15474b933e7e2371e93733b980e9d3c07cd3bf2f568e"
PYTHON_DEPENDENCIES = [
    "transformers==5.6.2", "hydra-core==1.3.2", "omegaconf==2.3.0",
    "numpy==2.1.3", "huggingface-hub==1.31.0", "safetensors", "pillow",
    "psutil", "einops", "packaging", "jinja2",
]
DOWNLOAD_CODE = """
import hashlib, json, pathlib, sys, urllib.request
destination = pathlib.Path(sys.argv[2])
digest = hashlib.sha256()
with urllib.request.urlopen(sys.argv[1], timeout=30) as response, destination.open('xb') as stream:
    while chunk := response.read(1024 * 1024):
        digest.update(chunk)
        stream.write(chunk)
actual = digest.hexdigest()
if actual != sys.argv[3]:
    raise RuntimeError('official FlashAttention wheel SHA256 mismatch')
print(json.dumps({'sha256': actual, 'bytes': destination.stat().st_size}))
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/semflow-native-dependencies"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/semflow-results/native-dependencies"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 600:
        parser.error("total timeout must be between 1 and 600 seconds")
    work, output = args.work_dir.resolve(), args.output_dir.resolve()
    if work == output or work in output.parents or output in work.parents:
        parser.error("work and output directories must be separate")
    if work.exists() or output.exists():
        parser.error("work and output directories must both be fresh; existing data is preserved")
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    report = {
        "schema_version": 1, "status": "running", "host_python": platform.python_version(),
        "upstream_url": UPSTREAM_URL, "upstream_revision": UPSTREAM_REVISION,
        "candidate_packages": EXPECTED_PACKAGES, "wheel_url": WHEEL_URL,
        "wheel_sha256": WHEEL_SHA256, "stages": [], "errors": [],
        "scope": "isolated dependency feasibility only; no model weights or authentication",
    }
    report_path = output / "egosteer_cloud_run.json"
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith(("UV_", "PIP_", "HF_", "HUGGINGFACE_", "HUGGING_FACE_"))
        and key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"}
    }
    environment.update({
        "UV_CACHE_DIR": str(work / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(work / "python"),
        "UV_NO_CONFIG": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1", "GIT_TERMINAL_PROMPT": "0",
        "HF_HOME": str(work / "empty-hf-cache"), "HF_TOKEN_PATH": str(work / "absent-token"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    })

    def save() -> None:
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def interrupted(_signum, _frame) -> None:
        raise InterruptedError("native dependency runner interrupted")

    def run(stage: str, command: list[str], limit: int = 120) -> Path:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("total deadline reached before " + stage)
        log = output / f"{stage}.log"
        record = {"stage": stage, "log": log.name, "status": "running"}
        report["stages"].append(record)
        save()
        print(f"Starting {stage}", flush=True)
        beginning = time.monotonic()
        process = None
        try:
            with log.open("xb") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                           cwd=work, env=environment, start_new_session=True)
                record["exit_code"] = process.wait(timeout=min(limit, remaining - 1))
                record["status"] = "passed" if record["exit_code"] == 0 else "failed"
        except BaseException as error:
            record["status"] = "timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed"
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=1)
            raise
        finally:
            record["seconds"] = time.monotonic() - beginning
            save()
        if record["status"] != "passed":
            raise RuntimeError(f"{stage} failed; inspect {log.name}")
        print(f"Completed {stage}: {record['seconds']:.1f}s", flush=True)
        return log

    try:
        save()
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("the pinned dependency environment requires Linux x86_64")
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        uv = shutil.which("uv")
        if uv is None:
            bootstrap = work / "bootstrap"
            run("bootstrap_uv", [sys.executable, "-m", "pip", "--isolated", "install",
                                 "--index-url", PYPI_INDEX, "--only-binary", ":all:",
                                 "--no-deps", "--target", str(bootstrap), UV_BOOTSTRAP], 45)
            uv = str(bootstrap / "bin/uv")
        run("uv_version", [uv, "--version"], 10)
        virtual = work / "venv"
        interpreter = virtual / "bin/python"
        run("python_environment", [uv, "venv", "--python", "3.12", "--managed-python", str(virtual)], 75)
        run("python_version", [str(interpreter), "-c", "import sys; print(sys.version); assert sys.version_info[:2] == (3, 12)"], 10)
        install = [uv, "pip", "install", "--python", str(interpreter)]
        run("install_torch", install + ["--index-url", PYTORCH_INDEX, "--only-binary", ":all:",
                                       "torch==2.10.0+cu130", "torchvision==0.25.0+cu130"], 240)
        # Hydra's antlr4 runtime is pure Python and may need wheel packaging;
        # CUDA/Torch/FlashAttention never fall back to compilation.
        run("install_python_dependencies", install + ["--index-url", PYPI_INDEX,
            "--only-binary", "torch,torchvision,flash-attn"] + PYTHON_DEPENDENCIES, 100)
        wheel = work / WHEEL_NAME
        wheel_log = run("download_verified_flash_wheel", [str(interpreter), "-c", DOWNLOAD_CODE,
                                                         WHEEL_URL, str(wheel), WHEEL_SHA256], 90)
        report["verified_wheel"] = json.loads(wheel_log.read_text())
        run("install_flash_wheel", install + ["--no-deps", "--no-index", "--only-binary", ":all:", str(wheel)], 30)
        checkout = work / "egosteer"
        run("git_init", ["git", "init", str(checkout)], 10)
        run("git_fetch", ["git", "-C", str(checkout), "fetch", "--depth=1", UPSTREAM_URL, UPSTREAM_REVISION], 60)
        run("git_checkout", ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], 10)
        inventory = run("installed_packages", [uv, "--quiet", "pip", "list", "--python", str(interpreter), "--format", "json"], 10)
        report["packages"] = json.loads(inventory.read_text())
        remaining = int(deadline - time.monotonic()) - 2
        if remaining < 10:
            raise TimeoutError("installation exhausted the total budget before dependency checks")
        probe_limit = min(90, remaining)
        probe_report = output / "native_dependencies.json"
        run("native_dependencies", [str(interpreter), str(ROOT / "tools/probe_native_dependencies.py"),
            "--upstream-root", str(checkout), "--output", str(probe_report),
            "--timeout-seconds", str(probe_limit)], probe_limit + 1)
        if json.loads(probe_report.read_text())["status"] != "passed":
            raise RuntimeError("native dependency probe did not report success")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error).split("\n")[0][:600]})
    finally:
        report["seconds"] = time.monotonic() - started
        save()
    print(json.dumps({"status": report["status"], "report": str(report_path), "seconds": report["seconds"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
