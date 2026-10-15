#!/usr/bin/env python3
"""Install an isolated simulator environment and run a bounded smoke in Colab.

Run with the host Python (including 3.13). No notebook environment packages are
replaced. The complete operation has a five-minute wall-clock deadline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
# Verified from the official commit page, rather than resolving a moving branch:
# https://github.com/Lifelong-Robot-Learning/LIBERO/commit/8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_SHA = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
UV_BOOTSTRAP = "uv==0.8.22"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/content/semflow-simulator"))
    parser.add_argument("--output-dir", type=Path, default=Path("/content/semflow-results/libero"))
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--install-only", action="store_true", help="Prepare the pinned simulator without instantiating any task")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 300:
        parser.error("total timeout must be between 1 and 300 seconds")
    work, output = args.work_dir.resolve(), args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    report = {
        "schema_version": 1, "status": "running", "host_python": platform.python_version(),
        "libero_url": LIBERO_URL, "libero_revision": LIBERO_SHA,
        "scope": "isolated simulator only; no robot demonstration downloads or policy training",
        "stages": [], "errors": [],
    }
    report_path = output / "libero_cloud_run.json"
    environment = os.environ.copy()
    environment.update({"UV_CACHE_DIR": str(work / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(work / "python"), "PYTHONNOUSERSITE": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})

    def save() -> None:
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(report_path)

    def run(stage: str, command: list[str], limit: int = 120) -> Path:
        remaining = deadline - time.monotonic()
        if remaining <= 1:
            raise TimeoutError("five-minute total deadline reached before " + stage)
        log = output / f"{stage}.log"
        record = {"stage": stage, "command": command, "log": log.name, "status": "running"}
        report["stages"].append(record)
        save()
        print(f"Starting {stage}", flush=True)
        stage_started = time.monotonic()
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            try:
                record["exit_code"] = process.wait(timeout=min(limit, remaining - 0.5))
                record["status"] = "passed" if record["exit_code"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                record["status"] = "timeout"
            finally:
                record["seconds"] = time.monotonic() - stage_started
                save()
        if record["status"] != "passed":
            raise RuntimeError(f"{stage} {record['status']}; inspect {log}")
        print(f"Completed {stage}: {record['seconds']:.1f}s", flush=True)
        return log

    try:
        save()
        if sys.platform != "linux":
            raise RuntimeError("the cloud simulator runner requires Linux")
        requirements = ROOT / "configs/requirements-simulator.txt"
        report["requirements_sha256"] = hashlib.sha256(requirements.read_bytes()).hexdigest()
        uv = shutil.which("uv")
        if uv is None:
            bootstrap = work / "bootstrap"
            candidate = bootstrap / "bin/uv"
            if not candidate.is_file():
                run("bootstrap_uv", [sys.executable, "-m", "pip", "install", "--target", str(bootstrap), UV_BOOTSTRAP], 45)
            uv = str(candidate)
        run("uv_version", [uv, "--version"], 10)
        virtual = work / "venv"
        interpreter = virtual / "bin/python"
        if not interpreter.is_file():
            run("python_environment", [uv, "venv", "--python", "3.11", "--managed-python", str(virtual)], 75)
        version_log = run("simulator_python", [str(interpreter), "-c", "import sys; print(sys.version); assert sys.version_info[:2] == (3, 11)"], 10)
        report["simulator_python"] = version_log.read_text().strip()
        checkout = work / "LIBERO"
        if not checkout.exists():
            run("git_init", ["git", "init", str(checkout)], 10)
            run("git_fetch", ["git", "-C", str(checkout), "fetch", "--depth=1", LIBERO_URL, LIBERO_SHA], 120)
            run("git_checkout", ["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], 20)
        revision = run("git_revision", ["git", "-C", str(checkout), "rev-parse", "HEAD"], 10).read_text().strip()
        if revision != LIBERO_SHA:
            raise RuntimeError(f"existing checkout has unexpected revision: {revision}")
        dirty = run("git_clean", ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], 10).read_text().strip()
        if dirty:
            raise RuntimeError("existing LIBERO checkout has tracked changes; preserving it and stopping")
        run("install_simulator", [uv, "pip", "install", "--python", str(interpreter), "-r", str(requirements)], 120)
        inventory = run("installed_packages", [uv, "--quiet", "pip", "list", "--python", str(interpreter), "--format", "json"], 10)
        packages = json.loads(inventory.read_text())
        report["packages"] = packages
        if any(package["name"].lower() == "torch" for package in packages):
            raise RuntimeError("unexpected torch installation in isolated simulator environment")
        if args.install_only:
            report["status"] = "passed"
            report["scope"] = "installation only; no tasks instantiated"
            return 0
        remaining = int(deadline - time.monotonic()) - 2
        if remaining < 10:
            raise TimeoutError("installation exhausted the total budget before the simulator probe")
        probe_limit = min(240, remaining)
        run("probe", [str(interpreter), str(ROOT / "tools/probe_libero.py"), "--libero-checkout", str(checkout), "--output-dir", str(output), "--timeout-seconds", str(probe_limit), "--gl", "egl"], probe_limit + 1)
        probe_report = json.loads((output / "libero_probe.json").read_text())
        if probe_report["status"] != "passed":
            raise RuntimeError("simulator probe did not report success")
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        traceback.print_exc()
    finally:
        report["seconds"] = time.monotonic() - started
        save()
    print(json.dumps({"status": report["status"], "report": str(report_path), "seconds": report["seconds"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
