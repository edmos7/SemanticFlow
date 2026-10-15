#!/usr/bin/env python3
"""Bounded headless simulator smoke; never runs a learned policy or tunes it.

Use a separate Python 3.10/3.11 environment with requirements-simulator.txt.
The official checkout is an input and is not modified. Target renders are
recording references and integration checks, not adaptation/evaluation data.
"""

from __future__ import annotations

import argparse
import ast
import collections
import collections.abc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def checkout_info(checkout: Path) -> dict:
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=True, timeout=15,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        text=True, capture_output=True, check=True, timeout=15,
    ).stdout.strip()
    if dirty:
        raise ValueError("LIBERO checkout contains tracked modifications; use a clean pinned checkout")
    return {"checkout": str(checkout), "revision": revision, "tracked_clean": True}


def read_task_map(checkout: Path) -> dict:
    """Read the data literal without importing LIBERO's torch-based benchmark."""
    path = checkout / "libero/libero/benchmark/libero_suite_task_map.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(name, ast.Name) and name.id == "libero_task_map" for name in statement.targets):
            mapping = ast.literal_eval(statement.value)
            if not isinstance(mapping, dict):
                break
            return mapping
    raise ValueError("official task map does not contain the expected literal")


def require_membership(manifest: dict, mapping: dict, checkout: Path) -> list[dict]:
    from semflow.config import validate_task_manifest

    validate_task_manifest(manifest)
    records = []
    for target in manifest["targets"]:
        candidates = [(manifest["target_suite"], target["task"], "target")]
        candidates.extend((manifest["source_suite"], source, "source") for source in target["source_tasks"])
        for suite, name, role in candidates:
            if name not in mapping.get(suite, []):
                raise ValueError(f"{role} task missing from official {suite}: {name}")
            bddl = checkout / "libero/libero/bddl_files" / suite / f"{name}.bddl"
            if not bddl.is_file():
                raise FileNotFoundError(bddl)
            records.append({"suite": suite, "task": name, "role": role, "bddl_sha256": hashlib.sha256(bddl.read_bytes()).hexdigest()})
    return records


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def probe_task(target: dict, checkout: Path, output: Path, seed: int, image_size: int,
               *, hard_reset: bool = False) -> dict:
    # Keep optional dependencies out of CLI parsing, metadata and membership work.
    import numpy as np
    from PIL import Image
    import robosuite.macros as macros
    from libero.libero.envs import OffScreenRenderEnv

    bddl = checkout / "libero/libero/bddl_files/libero_10" / f"{target['task']}.bddl"
    started = time.monotonic()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=image_size,
        camera_widths=image_size, camera_names=["agentview", "robot0_eye_in_hand"],
        control_freq=20, horizon=4, initialization_noise=None, hard_reset=hard_reset,
    )
    result = {"id": target["id"], "task": target["task"], "seed": seed, "passed": False}
    try:
        initializer_counts = [len(env.env.object_property_initializers)]
        # The upstream wrapper retries reset forever and suppresses exceptions.
        # Calling the same underlying operation directly makes failures visible.
        env.seed(seed)
        first = env.env.reset()
        initializer_counts.append(len(env.env.object_property_initializers))
        first_state = env.get_sim_state().copy()
        cameras = ("agentview_image", "robot0_eye_in_hand_image")
        first_images = {key: np.asarray(first[key]).copy() for key in cameras}
        success_initial = bool(env.check_success())
        lower, upper = (np.asarray(bound) for bound in env.env.action_spec)
        if lower.shape != (7,) or upper.shape != (7,):
            raise ValueError(f"expected 7D Panda action bounds, found {lower.shape}, {upper.shape}")
        action = np.zeros(7, dtype=np.float32)
        if not np.all((action >= lower) & (action <= upper)):
            raise ValueError("zero smoke action lies outside action bounds")
        steps = []
        for index in range(2):
            observation, reward, done, _ = env.step(action)
            if not np.isfinite(env.get_sim_state()).all():
                raise ValueError("nonfinite simulator state after zero action")
            steps.append({"index": index, "reward": float(reward), "done": bool(done), "success": bool(env.check_success())})
            if done:
                break
        env.seed(seed)
        second = env.env.reset()
        initializer_counts.append(len(env.env.object_property_initializers))
        initializers_stable = len(set(initializer_counts)) == 1
        second_state = env.get_sim_state().copy()
        state_equal = bool(np.allclose(first_state, second_state, atol=1e-10, rtol=0))
        images_equal = {key: bool(np.array_equal(first_images[key], second[key])) for key in cameras}
        artifacts = []
        convention = macros.IMAGE_CONVENTION
        if convention not in ("opengl", "opencv"):
            raise ValueError(f"unrecognized image convention: {convention}")
        for key in cameras:
            frame = first_images[key]
            if frame.shape != (image_size, image_size, 3) or frame.dtype != np.uint8:
                raise ValueError(f"unexpected image format {key}: {frame.shape}, {frame.dtype}")
            if int(frame.max()) == int(frame.min()):
                raise ValueError(f"rendered image is constant: {key}")
            # PNG viewers expect top-left origin. Do not change simulator tensors.
            display_frame = frame[::-1] if convention == "opengl" else frame
            name = f"{target['id']}_{key}.png"
            Image.fromarray(display_frame).save(output / name)
            artifacts.append({"file": name, "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(), "vertical_flip_for_png": convention == "opengl"})
        result.update({
            "passed": state_equal and all(images_equal.values()) and len(steps) == 2 and initializers_stable,
            "seconds": time.monotonic() - started,
            "instruction": env.language_instruction,
            "success_predicates": env.env.parsed_problem["goal_state"],
            "initial_success": success_initial,
            "observation_shapes": {key: list(np.asarray(value).shape) for key, value in first.items()},
            "action_dimension": int(lower.size), "action_lower": lower.tolist(), "action_upper": upper.tolist(),
            "smoke_action": action.tolist(), "steps": steps,
            "zero_action_note": "Zero controller input checks the API; physics still advances. This is not a learned policy.",
            "deterministic_reset": {"state_equal_atol_1e_10": state_equal, "state_max_abs_difference": float(np.max(np.abs(first_state - second_state))), "images_equal": images_equal},
            "reset_mode": "hard" if hard_reset else "soft",
            "property_initializer_counts": initializer_counts,
            "property_initializers_stable": initializers_stable,
            "initial_state_sha256": hashlib.sha256(first_state.tobytes()).hexdigest(),
            "raw_image_convention": convention, "images": artifacts,
        })
        return result
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-checkout", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=PROJECT_ROOT / "configs/tasks.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256, choices=(128, 256))
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--gl", choices=("egl", "osmesa"), default="egl")
    # Hard XML rebuilds in pinned LIBERO append articulation samplers on every
    # reset. Soft reset still resets physics and samples all initial placements.
    parser.add_argument("--hard-reset", action="store_true", help="diagnose upstream hard-reset sampler accumulation")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 600:
        parser.error("timeout must be between 1 and 600 seconds")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "libero_probe.json"
    report = {"schema_version": 1, "status": "running", "python": platform.python_version(), "purpose": "headless integration and recording references only; no policy evaluation or tuning", "tasks": [], "errors": []}
    started = time.monotonic()

    def on_timeout(signum, frame):
        raise TimeoutError(f"simulator probe exceeded {args.timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(args.timeout_seconds)
    try:
        write_report(report_path, report)
        if sys.version_info[:2] not in ((3, 10), (3, 11)):
            raise RuntimeError("this simulator probe requires a separate Python 3.10 or 3.11 environment")
        checkout = args.libero_checkout.expanduser().resolve()
        report["libero"] = checkout_info(checkout)
        manifest = json.loads(args.tasks.read_text(encoding="utf-8"))
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        report["membership"] = require_membership(manifest, read_task_map(checkout), checkout)
        names = ("numpy", "scipy", "numba", "mujoco", "robosuite", "bddl", "gym", "cloudpickle", "easydict", "PyYAML", "matplotlib", "Pillow", "opencv-python")
        report["packages"] = {name: importlib.metadata.version(name) for name in names}
        required = {"numpy": "1.26.4", "mujoco": "2.3.7", "robosuite": "1.4.0"}
        for name, version in required.items():
            if report["packages"][name] != version:
                raise RuntimeError(f"{name} must be {version}; found {report['packages'][name]}")
        # Older LIBERO uses this removed name. Limit compatibility to this
        # process, record it, and leave the external upstream checkout clean.
        if not hasattr(collections, "Iterable"):
            collections.Iterable = collections.abc.Iterable
            report["compatibility"] = ["collections.Iterable aliases collections.abc.Iterable in this process"]
        config_dir = output / "libero_config"
        config_dir.mkdir(exist_ok=True)
        benchmark = checkout / "libero/libero"
        unused_datasets = output / "unused_datasets"
        unused_datasets.mkdir(exist_ok=True)
        paths = {"benchmark_root": str(benchmark), "bddl_files": str(benchmark / "bddl_files"), "init_states": str(benchmark / "init_files"), "assets": str(benchmark / "assets"), "datasets": str(unused_datasets)}
        # JSON is valid YAML; create config before import to prevent auth-like
        # interactive dataset prompts and avoid modifying the user's home.
        (config_dir / "config.yaml").write_text(json.dumps(paths, indent=2), encoding="utf-8")
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        os.environ["MUJOCO_GL"] = args.gl
        os.environ["PYOPENGL_PLATFORM"] = args.gl
        os.environ.setdefault("NUMBA_CACHE_DIR", str(output / "numba_cache"))
        sys.path.insert(0, str(checkout))
        report["rendering_backend"] = args.gl
        for target in manifest["targets"]:
            print(f"Probing {target['id']}", flush=True)
            report["tasks"].append(probe_task(target, checkout, output, args.seed, args.image_size,
                                             hard_reset=args.hard_reset))
            write_report(report_path, report)
        report["status"] = "passed" if all(task["passed"] for task in report["tasks"]) else "failed"
    except Exception as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        traceback.print_exc()
    finally:
        signal.alarm(0)
        report["seconds"] = time.monotonic() - started
        write_report(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "seconds": report["seconds"]}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
