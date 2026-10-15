#!/usr/bin/env python3
"""LIBERO RPC; source-only by default, with explicit qualitative target mode."""

import argparse
import base64
import collections
import collections.abc
import json
import os
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET


def relocate_demo_assets(xml, robosuite_root, libero_assets):
    """Resolve recorded asset locations without importing training utilities.

    Only mesh/texture file attributes change; camera and physics attributes are
    preserved. Every resolved asset must exist beneath its installed root.
    """
    root = ET.fromstring(xml)
    for element in root.findall("./asset/mesh") + root.findall("./asset/texture"):
        filename = element.get("file")
        if filename is None:
            continue
        parts = Path(filename).parts
        if "robosuite" in parts:
            base, marker = Path(robosuite_root).resolve(), "robosuite"
        elif "assets" in parts:
            base, marker = Path(libero_assets).resolve(), "assets"
        else:
            raise ValueError("unrecognized recorded asset root")
        offset = len(parts) - 1 - parts[::-1].index(marker)
        asset = base.joinpath(*parts[offset + 1:]).resolve()
        if not asset.is_relative_to(base) or not asset.is_file():
            raise ValueError("recorded asset missing or outside installed root: " + asset.name)
        element.set("file", str(asset))
    return ET.tostring(root, encoding="unicode")


def validate_reset_request(request, approved, *, diagnostic=False, target=False):
    operation = request.get("operation")
    fields = {"operation", "task", "seed"}
    if target:
        if diagnostic or operation != "reset":
            raise ValueError("target mode rejects demonstration replay")
        fields.add("initial_state_id")
        if type(request.get("initial_state_id")) is not int or request["initial_state_id"] != 0:
            raise ValueError("qualitative target mode fixes initial_state_id=0")
    elif operation == "reset_demo":
        if not diagnostic:
            raise ValueError("recorded simulator states are diagnostic-only")
        fields |= {"initial_state", "model_xml"}
    elif operation != "reset":
        raise ValueError("unknown reset operation")
    if set(request) != fields or request["task"] not in approved:
        raise ValueError("reset rejects unapproved tasks or unexpected state inputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-checkout", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--target-initial-states", type=Path, help="Fixed benchmark states exported by native interpreter")
    parser.add_argument("--target-mode", action="store_true", help="Fixed qualitative LIBERO-10 trials; no replay")
    parser.add_argument("--diagnostic-dir", type=Path,
                        help="Opt-in trace export and demonstration replay, never policy/training inputs")
    args = parser.parse_args()
    if args.target_mode and (args.diagnostic_dir or args.target_initial_states is None):
        parser.error("target mode requires fixed benchmark states and forbids replay/trace")
    protocol = sys.stdout
    sys.stdout = sys.stderr  # upstream logs must not corrupt the JSON protocol
    import numpy as np
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    args.config_dir.mkdir(parents=True, exist_ok=True)
    base = args.libero_checkout / "libero/libero"
    paths = {"benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
             "init_states": str(base / "init_files"), "assets": str(base / "assets"),
             "datasets": str(args.config_dir / "unused_datasets")}
    (args.config_dir / "config.yaml").write_text(json.dumps(paths))
    os.environ.update(LIBERO_CONFIG_PATH=str(args.config_dir), MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                      NUMBA_CACHE_DIR=str(args.config_dir.parent / "numba_cache"))
    sys.path.insert(0, str(args.libero_checkout))
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.macros as macros
    tasks = json.loads(args.tasks.read_text())
    approved = {s for target in tasks["targets"] for s in target["source_tasks"]}
    if args.target_mode:
        approved = {t["task"] for t in tasks["targets"]}
    env, task, step = None, None, 0
    trace = []
    if args.diagnostic_dir:
        args.diagnostic_dir.mkdir(parents=True, exist_ok=False)

    def pack(obs, index):
        rgb = np.stack([obs[k] for k in ("agentview_image", "robot0_eye_in_hand_image")])
        if rgb.shape != (2, 128, 128, 3) or rgb.dtype != np.uint8:
            raise ValueError("unexpected simulator RGB format")
        return {"step": index, "rgb": base64.b64encode(rgb.tobytes()).decode(),
                "robot0_eef_pos": obs["robot0_eef_pos"].tolist(),
                "robot0_eef_quat": obs["robot0_eef_quat"].tolist(),
                "robot0_gripper_qpos": obs["robot0_gripper_qpos"].tolist()}

    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["operation"] == "close":
                    break
                if request["operation"] in ("reset", "reset_demo"):
                    validate_reset_request(request, approved, diagnostic=bool(args.diagnostic_dir), target=args.target_mode)
                    requested = request["task"]
                    if requested not in approved:
                        raise ValueError("RPC rejects unapproved tasks")
                    if requested != task:
                        if env is not None:
                            env.close()
                        suite = "libero_10" if args.target_mode else "libero_90"
                        env = OffScreenRenderEnv(bddl_file_name=str(base / "bddl_files" / suite / (requested + ".bddl")),
                            camera_heights=128, camera_widths=128, camera_names=["agentview", "robot0_eye_in_hand"],
                            control_freq=20, horizon=400, initialization_noise=None, hard_reset=False)
                        task = requested
                    env.seed(int(request["seed"]))
                    obs = env.env.reset()
                    if request["operation"] == "reset_demo":
                        import robosuite
                        state = np.asarray(request["initial_state"], dtype=np.float64)
                        if state.shape != env.get_sim_state().shape or not np.isfinite(state).all():
                            raise ValueError("invalid diagnostic initial state")
                        relocated = relocate_demo_assets(request["model_xml"],
                            Path(robosuite.__file__).parent, base / "assets")
                        env.reset_from_xml_string(relocated)
                        env.env.sim.reset()
                        obs = env.set_init_state(state)
                    reset_metadata = {}
                    if args.target_mode:
                        import hashlib
                        state_path = args.target_initial_states / (requested + ".npy")
                        state = np.load(state_path, allow_pickle=False)
                        state = np.asarray(state, dtype=np.float64)
                        if state.shape != env.get_sim_state().shape or not np.isfinite(state).all():
                            raise ValueError("invalid benchmark initial state")
                        obs = env.set_init_state(state)
                        reset_metadata = {"initial_state_id": request["initial_state_id"],
                            "initial_state_sha256": hashlib.sha256(state.astype('<f8').tobytes()).hexdigest(),
                            "bddl_sha256": hashlib.sha256((base / "bddl_files/libero_10" / (requested + ".bddl")).read_bytes()).hexdigest()}
                    step = 0
                    lower, upper = env.env.action_spec
                    if env.check_success():
                        raise ValueError("trial already successful at reset; cannot count a control success")
                    result = {"observations": [pack(obs, step)], "success": False,
                              "lower": lower.tolist(), "upper": upper.tolist(), "image_convention": macros.IMAGE_CONVENTION, **reset_metadata}
                elif request["operation"] == "step":
                    actions = np.asarray(request["actions"], dtype=np.float64)
                    if env is None or actions.ndim != 2 or actions.shape[1] != 7 or not 1 <= len(actions) <= 8:
                        raise ValueError("source RPC expects at most eight Panda actions")
                    if args.target_mode and step + len(actions) > 400:
                        raise ValueError("target step budget exceeded")
                    if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                        raise ValueError("unsafe controller commands")
                    frames, success, done = [], False, False
                    before_time = float(env.env.sim.data.time)
                    for action in actions:
                        obs, _, done, _ = env.step(action)
                        step += 1
                        success = bool(env.check_success())
                        if step % 4 == 0 or success or done or args.diagnostic_dir or args.target_mode:
                            frames.append(pack(obs, step))
                        if success or done:
                            break
                    result = {"observations": frames, "success": success, "done": bool(done), "step": step,
                              "simulated_seconds": float(env.env.sim.data.time) - before_time}
                else:
                    raise ValueError("unknown source RPC operation")
                if args.diagnostic_dir:
                    # Privileged state is written only to a diagnostic artifact.
                    # It never enters the response consumed by the policy.
                    trace.append({"operation": request["operation"], "task": task,
                                  "step": step, "simulator_state": env.get_sim_state().tolist(),
                                  "success": result["success"]})
                    (args.diagnostic_dir / "simulator-trace.json").write_text(json.dumps(trace))
                protocol.write(json.dumps({"ok": True, **result}) + "\n")
                protocol.flush()
            except Exception:
                protocol.write(json.dumps({"ok": False, "error": traceback.format_exc()}) + "\n")
                protocol.flush()
                return 1
    finally:
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
