"""Read-only source controller diagnostics; not a source-success evaluation.

Recorded simulator state is used solely to initialize demonstration replay.
Policy inputs contain measured RGB/proprioception only. No optimizer is run.
"""

import base64
import json
from pathlib import Path
import select
import subprocess
import time

import numpy as np

from semflow.source_batching import HISTORY_OFFSETS, panda_observation, controller_actions, source_window
from semflow.source_data import decode_episode, normalize


TASK = "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet"


def action_comparison(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != (32, 7) or target.shape != (32, 7):
        raise ValueError("comparison requires paired 32x7 normalized chunks")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("nonfinite action comparison")
    squared = (prediction - target) ** 2
    return {"mse": float(squared.mean()), "per_dimension_mse": squared.mean(0).tolist(),
            "first_eight_mse": float(squared[:8].mean()),
            "train_mean_baseline_mse": float((target ** 2).mean())}


def observed_window(response, history, step, convention, index):
    for frame in response["observations"]:
        rgb = np.frombuffer(base64.b64decode(frame["rgb"]), dtype=np.uint8).reshape(2, 128, 128, 3)
        obs = dict(frame, agentview_image=rgb[0], robot0_eye_in_hand_image=rgb[1])
        history[frame["step"]] = panda_observation(obs, index["normalization"], image_convention=convention)
    ids = [max(0, step + int(offset)) for offset in HISTORY_OFFSETS]
    return {"history_rgb": np.stack([history[i]["rgb"] for i in ids], axis=1),
            "current_rgb": history[step]["rgb"],
            "panda_states": np.stack([history[i]["state"] for i in ids]),
            "observation_times": (np.asarray(ids, dtype=np.float32) - step) / 20,
            "instruction": next(s["instruction"] for s in index["sources"] if s["task"] == TASK)}


def frame_png(path, rgb):
    from PIL import Image
    Image.fromarray(np.concatenate(rgb, axis=1)).save(path)


class Simulator:
    def __init__(self, args, directory):
        self.stderr = (directory / "simulator.log").open("w")
        self.worker = subprocess.Popen([str(args.simulator_python), str(Path(__file__).with_name("source_simulator_worker.py")),
            "--libero-checkout", str(args.libero_checkout), "--config-dir", str(directory / "libero_config"),
            "--tasks", str(args.pilot.parent / "tasks.json"), "--diagnostic-dir", str(directory / "privileged-diagnostic")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1)

    def call(self, request):
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        if not select.select([self.worker.stdout], [], [], 60)[0]:
            raise TimeoutError("diagnostic simulator reply exceeded 60 seconds")
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError("diagnostic simulator exited; inspect its log")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response

    def close(self):
        try:
            if self.worker.poll() is None:
                self.worker.stdin.write(json.dumps({"operation": "close"}) + "\n")
                self.worker.stdin.flush()
                try:
                    self.worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
        finally:
            self.stderr.close()


def diagnose(args, report, policy, processor, index, config, source_memory):
    import torch
    import probe_egosteer as native_probe

    directory = args.output.parent / "controller-diagnostic"
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    report["limitations"].append("One source task, fixed windows and paired rollouts; not source-gate or transfer evidence.")
    report["diagnostic"] = {"task": TASK, "optimizer_steps": 0, "offline": [], "rollouts": []}
    diagnostic = report["diagnostic"]
    train_actions = np.concatenate([source_memory.handles[TASK][e["episode_key"]]["actions"][1:]
        for e in index["manifest"]["episodes"] if e["task"] == TASK and e["split"] == "train"])
    diagnostic["source_train_action_distribution"] = {
        "frames": len(train_actions), "mean": train_actions.mean(0).tolist(),
        "std": train_actions.std(0).tolist(),
        "quantile_levels": [0, .05, .5, .95, 1],
        "quantiles": np.quantile(train_actions, [0, .05, .5, .95, 1], axis=0).tolist()}

    def persist():
        native_probe.write_report(report, args.output)

    def budget():
        if time.monotonic() - started > 720:
            raise TimeoutError("720-second controller diagnostic budget exhausted")

    records = {}
    for split in ("train", "validation"):
        records[split] = sorted((e for e in index["manifest"]["episodes"]
            if e["task"] == TASK and e["split"] == split
            and index["provenance"][e["sha256"]]["aligned_frames"] >= 61), key=lambda e: e["sha256"])[0]
        episode = decode_episode(source_memory.handles[TASK], index, records[split])
        window = source_window(episode, 20)
        budget()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            observation = policy.encode_observations(**processor([window], "cuda"))
            noise = torch.randn((1, 32, 7), device="cuda", generator=torch.Generator(device="cuda").manual_seed(4242))
            normal = policy.sample_actions(observation, noise, num_steps=10)
            zero = policy.sample_actions(observation, noise, num_steps=10, forecast_intervention="zero")
        prediction = normal[0].float().cpu().numpy()
        target = window["loss_targets"]["actions"]
        values = dict(window["provenance"], **action_comparison(prediction, target),
            normalized_prediction=prediction.tolist(), normalized_target=target.tolist(),
            zero_forecast_max_action_delta=float((normal - zero).abs().max()))
        diagnostic["offline"].append(values)
        frame_png(directory / (split + "-window.png"), window["current_rgb"])
        persist()

    # Replay raw actions from the corresponding pre-action state. The producer
    # stores observations AFTER actions[i], but states[i] BEFORE actions[i].
    record = records["train"]
    raw_episode = source_memory.handles[TASK][record["episode_key"]]
    reference = decode_episode(source_memory.handles[TASK], index, record)
    demo_dir = directory / "demonstration-replay"
    demo_dir.mkdir()
    simulator = Simulator(args, demo_dir)
    replay = {"episode_sha256": record["sha256"], "split": "train", "comparisons": [],
              "state_usage": "initialization only, never a policy input or training target"}
    diagnostic["demonstration_replay"] = replay
    try:
        initial_state = np.asarray(raw_episode["states"][0])
        if not np.array_equal(initial_state, raw_episode.attrs["init_state"]):
            raise ValueError("demonstration init_state differs from states[0]")
        response = simulator.call({"operation": "reset_demo", "task": TASK, "seed": 4242,
            "initial_state": initial_state.tolist(), "model_xml": str(raw_episode.attrs["model_file"])})
        convention = response["image_convention"]
        actions = np.asarray(raw_episode["actions"])
        for offset in range(0, len(actions), 8):
            budget()
            response = simulator.call({"operation": "step", "actions": actions[offset:offset + 8].tolist()})
            for frame in response["observations"]:
                row = frame["step"] - 1
                if row >= len(reference["actions"]):
                    continue
                rgb = np.frombuffer(base64.b64decode(frame["rgb"]), dtype=np.uint8).reshape(2, 128, 128, 3)
                measured = panda_observation(dict(frame, agentview_image=rgb[0], robot0_eye_in_hand_image=rgb[1]),
                                              index["normalization"], image_convention=convention)
                ref_rgb = np.stack([reference["rgb"][c][row] for c in ("agentview_rgb", "eye_in_hand_rgb")])
                replay["comparisons"].append({"row": row,
                    "state_max_abs_normalized_error": float(np.abs(measured["state"] - reference["panda_state"][row]).max()),
                    "rgb_mean_abs_error": float(np.abs(measured["rgb"].astype(float) - ref_rgb).mean())})
            last = response["observations"][-1]
            rgb = np.frombuffer(base64.b64decode(last["rgb"]), dtype=np.uint8).reshape(2, 128, 128, 3)
            frame_png(demo_dir / ("step-%04d.png" % response["step"]), rgb[:, ::-1] if convention == "opengl" else rgb)
            replay.update(executed_steps=response["step"], success=response["success"])
            persist()
            if response["success"] or response["done"]:
                break
    finally:
        simulator.close()

    # One fixed-seed pair, identical resets and caller-supplied flow noise.
    reset_reference = None
    rollout_dir = directory / "policy-pair"
    rollout_dir.mkdir()
    simulator = Simulator(args, rollout_dir)
    try:
        for intervention in ("none", "zero"):
            response = simulator.call({"operation": "reset", "task": TASK, "seed": 10000})
            if reset_reference is not None and reset_reference != response:
                raise RuntimeError("paired seeded resets differ")
            reset_reference = response
            convention, lower, upper = response["image_convention"], response["lower"], response["upper"]
            history, step = {}, 0
            generator = torch.Generator(device="cuda").manual_seed(10000)
            rollout = {"seed": 10000, "forecast_intervention": intervention, "chunks": []}
            diagnostic["rollouts"].append(rollout)
            while step < 400:
                budget()
                window = observed_window(response, history, step, convention, index)
                frame_png(rollout_dir / (intervention + "-%04d.png" % step), window["current_rgb"])
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    observation = policy.encode_observations(**processor([window], "cuda"))
                    noise = torch.randn((1, 32, 7), device="cuda", generator=generator)
                    prediction = policy.sample_actions(observation, noise, num_steps=10, forecast_intervention=intervention)
                    counterfactual = policy.sample_actions(observation, noise, num_steps=10, forecast_intervention="zero") if intervention == "none" and step == 0 else None
                normalized = prediction[0, :8].float().cpu().numpy()
                raw = normalize(normalized, index["normalization"]["action"], inverse=True)
                actions, clipped = controller_actions(normalized, index["normalization"], lower, upper)
                chunk = {"step": step, "panda_state_normalized": window["panda_states"][-1].tolist(),
                         "normalized_actions": normalized.tolist(), "raw_actions": raw.tolist(),
                         "controller_actions": actions.tolist(), "clipped_positions": np.argwhere((raw < -1) | (raw > 1)).tolist(),
                         "clipped_values": clipped}
                if counterfactual is not None:
                    chunk["zero_forecast_max_action_delta"] = float((prediction - counterfactual).abs().max())
                rollout["chunks"].append(chunk)
                response = simulator.call({"operation": "step", "actions": actions.tolist()})
                executed = response["step"] - step
                if not 1 <= executed <= 8 or not np.isclose(response["simulated_seconds"], executed / 20, atol=1e-7):
                    raise RuntimeError("diagnostic controller timing differs from 20 Hz")
                step = response["step"]
                rollout.update(steps=step, success=response["success"])
                persist()
                if step % 80 == 0:
                    print(json.dumps({"event": "diagnostic_progress", "intervention": intervention, "step": step}), flush=True)
                if response["success"] or response["done"]:
                    break
            # Include the final measured frame/state, even on early success.
            final_window = observed_window(response, history, step, convention, index)
            frame_png(rollout_dir / (intervention + "-%04d.png" % step), final_window["current_rgb"])
            rollout["final_panda_state_normalized"] = final_window["panda_states"][-1].tolist()
    finally:
        simulator.close()
    diagnostic.update(seconds=time.monotonic() - started, paired_reset_equal=True)
    report["status"] = "passed"
    persist()
