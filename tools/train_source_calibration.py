#!/usr/bin/env python3
"""Bounded native source calibration; robot demonstration files stay in RAM.

Cloud-only runner. Persistent local data and exported checkpoints belong on the
configured External HD. No target demonstrations or human clips are accepted.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import random
import select
import signal
import subprocess
import sys
import time
import urllib.request

import probe_egosteer as native_probe


def tensor_digest(tensors):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


class MemorySources:
    def __init__(self, lock):
        self.handles, self.buffers = {}, []
        import h5py
        from semflow.source_acquisition import source_url
        try:
            for entry in sorted(lock["files"], key=lambda e: e["task"]):
                print("Loading approved source into RAM: " + entry["task"], flush=True)
                buffer, digest, count = io.BytesIO(), hashlib.sha256(), 0
                self.buffers.append(buffer)
                with urllib.request.urlopen(source_url(lock, entry), timeout=60) as response:
                    while block := response.read(4 * 1024**2):
                        count += len(block)
                        if count > entry["bytes"]:
                            raise ValueError("RAM source exceeds pinned size")
                        buffer.write(block)
                        digest.update(block)
                if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
                    raise ValueError("RAM source size/hash mismatch")
                buffer.seek(0)
                self.handles[entry["task"]] = h5py.File(buffer, "r")
        except BaseException:
            self.close()
            raise

    def close(self):
        for handle in self.handles.values():
            handle.close()
        for buffer in self.buffers:
            buffer.close()


def save_checkpoint(path, payload):
    import torch
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def run(args, report):
    import numpy as np
    import torch
    from semflow.config import load_config, load_json
    from semflow.source_acquisition import validate_source_lock
    from semflow.source_batching import SourceWindowSampler
    from semflow.models import NativeSemFlowPolicy, inject_shared_lora
    from semflow.models.source_calibration import (SpatialReconstruction, enable_source_training,
        calibration_losses, checkpoint_payload, restore_checkpoint, trainable_delta)
    from semflow.native_batching import PandaBatchProcessor

    if not Path("/content").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("this bounded runner requires an explicit Colab GPU runtime")
    _, tasks = load_config(args.pilot)
    config = load_json(args.calibration)
    index = load_json(args.index)
    validate_source_lock(index["source_lock"], tasks)
    actual_hash = hashlib.sha256(args.index.read_bytes()).hexdigest()
    if actual_hash != args.index_sha256:
        raise ValueError("source index must match the verified External HD artifact")
    if (config["steps"] != 600 or config["source_trials_per_task"] != 10 or config["rollout_horizon"] != 400
            or config["control_hz"] != 20 or config["history_stride"] != 4 or config["flow_steps"] != 10):
        raise ValueError("this first bounded source candidate requires the reviewed fixed budget")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    report["calibration_configuration"] = config
    report["environment"] = native_probe.environment()
    native_probe.access_preflight(report, args)
    native_probe.prepare_native_imports(args)
    paths = native_probe.download_models(report, args)
    native, cfg = native_probe.load_model(report, args, paths)
    inject_shared_lora(native)
    policy = NativeSemFlowPolicy(native).cuda()
    reconstruction = SpatialReconstruction(policy.scene.dim, native.world_model_expert.hidden_size).cuda()
    parameters = enable_source_training(policy, reconstruction)
    adapters = [p for name, p in policy.named_parameters() if p.requires_grad and ".lora_" in name]
    identities = {id(p) for p in adapters}
    interfaces = [p for p in parameters if id(p) not in identities]
    optimizer = torch.optim.AdamW([
        {"params": interfaces, "lr": config["learning_rate"]},
        {"params": adapters, "lr": config["adapter_learning_rate"]}], weight_decay=config["weight_decay"], foreach=False)
    processor = PandaBatchProcessor(native, cfg)
    contract = {"base_artifacts": native_probe.ARTIFACTS, "upstream": native_probe.UPSTREAM_REVISION,
                "source_index_sha256": actual_hash, "normalization": index["normalization"], "configuration": config}
    # JSON normalizes tuples before save/load equality checks.
    contract = json.loads(json.dumps(contract))
    report["trainable_parameters"] = {name: p.numel() for name, p in policy.named_parameters() if p.requires_grad}
    report["reconstruction_parameters"] = sum(p.numel() for p in reconstruction.parameters())
    source_memory = None
    try:
        source_lock = index["source_lock"]
        if args.mode == "conditioning-audit":
            from probe_source_conditioning import TASK
            source_lock = dict(source_lock, files=[entry for entry in source_lock["files"] if entry["task"] == TASK])
            if len(source_lock["files"]) != 1:
                raise ValueError("conditioning requires exactly one pinned bowl source file")
        source_memory = MemorySources(source_lock) if args.mode != "evaluate" else None
        sampler = SourceWindowSampler(index, tasks, source_memory.handles.__getitem__, seed=seed) if source_memory else None

        def loss_step(windows, update=True, backward_only=False):
            if (update or backward_only) and any(w["provenance"]["split"] != "train" for w in windows):
                raise ValueError("optimizer rejected non-training source windows")
            observations = processor(windows, "cuda")
            future = torch.from_numpy(np.stack([w["loss_targets"]["future_rgb"] for w in windows])).cuda()
            actions = torch.from_numpy(np.stack([w["loss_targets"]["actions"] for w in windows])).float().cuda()
            with torch.set_grad_enabled(update or backward_only), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = policy.encode_observations(**observations)
                loss, terms = calibration_losses(policy, reconstruction, encoded, future, actions, weights=config["loss_weights"])
            if update or backward_only:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise RuntimeError("source training gradient is nonfinite")
                torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip"], error_if_nonfinite=True)
                if update:
                    optimizer.step()
            return {"total": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}}, windows

        checkpoint = args.output.parent / "common-source.pt"
        if args.mode == "conditioning-audit":
            from probe_source_conditioning import run_probe
            run_probe(args, report, policy, reconstruction, optimizer, sampler, processor, config, contract)
        elif args.mode in ("balanced-train", "balanced-panel"):
            from train_balanced_source import run_balanced
            run_balanced(args, report, policy, reconstruction, optimizer, sampler, processor,
                         config, contract, loss_step)
        elif args.mode in ("sampler-audit", "overfit", "overfit-audit"):
            from probe_source_actions import run_probe
            run_probe(args, report, policy, reconstruction, optimizer, sampler, processor,
                      index, config, contract, loss_step)
        elif args.mode == "smoke":
            before = tensor_digest(trainable_delta(policy))
            metrics, windows = loss_step(sampler.sample(config["batch_size"]), update=False, backward_only=True)
            groups = {}
            for name, parameter in policy.named_parameters():
                if not parameter.requires_grad and parameter.grad is not None:
                    raise RuntimeError("frozen base received a gradient: " + name)
                group = "shared_adapters" if ".lora_" in name else name.split(".")[0]
                if parameter.requires_grad and parameter.grad is not None:
                    groups[group] = groups.get(group, 0.) + float(parameter.grad.float().square().sum())
            groups["reconstruction"] = sum(float(p.grad.float().square().sum())
                for p in reconstruction.parameters() if p.grad is not None)
            required = {"shared_adapters", "forecast", "scene", "conditioner", "panda_state",
                        "panda_action_in", "panda_action_out", "reconstruction"}
            if not required.issubset(groups) or any(not np.isfinite(groups[n]) or groups[n] <= 0 for n in required):
                raise RuntimeError("real source backward lacks finite nonzero gradients: " + str(groups))
            if before != tensor_digest(trainable_delta(policy)) or optimizer.state:
                raise RuntimeError("smoke must not update weights or optimizer state")
            report["real_batch_smoke"] = {"losses": metrics, "windows": [w["provenance"] for w in windows],
                "gradient_norms": {n: value ** .5 for n, value in groups.items()},
                "optimizer_steps": 0, "parameters_unchanged": True, "memory": native_probe.memory()}
            native_probe.write_report(report, args.output)
            optimizer.zero_grad(set_to_none=True)
            evaluate(args, report, policy, processor, index, config, smoke=True)
        elif args.mode == "train":
            report["training"] = []
            started = time.monotonic()
            for step in range(1, config["steps"] + 1):
                if time.monotonic() - started > config["training_timeout_seconds"]:
                    raise TimeoutError("bounded source training budget exhausted")
                metrics, _ = loss_step(sampler.sample(config["batch_size"]))
                if step == 1 or step % 20 == 0:
                    record = {"step": step, "seconds": time.monotonic() - started, **metrics}
                    report["training"].append(record)
                    print(json.dumps(record), flush=True)
                    native_probe.write_report(report, args.output)
                if step % config["checkpoint_interval"] == 0 or step == config["steps"]:
                    save_checkpoint(checkpoint, checkpoint_payload(policy, reconstruction, optimizer, sampler, step=step, contract=contract))
            # Separate held-out source windows; never restore their RNG into train.
            validation = SourceWindowSampler(index, tasks, source_memory.handles.__getitem__, split="validation", seed=seed + 1)
            report["validation_losses"] = [loss_step(validation.sample(config["batch_size"]), update=False)[0]
                                            for _ in range(config["validation_batches"])]
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            # Fit scene statistics using only explicitly task-balanced source
            # training episodes after pooling calibration, never validation.
            from semflow.models.losses import SourceFeatureStatistics
            from semflow.source_batching import source_window
            features, feature_provenance = [], []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for task in sampler.tasks:
                    for record in sorted(sampler.episodes[task], key=lambda e: e["sha256"])[:4]:
                        episode = sampler.episode(record)
                        window = source_window(episode, 20)
                        encoded = policy.encode_observations(**processor([window], "cuda"))
                        features.append(encoded.current_scene.float().cpu())
                        feature_provenance.append(window["provenance"])
            features = torch.cat(features)
            stats = SourceFeatureStatistics.fit(features, torch.ones(len(features), dtype=torch.bool),
                source_manifest=actual_hash)
            saved["source_feature_statistics"] = {"mean": stats.mean, "scale": stats.scale,
                "source_manifest": actual_hash, "windows": feature_provenance, "fit_split": "train"}
            save_checkpoint(checkpoint, saved)
            report["source_feature_statistics"] = {"windows": len(features), "split": "train",
                "channels": len(stats.mean), "min_scale": float(stats.scale.min())}
            restore_checkpoint(saved, policy, reconstruction, optimizer, sampler, contract=contract)
            metrics, windows = loss_step(sampler.sample(config["batch_size"]))
            reference = {"losses": metrics, "windows": [w["provenance"] for w in windows],
                         "next_delta_sha256": tensor_digest(trainable_delta(policy))}
            (args.output.parent / "restart-reference.json").write_text(json.dumps(reference, indent=2) + "\n")
            report["checkpoint"] = {"file": checkpoint.name, "bytes": checkpoint.stat().st_size,
                                     "step": config["steps"], "format": "pinned-base plus complete trainable/interface delta, optimizer, RNG and source sampler"}
            report["status"] = "training_complete_evaluation_pending"
        elif args.mode == "restart":
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            restore_checkpoint(saved, policy, reconstruction, optimizer, sampler, contract=contract)
            metrics, windows = loss_step(sampler.sample(config["batch_size"]))
            actual = {"losses": metrics, "windows": [w["provenance"] for w in windows],
                      "next_delta_sha256": tensor_digest(trainable_delta(policy))}
            expected = load_json(args.output.parent / "restart-reference.json")
            report["restart"] = {"fresh_process": True, "next_step_equal": actual == expected,
                                  "expected": expected, "actual": actual}
            if actual != expected:
                raise RuntimeError("fresh-process next-step recovery differs")
            report["status"] = "passed"
        else:
            if args.mode == "diagnose":
                if args.checkpoint is None or args.checkpoint_sha256 is None:
                    raise ValueError("diagnostic requires a separately verified checkpoint")
                checkpoint = args.checkpoint
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
                    raise ValueError("diagnostic checkpoint hash mismatch")
            saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
            # Restore source RNG and all state with a sampler that need not fetch data.
            sampler = SourceWindowSampler(index, tasks, lambda _: None, seed=seed)
            restore_checkpoint(saved, policy, reconstruction, optimizer, sampler, contract=contract)
            policy.requires_grad_(False).eval()
            if args.mode == "diagnose":
                if saved["step"] != 600:
                    raise ValueError("diagnostic requires the durable step600 candidate")
                from diagnose_source_controller import diagnose
                before = tensor_digest(trainable_delta(policy))
                diagnose(args, report, policy, processor, index, config, source_memory)
                if before != tensor_digest(trainable_delta(policy)) or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
                    raise RuntimeError("diagnostic changed model or checkpoint")
                report.update(checkpoint_sha256=args.checkpoint_sha256, checkpoint_step=600,
                              optimizer_steps=0, parameters_unchanged=True)
            else:
                evaluate(args, report, policy, processor, index, config)
    finally:
        if source_memory is not None:
            source_memory.close()


def evaluate(args, report, policy, processor, index, config, *, smoke=False):
    import numpy as np
    import torch
    from semflow.source_batching import HISTORY_OFFSETS, panda_observation, controller_actions, source_success_gate
    stderr = (args.output.parent / "simulator.log").open("w")
    worker = subprocess.Popen([str(args.simulator_python), str(Path(__file__).with_name("source_simulator_worker.py")),
        "--libero-checkout", str(args.libero_checkout), "--config-dir", str(args.output.parent / "libero_config"),
        "--tasks", str(args.pilot.parent / "tasks.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=stderr, text=True, bufsize=1)
    started = time.monotonic()
    trials, successes = dict.fromkeys(index["coverage"], 0), dict.fromkeys(index["coverage"], 0)
    report["rollouts"] = []

    def call(request):
        worker.stdin.write(json.dumps(request) + "\n")
        worker.stdin.flush()
        if not select.select([worker.stdout], [], [], 60)[0]:
            raise TimeoutError("source simulator produced no response within 60 seconds")
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError("source simulator worker exited; inspect simulator.log")
        response = json.loads(line)
        if not response["ok"]:
            raise RuntimeError("source simulator error: " + response["error"])
        return response

    try:
        for task_index, task in enumerate(sorted(trials)):
            for trial in range(1 if smoke else config["source_trials_per_task"]):
                if time.monotonic() - started > config["evaluation_timeout_seconds"]:
                    raise TimeoutError("bounded source evaluation budget exhausted")
                seed = 10000 + task_index * 100 + trial
                report["active_trial"] = {"task": task, "seed": seed, "step": 0}
                native_probe.write_report(report, args.output)
                print(json.dumps({"event": "source_trial_started", **report["active_trial"]}), flush=True)
                response = call({"operation": "reset", "task": task, "seed": seed})
                if smoke:
                    repeated = call({"operation": "reset", "task": task, "seed": seed})
                    if response != repeated:
                        raise RuntimeError("source seeded soft reset changed observations, bounds or timing")
                lower, upper, convention = response["lower"], response["upper"], response["image_convention"]
                history, step, clipped, success = {}, 0, 0, False
                noise_rng = torch.Generator(device="cuda").manual_seed(seed)
                while step < (16 if smoke else config["rollout_horizon"]):
                    if time.monotonic() - started > config["evaluation_timeout_seconds"]:
                        raise TimeoutError("bounded source evaluation budget exhausted during a trial")
                    for frame in response["observations"]:
                        rgb = np.frombuffer(base64.b64decode(frame["rgb"]), dtype=np.uint8).reshape(2, 128, 128, 3)
                        obs = {**frame, "agentview_image": rgb[0], "robot0_eye_in_hand_image": rgb[1]}
                        history[frame["step"]] = panda_observation(obs, index["normalization"], image_convention=convention)
                    ids = [max(0, step + int(offset)) for offset in HISTORY_OFFSETS]
                    window = {"history_rgb": np.stack([history[i]["rgb"] for i in ids], axis=1),
                              "current_rgb": history[step]["rgb"],
                              "panda_states": np.stack([history[i]["state"] for i in ids]),
                              "observation_times": (np.asarray(ids, dtype=np.float32) - step) / 20,
                              "instruction": next(s["instruction"] for s in index["sources"] if s["task"] == task)}
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        observation = policy.encode_observations(**processor([window], "cuda"))
                        normalized = policy.sample_actions(observation, torch.randn((1, 32, 7), device="cuda", generator=noise_rng), num_steps=10)
                    actions, clipped_count = controller_actions(normalized[0, :8].float().cpu().numpy(), index["normalization"], lower, upper)
                    clipped += clipped_count
                    previous_step = step
                    response = call({"operation": "step", "actions": actions.tolist()})
                    executed = response["step"] - previous_step
                    if not 1 <= executed <= 8 or not np.isclose(response["simulated_seconds"], executed / 20, atol=1e-7):
                        raise RuntimeError("source controller observation/action timing differs from 20 Hz")
                    if not response["observations"] or response["observations"][-1]["step"] != response["step"]:
                        raise RuntimeError("source controller omitted the latest measured observation")
                    step, success = response["step"], response["success"]
                    report["active_trial"]["step"] = step
                    if step % 40 == 0:
                        native_probe.write_report(report, args.output)
                        print(json.dumps({"event": "source_trial_progress", **report["active_trial"],
                                          "evaluation_seconds": time.monotonic() - started}), flush=True)
                    if success or response["done"]:
                        break
                trials[task] += 1
                successes[task] += int(success)
                result = {"task": task, "seed": seed, "steps": step, "success": success,
                          "clipped_action_values": clipped, "controller_20hz_verified": True,
                          "reset_observations_repeated_exactly": True if smoke else None}
                report["rollouts"].append(result)
                report.pop("active_trial", None)
                print(json.dumps(result), flush=True)
                native_probe.write_report(report, args.output)
        if smoke:
            report["status"] = "passed"
            report["limitations"].append("Untrained controller smoke is not a source success measurement.")
        else:
            report["source_gate"] = source_success_gate(successes, trials, sorted(trials))
            report["status"] = "passed" if report["source_gate"]["passed"] else "source_gate_failed"
    finally:
        if worker.poll() is None:
            worker.stdin.write(json.dumps({"operation": "close"}) + "\n")
            worker.stdin.flush()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        stderr.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "train", "restart", "evaluate", "diagnose",
                                          "sampler-audit", "overfit", "overfit-audit",
                                          "balanced-train", "balanced-panel", "conditioning-audit"), required=True)
    parser.add_argument("--balanced-round", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--reference-panel", type=Path)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-path", type=Path)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--pilot", type=Path, default=Path("configs/pilot.json"))
    parser.add_argument("--calibration", type=Path, default=Path("configs/source_calibration.json"))
    parser.add_argument("--simulator-python", type=Path)
    parser.add_argument("--libero-checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    report = {"status": "running", "mode": args.mode, "limitations": ["Source-only calibration; no target adaptation or transfer result."]}
    started = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError("source calibration process exceeded its 3600-second deadline")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(3600)
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error).split("\n")[0][:600])
        import traceback
        traceback.print_exc()
    finally:
        signal.alarm(0)
        report["seconds"] = time.monotonic() - started
        native_probe.write_report(report, args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output), "seconds": report["seconds"]}), flush=True)
    return 0 if report["status"] in ("passed", "training_complete_evaluation_pending", "source_gate_failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
