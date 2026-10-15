"""Bounded source exposure with exact task balance and a frozen diagnostic panel."""

import hashlib
import json
import time

import numpy as np

from probe_source_actions import (CHECKPOINT_BYTES, CHECKPOINT_SHA256, SEEDS,
                                 metrics, optimizer_contract, window_digest)
from semflow.source_batching import source_window

BLOCK = 600
ROUNDS = 3
SCHEDULE_SEED = 424242


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class BalancedSchedule:
    """Round-robin tasks; hashed episode permutations and anchors, no global RNG."""

    max_updates = BLOCK * ROUNDS

    def __init__(self, source):
        if source.split != "train" or len(source.tasks) != 6:
            raise ValueError("balanced training requires six verified source train tasks")
        self.source = source
        self.tasks = sorted(source.tasks)
        self.episodes = {task: sorted(source.episodes[task], key=lambda e: e["sha256"]) for task in self.tasks}
        if any(not rows or any(e["split"] != "train" or e["task"] != task for e in rows)
               for task, rows in self.episodes.items()):
            raise ValueError("balanced schedule includes invalid training records")
        self.identity = digest(self.episodes)
        self.updates = 0

    def state_dict(self):
        return {"kind": "balanced-source-v1", "seed": SCHEDULE_SEED,
                "episode_order_sha256": self.identity, "updates": self.updates}

    def load_state_dict(self, state):
        updates = state.get("updates")
        expected = dict(self.state_dict(), updates=updates)
        if state != expected or type(updates) is not int or not 0 <= updates <= self.max_updates:
            raise ValueError("balanced schedule state/provenance differs")
        self.updates = updates

    def next_record(self):
        if self.updates >= self.max_updates:
            raise ValueError("balanced exposure budget exhausted")
        task = self.tasks[self.updates % 6]
        visit = self.updates // 6
        records = self.episodes[task]
        cycle, position = divmod(visit, len(records))
        ordered = sorted(records, key=lambda e: digest([SCHEDULE_SEED, task, cycle, e["sha256"]]))
        record = ordered[position]
        frames = self.source.index["provenance"][record["sha256"]]["aligned_frames"]
        anchor = 20 + int(digest([SCHEDULE_SEED, record["sha256"], visit]), 16) % (frames - 60)
        return record, anchor

    def sample(self):
        record, anchor = self.next_record()
        window = source_window(self.source.episode(record), anchor)
        if window["provenance"] != {"episode_sha256": record["sha256"], "anchor": anchor, "split": "train"}:
            raise ValueError("balanced window differs from scheduled source provenance")
        self.updates += 1
        return [window]


def panel_records(source):
    """Select before observing any model outcomes; never change across checkpoints."""
    selected = []
    for task in sorted(source.tasks):
        for split in ("train", "validation"):
            records = sorted((e for e in source.index["manifest"]["episodes"]
                              if e["task"] == task and e["split"] == split
                              and source.index["provenance"][e["sha256"]]["aligned_frames"] >= 61),
                             key=lambda e: e["sha256"])
            if len(records) < 2:
                raise ValueError("panel requires two distinct source episodes per task/split")
            for i, record in enumerate(records[:2]):
                last = source.index["provenance"][record["sha256"]]["aligned_frames"] - 41
                selected.append({"record": record, "anchor": 20 if i == 0 else (20 + last) // 2})
    if len({e["record"]["sha256"] for e in selected}) != 24:
        raise ValueError("panel episode identity overlaps tasks or splits")
    return selected


def declaration(source, scope):
    return {"kind": "balanced-exposure-v1", "initial_checkpoint_sha256": CHECKPOINT_SHA256,
            "initial_step": 600, "rounds": ROUNDS, "updates_per_round": BLOCK,
            "schedule_seed": SCHEDULE_SEED, "optimizer_parameter_names": scope,
            "optimizer": "resume complete AdamW and training RNG; unchanged four-term objective",
            "panel": panel_records(source), "noise_seeds": list(SEEDS), "flow_steps": 10,
            "rollouts": 0, "target_data": False}


def summarize_panel(rows):
    summary = {}
    for task in sorted({r["task"] for r in rows}):
        summary[task] = {}
        for split in ("train", "validation"):
            windows = [r for r in rows if r["task"] == task and r["split"] == split]
            if len(windows) != 2 or any([s["seed"] for s in w["samples"]] != list(SEEDS) for w in windows):
                raise ValueError("panel summary requires complete paired task/split coverage")
            samples = [s for w in windows for s in w["samples"]]
            result = {}
            for prefix in ("full", "first_eight"):
                result[prefix] = {key: np.mean([s[prefix][key] for s in samples], axis=0).tolist()
                                  for key in ("mse", "per_dimension_mse", "gripper_sign_accuracy", "gripper_mean", "clipped_values")}
                baseline = float(np.mean([w["train_mean_baseline"][prefix]["mse"] for w in windows]))
                result[prefix].update(train_mean_baseline_mse=baseline,
                                      beats_train_mean=result[prefix]["mse"] < baseline,
                                      paired_fraction_beating_baseline=float(np.mean([
                                          s[prefix]["mse"] < w["train_mean_baseline"][prefix]["mse"]
                                          for w in windows for s in w["samples"]])))
            summary[task][split] = result
    if len(summary) != 6:
        raise ValueError("panel summary requires exactly six tasks")
    return {"tasks": summary, "all_task_split_mean_errors_below_baseline": all(
        value[prefix]["beats_train_mean"] for task in summary.values() for value in task.values()
        for prefix in ("full", "first_eight")),
        "interpretation": "Offline action diagnostic only; gripper/clipping need review; no automatic rollout gate."}


def run_panel(policy, processor, source, persist, report, *, device="cuda"):
    import torch
    from train_source_calibration import tensor_digest

    before = tensor_digest(dict(policy.named_parameters()))
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all()
    started = time.monotonic()
    report["panel"] = []
    policy.eval()
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for item in panel_records(source):
            if time.monotonic() - started > 600:
                raise TimeoutError("source action panel exceeded 600 seconds")
            record, anchor = item["record"], item["anchor"]
            window = source_window(source.episode(record), anchor)
            target = window["loss_targets"]["actions"]
            row = {"task": record["task"], **window["provenance"], "window_sha256": window_digest(window),
                   "normalized_target": target.tolist(), "samples": [],
                   "train_mean_baseline": metrics(np.zeros_like(target), target, source.index["normalization"])}
            report["panel"].append(row)
            observation = policy.encode_observations(**processor([window], device))
            # One warmup; subsequent latency includes sampling only.
            policy.sample_actions(observation, torch.zeros((1, 32, 7), device=device), num_steps=10)
            for seed in SEEDS:
                noise = torch.randn((1, 32, 7), device=device, generator=torch.Generator(device=device).manual_seed(seed))
                noise_sha = tensor_digest({"noise": noise})
                synchronize()
                begin = time.perf_counter()
                prediction = policy.sample_actions(observation, noise, num_steps=10)
                synchronize()
                seconds = time.perf_counter() - begin
                if tensor_digest({"noise": noise}) != noise_sha:
                    raise RuntimeError("panel sampler mutated paired initial noise")
                row["samples"].append(dict(seed=seed, integration_steps=10, initial_noise_sha256=noise_sha,
                    seconds=seconds, **metrics(prediction[0].float().cpu().numpy(), target, source.index["normalization"])))
            persist()
    if (tensor_digest(dict(policy.named_parameters())) != before or not torch.equal(rng_before, torch.get_rng_state())
            or any(not torch.equal(a, b) for a, b in zip(cuda_before, torch.cuda.get_rng_state_all()))):
        raise RuntimeError("panel changed parameters or global training RNG")
    report.update(panel_summary=summarize_panel(report["panel"]), panel_seconds=time.monotonic() - started,
                  latency_scope="CUDA synchronized sampling only; one warmup per window; encoding excluded",
                  parameters_and_training_rng_unchanged=True)


def run_balanced(args, report, policy, reconstruction, optimizer, source, processor, config, contract, loss_step):
    import torch
    import probe_egosteer as native_probe
    from semflow.models.source_calibration import checkpoint_payload, restore_checkpoint
    from train_source_calibration import save_checkpoint

    round_id = args.balanced_round
    training = args.mode == "balanced-train"
    if type(round_id) is not int or not (1 if training else 0) <= round_id <= ROUNDS:
        raise ValueError("invalid bounded source round")
    previous = round_id - 1 if training else round_id
    checkpoint = args.checkpoint
    if checkpoint is None or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise ValueError("balanced source checkpoint hash differs")
    if previous == 0 and (args.checkpoint_sha256 != CHECKPOINT_SHA256 or checkpoint.stat().st_size != CHECKPOINT_BYTES):
        raise ValueError("balanced continuation must start at the immutable common step600")
    output = args.output.parent / (args.mode + "-" + str(round_id))
    output.mkdir(exist_ok=False)
    schedule = BalancedSchedule(source)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    expected_step = 600 + previous * BLOCK
    if saved["step"] != expected_step:
        raise ValueError("balanced checkpoint optimizer step differs")
    restore_checkpoint(saved, policy, reconstruction, optimizer, schedule if previous else source, contract=contract)
    scope = optimizer_contract(policy, reconstruction, optimizer, config, expected_step)
    declared = declaration(source, scope)
    if previous and (saved.get("balanced_exposure") != declared or schedule.updates != previous * BLOCK):
        raise ValueError("balanced checkpoint schedule/declaration differs")
    report.update(balanced_exposure=declared, round=round_id, input_checkpoint_sha256=args.checkpoint_sha256,
                  optimizer_steps=0, training=[])
    def persist():
        native_probe.write_report(report, args.output)
    persist()
    if training:
        if config["batch_size"] != 1:
            raise ValueError("balanced continuation requires batch size one")
        frozen = [(p, p._version) for p in policy.parameters() if not p.requires_grad]
        counts = dict.fromkeys(schedule.tasks, 0)
        started = time.monotonic()
        for local_step in range(1, BLOCK + 1):
            if time.monotonic() - started > 1800:
                raise TimeoutError("balanced source block exceeded 1800 seconds")
            record, anchor = schedule.next_record()
            windows = schedule.sample()
            fingerprint = window_digest(windows[0])
            values, returned = loss_step(windows)
            if returned is not windows or window_digest(windows[0]) != fingerprint:
                raise RuntimeError("training window changed during loss computation")
            if any(p._version != version or p.grad is not None for p, version in frozen):
                raise RuntimeError("balanced training changed frozen base scope")
            optimizer_contract(policy, reconstruction, optimizer, config, expected_step + local_step)
            counts[record["task"]] += 1
            row = {"step": expected_step + local_step, "task": record["task"], **windows[0]["provenance"],
                   "window_sha256": fingerprint, "seconds": time.monotonic() - started, **values}
            report["training"].append(row)
            report["optimizer_steps"] = local_step
            if local_step % 20 == 0:
                print(json.dumps(row), flush=True)
                persist()
        if set(counts.values()) != {100}:
            raise RuntimeError("source block did not expose each task exactly 100 times")
        payload = checkpoint_payload(policy, reconstruction, optimizer, schedule,
                                     step=600 + round_id * BLOCK, contract=contract)
        payload["balanced_exposure"] = declared
        destination = output / "source.pt"
        save_checkpoint(destination, payload)
        report.update(task_updates=counts, checkpoint={"file": str(destination.relative_to(args.output.parent)),
            "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "step": 600 + round_id * BLOCK, "source_gate_unmeasured": True,
            "feature_statistics": "not fitted; required before target adaptation"})
    else:
        run_panel(policy, processor, source, persist, report)
        optimizer_contract(policy, reconstruction, optimizer, config, expected_step)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise RuntimeError("balanced stage overwrote its input checkpoint")
    report["status"] = "passed"
    persist()
