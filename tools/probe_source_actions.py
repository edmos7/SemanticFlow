"""Fixed source-window exposure and paired integration diagnostics, never rollouts."""

import copy
import hashlib
import time

import numpy as np

from diagnose_source_controller import TASK, action_comparison
from semflow.source_batching import controller_actions, source_window
from semflow.source_data import normalize


EPISODE = "025e50246e1057d1d1fea41b9fec0f33aaff56e8742364a96045a4afac6149cd"
CHECKPOINT_SHA256 = "040a9e637c6dd621bc27efe3e70c7a4b579c698e0e02e6a1c76c09b12d124d45"
CHECKPOINT_BYTES = 49824199
WINDOW_SHA256 = "a7a08135e3deacf30299001969659a3eb19c089d42130df8d0726ccbe899fe45"
STEPS = (10, 20, 50, 100)
SEEDS = (4242, 4243, 4244, 4245)
UPDATES = 600
MODES = ("sampler-audit", "overfit", "overfit-audit")
PROVENANCE = {"episode_sha256": EPISODE, "anchor": 20, "split": "train"}


def window_digest(window):
    digest = hashlib.sha256()
    for name in ("history_rgb", "current_rgb", "panda_states", "observation_times"):
        value = np.ascontiguousarray(window[name])
        digest.update((name + str(value.shape) + str(value.dtype)).encode() + value.tobytes())
    for name in ("actions", "future_rgb"):
        value = np.ascontiguousarray(window["loss_targets"][name])
        digest.update((name + str(value.shape) + str(value.dtype)).encode() + value.tobytes())
    digest.update(window["instruction"].encode())
    return digest.hexdigest()


class FixedWindow:
    """An explicit immutable window; no random sampler or validation access."""

    def __init__(self, sampler):
        matches = [e for e in sampler.index["manifest"]["episodes"] if e["sha256"] == EPISODE]
        if len(matches) != 1 or matches[0]["task"] != TASK or matches[0]["split"] != "train":
            raise ValueError("fixed-window episode must uniquely match the declared source train task")
        self.window = source_window(sampler.episode(matches[0]), 20)
        if self.window["provenance"] != PROVENANCE:
            raise ValueError("fixed-window provenance differs")
        self.digest = window_digest(self.window)

    def state_dict(self):
        return {"kind": "fixed-source-window-v1", **PROVENANCE, "window_sha256": self.digest}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("fixed-window checkpoint provenance differs")

    def sample(self):
        if self.window["provenance"] != PROVENANCE or window_digest(self.window) != self.digest:
            raise ValueError("fixed-window contents or provenance changed")
        return [self.window]


def metrics(prediction, target, statistics):
    """Sign is measured in controller coordinates, not normalized coordinates."""
    result = action_comparison(prediction, target)
    raw = normalize(prediction, statistics["action"], inverse=True)
    truth = normalize(target, statistics["action"], inverse=True)
    commands, count = controller_actions(prediction, statistics, -np.ones(7), np.ones(7))
    clipped = (raw < -1) | (raw > 1)
    for label, selection in (("full", slice(None)), ("first_eight", slice(0, 8))):
        grip, expected = commands[selection, 6], truth[selection, 6]
        result[label] = {
            "mse": float(np.square(prediction[selection] - target[selection]).mean()),
            "per_dimension_mse": np.square(prediction[selection] - target[selection]).mean(0).tolist(),
            "gripper_sign_accuracy": float((np.sign(grip) == np.sign(expected)).mean()),
            "gripper_mean": float(grip.mean()), "gripper_min": float(grip.min()), "gripper_max": float(grip.max()),
            "gripper_negative_fraction": float((grip < 0).mean()),
            "gripper_zero_fraction": float((grip == 0).mean()),
            "gripper_positive_fraction": float((grip > 0).mean()),
            "clipped_values": int(clipped[selection].sum()),
            "clipped_by_dimension": clipped[selection].sum(0).tolist()}
    result.update(normalized_prediction=np.asarray(prediction).tolist(),
                  raw_actions=raw.tolist(), controller_actions=commands.tolist(),
                  clipped_positions=np.argwhere(clipped).tolist(), clipped_values=count)
    return result


def optimizer_contract(policy, reconstruction, optimizer, config, expected_step):
    """Audit exact group order/scope, AdamW settings, moments and counters."""
    import torch
    from semflow.models.adapters import named_shared_adapter_parameters

    if type(expected_step) is not int or expected_step < 0:
        raise ValueError("optimizer step must be a nonnegative integer")
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("fixed-window probe requires AdamW")
    adapter_ids = {id(p) for _, p in named_shared_adapter_parameters(policy.native)}
    prefixes = ("forecast.", "scene.", "conditioner.", "panda_state.", "panda_action_in.", "panda_action_out.")
    named = list(policy.named_parameters())
    intended = [(n, p) for n, p in named if n.startswith(prefixes) or id(p) in adapter_ids]
    actual = [(n, p) for n, p in named if p.requires_grad]
    if [n for n, _ in actual] != [n for n, _ in intended] or not all(p.requires_grad for p in reconstruction.parameters()):
        raise ValueError("probe trainable parameter scope differs")
    interfaces = [(n, p) for n, p in actual if id(p) not in adapter_ids]
    interfaces += [("reconstruction." + n, p) for n, p in reconstruction.named_parameters()]
    adapters = [(n, p) for n, p in actual if id(p) in adapter_ids]
    groups = (interfaces, adapters)
    if len(optimizer.param_groups) != 2 or any(not group for group in groups):
        raise ValueError("probe optimizer group count/scope differs")
    scope = []
    for actual_group, expected, lr in zip(optimizer.param_groups, groups,
                                         (config["learning_rate"], config["adapter_learning_rate"])):
        settings = {"lr": lr, "weight_decay": config["weight_decay"], "betas": (0.9, 0.999),
                    "eps": 1e-8, "amsgrad": False, "maximize": False, "foreach": False,
                    "capturable": False, "differentiable": False, "fused": None}
        if any(actual_group.get(k) != v for k, v in settings.items()):
            raise ValueError("probe optimizer hyperparameters differ")
        if [id(p) for p in actual_group["params"]] != [id(p) for _, p in expected]:
            raise ValueError("probe optimizer parameter order/scope differs")
        scope.append([n for n, _ in expected])
        for name, parameter in expected:
            state = optimizer.state.get(parameter, {})
            if expected_step == 0:
                if state:
                    raise ValueError("fresh optimizer contains moments")
                continue
            if set(state) != {"step", "exp_avg", "exp_avg_sq"} or float(state["step"]) != expected_step:
                raise ValueError("probe optimizer state/counter differs: " + name)
            for key in ("exp_avg", "exp_avg_sq"):
                if state[key].shape != parameter.shape or not torch.isfinite(state[key]).all():
                    raise ValueError("invalid optimizer moment: " + name)
    if set(optimizer.state) != ({p for group in groups for _, p in group} if expected_step else set()):
        raise ValueError("unexpected optimizer state outside declared scope")
    return scope


def probe_contract(fixed, scope):
    return {"kind": "fixed-source-overfit-v1", "window": fixed.state_dict(),
            "initial_checkpoint_sha256": CHECKPOINT_SHA256, "initial_step": 600,
            "updates": UPDATES, "optimizer": "resume complete AdamW state; no reset",
            "objective": "unchanged four-term source calibration", "optimizer_parameter_names": scope,
            "noise_seeds": list(SEEDS), "integration_steps": list(STEPS),
            "training_rng": "exact step600 RNG continuation", "pilot_flow_steps": 10}


def audit(policy, processor, fixed, statistics, persist, result, *, device="cuda"):
    """Encode once per checkpoint, reuse the same tensor noise for every NFE."""
    import torch
    from train_source_calibration import tensor_digest

    window = fixed.sample()[0]
    target = window["loss_targets"]["actions"]
    result.update(window=fixed.state_dict(), normalized_target=target.tolist(),
                  train_mean_baseline=metrics(np.zeros_like(target), target, statistics), samples=[],
                  latency_scope="CUDA synchronized sampler only; one warmup; encoding excluded",
                  controller_bounds=[[-1] * 7, [1] * 7], executed_actions=0)
    policy.eval()
    started = time.monotonic()
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        observation = policy.encode_observations(**processor([window], device))
        # Warm kernels without changing the paired seeds or encoding.
        policy.sample_actions(observation, torch.zeros((1, 32, 7), device=device), num_steps=10)
        for seed in SEEDS:
            noise = torch.randn((1, 32, 7), device=device, generator=torch.Generator(device=device).manual_seed(seed))
            noise_hash = tensor_digest({"noise": noise})
            for steps in STEPS:
                if time.monotonic() - started > 900:
                    raise TimeoutError("paired sampler audit exceeded 900 seconds")
                synchronize()
                begin = time.perf_counter()
                prediction = policy.sample_actions(observation, noise, num_steps=steps)
                synchronize()
                seconds = time.perf_counter() - begin
                if tensor_digest({"noise": noise}) != noise_hash:
                    raise RuntimeError("sampler mutated paired initial noise")
                result["samples"].append(dict(seed=seed, integration_steps=steps,
                    initial_noise_sha256=noise_hash, seconds=seconds,
                    **metrics(prediction[0].float().cpu().numpy(), target, statistics)))
                persist()
    result["seconds"] = time.monotonic() - started


def run_probe(args, report, policy, reconstruction, optimizer, sampler, processor,
              index, config, contract, loss_step):
    import torch
    import probe_egosteer as native_probe
    from semflow.models.source_calibration import checkpoint_payload, restore_checkpoint, trainable_delta
    from train_source_calibration import save_checkpoint, tensor_digest

    checkpoint = args.checkpoint
    if checkpoint is None or args.checkpoint_sha256 is None:
        raise ValueError("probe requires an explicit hash-verified input checkpoint")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise ValueError("probe input checkpoint hash differs")
    if args.mode != "overfit-audit" and (args.checkpoint_sha256 != CHECKPOINT_SHA256 or checkpoint.stat().st_size != CHECKPOINT_BYTES):
        raise ValueError("probe requires the immutable failed step600 checkpoint")
    directory = args.output.parent / args.mode
    directory.mkdir(exist_ok=False)
    fixed = FixedWindow(sampler)
    if fixed.digest != WINDOW_SHA256:
        raise ValueError("fixed window differs from the independently verified source bytes")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    expected_step = 600 + UPDATES if args.mode == "overfit-audit" else 600
    if saved["step"] != expected_step:
        raise ValueError("probe checkpoint step differs")
    restore_checkpoint(saved, policy, reconstruction, optimizer,
                       fixed if args.mode == "overfit-audit" else sampler, contract=contract)
    scope = optimizer_contract(policy, reconstruction, optimizer, config, expected_step)
    declaration = probe_contract(fixed, scope)
    if args.mode == "overfit-audit" and saved.get("action_probe") != declaration:
        raise ValueError("overfit checkpoint declaration differs")
    report.update(action_probe=declaration, input_checkpoint_sha256=args.checkpoint_sha256,
                  training=[], optimizer_steps=0)
    def persist():
        native_probe.write_report(report, args.output)
    persist()
    if args.mode == "overfit":
        if config["batch_size"] != 1:
            raise ValueError("fixed-window probe requires batch size one")
        # Guard frozen parameter writes as well as requires_grad scope.
        frozen = [(name, p, p._version) for name, p in policy.named_parameters() if not p.requires_grad]
        started = time.monotonic()
        for update in range(1, UPDATES + 1):
            if time.monotonic() - started > 1200:
                raise TimeoutError("fixed-window optimization exceeded 1200 seconds")
            values, windows = loss_step(fixed.sample())
            if len(windows) != 1 or windows[0] is not fixed.window:
                raise ValueError("optimizer used another window")
            fixed.sample()  # reject mutation by the loss/processor
            if any(p._version != version or p.grad is not None for _, p, version in frozen):
                raise RuntimeError("fixed-window optimization changed frozen scope")
            optimizer_contract(policy, reconstruction, optimizer, config, 600 + update)
            report["training"].append(dict(update=update, optimizer_step=600 + update,
                                           seconds=time.monotonic() - started, **values))
            report["optimizer_steps"] = update
            persist()
            if update % 20 == 0:
                print(report["training"][-1], flush=True)
            if update % 100 == 0 or update == UPDATES:
                payload = checkpoint_payload(policy, reconstruction, optimizer, fixed,
                                             step=600 + update, contract=contract)
                payload["action_probe"] = copy.deepcopy(declaration)
                # Do not carry stale scene statistics into this diagnostic checkpoint.
                destination = directory / ("fixed-window-step%d.pt" % update)
                if destination.exists():
                    raise FileExistsError(destination)
                save_checkpoint(destination, payload)
                report["checkpoint"] = {"file": str(destination.relative_to(args.output.parent)),
                    "bytes": destination.stat().st_size, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                    "step": 600 + update, "probe_updates": update, "diagnostic_only": True}
                persist()
    else:
        before = tensor_digest(trainable_delta(policy))
        report["sampler_audit"] = {}
        audit(policy, processor, fixed, index["normalization"], persist, report["sampler_audit"])
        if tensor_digest(trainable_delta(policy)) != before:
            raise RuntimeError("sampler audit changed model parameters")
        optimizer_contract(policy, reconstruction, optimizer, config, expected_step)
        report["parameters_unchanged"] = True
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != args.checkpoint_sha256:
        raise RuntimeError("probe changed its input checkpoint")
    report["status"] = "passed"
    persist()
