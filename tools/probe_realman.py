#!/usr/bin/env python3
"""Cloud-only RealMan/Panda compatibility gate, never training or evaluation.

One pinned source train window; fresh interfaces; frozen prediction readouts.
A pass establishes wiring, not learned action quality or source success.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import probe_egosteer as native_probe

ARTIFACTS = dict(native_probe.ARTIFACTS, egosteer=(
    "EgoSteer/EgoSteer-3B-RealMan", "c41ea92b7197855dc7ab2a95ed543ddad94f7596"))
CONFIG_SHA256 = "3a414358b4e1f248d45bfb32148871a832ea792791a18db1c2a164d5c08a4ae2"
WEIGHTS_SHA256 = "22a244597ccd3c8650bf0415f5331510ec9713e07f8cc718d9d99463768d07ab"
WEIGHTS_BYTES = 5577188775
INDEX_SHA256 = "45f87e0eb9b2ab90edb2e81d0e5945da26e8184c9285476f6b7f55895a879e38"
SEED = 4242


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def verify_files(paths):
    if (file_sha(paths["config.yaml"]) != CONFIG_SHA256
            or Path(paths["model_bf16.pt"]).stat().st_size != WEIGHTS_BYTES
            or file_sha(paths["model_bf16.pt"]) != WEIGHTS_SHA256):
        raise ValueError("RealMan config or weights differ from the pinned release")


def select_window(source):
    from probe_source_conditioning import TASK, WINDOWS
    from probe_source_actions import window_digest
    from semflow.source_batching import source_window
    episode, anchor, expected = WINDOWS[0]
    rows = [r for r in source.index["manifest"]["episodes"] if r["sha256"] == episode]
    if len(rows) != 1 or rows[0]["task"] != TASK or rows[0]["split"] != "train":
        raise ValueError("RealMan gate requires the pinned bowl source train window")
    window = source_window(source.episode(rows[0]), anchor)
    if (window["provenance"] != {"episode_sha256": episode, "anchor": anchor, "split": "train"}
            or window_digest(window) != expected):
        raise ValueError("RealMan source window fingerprint differs")
    return window


def positive_gradient(loss, parameters, label):
    import math
    import torch
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError(label + " gradient is nonfinite")
    norm = sum(g.float().square().sum().item() for g in gradients if g is not None) ** .5
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError(label + " gradient is absent or zero")
    return norm


def audit(policy, processor, window, parameters, *, device="cuda"):
    """Differentiate without an optimizer; hash all parameters/buffers before/after."""
    import torch
    from probe_source_actions import window_digest
    from train_source_calibration import tensor_digest
    from semflow.models import raw_future_feature_loss

    if policy.training or window["provenance"]["split"] != "train":
        raise ValueError("gate requires eval mode and a source training window")
    if ({id(p) for p in parameters} != {id(p) for p in policy.parameters() if p.requires_grad}
            or not parameters):
        raise ValueError("gradient parameters differ from the audited trainable set")
    before, window_before = tensor_digest(policy.state_dict()), window_digest(window)
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state_all() if device == "cuda" else []
    observations = processor([{k: v for k, v in window.items() if k != "loss_targets"}], device)
    teacher_calls = []

    def current_only(_module, inputs):
        expected = observations["current_rgb"].flatten(0, 1).unsqueeze(1)
        if not torch.equal(inputs[0], expected):
            raise RuntimeError("runtime teacher received non-current RGB")
        teacher_calls.append(list(inputs[0].shape))

    def forbidden(_module, _inputs):
        raise RuntimeError("privileged world action/motion encoder executed")

    hooks = [policy.native.frozen_teacher.register_forward_pre_hook(current_only)]
    hooks += [getattr(policy.native, name).register_forward_pre_hook(forbidden)
              for name in ("wm_action_encoder", "wm_motion_encoder")]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            encoded = policy.encode_observations(**observations)
    finally:
        for hook in hooks:
            hook.remove()
    if len(teacher_calls) != 1:
        raise RuntimeError("runtime must extract current-image DINO exactly once")
    if (encoded.raw_forecasts.shape[:3] != (1, 2, 3)
            or encoded.scene_forecasts.shape[:3] != (1, 3, policy.scene.num_queries)):
        raise RuntimeError("forecast layout differs from the Panda contract")
    future = torch.from_numpy(window["loss_targets"]["future_rgb"][None]).to(device)
    actions = torch.from_numpy(window["loss_targets"]["actions"][None]).float().to(device)
    noise = torch.randn((1, 32, 7), device=device,
                        generator=torch.Generator(device=device).manual_seed(SEED))
    noise_before = noise.clone()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        target = policy.native.frozen_teacher(future.flatten(0, 1)).reshape_as(encoded.raw_forecasts).detach()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        valid = encoded.view_valid[:, :, None].expand(-1, -1, 3).flatten(0, 1)
        raw_loss = raw_future_feature_loss(encoded.raw_forecasts.flatten(0, 1), target.flatten(0, 1), valid)
        velocity = policy.predict_velocity(encoded, (noise + actions) * .5,
                                           torch.full((1, 32), .5, device=device))
        action_loss = (velocity.float() - (actions - noise)).square().mean()
    gradients = {
        "raw_future_to_shared_adapters": positive_gradient(raw_loss, parameters, "raw future"),
        "action_loss_to_shared_adapters": positive_gradient(action_loss, parameters, "action"),
        "action_loss_to_predicted_scenes": positive_gradient(action_loss, [encoded.scene_forecasts], "scene"),
        "action_loss_to_raw_forecasts": positive_gradient(action_loss, [encoded.raw_forecasts], "raw forecast"),
    }
    losses = {"raw_future": raw_loss.item(), "action": action_loss.item()}
    shapes = {"raw": list(encoded.raw_forecasts.shape), "scene": list(encoded.scene_forecasts.shape)}
    del encoded, raw_loss, action_loss, velocity, target
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        encoded = policy.encode_observations(**observations)
        original = policy.sample_actions(encoded, noise, num_steps=10)
        repeat = policy.sample_actions(encoded, noise, num_steps=10)
        zero = policy.sample_actions(encoded, noise, num_steps=10, forecast_intervention="zero")
    difference = float((original.float() - zero.float()).abs().max())
    if (original.shape != (1, 32, 7) or not torch.equal(original, repeat)
            or not torch.isfinite(original).all() or not torch.isfinite(zero).all()
            or difference <= 0):
        raise RuntimeError("ten-step sampling is invalid, nondeterministic or forecast-insensitive")
    cuda_after = torch.cuda.get_rng_state_all() if device == "cuda" else []
    if (tensor_digest(policy.state_dict()) != before or window_digest(window) != window_before
            or not torch.equal(noise, noise_before) or not torch.equal(rng_before, torch.get_rng_state())
            or len(cuda_before) != len(cuda_after)
            or any(not torch.equal(a, b) for a, b in zip(cuda_before, cuda_after))
            or any(p.grad is not None for p in policy.parameters())):
        raise RuntimeError("preflight mutated weights, buffers, input, gradients or Torch RNG")
    return dict(gradients=gradients, losses=losses, shapes=shapes, runtime_teacher_calls=teacher_calls,
        optimizer_steps=0, rollouts=0, sampled_chunks=3, flow_steps=10, repeat_bitwise_equal=True,
        forecast_zero_max_action_difference=difference, parameters_buffers_inputs_torch_rng_unchanged=True,
        normalized_sample=original[0].float().cpu().tolist(), state_sha256_before_and_after=before,
        teacher_targets_detached=True, interfaces_source_calibrated=False)


def run(args, report):
    import torch
    from semflow.config import load_config, load_json
    from semflow.models import NativeSemFlowPolicy, inject_shared_lora, enable_shared_adapter_training
    from semflow.native_batching import PandaBatchProcessor
    from semflow.source_batching import SourceWindowSampler
    from train_source_calibration import MemorySources
    from probe_source_conditioning import TASK
    from probe_source_actions import window_digest

    report["environment"] = native_probe.environment()
    if not torch.cuda.is_available():
        raise RuntimeError("RealMan compatibility requires the reviewed CUDA runtime")
    if file_sha(args.index) != INDEX_SHA256:
        raise ValueError("source index differs from the verified original")
    _, tasks = load_config(Path(__file__).resolve().parents[1] / "configs/pilot.json")
    index = load_json(args.index)
    source = SourceWindowSampler(index, tasks, lambda _task: None, seed=SEED)
    native_probe.prepare_native_imports(args)
    native_probe.access_preflight(report, args, artifacts=ARTIFACTS)
    metadata = report["access"]["egosteer"]
    if metadata["bytes"] != WEIGHTS_BYTES or metadata["etag"] != WEIGHTS_SHA256:
        raise ValueError("RealMan remote weights metadata differs")
    paths = native_probe.download_models(report, args, artifacts=ARTIFACTS)
    verify_files(paths)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    native, cfg = native_probe.load_model(report, args, paths, synchronous=True)
    report["injected_projection_paths"] = list(inject_shared_lora(native))
    policy = NativeSemFlowPolicy(native).cuda()
    trainable = enable_shared_adapter_training(policy, policy=native)
    report["gradient_parameters"] = {n: p.numel() for n, p in trainable}
    report["observation_contract"] = dict(image_size=list(cfg.data.target_image_size),
        prompt_truncation=False, intrinsics="absent, not fabricated", camera_views=["workspace", "wrist"],
        history_seconds=[-1., -.8, -.6, -.4, -.2, 0.], rtc_enabled=bool(cfg.policy.rtc_config.enabled),
        native_normalizer_used=False, normalization="pinned source train-only Panda statistics",
        caveat="Panda interface audit, not native RealMan deployment; camera/history/state distributions differ.")
    processor = PandaBatchProcessor(native, cfg)
    source_lock = dict(index["source_lock"], files=[f for f in index["source_lock"]["files"] if f["task"] == TASK])
    if len(source_lock["files"]) != 1:
        raise ValueError("exactly one source bowl file is required")
    native_probe.write_report(report, args.output)
    memory = MemorySources(source_lock)
    try:
        source.open_source = memory.handles.__getitem__
        window = select_window(source)
        report["window"] = dict(window["provenance"], task=TASK, sha256=window_digest(window))
        report["audit"] = audit(policy, processor, window, [p for _, p in trainable])
    finally:
        memory.close()
    verify_files(paths)
    report.update(status="passed", memory=native_probe.memory(), checkpoint_files_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # No user-selectable host cache or root-disk fallback. This command is cloud-only.
    if not Path("/content").is_dir() or not all(Path(p).resolve().is_relative_to("/content")
            for p in (args.upstream_root, args.index, args.output)):
        parser.error("RealMan runner accepts only reviewed ephemeral /content paths")
    if args.output.exists():
        parser.error("report already exists; preserve the previous attempt")
    args.cache_dir, args.teacher_path = Path("/content/semflow-model-cache/hub"), None
    if os.environ.get("HF_HOME") != "/content/semflow-model-cache":
        parser.error("HF_HOME must be the explicit ephemeral runtime cache")
    report = dict(schema_version=1, status="running", stage="realman-audit", artifacts=ARTIFACTS,
        upstream_revision=native_probe.UPSTREAM_REVISION, source_index_sha256=INDEX_SHA256,
        weights_sha256=WEIGHTS_SHA256, config_sha256=CONFIG_SHA256, seed=SEED,
        optimizer_steps=0, rollouts=0, interpretation="Compatibility only; fresh interfaces; no source-success claim.")
    started = time.monotonic()

    def deadline(_signum, _frame):
        raise TimeoutError("RealMan compatibility exceeded 850 seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(850)
    native_probe.write_report(report, args.output)
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error).split("\n")[0][:600])
    finally:
        signal.alarm(0)
        report["seconds"] = time.monotonic() - started
        native_probe.write_report(report, args.output)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
