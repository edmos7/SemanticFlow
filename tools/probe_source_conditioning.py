"""Frozen two-window source conditioning crossover; no training or simulator."""

from dataclasses import replace
import hashlib
import pickle
import random
import time

import numpy as np

from probe_source_actions import SEEDS, metrics, optimizer_contract, window_digest
from semflow.source_batching import source_window
from train_balanced_source import BalancedSchedule, declaration


TASK = "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet"
CHECKPOINT_SHA256 = "0836b6006d2acce44e980ca1c2a8885e73f2dcac1ada2d8a01c0d764280e35fa"
CHECKPOINT_BYTES = 49833988
PANEL_SHA256 = "b845e0794541ab4fafe9ec29958a5d502e4e73d84040cd1cc7a9f7796f0674c7"
WINDOWS = (
    ("08da8f8d95ec52255fff2183055946d1e6e9783b9433904cbf18aba2592d9ddf", 20,
     "5957fde1d2298e298955c8410111dd8d886dcb126a5bc350e9d7b784e20a9da0"),
    ("094a97a14c656723f49ca71d822624bd07c43d4f77da71a08b5f2ab6af2f904c", 51,
     "3db329bbc358969d3b9e984d58d4539e768cadee9de270fa97df97fb217a5b23"),
)


def selected_windows(source, panel):
    """Resolve the two existing train windows and verify their original targets."""
    windows, references = [], []
    for episode, anchor, digest in WINDOWS:
        matches = [r for r in source.index["manifest"]["episodes"]
                   if r["sha256"] == episode]
        if len(matches) != 1 or matches[0]["task"] != TASK or matches[0]["split"] != "train":
            raise ValueError("conditioning window must be the pinned source train episode")
        window = source_window(source.episode(matches[0]), anchor)
        expected = {"episode_sha256": episode, "anchor": anchor, "split": "train"}
        if window["provenance"] != expected or window_digest(window) != digest:
            raise ValueError("conditioning source window differs from the frozen panel")
        rows = [r for r in panel["panel"] if r["episode_sha256"] == episode]
        if (len(rows) != 1 or rows[0]["task"] != TASK or rows[0]["window_sha256"] != digest
                or any(rows[0][k] != v for k, v in expected.items())):
            raise ValueError("conditioning reference panel provenance differs")
        row = rows[0]
        if (not np.array_equal(window["loss_targets"]["actions"], row["normalized_target"])
                or [s["seed"] for s in row["samples"]] != list(SEEDS)
                or any(s["integration_steps"] != 10 for s in row["samples"])):
            raise ValueError("conditioning panel target or paired seeds differ")
        windows.append(window)
        references.append(row)
    return windows, references


def crossed_observation(prefix, scene):
    """Keep prefix batch/cache/positions together; swap both decoder scene inputs.

    Raw forecasts and task tokens are retained with the prefix but are not used
    as decoder conditioning: sample_actions reads raw_forecasts only for shape.
    The donor scene tokens are predictions/current observations, never targets.
    """
    import torch
    if (prefix.current_scene.shape != scene.current_scene.shape
            or prefix.scene_forecasts.shape != scene.scene_forecasts.shape
            or not torch.equal(prefix.view_valid, scene.view_valid)):
        raise ValueError("conditioning donors have incompatible scene layouts")
    return replace(prefix, current_scene=scene.current_scene, scene_forecasts=scene.scene_forecasts)


def summarize(cells, targets):
    """Paired effects and empirical four-seed error decomposition, no seed selection."""
    predictions = np.asarray([[s["normalized_prediction"] for s in c["samples"]] for c in cells])
    if predictions.shape != (4, 4, 32, 7) or not np.isfinite(predictions).all():
        raise ValueError("conditioning summary requires four complete finite paired cells")
    if [(c["prefix_window"], c["scene_window"]) for c in cells] != [(0, 0), (0, 1), (1, 0), (1, 1)]:
        raise ValueError("conditioning cell order differs")
    if any([s["seed"] for s in c["samples"]] != list(SEEDS) for c in cells):
        raise ValueError("conditioning summary seed pairing differs")
    output = {}
    for label, count in (("full", 32), ("first_eight", 8)):
        x, y = predictions[:, :, :count], np.asarray(targets)[:, :count]
        means, variance = x.mean(1), x.var(1)
        errors = x[:, :, None] - y[None, None]
        mse = np.square(errors).mean((1, 3, 4))
        mean_error = np.square(means[:, None] - y[None]).mean((2, 3))
        if not np.allclose(mse, mean_error + variance.mean((1, 2))[:, None], rtol=1e-12, atol=1e-12):
            raise RuntimeError("seed variance decomposition failed")
        output[label] = {
            "cell_mse_against_each_target": mse.tolist(),
            "cell_mean_error_against_each_target": mean_error.tolist(),
            "cell_seed_variance": variance.mean((1, 2)).tolist(),
            "cell_per_dimension_mse_against_each_target": np.square(errors).mean((1, 3)).tolist(),
            "cell_per_dimension_mean_error_against_each_target": np.square(means[:, None] - y[None]).mean(2).tolist(),
            "cell_per_dimension_seed_variance": variance.mean(1).tolist(),
            "cell_signed_bias_against_each_target": (means[:, None] - y[None]).mean(2).tolist(),
            "train_mean_baseline_mse": np.square(y).mean((1, 2)).tolist(),
            # These are squared paired action changes, not explained-variance fractions.
            "prefix_swap_action_mse_by_scene": [float(np.square(x[2] - x[0]).mean()), float(np.square(x[3] - x[1]).mean())],
            "scene_swap_action_mse_by_prefix": [float(np.square(x[1] - x[0]).mean()), float(np.square(x[3] - x[2]).mean())],
            "interaction_action_mse": float(np.square(x[3] - x[2] - x[1] + x[0]).mean()),
            "diagonal_action_mse": float(np.square(x[3] - x[0]).mean()),
            "target_distance_mse": float(np.square(y[1] - y[0]).mean()),
        }
    return output


def run_crossover(policy, processor, windows, statistics, report, persist, *, device="cuda"):
    import torch
    from train_source_calibration import tensor_digest

    if (len(windows) != 2 or any(w["provenance"]["split"] != "train" for w in windows)
            or windows[0]["instruction"] != windows[1]["instruction"]):
        raise ValueError("conditioning crossover requires two same-task training windows")
    if any(p.requires_grad for p in policy.parameters()) or policy.training:
        raise ValueError("conditioning diagnostic requires a frozen eval policy")
    before = tensor_digest(policy.state_dict())
    fingerprints = [window_digest(w) for w in windows]
    torch_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == "cuda" else []
    host_rng = pickle.dumps((random.getstate(), np.random.get_state()))
    targets = [w["loss_targets"]["actions"] for w in windows]
    started = time.monotonic()
    report.update(cells=[], optimizer_steps=0, rollouts=0, flow_steps=10, noise_seeds=list(SEEDS))
    synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        # Strip loss targets before processing/encoding, including future RGB.
        observed = [{k: v for k, v in w.items() if k != "loss_targets"} for w in windows]
        encoded = [policy.encode_observations(**processor([w], device)) for w in observed]
        for prefix_id in range(2):
            for scene_id in range(2):
                observation = crossed_observation(encoded[prefix_id], encoded[scene_id])
                cell = {"prefix_window": prefix_id, "scene_window": scene_id, "samples": []}
                report["cells"].append(cell)
                for seed in SEEDS:
                    if time.monotonic() - started > 300:
                        raise TimeoutError("conditioning crossover exceeded 300 seconds")
                    noise = torch.randn((1, 32, 7), device=device,
                        generator=torch.Generator(device=device).manual_seed(seed))
                    noise_sha = tensor_digest({"noise": noise})
                    synchronize()
                    begin = time.monotonic()
                    prediction = policy.sample_actions(observation, noise, num_steps=10)
                    synchronize()
                    seconds = time.monotonic() - begin
                    if tensor_digest({"noise": noise}) != noise_sha:
                        raise RuntimeError("conditioning sampler mutated paired noise")
                    prediction = prediction[0].float().cpu().numpy()
                    cell["samples"].append({"seed": seed, "initial_noise_sha256": noise_sha,
                        "seconds": seconds, "normalized_prediction": prediction.tolist(),
                        "against_targets": [metrics(prediction, target, statistics) for target in targets]})
                    persist()
                print(f"Completed conditioning cell prefix={prefix_id} scene={scene_id}", flush=True)
    if ([window_digest(w) for w in windows] != fingerprints or tensor_digest(policy.state_dict()) != before
            or not torch.equal(torch_rng, torch.get_rng_state())
            or host_rng != pickle.dumps((random.getstate(), np.random.get_state()))
            or any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all() if device == "cuda" else []))):
        raise RuntimeError("conditioning diagnostic mutated weights, buffers, windows or global RNG")
    report.update(summary=summarize(report["cells"], targets), seconds_sampling_and_encoding=time.monotonic() - started,
                  parameters_buffers_windows_rng_unchanged=True, sampled_chunks=16,
                  interpretation="Frozen source-only sensitivity diagnostic; crossed inputs can be off-distribution; no controller gate.")
    persist()


def run_probe(args, report, policy, reconstruction, optimizer, source, processor, config, contract):
    import json
    import torch
    import probe_egosteer as native_probe
    from semflow.models.source_calibration import restore_checkpoint

    if (args.checkpoint is None or args.checkpoint_sha256 != CHECKPOINT_SHA256
            or args.checkpoint.stat().st_size != CHECKPOINT_BYTES
            or hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256):
        raise ValueError("conditioning requires the immutable balanced step2400 checkpoint")
    if args.reference_panel is None or hashlib.sha256(args.reference_panel.read_bytes()).hexdigest() != PANEL_SHA256:
        raise ValueError("conditioning requires the immutable final balanced panel")
    panel = json.loads(args.reference_panel.read_text())
    windows, references = selected_windows(source, panel)
    schedule = BalancedSchedule(source)
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if saved["step"] != 2400:
        raise ValueError("conditioning checkpoint is not step2400")
    restore_checkpoint(saved, policy, reconstruction, optimizer, schedule, contract=contract)
    scope = optimizer_contract(policy, reconstruction, optimizer, config, 2400)
    if saved.get("balanced_exposure") != declaration(source, scope) or schedule.updates != 1800:
        raise ValueError("conditioning checkpoint exposure provenance differs")
    del saved
    policy.requires_grad_(False).eval()
    reconstruction.requires_grad_(False).eval()
    report.update(checkpoint_sha256=CHECKPOINT_SHA256, checkpoint_step=2400, reference_panel_sha256=PANEL_SHA256,
        windows=[{"task": TASK, **w["provenance"], "window_sha256": window_digest(w),
                  "normalized_target": w["loss_targets"]["actions"].tolist()} for w in windows])
    persist = lambda: native_probe.write_report(report, args.output)
    run_crossover(policy, processor, windows, source.index["normalization"], report, persist)
    comparisons = []
    for window_id, cell_id in ((0, 0), (1, 3)):
        for current, original in zip(report["cells"][cell_id]["samples"], references[window_id]["samples"]):
            a, b = np.asarray(current["normalized_prediction"]), np.asarray(original["normalized_prediction"])
            comparisons.append({"window": window_id, "seed": current["seed"],
                "max_abs_difference": float(np.abs(a - b).max()),
                "passed": bool(current["initial_noise_sha256"] == original["initial_noise_sha256"]
                               and np.allclose(a, b, rtol=0, atol=1e-5))})
    report["diagonal_reproduction"] = comparisons
    persist()
    if not all(c["passed"] for c in comparisons):
        raise RuntimeError("conditioning diagonal differs from the saved panel; do not interpret crossover")
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("conditioning diagnostic changed its input checkpoint")
    report["status"] = "passed"
    persist()
