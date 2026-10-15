#!/usr/bin/env python3
"""Bounded synthetic CUDA/BF16 composition and checkpoint-recovery check.

No pretrained model or real data is loaded. This checks research-interface
plumbing and recovery, not EgoSteer feasibility or manipulation performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_probe(output: Path, seed: int, report: dict) -> None:
    # Set before importing torch/CUDA, to support deterministic CUDA matmuls.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch import nn
    from torch.nn.attention import SDPBackend, sdpa_kernel

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from semflow.models import (
        CausalForecastReadout,
        ForecastActionConditioner,
        SceneResampler,
        SourceFeatureStatistics,
        raw_future_feature_loss,
        semantic_flow_loss,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this probe has no CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native CUDA BF16 support is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats(device)
    report.update({
        "torch_version": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "autocast_dtype": "bfloat16",
        "attention_backend": "math",
        "seed": seed,
        "dimensions": {"batch": 2, "history_steps": 3, "patches": 16,
                       "patch_dim": 64, "scene_queries": 16, "scene_dim": 256,
                       "action_tokens": 8, "action_dim": 7},
    })

    class ProbeModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Synthetic hidden adapter is deliberately not called a backbone.
            self.hidden_adapter = nn.Linear(64, 64)
            self.forecast = CausalForecastReadout(64, 64, num_tokens=16, dim=64)
            self.scene = SceneResampler(64, 32, dim=256, num_queries=16)
            self.conditioner = ForecastActionConditioner(32, 256, 256, dim=64)
            self.action_head = nn.Linear(32, 7)
            self.forecast.requires_grad_(False).eval()
            self.scene.requires_grad_(False).eval()

        def resample(self, patches, task):
            return self.scene(
                patches.float(), task.float(),
                patch_valid=torch.ones(patches.shape[:2], dtype=torch.bool, device=device),
                task_valid=torch.ones(task.shape[:2], dtype=torch.bool, device=device),
            )

        def forward(self, batch):
            hidden = self.hidden_adapter(batch["history"].mean(dim=2))
            raw_forecast = self.forecast(
                hidden,
                observation_times=batch["observation_times"],
                forecast_origin=batch["forecast_origin"],
                history_valid=batch["history_valid"],
            )
            task_future = batch["task"][:, None].expand(-1, 3, -1, -1).flatten(0, 1)
            predicted_scene = self.resample(raw_forecast.flatten(0, 1), task_future).reshape(2, 3, 16, 256)
            current = self.resample(batch["history"][:, -1], batch["task"])
            actions = self.condition_actions(batch["action_tokens"], current, predicted_scene, batch["future_valid"])
            return raw_forecast, predicted_scene, current, actions

        def condition_actions(self, action_tokens, current, futures, future_valid):
            conditioned = self.conditioner(
                action_tokens.float(), current.float(), futures.float(),
                current_valid=torch.ones(current.shape[:2], dtype=torch.bool, device=device),
                future_valid=future_valid,
            )
            return self.action_head(conditioned)

    def make_optimizer(model):
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, foreach=False)

    def make_batch():
        # All three RNG streams affect the batch and must be restored.
        scale = 0.75 + random.random() * 0.25 + torch.rand(()).item() * 0.25
        return {
            "history": torch.randn(2, 3, 16, 64, device=device) * scale,
            "task": torch.randn(2, 4, 32, device=device),
            "future_patches": torch.randn(2, 3, 16, 64, device=device),
            "action_tokens": torch.randn(2, 8, 32, device=device),
            "target_actions": torch.randn(2, 8, 7, device=device),
            "observation_times": torch.tensor([[-1., -0.5, 0.], [-1., -0.5, 0.]], device=device),
            "forecast_origin": torch.zeros(2, device=device),
            "history_valid": torch.ones(2, 3, device=device, dtype=torch.bool),
            "future_valid": torch.tensor([[True, True, True], [True, True, False]], device=device),
        }

    def one_step(model, optimizer, statistics):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        batch = make_batch()
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
            raw_forecast, predicted_scene, _, actions = model(batch)
            with torch.no_grad():
                target_task = batch["task"][:, None].expand(-1, 3, -1, -1).flatten(0, 1)
                target_scene = model.resample(batch["future_patches"].flatten(0, 1), target_task).reshape(2, 3, 16, 256)
            raw_loss = raw_future_feature_loss(raw_forecast, batch["future_patches"], batch["future_valid"])
            semantic = semantic_flow_loss(predicted_scene, target_scene, batch["future_valid"], statistics)
            action_loss = (actions.float() - batch["target_actions"]).square().mean()
            loss = raw_loss + semantic.total * 0.1 + action_loss
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all().item()):
                raise RuntimeError(f"Missing or nonfinite trainable gradient: {name}")
            if not parameter.requires_grad and parameter.grad is not None:
                raise RuntimeError(f"Frozen parameter received a gradient: {name}")
        adapter_grad = model.hidden_adapter.weight.grad.norm().item()
        if adapter_grad <= 0:
            raise RuntimeError("Frozen prediction-side modules blocked adapter gradients")
        optimizer.step()
        torch.cuda.synchronize(device)
        return batch, {
            "loss": loss.item(), "raw_feature_loss": raw_loss.item(),
            "semantic_state_loss": semantic.state.item(),
            "semantic_transition_loss": semantic.transition.item(),
            "action_loss": action_loss.item(), "adapter_gradient_norm": adapter_grad,
            "forecast_dtype": str(raw_forecast.dtype), "action_dtype": str(actions.dtype),
            "elapsed_seconds": time.perf_counter() - started,
        }

    model = ProbeModel().to(device)
    optimizer = make_optimizer(model)
    report["parameters"] = {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        source_scene = model.resample(torch.randn(16, 16, 64, device=device), torch.randn(16, 4, 32, device=device))
    statistics = SourceFeatureStatistics.fit(
        source_scene, torch.ones(16, dtype=torch.bool, device=device),
        source_manifest=f"synthetic-probe-only:source:seed={seed}",
    )
    _, first = one_step(model, optimizer, statistics)
    report["first_step"] = first
    checkpoint_path = output / "semantic_checkpoint.pt"
    checkpoint = {
        "schema_version": 1, "purpose": "synthetic-semantic-gpu-probe", "step": 1,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "source_statistics": {"mean": statistics.mean.cpu(), "scale": statistics.scale.cpu(),
                              "source_manifest": statistics.source_manifest},
        "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all()},
    }
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(checkpoint_path)
    report["checkpoint"] = {"file": checkpoint_path.name, "bytes": checkpoint_path.stat().st_size,
                            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
    # Continue first, then reconstruct objects exclusively from the file.
    uninterrupted_batch, uninterrupted = one_step(model, optimizer, statistics)
    resumed_model = ProbeModel().to(device)
    resumed_optimizer = make_optimizer(resumed_model)
    recovered = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resumed_model.load_state_dict(recovered["model"])
    resumed_optimizer.load_state_dict(recovered["optimizer"])
    resumed_statistics = SourceFeatureStatistics(**recovered["source_statistics"])
    random.setstate(recovered["rng"]["python"])
    torch.set_rng_state(recovered["rng"]["torch_cpu"])
    torch.cuda.set_rng_state_all(recovered["rng"]["torch_cuda"])
    resumed_batch, resumed = one_step(resumed_model, resumed_optimizer, resumed_statistics)
    batch_equal = all(torch.equal(value, resumed_batch[name]) for name, value in uninterrupted_batch.items())
    model_equal = all(torch.equal(value, resumed_model.state_dict()[name]) for name, value in model.state_dict().items())

    def nested_equal(left, right):
        if isinstance(left, torch.Tensor):
            return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
        if isinstance(left, dict):
            return isinstance(right, dict) and left.keys() == right.keys() and all(nested_equal(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
        return left == right

    optimizer_equal = nested_equal(optimizer.state_dict(), resumed_optimizer.state_dict())
    losses_equal = uninterrupted["loss"] == resumed["loss"]
    report["resume"] = {
        "mode": "fresh_objects_from_disk_in_same_cuda_process",
        "next_batch_bitwise_equal": batch_equal, "next_loss_bitwise_equal": losses_equal,
        "next_model_bitwise_equal": model_equal, "next_optimizer_bitwise_equal": optimizer_equal,
        "uninterrupted_step": uninterrupted, "resumed_step": resumed,
    }
    if not all((batch_equal, losses_equal, model_equal, optimizer_equal)):
        raise RuntimeError("Checkpoint restoration failed exact next-step equivalence")
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", dtype=torch.bfloat16):
        _, predicted_scene, current, actions = resumed_model(resumed_batch)
        # Intervene on actual model forecasts while keeping observations fixed.
        shuffled = predicted_scene[:, [1, 0, 2]]
        static = current[:, None].expand(-1, 3, -1, -1)
        shuffled_actions = resumed_model.condition_actions(resumed_batch["action_tokens"], current, shuffled, resumed_batch["future_valid"])
        static_actions = resumed_model.condition_actions(resumed_batch["action_tokens"], current, static, resumed_batch["future_valid"])
    shuffled_change = (actions.float() - shuffled_actions.float()).abs().max().item()
    static_change = (actions.float() - static_actions.float()).abs().max().item()
    report["forecast_interventions"] = {
        "shuffle_first_two_valid_horizons_max_action_difference": shuffled_change,
        "replace_with_current_scene_max_action_difference": static_change,
    }
    if shuffled_change <= 0 or static_change <= 0:
        raise RuntimeError("A forecast intervention did not change predicted actions")
    torch.cuda.synchronize(device)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    report["status"] = "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "semantic_gpu.json"
    report = {
        "schema_version": 1, "stage": "synthetic_semantic_gpu", "status": "running",
        "limitations": ["Synthetic features and actions only; no pretrained model or real data.",
                        "Frozen randomly initialized readouts are plumbing fixtures, not trained teachers.",
                        "Same-process recovery of fresh objects, not a restarted cloud runtime.",
                        "No claim about EgoSteer resource requirements or robot task success."],
    }
    started = time.perf_counter()

    def timed_out(_signum, _frame):
        raise TimeoutError("Semantic GPU probe exceeded its 55-second execution budget")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(55)
    try:
        run_probe(args.output_dir, args.seed, report)
    except Exception as error:
        report.update({"status": "failed", "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc()})
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(report_path, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
