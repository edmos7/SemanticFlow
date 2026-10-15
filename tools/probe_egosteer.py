#!/usr/bin/env python3
"""Bounded native EgoSteer feasibility checks; run on the authenticated GPU host.

No dataset downloads, fake teacher weights, robot claims, or full-model optimizer.
The backward stage audits one existing shared parameter, not SemFlow or LoRA.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

UPSTREAM_REVISION = "5e646d9dd278280af14e5050993862bead94cc19"
ARTIFACTS = {
    "egosteer": ("EgoSteer/EgoSteer-3B-Base", "3e299fb4e2abc17244168be36d95a2920a9c9ad3"),
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203"),
    "teacher": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "ea8dc2863c51be0a264bab82070e3e8836b02d51"),
}
SHARED_PARAMETER = "backbone.language_model.layers.0.input_layernorm.weight"
PACKAGES = ("torch", "torchvision", "transformers", "hydra-core", "omegaconf",
            "huggingface-hub", "safetensors", "flash-attn", "numpy", "Pillow")


def write_report(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def environment() -> dict:
    result = {"python": platform.python_version(), "packages": {}, "cuda_available": False}
    for package in PACKAGES:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    if importlib.util.find_spec("torch") is not None:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu"] = torch.cuda.get_device_name(0)
            result["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            result["bf16_supported"] = torch.cuda.is_bf16_supported()
    return result


def access_preflight(report: dict, args: argparse.Namespace, *, artifacts=None) -> None:
    """HEAD requests only; teacher access is checked before large downloads."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    artifacts = ARTIFACTS if artifacts is None else artifacts
    report["access"] = {}
    for name in ("teacher", "qwen", "egosteer"):
        report["operation"] = f"access:{name}"
        write_report(report, args.output)
        repo, revision = artifacts[name]
        if name == "teacher" and args.teacher_path:
            path = args.teacher_path.resolve()
            if not (path / "config.json").is_file():
                raise FileNotFoundError("--teacher-path must contain config.json and real DINOv3 weights")
            if not (list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))):
                raise FileNotFoundError("No teacher weights found under --teacher-path")
            report["access"][name] = {"source": "provided_local_teacher", "revision": "user_supplied"}
            continue
        filename = "model_bf16.pt" if name == "egosteer" else "model.safetensors"
        # Both pinned dependent models currently use a single safetensors file.
        metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
        report["access"][name] = {
            "repository": repo, "revision": revision, "filename": filename,
            "bytes": metadata.size, "etag": metadata.etag, "accessible": True,
        }
        write_report(report, args.output)


def download_models(report: dict, args: argparse.Namespace, *, artifacts=None) -> dict:
    from huggingface_hub import hf_hub_download, snapshot_download

    artifacts = ARTIFACTS if artifacts is None else artifacts
    paths = {}
    repo, revision = artifacts["egosteer"]
    for filename in ("config.yaml", "model_bf16.pt"):
        paths[filename] = hf_hub_download(repo, filename, revision=revision, cache_dir=args.cache_dir)
    # The native Qwen wrapper initializes with from_pretrained before loading the
    # complete EgoSteer state. Its initial Qwen weights are therefore required.
    for name in ("qwen", "teacher"):
        if name == "teacher" and args.teacher_path:
            paths[name] = str(args.teacher_path.resolve())
            continue
        repo, revision = artifacts[name]
        paths[name] = snapshot_download(
            repo, revision=revision, cache_dir=args.cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        )
    report["weights_downloaded"] = True
    return paths


def memory() -> dict:
    import torch
    result = {"process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    if torch.cuda.is_available():
        result.update(
            cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    return result


def prepare_native_imports(args: argparse.Namespace) -> None:
    """Reject a wrong checkout or incompatible dependencies before downloading."""
    if args.upstream_root is None:
        raise ValueError("--upstream-root is required for model stages")
    upstream = args.upstream_root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise RuntimeError(f"Upstream revision differs: expected {UPSTREAM_REVISION}, got {revision}")
    if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=upstream, text=True).strip():
        raise RuntimeError("Upstream tracked files have modifications; use a clean pinned checkout")
    sys.path.insert(0, str(upstream))
    # Native source unconditionally imports and applies its FlashAttention vision
    # patch. Setting a YAML SDPA flag does not remove this requirement.
    for module in (
        "hydra", "omegaconf", "src.utils.checkpoint_util", "src.policy.egosteer",
        "src.model.vlm.qwen3_vl_backbone", "src.model.vlm.qwen3_expert",
        "src.model.world_model.frozen_teacher", "src.dataset.qwen3_vl_batching",
    ):
        importlib.import_module(module)


def load_model(report: dict, args: argparse.Namespace, paths: dict, *, synchronous=False):
    import hydra
    from omegaconf import OmegaConf
    import torch
    from src.utils.checkpoint_util import load_state_dict_checked

    cfg = OmegaConf.load(paths["config.yaml"])
    cfg.pretrained.model_name_or_path = paths["qwen"]
    cfg.world_model.frozen_teacher.model_name_or_path = paths["teacher"]
    cfg.pretrained.trust_remote_code = False
    cfg.pretrained.text_attn_implementation = "sdpa"
    # Retain every native module and tensor dimension, so checkpoint keys can be
    # verified; turn off privileged future conditioning in the RGB-only path.
    cfg.world_model.action_conditioning = False
    cfg.world_model.motion_conditioning = False
    cfg.world_model.mask_loss_by_view_mask = True
    if synchronous:
        cfg.policy.rtc_config.enabled = False
    cfg.data.target_pixels = int(cfg.data.target_image_size[0]) * int(cfg.data.target_image_size[1])
    policy_config = OmegaConf.to_container(cfg.policy, resolve=True)
    report["configuration"] = {
        "native_state_dim": int(cfg.dims.state_dim), "native_action_dim": int(cfg.dims.action_dim),
        "action_horizon": int(cfg.data.shape_meta.action.horizon),
        "rgb_history_frames": int(cfg.data.shape_meta.obs.rgb.horizon),
        "future_frames": int(cfg.world_model.future_frame.horizon),
        "teacher_resolution": list(cfg.world_model.teacher_image_size),
        "text_attention": "sdpa", "vision_attention": "native_flash_attention_2_patch",
        "future_action_conditioning": False, "future_camera_motion_conditioning": False,
        "all_native_module_dimensions_preserved": True,
    }
    started = time.perf_counter()
    model = hydra.utils.instantiate(OmegaConf.create(policy_config))
    # mmap avoids materializing another 5.6 GB CPU copy. weights_only rejects
    # arbitrary pickle objects; no fallback to unsafe or partial loading.
    state = torch.load(paths["model_bf16.pt"], map_location="cpu", weights_only=True, mmap=True)
    for wrapper in ("model", "module", "model_state_dict"):
        if wrapper in state:
            state = state[wrapper]
            break
    groups = ("backbone.", "flow_expert.", "world_model_expert.", "wm_head.",
              "wm_action_encoder.", "wm_motion_encoder.")
    group_counts = {prefix: sum(key.startswith(prefix) for key in state) for prefix in groups}
    if not all(group_counts.values()):
        raise RuntimeError(f"Checkpoint is missing a required native parameter group: {group_counts}")
    load_state_dict_checked(model, state, allow_missing_prefixes=("frozen_teacher.",))
    del state
    gc.collect()
    model.to(device="cuda", dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    torch.cuda.synchronize()
    report["checkpoint_key_groups"] = group_counts
    report["model_loaded"] = True
    report["load_seconds"] = time.perf_counter() - started
    report["parameter_count"] = sum(p.numel() for p in model.parameters())
    report["parameter_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    report["memory_after_load"] = memory()
    return model, cfg


def synthetic_rgb_batch(model, cfg):
    """A deterministic moving-square video; no human or robot labels invented."""
    import torch
    from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor, Qwen3VLChatFormatter

    count = int(cfg.data.shape_meta.obs.rgb.horizon)
    height, width = map(int, cfg.data.target_image_size)
    clip = torch.full((count + 1, height, width, 3), 64, dtype=torch.uint8)
    for index, frame in enumerate(clip):
        x = 30 + index * 16
        frame[100:160, x:x + 60, 0] = 230
        frame[100:160, x:x + 60, 1:] = 30
    sample = {
        "vision_type": "video", "images": clip[:-1], "active_views": ["head"],
        "video_fps": torch.tensor(float(cfg.data.video_base_fps) / int(cfg.data.shape_meta.obs.rgb.stride)),
        "is_vla_data": torch.tensor(False), "question": "Predict how this scene will change.", "answer": "",
    }
    processor = Qwen3VLBatchProcessor(
        model_name_or_path=str(cfg.pretrained.model_name_or_path), processor=model.backbone.processor,
        processor_call_kwargs={"min_pixels": height * width, "max_pixels": height * width,
                               "text_kwargs": {"padding": "longest", "truncation": False, "return_tensors": "pt"}},
    )
    formatter = Qwen3VLChatFormatter()
    messages = formatter.build_messages(sample, prompt_only=True)
    batch = processor.encode_messages([messages], [sample], add_generation_prompt=True)
    batch["answer_start_idx"] = processor.find_answer_start_idx(batch["input_ids"])
    if not torch.equal(batch["answer_start_idx"], batch["attention_mask"].sum(-1)):
        raise RuntimeError("Synthetic probe must contain only a complete causal prompt")
    future = clip[-1:].unsqueeze(0)
    batch.update(future_frames=future, chest_future_frames=torch.zeros_like(future),
                 n_future_frames=torch.tensor([1]), view_mask=torch.tensor([[True, False]]))
    return {key: value.to("cuda") if torch.is_tensor(value) else value for key, value in batch.items()}


def run_streams(model, cfg, report: dict, args: argparse.Namespace) -> None:
    import torch
    from src.policy.egosteer_loss import compute_wm_loss

    if int(cfg.world_model.future_frame.horizon) != 1:
        raise RuntimeError("Probe expects the pinned checkpoint's single future frame")
    is_backward = args.stage == "backward"
    selected = dict(model.named_parameters())[SHARED_PARAMETER]
    if is_backward:
        selected.requires_grad_(True)
    report["trainable_parameters"] = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    batch = synthetic_rgb_batch(model, cfg)
    report["input"] = {
        "kind": "synthetic_rgb_video", "token_count": int(batch["input_ids"].shape[-1]),
        "absent_modalities": ["states", "actions", "future_head_motion", "future_chest_motion"],
        "inactive_chest_is_masked_padding": True,
    }
    captured = {}
    # Capture actual native stream outputs without duplicating teacher inference.
    handle = model.wm_head.output_proj.register_forward_hook(
        lambda _module, _inputs, output: captured.update(projection_shape=list(output.shape)))
    started = time.perf_counter()
    try:
        with torch.set_grad_enabled(is_backward), torch.autocast("cuda", dtype=torch.bfloat16):
            backbone = model.forward_backbone_stream(batch, model.build_slot_embeddings(batch))
            loss = compute_wm_loss(model, batch, backbone)
        if not torch.isfinite(loss):
            raise RuntimeError("Native world-model loss is not finite")
        if is_backward:
            loss.backward()
            gradient = selected.grad
            if gradient is None or not torch.isfinite(gradient).all() or gradient.float().norm() == 0:
                raise RuntimeError("World-model loss did not produce a finite nonzero shared-backbone gradient")
            report["shared_gradient_norm"] = float(gradient.float().norm())
        torch.cuda.synchronize()
        report["world_stream_seconds"] = time.perf_counter() - started
        report["world_loss"] = float(loss.detach())
        report["world_projection_shape"] = captured["projection_shape"]
        report["backbone_hidden_shape"] = list(backbone.last_hidden_states.shape)
        report["teacher_frozen"] = all(not p.requires_grad and p.grad is None for p in model.frozen_teacher.parameters())
        if not report["teacher_frozen"]:
            raise RuntimeError("Frozen teacher unexpectedly received gradients")
        if not args.skip_action_forward:
            # This is the native action expert's velocity field, not Panda control
            # or a simulator rollout. Noise is a model input, not an action label.
            action_batch = dict(batch)
            shape = (1, model.action_horizon, model.action_dim)
            action_batch["actions_valid_mask"] = torch.ones(shape, device="cuda", dtype=torch.bool)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                action = model.forward_flow_stream(
                    action_batch, backbone,
                    {"noisy_actions": torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                     "time_for_model": torch.full(shape[:2], 0.5, device="cuda", dtype=torch.bfloat16)},
                    num_parallel_chunks=1,
                )["pred_v"]
            if not torch.isfinite(action).all():
                raise RuntimeError("Native action expert produced non-finite values")
            report["native_action_velocity_shape"] = list(action.shape)
        if is_backward:
            before = selected.detach().clone()
            optimizer = torch.optim.AdamW([selected], lr=0.01)
            optimizer.step()
            if torch.equal(selected, before):
                raise RuntimeError("Audit optimizer step made no representable parameter update")
            checkpoint = args.output.with_suffix(".audit.pt")
            torch.save({"parameter_name": SHARED_PARAMETER, "parameter": selected.detach().cpu(),
                        "optimizer": optimizer.state_dict(), "seed": args.seed,
                        "purpose": "single-parameter feasibility audit; not LoRA or trained SemFlow"}, checkpoint)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            restored = torch.nn.Parameter(saved["parameter"].to(selected.device))
            resumed_optimizer = torch.optim.AdamW([restored], lr=0.01)
            resumed_optimizer.load_state_dict(saved["optimizer"])
            if not torch.equal(restored, selected):
                raise RuntimeError("Audit checkpoint round trip changed its parameter")
            report["audit_optimizer_step"] = True
            report["audit_checkpoint_roundtrip"] = True
            report["runtime_restart_recovery_tested"] = False
        report["memory_after_streams"] = memory()
    finally:
        handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("environment", "access", "model-load", "forward", "backward"), default="model-load")
    parser.add_argument("--upstream-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--teacher-path", type=Path, help="Already authorized, complete local DINOv3 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-action-forward", action="store_true")
    args = parser.parse_args()
    report = {
        "schema_version": 1, "stage": args.stage, "status": "running",
        "upstream_revision": UPSTREAM_REVISION,
        "artifacts": {name: {"repository": repo, "revision": rev} for name, (repo, rev) in ARTIFACTS.items()},
        "model_loaded": False, "weights_downloaded": False, "seed": args.seed,
        "scope": "Native feasibility only; no task training, LoRA, SemFlow, or LIBERO rollout",
    }
    started = time.perf_counter()
    write_report(report, args.output)
    try:
        report["environment"] = environment()
        write_report(report, args.output)
        if args.stage != "environment":
            access_preflight(report, args)
        if args.stage not in ("environment", "access"):
            import torch
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise RuntimeError("Native probe requires a CUDA GPU with bfloat16 support")
            if importlib.util.find_spec("flash_attn") is None:
                raise RuntimeError("Pinned upstream requires flash-attn, including with SDPA text attention")
            report["operation"] = "native-imports"
            write_report(report, args.output)
            prepare_native_imports(args)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
            report["operation"] = "download"
            write_report(report, args.output)
            paths = download_models(report, args)
            report["operation"] = "model-load"
            write_report(report, args.output)
            model, cfg = load_model(report, args, paths)
            report["operation"] = args.stage
            write_report(report, args.output)
            if args.stage in ("forward", "backward"):
                run_streams(model, cfg, report, args)
        report["status"] = "passed"
        report["operation"] = "completed"
        return_code = 0
    except Exception as error:
        # Preserve the artifact even for access gates and OOM. Avoid response
        # bodies/URLs, which can contain signed links or credential details.
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        if type(error).__name__ == "GatedRepoError":
            report["error"] = "Required model is gated; grant this account DINOv3 access and authenticate on the runtime."
        elif isinstance(error, (RuntimeError, ValueError, FileNotFoundError, ImportError)):
            report["error"] = str(error).split("\n")[0][:600]
        else:
            report["error"] = "Probe failed; inspect the error type and last completed report stage."
        return_code = 1
    report["elapsed_seconds"] = time.perf_counter() - started
    write_report(report, args.output)
    print(json.dumps({"stage": args.stage, "status": report["status"], "report": str(args.output),
                      "error_type": report.get("error_type"), "model_loaded": report["model_loaded"]}))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
