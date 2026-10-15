#!/usr/bin/env python3
"""Check pinned native imports and a small CUDA kernel; never load model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

from probe_egosteer import UPSTREAM_REVISION, prepare_native_imports, write_report


EXPECTED_PACKAGES = {
    "torch": "2.10.0+cu130", "torchvision": "0.25.0+cu130",
    "transformers": "5.6.2", "flash-attn": "2.8.3",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 120:
        parser.error("timeout must be between 1 and 120 seconds")
    if args.output.exists():
        parser.error("output must be a fresh report path")
    report = {
        "schema_version": 1, "status": "running", "checks": [],
        "upstream_revision": UPSTREAM_REVISION, "model_loaded": False,
        "weights_downloaded": False,
        "scope": "dependency imports and synthetic attention only; no model construction or access requests",
    }
    started = time.monotonic()

    def save() -> None:
        write_report(report, args.output)

    def timed_out(_signum, _frame) -> None:
        raise TimeoutError("native dependency probe deadline reached")

    def check(name: str, operation) -> None:
        record = {"check": name, "status": "running"}
        report["checks"].append(record)
        save()
        beginning = time.monotonic()
        try:
            operation()
            record["status"] = "passed"
        except BaseException:
            record["status"] = "failed"
            raise
        finally:
            record["seconds"] = time.monotonic() - beginning
            save()

    def environment() -> None:
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("the pinned wheels require Linux x86_64")
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError("the pinned FlashAttention wheel requires Python 3.12")
        libc, version = platform.libc_ver()
        if libc != "glibc" or tuple(map(int, version.split(".")[:2])) < (2, 28):
            raise RuntimeError("the pinned Torch wheel requires glibc >= 2.28")
        report["python"] = platform.python_version()
        report["glibc"] = version
        packages = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
        report["packages"] = packages
        for name, expected in EXPECTED_PACKAGES.items():
            actual = packages[name].split("+")[0] if name == "flash-attn" else packages[name]
            if actual != expected:
                raise RuntimeError(f"{name} version differs: expected {expected}, got {packages[name]}")
        import torch
        import torchvision
        report["torchvision_imported"] = torchvision.__version__
        report["cxx11_abi"] = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
        if not report["cxx11_abi"] or torch.version.cuda != "13.0":
            raise RuntimeError("FlashAttention wheel requires Torch CUDA 13.0 and CXX11 ABI TRUE")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("a CUDA GPU with native BF16 support is required")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        report["torch_cuda"] = torch.version.cuda

    def native_imports() -> None:
        prepare_native_imports(args)
        # Resolve the lazy Transformers exports without constructing anything.
        from transformers import AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration
        report["native_exports"] = [item.__name__ for item in (AutoModel, AutoProcessor, Qwen3VLForConditionalGeneration)]

    def flash_attention() -> None:
        import torch
        from flash_attn import flash_attn_varlen_func
        from torch.nn.attention import SDPBackend, sdpa_kernel

        torch.manual_seed(17)
        torch.cuda.manual_seed_all(17)
        tensors = [torch.randn(96, 4, 64, device="cuda", dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3)]
        offsets = torch.tensor([0, 32, 96], device="cuda", dtype=torch.int32)
        output = flash_attn_varlen_func(
            *tensors, cu_seqlens_q=offsets, cu_seqlens_k=offsets,
            max_seqlen_q=64, max_seqlen_k=64, dropout_p=0.0, causal=False,
        )
        if output.shape != tensors[0].shape or not torch.isfinite(output).all():
            raise RuntimeError("FlashAttention produced invalid output")
        # Compare actual variable-length outputs against independent math SDPA.
        expected = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            for start, end in ((0, 32), (32, 96)):
                q, k, v = [value[start:end].transpose(0, 1).unsqueeze(0).float() for value in tensors]
                expected.append(torch.nn.functional.scaled_dot_product_attention(q, k, v).squeeze(0).transpose(0, 1))
        expected = torch.cat(expected)
        torch.testing.assert_close(output.float(), expected, atol=0.03, rtol=0.03)
        loss = output.float().square().mean()
        loss.backward()
        norms = []
        for value in tensors:
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise RuntimeError("FlashAttention backward returned missing or nonfinite gradients")
            norm = value.grad.float().norm().item()
            if norm <= 0:
                raise RuntimeError("FlashAttention backward returned zero gradients")
            norms.append(norm)
        torch.cuda.synchronize()
        report["attention"] = {
            "dtype": "bfloat16", "shape": list(output.shape), "sequence_lengths": [32, 64],
            "max_reference_error": (output.float() - expected).abs().max().item(),
            "qkv_gradient_norms": norms, "loss": loss.item(),
        }

    try:
        save()
        if sys.platform != "linux":
            raise RuntimeError("native dependency probe requires Linux")
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        check("environment", environment)
        check("native_imports", native_imports)
        check("flash_attention_bf16", flash_attention)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error_type"] = type(error).__name__
        report["error"] = str(error).split("\n")[0][:600]
    finally:
        if sys.platform == "linux":
            signal.setitimer(signal.ITIMER_REAL, 0)
        report["seconds"] = time.monotonic() - started
        save()
    print(json.dumps({"status": report["status"], "report": str(args.output)}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
