#!/usr/bin/env python3
"""Measure an existing cloud runtime without downloading weights or datasets."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def main() -> None:
    report = {
        "schema_version": 1,
        "stage": "environment",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_free_gib": round(shutil.disk_usage('/content').free / 2**30, 2),
        "packages": {},
    }
    for package in ('torch', 'torchvision', 'transformers', 'huggingface-hub', 'numpy', 'mujoco', 'robosuite'):
        try:
            report['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report['packages'][package] = None
    report['memory_info'] = Path('/proc/meminfo').read_text().splitlines()[:3]
    gpu = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
        text=True, capture_output=True, check=True,
    )
    report['gpu'] = gpu.stdout.strip()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no GPU feasibility claim is possible')
    report['torch_cuda'] = torch.version.cuda
    report['bf16_supported'] = torch.cuda.is_bf16_supported()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    x = torch.randn(32, 256, device='cuda', requires_grad=True)
    layer = torch.nn.Linear(256, 256, device='cuda')
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = layer(x).square().mean()
    loss.backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError('GPU backward pass produced non-finite gradients')
    optimizer.step()
    torch.cuda.synchronize()
    report['synthetic_gpu_optimizer_step'] = {
        'elapsed_seconds': round(time.perf_counter() - started, 4),
        'loss': float(loss.detach()),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
    }
    report['limitations'] = ['Synthetic runtime check only; not an EgoSteer or LIBERO result.']
    output = Path('/content/semflow-results/environment.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
