#!/usr/bin/env python3
"""Bounded real-backbone SemFlow integration audit with synthetic RGB only.

New scene/Panda/horizon interfaces are untrained fixtures. This verifies native
execution, adapter gradients and forecast influence, not robot task performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

import probe_egosteer as native_probe


def run(args, report):
    import torch
    from semflow.models import (NativeSemFlowPolicy, SourceFeatureStatistics,
        enable_shared_adapter_training, inject_shared_lora,
        raw_future_feature_loss, semantic_flow_loss)
    from semflow.models.native_policy import OBSERVATION_KEYS

    report['environment'] = native_probe.environment()
    native_probe.access_preflight(report, args)
    native_probe.prepare_native_imports(args)
    paths = native_probe.download_models(report, args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    native, cfg = native_probe.load_model(report, args, paths)
    synthetic = native_probe.synthetic_rgb_batch(native, cfg)
    view_valid = synthetic['view_mask']
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        prefix = native.forward_backbone_stream(synthetic, native.build_slot_embeddings(synthetic))
        baseline = native.forward_world_model_stream(synthetic, prefix)['pred'][:, 0, 0].clone()
    del prefix
    observed = {key: value for key, value in synthetic.items() if key in OBSERVATION_KEYS}
    report['observed_batch_keys'] = sorted(observed)
    report['injected_projection_paths'] = list(inject_shared_lora(native))
    model = NativeSemFlowPolicy(native).to('cuda')
    trainable = enable_shared_adapter_training(model, policy=native)
    parameters = [parameter for _, parameter in trainable]
    report['adaptation_parameters'] = {name: parameter.numel() for name, parameter in trainable}
    report['adaptation_parameter_count'] = sum(parameter.numel() for parameter in parameters)
    report['rank'] = 8
    report['shared_projection_selection'] = ['q_proj', 'v_proj']
    report['new_interfaces_source_calibrated'] = False

    height, width = map(int, cfg.data.target_image_size)
    def frame(offset):
        image = torch.full((height, width, 3), 64, dtype=torch.uint8, device='cuda')
        x = int(30 + (5 + offset) * 16)
        image[100:160, x:x + 60, 0] = 230
        image[100:160, x:x + 60, 1:] = 30
        return image
    current_rgb = torch.stack((frame(0), torch.zeros_like(frame(0))))[None]
    times = torch.arange(-5, 1, device='cuda', dtype=torch.float32)[None]
    origin = torch.zeros(1, device='cuda')
    teacher_calls = []
    def check_current_teacher(_module, inputs):
        expected = current_rgb.flatten(0, 1).unsqueeze(1)
        if not torch.equal(inputs[0], expected):
            raise RuntimeError('Runtime teacher received something other than the current observation')
        teacher_calls.append(list(inputs[0].shape))
    def forbidden_world_stream(*_args, **_kwargs):
        raise RuntimeError('Integration called native training world stream with teacher targets')
    native.forward_world_model_stream = forbidden_world_stream
    handle = native.frozen_teacher.register_forward_pre_hook(check_current_teacher)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = model.encode_observations(observed, current_rgb, view_valid,
                observation_times=times, forecast_origin=origin)
    finally:
        handle.remove()
    if len(teacher_calls) != 1:
        raise RuntimeError('Runtime must extract only current-image DINO once')
    report['runtime_current_teacher_calls'] = teacher_calls
    actual = context.raw_forecasts[:, 0, 1]
    report['native_one_second_max_abs_difference'] = (actual.float() - baseline.float()).abs().max().item()
    if not torch.equal(actual, baseline):
        raise RuntimeError('The observed-view 1-second forecast differs from the native checkpoint path')
    report['raw_forecast_shape'] = list(context.raw_forecasts.shape)
    report['scene_forecast_shape'] = list(context.scene_forecasts.shape)
    if context.raw_forecasts[:, 1].count_nonzero().item():
        raise RuntimeError('Missing view was not masked')

    # Supervised RGB is separate from the observation API and enters losses only.
    target_rgb = torch.stack([frame(h) for h in (0.5, 1.0, 2.0)])[None, None]
    target_rgb = torch.cat((target_rgb, torch.zeros_like(target_rgb)), dim=1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        target_raw = native.frozen_teacher(target_rgb.flatten(0, 1)).reshape_as(context.raw_forecasts)
        target_scene = model.pool_raw(target_raw, view_valid, context.task_tokens.detach(), context.task_valid)
    statistics = SourceFeatureStatistics.fit(target_scene.flatten(0, 1),
        torch.ones(3, device='cuda', dtype=torch.bool), source_manifest='synthetic-native-integration-probe-only')
    feature_valid = view_valid[:, :, None].expand(-1, -1, 3).flatten(0, 1)
    future_valid = torch.ones((1, 3), device='cuda', dtype=torch.bool)
    noise = torch.randn(1, 32, 7, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        raw_loss = raw_future_feature_loss(context.raw_forecasts.flatten(0, 1), target_raw.flatten(0, 1), feature_valid)
        semantic = semantic_flow_loss(context.scene_forecasts, target_scene, future_valid, statistics)
        velocity = model.predict_velocity(context, noise, torch.full((1, 32), 0.5, device='cuda'))
        action_energy = velocity.float().square().mean()

    def adapter_gradient_norm(loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        if any(value is not None and not torch.isfinite(value).all() for value in gradients):
            raise RuntimeError('Nonfinite adapter gradient')
        norm = sum(value.float().square().sum().item() for value in gradients if value is not None) ** 0.5
        if norm <= 0:
            raise RuntimeError('Frozen prediction readouts blocked adapter gradients')
        return norm
    report['raw_loss_adapter_gradient_norm'] = adapter_gradient_norm(raw_loss)
    report['semantic_loss_adapter_gradient_norm'] = adapter_gradient_norm(semantic.total)
    forecast_gradient = torch.autograd.grad(action_energy, context.scene_forecasts, retain_graph=True)[0]
    report['action_to_forecast_gradient_norm'] = forecast_gradient.float().norm().item()
    if not torch.isfinite(forecast_gradient).all() or report['action_to_forecast_gradient_norm'] <= 0:
        raise RuntimeError('Action path does not differentiate through predicted forecasts')

    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    before = [parameter.detach().clone() for parameter in parameters]
    loss = raw_loss + 0.1 * semantic.total + 0.01 * action_energy
    loss.backward()
    if any(parameter.grad is not None for parameter in model.parameters() if not parameter.requires_grad):
        raise RuntimeError('A frozen parameter received a gradient')
    optimizer.step()
    changed = sum(not torch.equal(old, parameter) for old, parameter in zip(before, parameters))
    if not changed:
        raise RuntimeError('Optimizer did not change any shared adapter')
    report['updated_adapter_tensors'] = changed
    report['losses'] = {'raw': raw_loss.item(), 'semantic_state': semantic.state.item(),
                        'semantic_transition': semantic.transition.item(), 'action_energy': action_energy.item()}
    checkpoint_path = args.output.with_suffix('.adapters.pt')
    torch.save({'purpose': 'synthetic integration audit; not source-calibrated',
        'adapters': {name: p.detach().cpu() for name, p in trainable},
        'optimizer': optimizer.state_dict(), 'rank': 8, 'seed': args.seed}, checkpoint_path)
    restored = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if any(not torch.equal(restored['adapters'][name], p.detach().cpu()) for name, p in trainable):
        raise RuntimeError('Adapter checkpoint roundtrip differs')
    report['adapter_checkpoint'] = {'file': checkpoint_path.name, 'bytes': checkpoint_path.stat().st_size,
        'sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), 'parameter_roundtrip_equal': True}
    report['teacher_targets_detached'] = not target_raw.requires_grad and not target_scene.requires_grad

    del context, raw_loss, semantic, action_energy, loss, velocity, before, baseline, actual, forecast_gradient
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    # Runtime actions reuse only observation-derived forecasts and paired noise.
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        context = model.encode_observations(observed, current_rgb, view_valid,
            observation_times=times, forecast_origin=origin)
        original = model.sample_actions(context, noise, num_steps=native.num_inference_steps)
        changes = {}
        for intervention in ('zero', 'static', 'shuffle'):
            actions = model.sample_actions(context, noise, num_steps=native.num_inference_steps,
                                           forecast_intervention=intervention)
            changes[intervention] = (original - actions).abs().max().item()
        repeat = model.sample_actions(context, noise, num_steps=native.num_inference_steps)
    if not torch.equal(original, repeat) or any(value <= 0 for value in changes.values()):
        raise RuntimeError('Paired action sampling is nondeterministic or forecast intervention has no effect')
    report['sampling'] = {'action_shape': list(original.shape), 'execute_prefix_shape': list(original[:, :8].shape),
        'flow_steps': native.num_inference_steps, 'repeat_bitwise_equal': True, 'intervention_max_action_difference': changes,
        'coordinate_system': 'normalized Panda interface; no simulator commands emitted'}
    report['memory_after_integration'] = native_probe.memory()
    report['status'] = 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--teacher-path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    report = {'schema_version': 1, 'status': 'running', 'stage': 'native_semflow_integration',
        'upstream_revision': native_probe.UPSTREAM_REVISION,
        'limitations': ['Synthetic RGB only; new interfaces and teacher pooling are not source-calibrated.',
            'No human-video adaptation, task training or robot rollout.',
            'Adapter serialization only; no full-runtime restart recovery.']}
    started = time.monotonic()
    def deadline(_signum, _frame):
        raise TimeoutError('Native integration probe exceeded 600 seconds')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    native_probe.write_report(report, args.output)
    try:
        run(args, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error).split('\n')[0][:600])
    finally:
        signal.alarm(0)
        report['seconds'] = time.monotonic() - started
        native_probe.write_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
