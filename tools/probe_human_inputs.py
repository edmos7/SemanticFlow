#!/usr/bin/env python3
"""Bounded three-clip native compatibility audit; zero updates and rollouts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import time

import probe_egosteer as native_probe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(args):
    from semflow.config import load_config
    from semflow.data import validate_episode_manifest
    request = json.loads(args.request.read_text())
    expected_keys = {'manifest_sha256', 'source_index_sha256', 'checkpoint_sha256', 'windows'}
    if set(request) != expected_keys:
        raise ValueError('unexpected human preflight request fields')
    for path, key in ((args.manifest, 'manifest_sha256'), (args.index, 'source_index_sha256'),
                      (args.checkpoint, 'checkpoint_sha256')):
        if sha(path) != request[key]:
            raise ValueError('input hash mismatch: ' + key)
    receipt = json.loads(args.checkpoint_receipt.read_text())
    if receipt.get('stage') != 'train' or receipt.get('checkpoint_sha256') != request['checkpoint_sha256']:
        raise ValueError('common-source checkpoint lacks matching original training receipt')
    _, tasks = load_config(args.pilot)
    manifest, index = json.loads(args.manifest.read_text()), json.loads(args.index.read_text())
    counts = validate_episode_manifest(manifest, tasks, require_complete=True)
    if index['manifest']['human_clips'] or manifest['episodes'] != index['manifest']['episodes']:
        raise ValueError('source-only index or merged episode identity differs')
    if (len(request['windows']) != 3 or set(request['windows']) != {c['target_id'] for c in manifest['human_clips']}
            or any(type(a) is not int for a in request['windows'].values())):
        raise ValueError('require one fixed anchor for each of three clips')
    return request, manifest, index, counts


def run(args, report):
    import torch
    from semflow.config import load_json
    from semflow.human_batching import HumanBatchProcessor, read_human_window
    from semflow.models import (NativeSemFlowPolicy, SourceFeatureStatistics, inject_shared_lora,
        enable_shared_adapter_training, raw_future_feature_loss, semantic_flow_loss)
    from semflow.models.source_calibration import SpatialReconstruction, enable_source_training, trainable_delta
    from train_source_calibration import tensor_digest

    request, manifest, index, counts = validate_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError('native human compatibility audit requires verified GPU environment')
    report.update(counts=counts, input_hashes={k:v for k,v in request.items() if k != 'windows'}, clips=[])
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    contract = {'base_artifacts': native_probe.ARTIFACTS, 'upstream': native_probe.UPSTREAM_REVISION,
                'source_index_sha256': request['source_index_sha256'], 'normalization': index['normalization'],
                'configuration': load_json(args.calibration)}
    contract = json.loads(json.dumps(contract))
    if saved.get('schema_version') != 1 or saved['contract'] != contract or saved['step'] != 600:
        raise ValueError('requires original Base step600 checkpoint and exact calibration contract')
    stats = saved['source_feature_statistics']
    train_hashes = {e['sha256'] for e in index['manifest']['episodes'] if e['split'] == 'train'}
    if (stats['source_manifest'] != request['source_index_sha256'] or stats['fit_split'] != 'train'
            or not stats['windows'] or any(w['split'] != 'train' or w['episode_sha256'] not in train_hashes
                                          for w in stats['windows'])):
        raise ValueError('scene statistics must have source-train-only provenance')
    statistics = SourceFeatureStatistics(stats['mean'], stats['scale'], stats['source_manifest'])
    report['environment'] = native_probe.environment()
    native_probe.access_preflight(report, args)
    native_probe.prepare_native_imports(args)
    paths = native_probe.download_models(report, args)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    native, cfg = native_probe.load_model(report, args, paths)
    inject_shared_lora(native)
    policy = NativeSemFlowPolicy(native).cuda()
    reconstruction = SpatialReconstruction(policy.scene.dim, native.world_model_expert.hidden_size).cuda()
    enable_source_training(policy, reconstruction)
    expected = set(trainable_delta(policy))
    state = policy.state_dict()
    if (set(saved['model_delta']) != expected
            or saved['trainable_names'] != [n for n,p in policy.named_parameters() if p.requires_grad]):
        raise ValueError('checkpoint does not exactly cover source-trained interfaces and adapters')
    for name, value in saved['model_delta'].items():
        if value.shape != state[name].shape or value.dtype != state[name].dtype or not torch.isfinite(value).all():
            raise ValueError('invalid checkpoint tensor: ' + name)
    policy.load_state_dict(saved['model_delta'], strict=False)
    del reconstruction, saved, state
    entries = enable_shared_adapter_training(policy, policy=native)
    parameters = [p for _,p in entries]
    report['adaptation_parameters'] = {n:p.numel() for n,p in entries}
    before = tensor_digest(trainable_delta(policy))
    versions = {n: t._version for n,t in (*policy.named_parameters(), *policy.named_buffers())}
    processor = HumanBatchProcessor(native, cfg)
    image_size = tuple(map(int, cfg.data.target_image_size))
    torch.cuda.reset_peak_memory_stats()
    for clip in manifest['human_clips']:
        window = read_human_window(args.data_root, clip, request['windows'][clip['target_id']], image_size=image_size)
        inputs = processor([window['observations']], 'cuda')
        times = window['observations']['observation_times']
        expected_tags = [f'{((times[i]-times[0])+(times[i+1]-times[0]))/2:.1f}' for i in (0,2,4)]
        prompt = native.backbone.processor.tokenizer.decode(inputs['batch']['input_ids'][0], skip_special_tokens=False)
        if re.findall(r'<([0-9]+\.[0-9]) seconds>', prompt) != expected_tags:
            raise RuntimeError('native video prompt timestamps differ from measured PTS pair averages')
        expected_current = inputs['current_rgb'].flatten(0, 1).unsqueeze(1)
        calls = []
        def check_current(_module, arguments):
            if not torch.equal(arguments[0], expected_current):
                raise RuntimeError('future or noncurrent RGB entered observation teacher path')
            calls.append(True)
        hook = native.frozen_teacher.register_forward_pre_hook(check_current)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                observed = policy.encode_observations(**inputs)
        finally:
            hook.remove()
        if len(calls) != 1 or observed.raw_forecasts[:, 0].count_nonzero():
            raise RuntimeError('current-image teacher or absent-view forecast contract failed')
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            future = torch.from_numpy(window['loss_targets']['future_rgb']).cuda()[None]
            chest_raw = native.frozen_teacher(future)
            target_raw = torch.stack((torch.zeros_like(chest_raw), chest_raw), dim=1)
            target_scene = policy.pool_raw(target_raw, observed.view_valid,
                observed.task_tokens.detach(), observed.task_valid)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            valid = observed.view_valid[:, :, None].expand(-1, -1, 3).flatten(0, 1)
            raw = raw_future_feature_loss(observed.raw_forecasts.flatten(0, 1), target_raw.flatten(0, 1), valid)
            semantic = semantic_flow_loss(observed.scene_forecasts, target_scene,
                torch.ones((1,3), dtype=torch.bool, device='cuda'), statistics)
        norms = {}
        for label, loss in (('raw', raw), ('semantic', semantic.total)):
            gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            if any(g is not None and not torch.isfinite(g).all() for g in gradients):
                raise RuntimeError('nonfinite adapter gradient')
            norms[label] = sum(g.float().square().sum().item() for g in gradients if g is not None)**.5
            if norms[label] <= 0:
                raise RuntimeError('frozen readout blocked prediction gradients')
        del gradients
        changed = {**inputs, 'current_rgb': inputs['current_rgb'].clone()}
        changed['current_rgb'][:, 0] = 255
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            other = policy.encode_observations(**changed)
        for first, second in ((observed.raw_forecasts, other.raw_forecasts),
                              (observed.current_scene, other.current_scene),
                              (observed.scene_forecasts, other.scene_forecasts)):
            if not torch.equal(first, second):
                raise RuntimeError('absent view affected observed scene or forecast')
        report['clips'].append({'provenance': window['provenance'], 'gradient_norms': norms,
            'finite_losses': bool(torch.isfinite(raw) and torch.isfinite(semantic.total)),
            'raw_forecast_shape': list(observed.raw_forecasts.shape),
            'native_time_tags_seconds': expected_tags,
            'missing_view_invariance': True, 'state_action_slots_absent': True})
        native_probe.write_report(report, args.output)
        print(json.dumps({'event': 'human_clip_passed', 'target_id': clip['target_id']}), flush=True)
        del observed, other, raw, semantic, target_raw, target_scene, inputs, changed, loss
    if (tensor_digest(trainable_delta(policy)) != before or
            any(t._version != versions[n] for n,t in (*policy.named_parameters(), *policy.named_buffers()))
            or any(p.grad is not None for p in policy.parameters())):
        raise RuntimeError('preflight mutated model state or accumulated parameter gradients')
    report.update(status='passed', model_state_unchanged=True, memory=native_probe.memory())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'manifest', 'index', 'checkpoint', 'checkpoint-receipt', 'data-root', 'upstream-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=Path('configs/pilot.json'))
    parser.add_argument('--calibration', type=Path, default=Path('configs/source_calibration.json'))
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/semflow-model-cache/hub'))
    parser.add_argument('--teacher-path', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior human audit output')
    report = {'status': 'running', 'optimizer_steps': 0, 'rollouts': 0,
              'scope': 'Human input compatibility only; no adaptation or effectiveness scores.'}
    started = time.monotonic()
    def timeout(_signum, _frame):
        raise TimeoutError('human model probe exceeded 900 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    try:
        run(args, report)
    except Exception as error:
        report.update(status='failed', error=str(error).split('\n')[0][:600])
        raise
    finally:
        signal.alarm(0)
        report['seconds'] = time.monotonic()-started
        native_probe.write_report(report, args.output)


if __name__ == '__main__':
    main()
