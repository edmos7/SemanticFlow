#!/usr/bin/env python3
"""Paired sampled-action evaluation of the common and adapted source policies."""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import probe_egosteer as native_probe
from rehearse_adaptation import sha, validate_request, load_common, validate_matched_reference
from probe_source_actions import SEEDS, metrics, window_digest
from compare_source_initializations import ObservedProcessor

PANEL_CONFIG = {'split':'validation', 'windows':12, 'seeds':list(SEEDS),
                'integration_steps':[5,10,20], 'action_horizon':32, 'execute_prefix':8,
                'optimizer_steps':0, 'rollouts':0}
ARMS = ('frozen','standard','semflow')


def validate_inputs(args):
    request,index,tasks,plan = validate_request(args)
    spec = json.loads(args.panel_request.read_text())
    if set(spec) != {'config','rehearsal_request_sha256','artifacts'} or spec['config'] != PANEL_CONFIG:
        raise ValueError('sampled-action panel declaration differs')
    if sha(args.request) != spec['rehearsal_request_sha256'] or set(spec['artifacts']) != {'standard','semflow'}:
        raise ValueError('panel input declaration differs')
    reports = {}
    for arm in ('standard','semflow'):
        pins = spec['artifacts'][arm]
        if set(pins) != {'checkpoint','report','receipt'}:
            raise ValueError('incomplete adapted artifact pins')
        paths = {key:args.input_root/(arm+suffix) for key,suffix in
                 (('checkpoint','.pt'),('report','-report.json'),('receipt','-receipt.json'))}
        if any(sha(paths[k]) != pins[k] for k in pins):
            raise ValueError('adapted input hash mismatch')
        report = json.loads(paths['report'].read_text())
        receipt = json.loads(paths['receipt'].read_text())
        if (report['status'] != 'passed' or report['condition'] != arm or report['mode'] != 'train'
                or report['optimizer_steps'] != 60 or report['request'] != request or report['plan'] != plan
                or report['checkpoint']['sha256'] != pins['checkpoint']
                or report['checkpoint']['bytes'] != paths['checkpoint'].stat().st_size
                or receipt['stage'] != 'rehearsal-'+arm+'-train'
                or receipt['checkpoint_sha256'] != pins['checkpoint']
                or receipt['comparison_report_sha256'] != pins['report']):
            raise ValueError('adapted report/receipt contract differs')
        reports[arm] = report
    validate_matched_reference(reports['semflow'],reports['standard'])
    return request,index,tasks,plan,spec,reports


def panel_identity(report):
    return {'request':report['request'], 'panel_config':report['panel_config'],
            'initial_adapter_sha256':report['initial_adapter_sha256'],
            'processed_layouts':report['processed_layouts'],
            'rows':[{k:row[k] for k in ('task','episode_sha256','split','anchor','window_sha256','normalized_target')}
                    | {'noise':[(s['integration_steps'],s['seed'],s['initial_noise_sha256']) for s in row['samples']]}
                    for row in report['panel']]}


def summarize(rows):
    tasks = sorted({r['task'] for r in rows})
    expected = [(steps,seed) for steps in PANEL_CONFIG['integration_steps'] for seed in SEEDS]
    if (len(rows) != 12 or len(tasks) != 6 or len({r['episode_sha256'] for r in rows}) != 12
            or any(r['split'] != 'validation' for r in rows)
            or any(sum(r['task']==task for r in rows)!=2 for task in tasks)
            or any([(s['integration_steps'],s['seed']) for s in r['samples']] != expected for r in rows)):
        raise ValueError('incomplete or unpaired validation panel')
    result = {}
    for steps in PANEL_CONFIG['integration_steps']:
        by_task = {}
        for task in ['all']+tasks:
            selected = [r for r in rows if task=='all' or r['task']==task]
            samples = [[s for s in r['samples'] if s['integration_steps']==steps] for r in selected]
            pred = np.asarray([[s['normalized_prediction'] for s in ss] for ss in samples],dtype=np.float64)
            target = np.asarray([r['normalized_target'] for r in selected],dtype=np.float64)
            if (pred.shape!=(len(selected),4,32,7) or target.shape!=(len(selected),32,7)
                    or not np.isfinite(pred).all() or not np.isfinite(target).all()):
                raise ValueError('invalid sampled action arrays')
            horizons = {}
            for label,count in (('full',32),('first_eight',8)):
                x,y = pred[:,:,:count],target[:,:count]
                mse = np.square(x-y[:,None]).mean((0,1,2))
                mean_error = np.square(x.mean(1)-y).mean((0,1))
                variance = x.var(1).mean((0,1))
                if not np.allclose(mse,mean_error+variance,rtol=1e-10,atol=1e-10):
                    raise ValueError('seed decomposition failed')
                flat = [s[label] for ss in samples for s in ss]
                horizons[label] = {'sampled_mse':float(mse.mean()),
                    'mean_prediction_mse':float(mean_error.mean()),'seed_variance':float(variance.mean()),
                    'per_dimension_mse':mse.tolist(),'per_dimension_mean_error':mean_error.tolist(),
                    'per_dimension_seed_variance':variance.tolist(),
                    'train_mean_baseline_mse':float(np.square(y).mean()),
                    'gripper_sign_accuracy':float(np.mean([s['gripper_sign_accuracy'] for s in flat])),
                    'clipped_values':float(np.mean([s['clipped_values'] for s in flat])),
                    'clipped_by_dimension':np.mean([s['clipped_by_dimension'] for s in flat],axis=0).tolist()}
            by_task[task] = horizons
        result[str(steps)] = by_task
    return result


def compare_panels(reports):
    if set(reports)!=set(ARMS) or any(r['status']!='passed' or r['condition']!=arm for arm,r in reports.items()):
        raise ValueError('three passed action panels required')
    reference = panel_identity(reports['frozen'])
    if any(panel_identity(r)!=reference for r in reports.values()):
        raise ValueError('panel inputs, noise or processing differ across arms')
    summaries = {arm:summarize(r['panel']) for arm,r in reports.items()}
    deltas = {}
    for steps in map(str,PANEL_CONFIG['integration_steps']):
        deltas[steps] = {}
        for horizon in ('full','first_eight'):
            values = {arm:s[steps]['all'][horizon]['sampled_mse'] for arm,s in summaries.items()}
            deltas[steps][horizon] = {'standard_minus_frozen':values['standard']-values['frozen'],
                'semflow_minus_frozen':values['semflow']-values['frozen'],
                'semflow_minus_standard':values['semflow']-values['standard']}
    return {'paired':True,'summaries':summaries,'deltas':deltas,
        'interpretation':'Fixed source panel, one adaptation seed. No seed or integration-step selection, robot-success or target-transfer claim.'}


def sample_panel(policy, processor, windows, records, normalization, report, persist, *, device='cuda'):
    import torch
    from train_source_calibration import tensor_digest
    policy.requires_grad_(False).eval()
    before = tensor_digest(dict(policy.named_parameters()) | dict(policy.named_buffers()))
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device=='cuda' else []
    report['panel'] = []
    start = time.monotonic()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
        for window,item in zip(windows,records,strict=True):
            target = window['loss_targets']['actions']
            row = {'task':item['record']['task'],**window['provenance'],
                   'window_sha256':window_digest(window),'normalized_target':target.tolist(),
                   'train_mean_baseline':metrics(np.zeros_like(target),target,normalization),'samples':[]}
            observed = {k:v for k,v in window.items() if k!='loss_targets'}
            encoded = policy.encode_observations(**processor([observed],device))
            # Warm the sampling path; report synchronized sampling latency only.
            policy.sample_actions(encoded,torch.zeros((1,32,7),device=device),num_steps=10)
            for steps in PANEL_CONFIG['integration_steps']:
                for seed in SEEDS:
                    noise = torch.randn((1,32,7),device=device,dtype=torch.float32,
                                        generator=torch.Generator(device=device).manual_seed(seed))
                    noise_sha = tensor_digest({'noise':noise})
                    if device=='cuda': torch.cuda.synchronize()
                    begin = time.perf_counter()
                    prediction = policy.sample_actions(encoded,noise,num_steps=steps)
                    if device=='cuda': torch.cuda.synchronize()
                    seconds = time.perf_counter()-begin
                    if tensor_digest({'noise':noise})!=noise_sha:
                        raise RuntimeError('sampler changed initial noise')
                    row['samples'].append({'seed':seed,'integration_steps':steps,
                        'initial_noise_sha256':noise_sha,'seconds':seconds,
                        **metrics(prediction[0].float().cpu().numpy(),target,normalization)})
            report['panel'].append(row)
            print(json.dumps({'event':'action_panel_progress','condition':report['condition'],
                              'windows':len(report['panel']),'of':12}),flush=True)
            persist()
    if (before!=tensor_digest(dict(policy.named_parameters()) | dict(policy.named_buffers()))
            or not torch.equal(cpu_rng,torch.get_rng_state())
            or (device=='cuda' and any(not torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)))
            or any(p.grad is not None for p in policy.parameters())):
        raise RuntimeError('read-only sampling changed model state or global RNG')
    report.update(panel_summary=summarize(report['panel']),panel_seconds=time.monotonic()-start,
                  parameters_buffers_rng_unchanged=True,
                  latency_scope='CUDA synchronized sampling only; encoding excluded; one warmup per window')


def run(args,report):
    import torch
    from semflow.source_batching import SourceWindowSampler,source_window
    from train_source_calibration import MemorySources,tensor_digest
    request,index,tasks,plan,spec,training_reports = validate_inputs(args)
    report.update(request=request,panel_config=PANEL_CONFIG,panel_request=spec,condition=args.condition)
    if not torch.cuda.is_available(): raise RuntimeError('reviewed GPU runtime required')
    policy,processor,entries,_ = load_common(args,index,request,report)
    initial = tensor_digest({n:p for n,p in entries})
    if initial != training_reports['standard']['initial_adapter_sha256']:
        raise ValueError('common adapter initialization differs')
    report['initial_adapter_sha256'] = initial
    if args.condition!='frozen':
        training = training_reports[args.condition]
        saved = torch.load(args.input_root/(args.condition+'.pt'),map_location='cpu',weights_only=True)
        expected = {k:training[k] for k in ('request','condition','teacher_cache_sha256','initial_adapter_sha256','semantic_calibration')}
        if (saved['schema_version']!=1 or saved['step']!=60 or saved['contract']!=expected
                or set(saved['adapters'])!={n for n,p in entries}):
            raise ValueError('adapted checkpoint contract differs')
        with torch.no_grad():
            for name,p in entries:
                value=saved['adapters'][name]
                if value.dtype!=p.dtype or value.shape!=p.shape or not torch.isfinite(value).all():
                    raise ValueError('invalid adapted tensor')
                p.copy_(value)
        if tensor_digest({n:p for n,p in entries})!=tensor_digest(saved['adapters']):
            raise ValueError('restored adapters differ')
        report['adapted_checkpoint_sha256']=spec['artifacts'][args.condition]['checkpoint']
        del saved
    policy.requires_grad_(False).eval()
    processor = ObservedProcessor(processor)
    sources = MemorySources(index['source_lock'])
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,split='validation',cache_size=2)
        rows=plan['validation']
        windows=[source_window(sampler.episode(item['record']),item['anchor']) for item in rows]
        sample_panel(policy,processor,windows,rows,index['normalization'],report,
                     lambda:native_probe.write_report(report,args.output))
        report.update(processed_layouts=processor.layouts,status='passed',memory=native_probe.memory())
        if args.condition != 'frozen':
            if args.frozen_report is None: raise ValueError('verified frozen panel required')
            frozen = json.loads(args.frozen_report.read_text())
            if (frozen['status']!='passed' or frozen['condition']!='frozen'
                    or panel_identity(report)!=panel_identity(frozen)):
                raise ValueError('frozen panel inputs or paired noise differ')
            report['frozen_report_sha256'] = sha(args.frozen_report)
            if args.condition == 'semflow':
                if args.standard_report is None: raise ValueError('verified standard panel required')
                standard = json.loads(args.standard_report.read_text())
                report['comparison'] = compare_panels(dict(frozen=frozen,standard=standard,semflow=report))
                report['standard_report_sha256'] = sha(args.standard_report)
    finally: sources.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','panel-request','input-root','index','checkpoint','checkpoint-receipt','upstream-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--condition',choices=ARMS,required=True)
    parser.add_argument('--frozen-report',type=Path)
    parser.add_argument('--standard-report',type=Path)
    parser.add_argument('--pilot',type=Path,default=Path('configs/pilot.json'))
    parser.add_argument('--calibration',type=Path,default=Path('configs/source_calibration.json'))
    parser.add_argument('--cache-dir',type=Path,default=Path('/content/semflow-model-cache/hub'))
    parser.add_argument('--teacher-path',type=Path)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('preserve existing panel result')
    report={'status':'running','optimizer_steps':0,'rollouts':0,'no_target_data':True}
    start=time.monotonic()
    def timeout(_signum,_frame): raise TimeoutError('action panel exceeded900s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:run(args,report)
    except Exception as error:
        report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-start;native_probe.write_report(report,args.output)


if __name__=='__main__':main()
