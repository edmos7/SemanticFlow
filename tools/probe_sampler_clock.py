#!/usr/bin/env python3
"""Trace indexed/native flow clocks without changing production sampling."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time
import numpy as np
import probe_egosteer as native_probe
from rehearse_adaptation import load_common,validate_request,sha
from decision_diagnostics import native_sample
from probe_source_actions import SEEDS,metrics,window_digest
from train_source_calibration import tensor_digest
from compare_source_initializations import ObservedProcessor,assert_nested_equal
from semflow.models.sampling import accumulated_sample

CLOCK_CONFIG={'validation_windows':6,'seed':SEEDS[0],'integration_steps':[5,10,20],
    'implementations':['indexed','clock_only','accumulated','upstream'],'parity_atol':1e-4,
    'previous_reproduction_atol':1e-5,'optimizer_steps':0,'rollouts':0}


def validate_inputs(args):
    request,index,tasks,plan=validate_request(args)
    spec=json.loads(args.clock_request.read_text())
    if (set(spec)!={'config','source_request_sha256','previous_report_sha256','previous_receipt_sha256'}
            or spec['config']!=CLOCK_CONFIG or spec['source_request_sha256']!=sha(args.request)
            or spec['previous_report_sha256']!=sha(args.previous_report)
            or spec['previous_receipt_sha256']!=sha(args.previous_receipt)):
        raise ValueError('clock audit input declaration differs')
    previous=json.loads(args.previous_report.read_text());receipt=json.loads(args.previous_receipt.read_text())
    if (previous['status']!='passed' or previous['condition']!='frozen'
            or previous['request']!=request or receipt['stage']!='humanaction-frozen'
            or receipt['comparison_report_sha256']!=spec['previous_report_sha256']):
        raise ValueError('previous frozen sample contract differs')
    return request,index,tasks,plan,spec,previous


@contextmanager
def capture_flow(policy):
    captured={k:[] for k in ('times','embedding','noisy','velocity')}
    def before(name):
        return lambda module,args:captured[name].append(args[0].detach().cpu().clone()) and None
    def after(name):
        return lambda module,args,value:captured[name].append(value.detach().cpu().clone()) and None
    hooks=[policy.native.time_embedding.register_forward_pre_hook(before('times')),
           policy.native.time_embedding.register_forward_hook(after('embedding')),
           policy.panda_action_in.register_forward_pre_hook(before('noisy')),
           policy.panda_action_out.register_forward_hook(after('velocity'))]
    try:yield captured
    finally:
        for hook in reversed(hooks):hook.remove()


def compare_traces(a,b,steps):
    import torch
    if any(len(x[k])!=steps for x in (a,b) for k in ('times','embedding','noisy','velocity')):
        raise ValueError('incomplete per-step flow trace')
    rows=[]
    for i in range(steps):
        row={'step':i,'left_time':float(a['times'][i][0]),'right_time':float(b['times'][i][0])}
        for key in a:
            if a[key][i].shape!=b[key][i].shape:raise ValueError('trace shape mismatch')
            row[key+'_max_abs']=float((a[key][i].float()-b[key][i].float()).abs().max())
        rows.append(row)
    return rows


def run(args,report):
    import torch
    from semflow.source_batching import SourceWindowSampler,source_window
    from train_source_calibration import MemorySources
    request,index,tasks,plan,spec,previous=validate_inputs(args)
    report.update(request=request,clock_request=spec,rows=[])
    policy,processor,entries,_=load_common(args,index,request,report)
    initial=tensor_digest({n:p for n,p in entries})
    if initial!=previous['initial_adapter_sha256']:raise ValueError('common adapters differ')
    common=torch.load(args.checkpoint,map_location='cpu',weights_only=True);state=policy.state_dict()
    if tensor_digest(common['model_delta'])!=tensor_digest({k:state[k] for k in common['model_delta']}):raise ValueError('loaded checkpoint differs')
    del common,state
    policy.requires_grad_(False).eval();before=tensor_digest(policy.state_dict())
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
    from src.policy.egosteer_inference import infer_flow_action
    report['upstream_inference_sha256']=sha(args.upstream_root/'src/policy/egosteer_inference.py')
    sources=MemorySources(index['source_lock']);observed=ObservedProcessor(processor)
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,split='validation',cache_size=2)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i in range(0,12,2):
                record=plan['validation'][i];window=source_window(sampler.episode(record['record']),record['anchor'])
                prior=previous['panels']['frozen']['panel'][i]
                if window_digest(window)!=prior['window_sha256']:raise ValueError('prior source window differs')
                encoded=policy.encode_observations(**observed([{k:v for k,v in window.items() if k!='loss_targets'}],'cuda'))
                noise=torch.randn((1,32,7),device='cuda',dtype=torch.float32,generator=torch.Generator(device='cuda').manual_seed(SEEDS[0]))
                noise_sha=tensor_digest({'noise':noise})
                for steps in CLOCK_CONFIG['integration_steps']:
                    outputs={};traces={}
                    for implementation in CLOCK_CONFIG['implementations']:
                        with capture_flow(policy) as captured:
                            if implementation=='indexed':value=policy.sample_actions(encoded,noise,num_steps=steps)
                            elif implementation in ('clock_only','accumulated'):
                                value=accumulated_sample(policy,encoded,noise,steps,native_update=implementation=='accumulated')
                            else:value=native_sample(policy,encoded,noise,steps,infer_flow_action)
                        if tensor_digest({'noise':noise})!=noise_sha:raise ValueError('sampler changed paired noise')
                        outputs[implementation]=value[0].float().cpu().numpy();traces[implementation]=captured
                    old=next(s for s in prior['samples'] if s['seed']==SEEDS[0] and s['integration_steps']==steps)
                    reproduction=float(np.abs(outputs['indexed']-np.asarray(old['normalized_prediction'])).max())
                    parity=float(np.abs(outputs['accumulated']-outputs['upstream']).max())
                    row={'task':record['record']['task'],**window['provenance'],'steps':steps,'noise_sha256':noise_sha,
                         'previous_max_abs':reproduction,'matched_clock_vs_upstream_max_abs':parity,
                         'clock_only_vs_upstream_max_abs':float(np.abs(outputs['clock_only']-outputs['upstream']).max()),
                         'indexed_vs_upstream_max_abs':float(np.abs(outputs['indexed']-outputs['upstream']).max()),
                         'indexed_vs_upstream_rms':float(np.square(outputs['indexed']-outputs['upstream']).mean())**.5,
                         'indexed_vs_upstream_trace':compare_traces(traces['indexed'],traces['upstream'],steps),
                         'matched_vs_upstream_trace':compare_traces(traces['accumulated'],traces['upstream'],steps),
                         'metrics':{k:metrics(v,window['loss_targets']['actions'],index['normalization']) for k,v in outputs.items()},
                         'passed':reproduction<=CLOCK_CONFIG['previous_reproduction_atol'] and parity<=CLOCK_CONFIG['parity_atol']}
                    report['rows'].append(row);native_probe.write_report(report,args.output)
                    print(json.dumps({'event':'sampler_clock_progress','window':i,'steps':steps,'passed':row['passed'],
                        'indexed_vs_native_max_abs':row['indexed_vs_upstream_max_abs'],'matched_vs_native_max_abs':parity}),flush=True)
                    # Record all failed comparisons before stopping; do not loosen tolerances.
                    if not row['passed']:raise ValueError('matched clock parity or previous reproduction failed; traces saved')
        if before!=tensor_digest(policy.state_dict()):raise ValueError('sampler audit changed model')
        assert_nested_equal(rng,{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()})
        if any(p.grad is not None for p in policy.parameters()):raise ValueError('unexpected gradients')
        report.update(status='passed',parameters_buffers_rng_unchanged=True,memory=native_probe.memory(),
            clock_only_explains_mismatch=all(r['clock_only_vs_upstream_max_abs']<=CLOCK_CONFIG['parity_atol'] for r in report['rows']),
            conclusion='Accumulated native clock tested against original inference; clock-only path separates time-grid from update arithmetic. Indexed production path preserved and prior samples reproduced. No controller-success claim.')
    finally:sources.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','clock-request','previous-report','previous-receipt','index','checkpoint','checkpoint-receipt','upstream-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pilot',type=Path,default=Path('configs/pilot.json'));p.add_argument('--calibration',type=Path,default=Path('configs/source_calibration.json'))
    p.add_argument('--cache-dir',type=Path,default=Path('/content/semflow-model-cache/hub'));p.add_argument('--teacher-path',type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve clock audit output')
    report={'status':'running','optimizer_steps':0,'rollouts':0};start=time.monotonic()
    def timeout(s,f):raise TimeoutError('clock audit exceeded600s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:run(args,report)
    except Exception as error:report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:signal.alarm(0);report['seconds']=time.monotonic()-start;native_probe.write_report(report,args.output)
if __name__=='__main__':main()
