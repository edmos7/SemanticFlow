#!/usr/bin/env python3
"""Matched one-clip adaptation with separate task resets and durable recovery."""
import argparse
import copy
import json
from pathlib import Path
import signal
import time
import numpy as np
import probe_egosteer as native_probe
from rehearse_adaptation import load_common, sha, validate_request
from semflow.adaptation_plan import CONFIG, calibrated_semantic_weight, digest
from semflow.human_adaptation_plan import HUMAN_CONFIG, human_plan


def validate_inputs(args):
    from semflow.config import load_config
    from semflow.data import validate_episode_manifest
    request,index,tasks,source = validate_request(args)
    human = json.loads(args.human_request.read_text())
    if set(human) != {'manifest_sha256','source_request_sha256','config','plan_sha256'}:
        raise ValueError('unexpected human adaptation request fields')
    if (human['manifest_sha256'] != sha(args.manifest) or human['source_request_sha256'] != sha(args.request)
            or human['config'] != HUMAN_CONFIG):
        raise ValueError('human adaptation request differs from inputs/configuration')
    manifest=json.loads(args.manifest.read_text())
    validate_episode_manifest(manifest,tasks,require_complete=True)
    if manifest['episodes'] != index['manifest']['episodes']:
        raise ValueError('human manifest source identities differ')
    plan=human_plan(manifest)
    if digest(plan)!=human['plan_sha256']:
        raise ValueError('human window plan differs')
    return request,index,tasks,source,human,manifest,plan


def reset_adapters(entries, values):
    import torch
    if set(values) != {n for n,_ in entries}:
        raise ValueError('adapter scope differs')
    with torch.no_grad():
        for name,parameter in entries:
            value=values[name]
            if value.shape!=parameter.shape or value.dtype!=parameter.dtype or not torch.isfinite(value).all():
                raise ValueError('invalid adapter tensor')
            parameter.copy_(value)
            parameter.grad=None


def validate_pair(report, reference):
    if reference.get('status')!='passed' or reference.get('condition')!='standard' or reference.get('mode')!='train':
        raise ValueError('passed standard training reference required')
    for key in ('request','human_request','plan','source_plan','initial_adapter_sha256',
                'teacher_cache_sha256','semantic_calibration','trainable_parameters'):
        if report[key]!=reference[key]:
            raise ValueError('paired human adaptation differs: '+key)


def run(args,report):
    import torch
    from semflow.human_batching import read_human_window
    from semflow.models.adaptation import fixed_forecast_targets, forecast_objective, source_action_retention
    from semflow.source_batching import SourceWindowSampler,source_window
    from train_source_calibration import MemorySources,save_checkpoint,tensor_digest
    from compare_source_initializations import assert_nested_equal
    request,index,tasks,source_plan,human_request,manifest,plan=validate_inputs(args)
    report.update(condition=args.condition,mode=args.mode,request=request,human_request=human_request,
                  plan=plan,source_plan=source_plan,optimizer_steps=0,tasks={})
    if not torch.cuda.is_available(): raise RuntimeError('human adaptation requires native GPU environment')
    policy,source_processor,entries,statistics,human_processor=load_common(
        args,index,request,report,include_human_processor=True)
    parameters=[p for _,p in entries]
    initial={n:p.detach().cpu().clone() for n,p in entries}
    initial_sha=tensor_digest(initial)
    frozen_versions={n:t._version for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n not in initial}
    report['trainable_parameters']={n:p.numel() for n,p in entries}
    sources=MemorySources(index['source_lock'])
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,cache_size=4)
        source_windows={g:[source_window(sampler.episode(r['record']),r['anchor']) for r in rows]
                        for g,rows in source_plan.items()}
        human_windows={c['target_id']:[read_human_window(args.data_root,c,a,image_size=human_processor.image_size)
                        for a in plan[c['target_id']]['anchors']] for c in manifest['human_clips']}
        def encode_source(window):
            return policy.encode_observations(**source_processor([{k:v for k,v in window.items() if k!='loss_targets'}],'cuda'))
        def encode_human(window):
            return policy.encode_observations(**human_processor([window['observations']],'cuda'))
        # Fix both DINO features and task-conditioned semantic targets at Base600.
        # No labels are recomputed after either a task update or checkpoint load.
        targets={}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for target in sorted(plan):
                targets[target]=[]
                for window in human_windows[target]:
                    rgb=torch.from_numpy(window['loss_targets']['future_rgb'])[None].cuda()
                    future=torch.stack((torch.zeros_like(rgb),rgb),dim=1)
                    targets[target].append(fixed_forecast_targets(policy,encode_human(window),future))
            source_targets=[]
            for window in source_windows['adaptation'][::2]:
                source_targets.append(fixed_forecast_targets(policy,encode_source(window),
                    torch.from_numpy(window['loss_targets']['future_rgb'])[None].cuda()))
        def teacher_sha():
            return tensor_digest({f'{g}/{i}/{k}':v for g,rows in targets.items() for i,row in enumerate(rows) for k,v in row.items()})
        cache_sha=teacher_sha()
        norms=[]
        for window,label in zip(source_windows['adaptation'][::2],source_targets):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                observation=encode_source(window)
                _,parts=forecast_objective(observation,label,statistics,condition='semflow',semantic_weight=1.)
            pair={}
            for name,loss in (('raw',.1*parts['raw']),('semantic',parts['semantic'])):
                grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in grads):
                    raise ValueError('nonfinite source calibration gradient')
                pair[name]=sum(g.float().square().sum().item() for g in grads if g is not None)**.5
            norms.append(pair)
            del observation,parts,loss,grads,_
        weight=calibrated_semantic_weight(norms)
        report.update(initial_adapter_sha256=initial_sha,teacher_cache_sha256=cache_sha,
                      semantic_calibration=dict(weight,source_training_gradient_norms=norms))
        if args.condition=='semflow':
            if args.reference_report is None:raise ValueError('exported standard reference required')
            validate_pair(report,json.loads(args.reference_report.read_text()))
            report['matched_reference_sha256']=sha(args.reference_report)
        contract={k:report[k] for k in ('request','human_request','condition','initial_adapter_sha256',
                                       'teacher_cache_sha256','semantic_calibration')}
        saved=None
        if args.mode=='evaluate':
            if args.arm_checkpoint is None or sha(args.arm_checkpoint)!=args.arm_checkpoint_sha256:
                raise ValueError('requires hash-verified exported human checkpoint')
            saved=torch.load(args.arm_checkpoint,map_location='cpu',weights_only=True)
            if saved['schema_version']!=1 or saved['contract']!=contract or set(saved['tasks'])!=set(plan):
                raise ValueError('human recovery contract differs')
        def snapshot():return {n:p.detach().cpu().clone() for n,p in entries}
        def losses(target,a,r,seed):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                forecast,parts=forecast_objective(encode_human(human_windows[target][a]),targets[target][a],statistics,
                    condition=args.condition,semantic_weight=weight['weight'])
                window=source_windows['retention'][r]
                action=source_action_retention(policy,encode_source(window),
                    torch.from_numpy(window['loss_targets']['actions'])[None].cuda(),seed=seed)
            return forecast+action,{**parts,'retention':action}
        def reference(target):
            with torch.no_grad():total,parts=losses(target,0,0,CONFIG['flow_seed'])
            return {'total':total.item(),**{k:v.item() for k,v in parts.items()}}
        def evaluate(target):
            fit,source=[],[]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                for i,window in enumerate(human_windows[target]):
                    _,parts=forecast_objective(encode_human(window),targets[target][i],statistics,
                        condition=args.condition,semantic_weight=weight['weight'])
                    fit.append({'provenance':window['provenance'],**{k:v.item() for k,v in parts.items()}})
                for i,window in enumerate(source_windows['validation']):
                    loss=source_action_retention(policy,encode_source(window),
                        torch.from_numpy(window['loss_targets']['actions'])[None].cuda(),seed=CONFIG['flow_seed']+1000+i)
                    source.append({**window['provenance'],'action_flow_loss':loss.item()})
            return {'human_training_window_fit':fit,'source_validation':source}
        payload={'schema_version':1,'contract':contract,'tasks':{}}
        for target in sorted(plan):
            reset_adapters(entries,initial)
            torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
            if tensor_digest(snapshot())!=initial_sha:raise ValueError('task did not reset to common initialization')
            optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'],foreach=False)
            row={'initial_adapter_sha256':initial_sha,'optimizer_steps':0}
            report['tasks'][target]=row
            if args.mode=='train':
                row['training']=[]
                start=time.monotonic()
                for step in range(CONFIG['updates']):
                    a=step%HUMAN_CONFIG['windows_per_clip'];r=(step+step//12)%12
                    if source_windows['retention'][r]['provenance']['split']!='train':
                        raise ValueError('validation cannot enter training')
                    optimizer.zero_grad(set_to_none=True)
                    total,parts=losses(target,a,r,CONFIG['flow_seed']+step)
                    if not torch.isfinite(total):raise ValueError('nonfinite human objective')
                    total.backward()
                    norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['gradient_clip'],error_if_nonfinite=True)
                    if not norm.item()>0:raise ValueError('vanishing adaptation gradient')
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):raise ValueError('nonfinite adapter')
                    row['optimizer_steps']=step+1;report['optimizer_steps']+=1
                    row['training'].append({'step':step+1,'human_window':a,'retention_window':r,'total':total.item(),
                        'gradient_norm_before_clip':norm.item(),**{k:v.item() for k,v in parts.items()}})
                    if step==0 or (step+1)%12==0:
                        print(json.dumps({'event':'human_adaptation_progress','target':target,'condition':args.condition,'step':step+1}),flush=True)
                        native_probe.write_report(report,args.output)
                    del total,parts,norm
                optimizer.zero_grad(set_to_none=True)
                adapters=snapshot()
                if tensor_digest(adapters)==initial_sha:raise ValueError('adapters unchanged after training')
                payload['tasks'][target]={'step':CONFIG['updates'],'adapters':adapters,
                    'optimizer':copy.deepcopy(optimizer.state_dict()),
                    'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'reference':reference(target)}
                row.update(training_seconds=time.monotonic()-start,adapter_sha256=tensor_digest(adapters),
                           recovery_reference=payload['tasks'][target]['reference'])
            else:
                row['frozen']=evaluate(target)
                task=saved['tasks'][target]
                if task['step']!=CONFIG['updates']:raise ValueError('recovery training budget differs')
                reset_adapters(entries,task['adapters']);optimizer.load_state_dict(task['optimizer'])
                assert_nested_equal(optimizer.state_dict(),task['optimizer'])
                assert_nested_equal(snapshot(),task['adapters'])
                torch.set_rng_state(task['rng']['torch']);torch.cuda.set_rng_state_all(task['rng']['cuda'])
                assert_nested_equal({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},task['rng'])
                actual=reference(target)
                if any(not np.isclose(actual[k],v,rtol=1e-6,atol=1e-7) for k,v in task['reference'].items()):
                    raise ValueError('fresh-process human loss reference differs')
                row.update(checkpoint_roundtrip_equal=True,recovery_reference=actual,
                           recovered_training_steps=task['step'],adapted=evaluate(target))
            native_probe.write_report(report,args.output)
            print(json.dumps({'event':'human_task_complete','target':target,'condition':args.condition,'mode':args.mode}),flush=True)
            del optimizer
        if teacher_sha()!=cache_sha:raise ValueError('immutable human targets changed')
        if any(t._version!=frozen_versions[n] for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n in frozen_versions):
            raise ValueError('frozen model state mutated')
        if any(p.grad is not None for p in policy.parameters()):raise ValueError('unexpected accumulated gradient')
        if args.mode=='train':
            destination=args.output.with_suffix('.pt')
            if destination.exists():raise FileExistsError('preserve human checkpoint')
            save_checkpoint(destination,payload)
            report['checkpoint']={'sha256':sha(destination),'bytes':destination.stat().st_size,'steps_per_task':CONFIG['updates']}
        report.update(status='passed',frozen_state_unchanged=True,teacher_targets_unchanged=True,memory=native_probe.memory())
    finally:sources.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','human-request','manifest','data-root','index','checkpoint','checkpoint-receipt','upstream-root','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mode',choices=('train','evaluate'),required=True)
    parser.add_argument('--condition',choices=('standard','semflow'),required=True)
    parser.add_argument('--reference-report',type=Path)
    parser.add_argument('--arm-checkpoint',type=Path)
    parser.add_argument('--arm-checkpoint-sha256')
    parser.add_argument('--pilot',type=Path,default=Path('configs/pilot.json'))
    parser.add_argument('--calibration',type=Path,default=Path('configs/source_calibration.json'))
    parser.add_argument('--cache-dir',type=Path,default=Path('/content/semflow-model-cache/hub'))
    parser.add_argument('--teacher-path',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('preserve human adaptation report')
    report={'status':'running','optimizer_steps':0,'rollouts':0,'no_target_robot_data':True,
            'scope':'Human one-clip forecast fitting and source retention; no held-out human or robot-transfer claim. Source controller gate remains unmet.'}
    start=time.monotonic()
    def timeout(_signum,_frame):raise TimeoutError('human adaptation stage exceeded900s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:run(args,report)
    except Exception as error:
        report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-start;native_probe.write_report(report,args.output)

if __name__=='__main__':main()
