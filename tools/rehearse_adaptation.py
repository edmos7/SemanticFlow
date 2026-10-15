#!/usr/bin/env python3
"""Source-only matched adaptation rehearsal, exported before fresh recovery."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import probe_egosteer as native_probe

from semflow.adaptation_plan import CONFIG, calibrated_semantic_weight, digest, source_plan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_request(args):
    from semflow.config import load_config
    from semflow.data import validate_episode_manifest
    from semflow.source_acquisition import validate_source_lock
    request = json.loads(args.request.read_text())
    if set(request) != {'source_index_sha256','checkpoint_sha256','checkpoint_receipt_sha256','plan_sha256','config'}:
        raise ValueError('unexpected rehearsal request fields')
    for path,key in ((args.index,'source_index_sha256'),(args.checkpoint,'checkpoint_sha256'),
                     (args.checkpoint_receipt,'checkpoint_receipt_sha256')):
        if sha(path)!=request[key]: raise ValueError('rehearsal input hash mismatch: '+key)
    receipt=json.loads(args.checkpoint_receipt.read_text())
    if receipt.get('stage')!='train' or receipt.get('checkpoint_sha256')!=request['checkpoint_sha256']:
        raise ValueError('original training receipt required')
    index=json.loads(args.index.read_text())
    _,tasks=load_config(args.pilot)
    validate_source_lock(index['source_lock'],tasks)
    validate_episode_manifest(index['manifest'],tasks)
    plan=source_plan(index)
    if digest(plan)!=request['plan_sha256'] or request['config']!=CONFIG:
        raise ValueError('rehearsal plan/configuration differs')
    return request,index,tasks,plan


def validate_matched_reference(report, reference):
    if reference.get('status') != 'passed' or reference.get('condition') != 'standard' or reference.get('mode') != 'train':
        raise ValueError('passed standard training reference required')
    for key in ('request','plan','initial_adapter_sha256','teacher_cache_sha256','semantic_calibration','trainable_parameters'):
        if report[key] != reference[key]:
            raise ValueError('matched rehearsal reference differs: '+key)


def load_common(args,index,request,report, *, include_human_processor=False):
    import torch
    from semflow.models import NativeSemFlowPolicy, inject_shared_lora, enable_shared_adapter_training
    from semflow.models.losses import SourceFeatureStatistics
    from semflow.models.source_calibration import SpatialReconstruction, enable_source_training, trainable_delta
    from semflow.native_batching import PandaBatchProcessor
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    contract={'base_artifacts':native_probe.ARTIFACTS,'upstream':native_probe.UPSTREAM_REVISION,
              'source_index_sha256':request['source_index_sha256'],'normalization':index['normalization'],
              'configuration':json.loads(args.calibration.read_text())}
    if saved['schema_version']!=1 or saved['step']!=600 or saved['contract']!=json.loads(json.dumps(contract)):
        raise ValueError('requires original Base step600 with exact source contract')
    stats=saved['source_feature_statistics']
    train={e['sha256'] for e in index['manifest']['episodes'] if e['split']=='train'}
    if (stats['fit_split']!='train' or stats['source_manifest']!=request['source_index_sha256']
            or not stats['windows'] or any(w['split']!='train' or w['episode_sha256'] not in train for w in stats['windows'])):
        raise ValueError('invalid source feature statistics provenance')
    statistics=SourceFeatureStatistics(stats['mean'],stats['scale'],stats['source_manifest'])
    report['environment']=native_probe.environment()
    native_probe.access_preflight(report,args)
    native_probe.prepare_native_imports(args)
    paths=native_probe.download_models(report,args)
    torch.manual_seed(CONFIG['seed']);torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    native,cfg=native_probe.load_model(report,args,paths)
    inject_shared_lora(native)
    policy=NativeSemFlowPolicy(native).cuda()
    reconstruction=SpatialReconstruction(policy.scene.dim,native.world_model_expert.hidden_size).cuda()
    enable_source_training(policy,reconstruction)
    expected=set(trainable_delta(policy));state=policy.state_dict()
    if set(saved['model_delta'])!=expected or saved['trainable_names']!=[n for n,p in policy.named_parameters() if p.requires_grad]:
        raise ValueError('incomplete common-source delta')
    for name,value in saved['model_delta'].items():
        if value.shape!=state[name].shape or value.dtype!=state[name].dtype or not torch.isfinite(value).all():
            raise ValueError('invalid common-source tensor: '+name)
    policy.load_state_dict(saved['model_delta'],strict=False)
    del saved,state,reconstruction
    entries=enable_shared_adapter_training(policy,policy=native)
    result = policy,PandaBatchProcessor(native,cfg),entries,statistics
    if include_human_processor:
        from semflow.human_batching import HumanBatchProcessor
        return (*result, HumanBatchProcessor(native,cfg))
    return result


def run(args,report):
    import torch
    from semflow.models.adaptation import fixed_forecast_targets, forecast_objective, source_action_retention
    from semflow.source_batching import SourceWindowSampler,source_window
    from train_source_calibration import MemorySources,save_checkpoint,tensor_digest
    from compare_source_initializations import assert_nested_equal
    request,index,tasks,plan=validate_request(args)
    report.update(condition=args.condition,mode=args.mode,request=request,plan=plan,optimizer_steps=0,rollouts=0)
    if not torch.cuda.is_available(): raise RuntimeError('rehearsal requires verified native GPU environment')
    policy,processor,entries,statistics=load_common(args,index,request,report)
    parameters=[p for _,p in entries]
    report['trainable_parameters']={n:p.numel() for n,p in entries}
    initial_adapter={n:p.detach().cpu().clone() for n,p in entries}
    initial_sha=tensor_digest(initial_adapter)
    frozen_versions={n:t._version for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n not in initial_adapter}
    optimizer=torch.optim.AdamW(parameters,lr=CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay'],foreach=False)
    sources=MemorySources(index['source_lock'])
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,cache_size=4)
        windows={group:[source_window(sampler.episode(item['record']),item['anchor']) for item in rows]
                 for group,rows in plan.items()}
        targets={}
        def encode(window):
            observed={k:v for k,v in window.items() if k!='loss_targets'}
            return policy.encode_observations(**processor([observed],'cuda'))
        # This happens only at the common source initialization, including in
        # recovery. Targets are never regenerated after loading adapted weights.
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for group in ('adaptation','validation'):
                targets[group]=[]
                for window in windows[group]:
                    observed=encode(window)
                    rgb=torch.from_numpy(window['loss_targets']['future_rgb'])[None].cuda()
                    targets[group].append(fixed_forecast_targets(policy,observed,rgb))
        def target_sha():
            return tensor_digest({f'{g}/{i}/{k}':v for g,rows in targets.items() for i,row in enumerate(rows) for k,v in row.items()})
        teacher_sha=target_sha()
        norms=[]
        for i in range(0,12,2):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                observation=encode(windows['adaptation'][i])
                _,components=forecast_objective(observation,targets['adaptation'][i],statistics,
                    condition='semflow',semantic_weight=1.)
            pair={}
            for label,loss in (('raw',.1*components['raw']),('semantic',components['semantic'])):
                grads=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
                if any(g is not None and not torch.isfinite(g).all() for g in grads): raise ValueError('nonfinite calibration gradient')
                pair[label]=sum(g.float().square().sum().item() for g in grads if g is not None)**.5
            norms.append(pair)
            del observation,components,loss,grads,_
        weight=calibrated_semantic_weight(norms)
        report.update(initial_adapter_sha256=initial_sha,teacher_cache_sha256=teacher_sha,
                      semantic_calibration=dict(weight,source_training_gradient_norms=norms))
        if args.condition == 'semflow':
            if args.reference_report is None:
                raise ValueError('SemFlow requires exported standard arm reference before updates')
            validate_matched_reference(report,json.loads(args.reference_report.read_text()))
            report['matched_reference_sha256']=sha(args.reference_report)
        def losses(window,target,retention,seed):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                observation=encode(window)
                forecast,parts=forecast_objective(observation,target,statistics,
                    condition=args.condition,semantic_weight=weight['weight'])
                retain=encode(retention)
                actions=torch.from_numpy(retention['loss_targets']['actions'])[None].cuda()
                action=source_action_retention(policy,retain,actions,seed=seed)
                total=forecast+action
            return total,{**parts,'retention':action}
        def reference():
            with torch.no_grad():
                total,parts=losses(windows['adaptation'][0],targets['adaptation'][0],windows['retention'][0],CONFIG['flow_seed'])
            return {'total':total.item(),**{k:v.item() for k,v in parts.items()}}
        checkpoint_contract={'request':request,'condition':args.condition,'teacher_cache_sha256':teacher_sha,
                             'initial_adapter_sha256':initial_sha,'semantic_calibration':report['semantic_calibration']}
        if args.mode=='train':
            report['training']=[]
            training_start=time.monotonic()
            for step in range(CONFIG['updates']):
                a=step%12;r=(step+step//12)%12
                if windows['adaptation'][a]['provenance']['split']!='train' or windows['retention'][r]['provenance']['split']!='train':
                    raise ValueError('validation cannot enter an optimizer step')
                optimizer.zero_grad(set_to_none=True)
                total,parts=losses(windows['adaptation'][a],targets['adaptation'][a],windows['retention'][r],CONFIG['flow_seed']+step)
                if not torch.isfinite(total): raise ValueError('nonfinite adaptation objective')
                total.backward()
                norm=torch.nn.utils.clip_grad_norm_(parameters,CONFIG['gradient_clip'],error_if_nonfinite=True)
                if not norm.item()>0: raise ValueError('adaptation gradient vanished')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters): raise ValueError('nonfinite adapted parameter')
                report['optimizer_steps']=step+1
                report['training'].append({'step':step+1,'adaptation_window':a,'retention_window':r,
                    'total':total.item(),'gradient_norm_before_clip':norm.item(),**{k:v.item() for k,v in parts.items()}})
                if (step+1)%12==0:
                    print(json.dumps({'event':'rehearsal_progress','condition':args.condition,'step':step+1}),flush=True)
                    native_probe.write_report(report,args.output)
                del total,parts,norm
            optimizer.zero_grad(set_to_none=True)
            report['training_seconds']=time.monotonic()-training_start
            adapters={n:p.detach().cpu().clone() for n,p in entries}
            if tensor_digest(adapters)==initial_sha: raise ValueError('optimizer did not change shared adapters')
            payload={'schema_version':1,'contract':checkpoint_contract,'step':CONFIG['updates'],
                     'adapters':adapters,'optimizer':copy.deepcopy(optimizer.state_dict()),
                     'rng':{'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},'reference':reference()}
            destination=args.output.with_suffix('.pt')
            if destination.exists(): raise FileExistsError('preserve rehearsal checkpoint')
            save_checkpoint(destination,payload)
            report['checkpoint']={'sha256':sha(destination),'bytes':destination.stat().st_size,'step':CONFIG['updates']}
            report['recovery_reference']=payload['reference']
        else:
            if args.arm_checkpoint is None or sha(args.arm_checkpoint)!=args.arm_checkpoint_sha256:
                raise ValueError('fresh recovery requires hash-verified exported arm checkpoint')
            saved=torch.load(args.arm_checkpoint,map_location='cpu',weights_only=True)
            if saved['schema_version']!=1 or saved['contract']!=checkpoint_contract or saved['step']!=CONFIG['updates']:
                raise ValueError('recovery common initialization/teacher/plan contract differs')
            def evaluate():
                rows=[]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    for i,window in enumerate(windows['validation']):
                        observation=encode(window)
                        _,parts=forecast_objective(observation,targets['validation'][i],statistics,
                            condition=args.condition,semantic_weight=weight['weight'])
                        action=source_action_retention(policy,observation,
                            torch.from_numpy(window['loss_targets']['actions'])[None].cuda(),seed=CONFIG['flow_seed']+1000+i)
                        rows.append({'task':plan['validation'][i]['record']['task'],**window['provenance'],
                                     **{k:v.item() for k,v in parts.items()},'action_flow_loss':action.item()})
                return rows
            report['frozen_validation']=evaluate()
            if set(saved['adapters'])!=set(initial_adapter): raise ValueError('recovery adapter scope differs')
            with torch.no_grad():
                for name,parameter in entries:
                    value=saved['adapters'][name]
                    if value.shape!=parameter.shape or value.dtype!=parameter.dtype or not torch.isfinite(value).all():
                        raise ValueError('invalid adapted tensor')
                    parameter.copy_(value)
            optimizer.load_state_dict(saved['optimizer'])
            assert_nested_equal(optimizer.state_dict(),saved['optimizer'])
            torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
            assert_nested_equal({n:p.detach().cpu() for n,p in entries},saved['adapters'])
            assert_nested_equal({'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()},saved['rng'])
            actual=reference()
            if any(not np.isclose(actual[k],v,rtol=1e-6,atol=1e-7) for k,v in saved['reference'].items()):
                raise ValueError('fresh-process loss reference differs')
            report.update(checkpoint_roundtrip_equal=True,recovery_reference=actual,optimizer_steps=0,
                          recovered_training_steps=saved['step'],adapted_validation=evaluate())
        if target_sha()!=teacher_sha: raise ValueError('immutable teacher cache changed')
        if any(t._version!=frozen_versions[n] for n,t in (*policy.named_parameters(),*policy.named_buffers()) if n in frozen_versions):
            raise ValueError('frozen model state mutated')
        if any(p.grad is not None for p in policy.parameters()): raise ValueError('unexpected accumulated gradient after cleanup')
        report.update(status='passed',frozen_state_unchanged=True,teacher_targets_unchanged=True,memory=native_probe.memory())
    finally:
        sources.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('request','index','checkpoint','checkpoint-receipt','upstream-root','output'):
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
    if args.output.exists(): raise FileExistsError('preserve existing rehearsal result')
    report={'status':'running','no_target_data':True,'rollouts':0,'optimizer_steps':0,
            'scope':'Source-only engineering rehearsal; not source controller gate or target adaptation.'}
    start=time.monotonic()
    def timeout(_signum,_frame): raise TimeoutError('source rehearsal stage exceeded900s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try: run(args,report)
    except Exception as error:
        report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-start;native_probe.write_report(report,args.output)


if __name__=='__main__': main()
