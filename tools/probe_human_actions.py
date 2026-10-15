#!/usr/bin/env python3
"""Paired source-action retention panel after independent human adaptations."""
import argparse
import json
from pathlib import Path
import signal
import time

import probe_egosteer as native_probe
from rehearse_adaptation import sha,validate_request,load_common
from adapt_human import validate_pair,reset_adapters
from probe_adapted_actions import PANEL_CONFIG,ARMS,sample_panel,panel_identity,compare_panels
from compare_source_initializations import ObservedProcessor
from semflow.human_adaptation_plan import HUMAN_CONFIG

SUFFIXES={'checkpoint':'.pt','report':'-report.json','receipt':'-receipt.json',
          'evaluation':'-evaluation.json','evaluation_receipt':'-evaluation-receipt.json'}


def validate_training_pair(request,plan,tasks,reports,evaluations):
    targets={t['id'] for t in tasks['targets']}
    for arm in ('standard','semflow'):
        report=reports[arm];evaluation=evaluations[arm]
        if (report['status']!='passed' or report['condition']!=arm or report['mode']!='train'
                or report['optimizer_steps']!=180 or report['request']!=request or report['source_plan']!=plan
                or report['human_request']['config']!=HUMAN_CONFIG or set(report['tasks'])!=targets
                or set(report['plan'])!=targets):
            raise ValueError('human training report scope differs')
        if (evaluation['status']!='passed' or evaluation['condition']!=arm or evaluation['mode']!='evaluate'
                or evaluation['optimizer_steps']!=0 or set(evaluation['tasks'])!=targets):
            raise ValueError('human evaluation report scope differs')
        for key in ('request','human_request','plan','source_plan','initial_adapter_sha256',
                    'teacher_cache_sha256','semantic_calibration','trainable_parameters'):
            if report[key]!=evaluation[key]:raise ValueError('human recovery contract differs: '+key)
        for target in targets:
            if (report['tasks'][target]['optimizer_steps']!=60
                    or report['tasks'][target]['initial_adapter_sha256']!=report['initial_adapter_sha256']
                    or not evaluation['tasks'][target]['checkpoint_roundtrip_equal']
                    or evaluation['tasks'][target]['recovered_training_steps']!=60):
                raise ValueError('every task requires complete independent training and recovery')
        if not all(report[k] and evaluation[k] for k in ('frozen_state_unchanged','teacher_targets_unchanged')):
            raise ValueError('human frozen-state/teacher contract failed')
    validate_pair(reports['semflow'],reports['standard'])
    if any(evaluations['standard']['tasks'][t]['frozen']!=evaluations['semflow']['tasks'][t]['frozen'] for t in targets):
        raise ValueError('human arms have different frozen evaluation')
    return sorted(targets)


def validate_inputs(args):
    request,index,tasks,plan=validate_request(args)
    spec=json.loads(args.panel_request.read_text())
    if (set(spec)!={'config','source_request_sha256','artifacts'} or spec['config']!=PANEL_CONFIG
            or sha(args.request)!=spec['source_request_sha256'] or set(spec['artifacts'])!={'standard','semflow'}):
        raise ValueError('human action panel declaration differs')
    reports={};evaluations={}
    for arm in ('standard','semflow'):
        pins=spec['artifacts'][arm]
        if set(pins)!=set(SUFFIXES):raise ValueError('incomplete human artifact pins')
        paths={k:args.input_root/(arm+suffix) for k,suffix in SUFFIXES.items()}
        if any(sha(paths[k])!=pins[k] for k in pins):raise ValueError('human artifact hash mismatch')
        report=json.loads(paths['report'].read_text());receipt=json.loads(paths['receipt'].read_text())
        evaluation=json.loads(paths['evaluation'].read_text());evaluation_receipt=json.loads(paths['evaluation_receipt'].read_text())
        if (report['checkpoint']['sha256']!=pins['checkpoint']
                or report['checkpoint']['bytes']!=paths['checkpoint'].stat().st_size
                or receipt['stage']!='humanadapt-'+arm+'-train'
                or receipt['checkpoint_sha256']!=pins['checkpoint']
                or receipt['comparison_report_sha256']!=pins['report']
                or evaluation_receipt['stage']!='humanadapt-'+arm+'-evaluate'
                or evaluation_receipt['comparison_report_sha256']!=pins['evaluation']
                or evaluation_receipt['bundle_sha256']!=receipt['bundle_sha256']):
            raise ValueError('human artifact receipt differs')
        reports[arm]=report;evaluations[arm]=evaluation
    targets=validate_training_pair(request,plan,tasks,reports,evaluations)
    return request,index,tasks,plan,spec,reports,targets


def restore_task(entries,saved,training,target):
    from train_source_calibration import tensor_digest
    expected={k:training[k] for k in ('request','human_request','condition','initial_adapter_sha256',
                                    'teacher_cache_sha256','semantic_calibration')}
    if (saved['schema_version']!=1 or saved['contract']!=expected or set(saved['tasks'])!=set(training['tasks'])
            or target not in saved['tasks']):
        raise ValueError('human checkpoint contract or target differs')
    task=saved['tasks'][target]
    if task['step']!=60 or tensor_digest(task['adapters'])!=training['tasks'][target]['adapter_sha256']:
        raise ValueError('human task checkpoint identity differs')
    reset_adapters(entries,task['adapters'])
    if tensor_digest({n:p for n,p in entries})!=training['tasks'][target]['adapter_sha256']:
        raise ValueError('restored human adapters differ')


def compare_human_panels(frozen,standard,semflow,targets):
    if (frozen['status']!='passed' or frozen['condition']!='frozen' or set(frozen['panels'])!={'frozen'}
            or standard['status']!='passed' or standard['condition']!='standard'
            or semflow['status']!='passed' or semflow['condition']!='semflow'
            or set(standard['panels'])!=set(targets) or set(semflow['panels'])!=set(targets)
            or frozen['panel_request']!=standard['panel_request'] or frozen['panel_request']!=semflow['panel_request']):
        raise ValueError('incomplete or differently declared human action panels')
    comparisons={}
    for target in targets:
        if any(r['panels'][target]['adaptation_target']!=target for r in (standard,semflow)):
            raise ValueError('human adaptation target pairing differs')
        comparisons[target]=compare_panels({'frozen':frozen['panels']['frozen'],
            'standard':standard['panels'][target],'semflow':semflow['panels'][target]})
    return {'paired':True,'by_human_adaptation':comparisons,
            'interpretation':'Source-action retention after human adaptation; one adaptation seed. No target-composition rollouts or robot-transfer claim.'}


def run(args,report):
    import torch
    from semflow.source_batching import SourceWindowSampler,source_window
    from train_source_calibration import MemorySources,tensor_digest
    request,index,tasks,plan,spec,training,targets=validate_inputs(args)
    report.update(request=request,panel_request=spec,panel_config=PANEL_CONFIG,condition=args.condition,
                  panels={},human_adaptation_targets=targets)
    if not torch.cuda.is_available():raise RuntimeError('native GPU runtime required')
    policy,processor,entries,_=load_common(args,index,request,report)
    initial={n:p.detach().cpu().clone() for n,p in entries};initial_sha=tensor_digest(initial)
    if initial_sha!=training['standard']['initial_adapter_sha256']:raise ValueError('common initialization differs')
    report['initial_adapter_sha256']=initial_sha
    frozen=None
    if args.condition!='frozen':
        if args.frozen_report is None:raise ValueError('exported frozen report required')
        frozen=json.loads(args.frozen_report.read_text())
        if frozen['status']!='passed' or frozen['condition']!='frozen' or frozen['panel_request']!=spec:
            raise ValueError('frozen reference contract differs')
        saved=torch.load(args.input_root/(args.condition+'.pt'),map_location='cpu',weights_only=True)
        report['adapted_checkpoint_sha256']=spec['artifacts'][args.condition]['checkpoint']
    sources=MemorySources(index['source_lock'])
    try:
        sampler=SourceWindowSampler(index,tasks,sources.handles.__getitem__,split='validation',cache_size=2)
        rows=plan['validation'];windows=[source_window(sampler.episode(r['record']),r['anchor']) for r in rows]
        for target in (['frozen'] if args.condition=='frozen' else targets):
            reset_adapters(entries,initial)
            if tensor_digest({n:p for n,p in entries})!=initial_sha:raise ValueError('panel reset failed')
            if args.condition!='frozen':restore_task(entries,saved,training[args.condition],target)
            observed=ObservedProcessor(processor)
            sub={'request':request,'panel_config':PANEL_CONFIG,'condition':args.condition,
                 'initial_adapter_sha256':initial_sha,'adaptation_target':target,'status':'running'}
            report['panels'][target]=sub
            sample_panel(policy,observed,windows,rows,index['normalization'],sub,
                         lambda:native_probe.write_report(report,args.output))
            sub.update(processed_layouts=observed.layouts,status='passed')
            if frozen is not None and panel_identity(sub)!=panel_identity(frozen['panels']['frozen']):
                raise ValueError('human action panel inputs/layout/noise differ from frozen')
            print(json.dumps({'event':'human_action_panel_complete','condition':args.condition,'adaptation_target':target}),flush=True)
            native_probe.write_report(report,args.output)
        report.update(status='passed',memory=native_probe.memory())
        if args.condition=='semflow':
            if args.standard_report is None:raise ValueError('exported standard panel required')
            report['comparison']=compare_human_panels(frozen,json.loads(args.standard_report.read_text()),report,targets)
        if frozen is not None:report['frozen_report_sha256']=sha(args.frozen_report)
        if args.standard_report is not None:report['standard_report_sha256']=sha(args.standard_report)
    finally:sources.close()


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
    if args.output.exists():raise FileExistsError('preserve human action panel')
    report={'status':'running','optimizer_steps':0,'rollouts':0,'human_rgb_uploaded':False,
            'scope':'Held-out source actions after human adaptation; no target robot demonstrations or rollout tuning.'}
    started=time.monotonic()
    def timeout(_signum,_frame):raise TimeoutError('human action panel exceeded900s')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:run(args,report)
    except Exception as error:
        report.update(status='failed',error=str(error).split('\n')[0][:600]);raise
    finally:
        signal.alarm(0);report['seconds']=time.monotonic()-started;native_probe.write_report(report,args.output)

if __name__=='__main__':main()
