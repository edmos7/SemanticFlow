#!/usr/bin/env python3
"""Export explicitly selected numeric evidence; never copy raw logs or paths."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def export(root,out):
    root=root.resolve(strict=True)
    mount=next((p for p in (root,*root.parents) if p.is_mount()),None)
    if mount is None or mount==Path('/'):
        raise ValueError('input data must remain on its mounted volume; no fallback')
    out.mkdir(parents=True,exist_ok=True)
    evidence={};cache={}
    def verified(run,stage,alias):
        key=(run,stage)
        if key in cache:return cache[key]
        base=root/'runs'/run;receipt=json.loads((base/('export-receipt-'+stage+'.json')).read_text())
        outer=base/stage/('source-calibration-'+stage+'.json');inner=base/stage/'calibration'/(stage+'.json')
        if sha(outer)!=receipt['report_sha256'] or sha(base/('cloud-'+stage+'.tar.gz'))!=receipt['archive_sha256']:
            raise ValueError('export receipt hash mismatch: '+alias)
        if sha(inner)!=receipt['comparison_report_sha256']:raise ValueError('inner report hash mismatch: '+alias)
        data=json.loads(inner.read_text())
        if data['status']!='passed':raise ValueError('scientific report did not pass: '+alias)
        if 'checkpoint' in data and sha(inner.with_suffix('.pt'))!=receipt['checkpoint_sha256']:
            raise ValueError('checkpoint receipt mismatch: '+alias)
        evidence[alias]={'report_sha256':sha(inner),'archive_sha256':receipt['archive_sha256'],
                         'bundle_sha256':receipt['bundle_sha256'],'stage':stage}
        cache[key]=data;return data
    human_run='2026-10-15-human-adaptation-01';panel_run='2026-10-15-human-action-panel-02';decision_run='2026-10-15-decision-diagnostics-03'
    human={arm:{mode:verified(human_run,'humanadapt-'+arm+'-'+mode,'human_'+arm+'_'+mode)
                for mode in ('train','evaluate')} for arm in ('standard','semflow')}
    panels={arm:verified(panel_run,'humanaction-'+arm,'actions_'+arm) for arm in ('frozen','standard','semflow')}
    audit=verified(decision_run,'decision-audit','decision_audit')
    retention_train=verified(decision_run,'decision-retention-train','retention_train')
    retention=verified(decision_run,'decision-retention-evaluate','retention_evaluate')
    semantic=verified(decision_run,'decision-semflow','decision')
    clock=verified('2026-10-15-sampler-clock-01','clock-audit','clock_audit')
    if not semantic['decision']['implementation_audit_passed'] or not retention['checkpoint_roundtrip_equal']:
        raise ValueError('decision or checkpoint recovery failed')
    action_rows=[];baseline=[]
    def action_panel(model,task,panel):
        if not panel['parameters_buffers_rng_unchanged']:raise ValueError('sampling state changed')
        for row in panel['panel']:
            common={'model':model,'human_task':task,'source_task':row['task'],'episode_sha256':row['episode_sha256'],'anchor':row['anchor'],'split':row['split']}
            if model=='frozen':baseline.append({**common,'full_mse':row['train_mean_baseline']['full']['mse'],
                'first_eight_mse':row['train_mean_baseline']['first_eight']['mse']})
            for sample in row['samples']:
                action_rows.append({**common,'steps':sample['integration_steps'],'noise_seed':sample['seed'],
                    'noise_sha256':sample['initial_noise_sha256'],**{h+'_'+k:sample[h][k]
                      for h in ('full','first_eight') for k in ('mse','gripper_sign_accuracy','clipped_values')}})
    action_panel('frozen','none',panels['frozen']['panels']['frozen'])
    for arm in ('standard','semflow'):
        for target,panel in panels[arm]['panels'].items():action_panel(arm,target,panel)
    action_panel('retention','none',retention)
    if len(action_rows)!=1152:raise ValueError('action sample coverage differs')
    ablations=[]
    for name,panel in {'frozen':audit['ablations'],**semantic['ablations']}.items():
        if not panel['parameters_buffers_rng_unchanged']:raise ValueError('ablation changed state')
        for row in panel['rows']:
            for variant,samples in row['samples'].items():
                for sample in samples:
                    normal=next(s for s in row['samples']['none'] if s['seed']==sample['seed'])
                    if normal['initial_noise_sha256']!=sample['initial_noise_sha256']:
                        raise ValueError('ablation initial noise differs')
                    effect={h+'_paired_action_mse':statistics.mean((a-b)**2
                        for x,y in zip(sample['normalized_prediction'][:n],normal['normalized_prediction'][:n],strict=True)
                        for a,b in zip(x,y,strict=True)) for h,n in (('full',32),('first_eight',8))}
                    ablations.append({'model':name,'source_task':row['task'],'episode_sha256':row['episode_sha256'],
                        'anchor':row['anchor'],'split':row['split'],'intervention':variant,'noise_seed':sample['seed'],**effect,
                        'noise_sha256':sample['initial_noise_sha256'],**{h+'_'+k:sample[h][k]
                         for h in ('full','first_eight') for k in ('mse','gripper_sign_accuracy','clipped_values')}})
    if len(ablations)!=960:raise ValueError('ablation coverage differs')
    fitting=[];curves=[]
    for arm,reports in human.items():
        for target,row in reports['evaluate']['tasks'].items():
            if not row['checkpoint_roundtrip_equal']:raise ValueError('human checkpoint recovery failed')
            for state in ('frozen','adapted'):
                for entry in row[state]['human_training_window_fit']:
                    fitting.append({'arm':arm,'human_task':target,'state':state,'anchor':entry['provenance']['anchor'],
                        **{k:entry[k] for k in ('raw','semantic','state','transition')}})
                    # Use distinct names for model state and semantic state loss.
                    fitting[-1]['model_state']=state
                    fitting[-1]['state_loss']=entry['state'];del fitting[-1]['state']
        for target,row in reports['train']['tasks'].items():
            for entry in row['training']:curves.append({'arm':arm,'human_task':target,**entry})
    curves += [{'arm':'retention','human_task':'none',**entry} for entry in retention_train['training']]
    manifest=json.loads((root/'datasets/human-ego-v1/ingestion-2026-10-14/manifest.json').read_text())
    clips=[{'task':c['target_id'],'frames':len(c['timestamps_seconds']),
            'duration_seconds':c['timestamps_seconds'][-1]-c['timestamps_seconds'][0],
            'source_video_sha256':c['sha256'],'viewpoint':'chest-level egocentric RGB',
            'asset':'assets/videos/human-'+c['target_id']+'.mp4'} for c in manifest['human_clips']]
    source_index=json.loads((root/'datasets/libero90-source-v1/index.json').read_text())
    evaluation=json.loads((root/'runs/2026-10-06-source-calibration-validation-01/evaluate/calibration/evaluate.json').read_text())
    sim_summary={'status':evaluation['status'],'error_type':evaluation.get('error_type'),
        'completed_source_trials':len(evaluation['rollouts']),'successful_completed_trials':sum(x['success'] for x in evaluation['rollouts']),
        'interpretation':'Incomplete source evaluation, stopped by budget during a further trial; not a completed six-task success rate. No human-adapted target rollout matrix.'}
    diag=json.loads((root/'runs/2026-10-06-controller-diagnostic-02/diagnose/calibration/diagnose.json').read_text())
    sim_summary['diagnostic_keys']=list(diag['diagnostic'])
    evidence['source_simulation']={'report_sha256':sha(root/'runs/2026-10-06-source-calibration-validation-01/evaluate/calibration/evaluate.json')}
    stages=[]
    for (run,stage),data in cache.items():
        mem=data.get('memory',{})
        stages.append({'run':run,'stage':stage,'seconds':data['seconds'],'optimizer_steps':data.get('optimizer_steps',0),
            'peak_cuda_allocated_gib':mem.get('cuda_peak_allocated_bytes',0)/2**30,
            'gpu':data.get('environment',{}).get('gpu'),'python':data.get('environment',{}).get('python'),
            'training_loop_seconds':sum(x.get('training_seconds',0.) for x in data.get('tasks',{}).values())})
    compute=[]
    for closure in sorted((root/'runs').glob('*/closure.json')):
        name=closure.parent.name
        d=json.loads(closure.read_text());initial=closure.parent/'initial-usage.log'
        final=d.get('cleanup',{}).get('final-usage',{}).get('output','')
        first=re.search(r'Current balance: ([0-9.]+) compute units',initial.read_text()) if initial.exists() else None
        last=re.search(r'Current balance: ([0-9.]+) compute units',final)
        compute.append({'run':name,'displayed_cu_difference':round(float(first[1])-float(last[1]),2) if first and last else None,
            'host_error':bool(d.get('error')),'shutdown_confirmed':d.get('cleanup',{}).get('runtime-stop',{}).get('returncode')==0,
            'scope':'Displayed balance difference for this session, not precise billed cost; missing records are null.'})
    for name,rows in (('actions.csv',action_rows),('action_baseline.csv',baseline),('ablations.csv',ablations),('human_fit.csv',fitting),('training.csv',curves),('compute.csv',compute),('stages.csv',stages)):
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=columns,lineterminator="\n");w.writeheader();w.writerows(rows)
    summary={'schema_version':1,'research_question':'Can semantic world information learned from an unlabeled human video improve an action model without human-to-robot action retargeting?',
        'operational_test':'One-clip future-feature versus semantic-state/transition adaptation, measured on forecast fitting and held-out source-action diagnostics.',
        'source_counts':source_index['counts'],'human_clips':clips,'adaptation_config':human['standard']['train']['human_request']['config'],
        'semantic_calibration':human['semflow']['train']['semantic_calibration'],
        'adaptation_parameters':sum(human['standard']['train']['trainable_parameters'].values()),
        'model_artifacts':human['standard']['train']['request'],'environment':human['standard']['train']['environment'],
        'decision':semantic['decision'],'source_simulation':sim_summary,
        'clock_audit':{'passed':clock['status']=='passed','clock_only_explains_mismatch':clock['clock_only_explains_mismatch'],
            'rows':[{k:r[k] for k in ('task','steps','previous_max_abs','matched_clock_vs_upstream_max_abs','clock_only_vs_upstream_max_abs','indexed_vs_upstream_max_abs','indexed_vs_upstream_rms')} for r in clock['rows']]},
        'constraints':{'hardware':'No robot hardware available; simulation only. Local CPU for preparation; cloud GPU for model execution.',
            'challenge_window':'Two weeks; public-repository deadline 16 October 2026 23:59 BST.',
            'main_adaptation_seed':42,'noise_seeds':[4242,4243,4244,4245],'source_steps':600,'human_steps_total':360,'retention_steps':60,
            'human_held_out_videos':0,'target_task_rollout_trials':0,'large_run_host_cap_seconds':3600,'clock_audit_cap_seconds':1200},
        'evidence':evidence}
    # Never export raw environment paths, access records, authentication URLs or logs.
    environment=summary['environment'];summary['environment']={k:environment[k] for k in ('gpu','gpu_total_bytes','python','torch_cuda','packages') if k in environment}
    write_json(out/'summary.json',summary)
    write_json(out/'MANIFEST.json',{'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='MANIFEST.json'}})
    print(json.dumps({'action_samples':len(action_rows),'ablation_samples':len(ablations),'human_fit_rows':len(fitting),'training_rows':len(curves),'reports_verified':len(evidence)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results'))
    a=p.parse_args();export(a.data_root,a.output)
if __name__=='__main__':main()
