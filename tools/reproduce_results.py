#!/usr/bin/env python3
"""Recompute the submission's comparisons and decision using only scalar CSVs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

TARGETS=('basket_two_foods','bowl_drawer_close','mugs_two_plates')


def verify_manifest(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe evidence manifest path')
        if hashlib.sha256((root/p).read_bytes()).hexdigest()!=digest:raise ValueError('evidence hash mismatch: '+name)
    return manifest


def rows(root,name):
    with (root/name).open(newline='') as h:return list(csv.DictReader(h))


def mean(data,key):
    values=[float(r[key]) for r in data]
    if not values or not all(math.isfinite(x) for x in values):raise ValueError('missing or nonfinite evidence')
    return statistics.mean(values)


def validate_actions(data):
    expected={('frozen','none'),('retention','none')}|{(a,t) for a in ('standard','semflow') for t in TARGETS}
    if {(r['model'],r['human_task']) for r in data}!=expected:raise ValueError('action model coverage differs')
    identities=None
    for model,target in sorted(expected):
        selected=[r for r in data if (r['model'],r['human_task'])==(model,target)]
        ids=[(r['source_task'],r['episode_sha256'],r['anchor'],r['steps'],r['noise_seed'],r['noise_sha256']) for r in selected]
        if len(selected)!=144 or len(set(ids))!=144 or any(r['split']!='validation' for r in selected):raise ValueError('action sample coverage differs')
        windows={(r['source_task'],r['episode_sha256'],r['anchor']) for r in selected}
        if len(windows)!=12 or len({w[0] for w in windows})!=6 or len({w[1] for w in windows})!=12:raise ValueError('action window coverage differs')
        if {r['steps'] for r in selected}!={'5','10','20'} or {r['noise_seed'] for r in selected}!={'4242','4243','4244','4245'}:raise ValueError('action integration/noise coverage differs')
        if identities is None:identities=set(ids)
        elif identities!=set(ids):raise ValueError('action models are not paired')


def analyze(root):
    root=Path(root);verify_manifest(root);source=json.loads((root/'summary.json').read_text())
    actions=rows(root,'actions.csv');validate_actions(actions)
    ablations=rows(root,'ablations.csv');fitting=rows(root,'human_fit.csv');baseline=rows(root,'action_baseline.csv')
    result={'action_samples':len(actions),'ablation_samples':len(ablations),'actions':{},'baseline':{},'forecast_fit':{},'ablations':{}}
    for steps in ('5','10','20'):
        result['actions'][steps]={}
        for model,target in sorted({(r['model'],r['human_task']) for r in actions}):
            key=model if target=='none' else model+':'+target
            selected=[r for r in actions if (r['model'],r['human_task'],r['steps'])==(model,target,steps)]
            result['actions'][steps][key]={h:mean(selected,h+'_mse') for h in ('full','first_eight')}
    result['baseline']={h:mean(baseline,h+'_mse') for h in ('full','first_eight')}
    if len(ablations)!=960:raise ValueError('ablation sample coverage differs')
    identities=None
    for model in ('frozen',*TARGETS):
        model_rows=[r for r in ablations if r['model']==model]
        if len(model_rows)!=240 or any(r['split']!='validation' for r in model_rows):raise ValueError('ablation model coverage differs')
        ids=[(r['source_task'],r['episode_sha256'],r['anchor'],r['noise_seed'],r['noise_sha256'],r['intervention']) for r in model_rows]
        if len(set(ids))!=240:raise ValueError('duplicate ablation sample')
        if identities is None:identities=set(ids)
        elif identities!=set(ids):raise ValueError('ablation models are not paired')
        result['ablations'][model]={}
        for group in ('all',*sorted({r['source_task'] for r in model_rows})):
            selected=[r for r in model_rows if group=='all' or r['source_task']==group]
            result['ablations'][model][group]={}
            for variant in ('none','zero','static','shuffle','bypass'):
                cells=[r for r in selected if r['intervention']==variant]
                result['ablations'][model][group][variant]={h:{'mse':mean(cells,h+'_mse'),
                    'action_rms':mean(cells,h+'_paired_action_mse')**.5} for h in ('full','first_eight')}
    for arm in ('standard','semflow'):
        result['forecast_fit'][arm]={}
        for target in TARGETS:
            states={state:{loss:mean([r for r in fitting if (r['arm'],r['human_task'],r['model_state'])==(arm,target,state)],loss)
                            for loss in ('raw','semantic','state_loss','transition')} for state in ('frozen','adapted')}
            result['forecast_fit'][arm][target]=states
    control=result['actions']['10']['retention']['first_eight']
    gains={arm:{t:(control-result['actions']['10'][arm+':'+t]['first_eight'])/control for t in TARGETS} for arm in ('standard','semflow')}
    useful={}
    for model,summary in result['ablations'].items():
        normal=summary['all']['none']['first_eight']['mse'];checks={}
        for variant in ('static','shuffle'):
            alt=summary['all'][variant]['first_eight'];gain=(alt['mse']-normal)/alt['mse']
            wins=sum(s[variant]['first_eight']['mse']>s['none']['first_eight']['mse'] for task,s in summary.items() if task!='all')
            checks[variant]={'relative_gain_of_predicted':gain,'task_wins':wins,'action_rms':alt['action_rms'],
                'passed':gain>=.01 and wins>=4 and alt['action_rms']>=.001}
        useful[model]={'comparisons':checks,'passed':all(c['passed'] for c in checks.values())}
    human_gain=statistics.mean(gains['semflow'].values());human_pass=human_gain>=.01 and sum(v>0 for v in gains['semflow'].values())>=2
    recommendation='continue_with_source_controller_repair' if human_pass or any(v['passed'] for v in useful.values()) else 'pause_current_approach'
    result['decision']={'recommendation':recommendation,'mean_semflow_relative_gain':human_gain,
        'incremental_human_gain_vs_retention':gains,'human_increment_passed':human_pass,'useful_forecast_checks':useful,
        'useful_forecast_passed':any(v['passed'] for v in useful.values())}
    original=source['decision']
    if recommendation!=original['recommendation'] or not math.isclose(human_gain,original['mean_semflow_relative_gain'],abs_tol=1e-12):
        raise ValueError('recomputed decision differs from verified GPU report')
    for model,entry in useful.items():
        for variant,check in entry['comparisons'].items():
            old=original['useful_forecast_checks'][model]['comparisons'][variant]
            if check['passed']!=old['passed'] or check['task_wins']!=old['task_wins'] or not math.isclose(check['relative_gain_of_predicted'],old['relative_gain_of_predicted'],abs_tol=1e-12) or not math.isclose(check['action_rms'],old['action_rms'],abs_tol=1e-12):
                raise ValueError('recomputed forecast control differs from GPU report')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,default=Path('results'));p.add_argument('--output',type=Path)
    a=p.parse_args();result=analyze(a.results)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'verified':True,'action_samples':result['action_samples'],'ablation_samples':result['ablation_samples'],
        'decision':result['decision']['recommendation'],'human_increment_percent':100*result['decision']['mean_semflow_relative_gain']}))
if __name__=='__main__':main()
