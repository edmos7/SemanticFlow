"""Fixed source-only rehearsal plan, declared before model scores."""

import hashlib
import json
import math


CONFIG = {'updates':60, 'batch_size':1, 'learning_rate':5e-5, 'weight_decay':.01,
          'gradient_clip':1., 'raw_weight':.1, 'retention_weight':1.,
          'semantic_weight_bounds':[1e-4,1.], 'seed':42, 'flow_seed':630000}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_plan(index):
    if index['manifest']['human_clips']:
        raise ValueError('source rehearsal cannot consume human clips')
    tasks = sorted(index['coverage'])
    if len(tasks) != 6:
        raise ValueError('rehearsal requires six source components')
    plan = {'adaptation':[], 'retention':[], 'validation':[]}
    for task in tasks:
        for split in ('train','validation'):
            rows = sorted((e for e in index['manifest']['episodes'] if e['task']==task and e['split']==split
                           and index['provenance'][e['sha256']]['aligned_frames']>=61),key=lambda e:e['sha256'])
            needed = 4 if split=='train' else 2
            if len(rows)<needed:
                raise ValueError('insufficient distinct source episodes for rehearsal')
            for i,record in enumerate(rows[:needed]):
                group = 'validation' if split=='validation' else 'adaptation' if i<2 else 'retention'
                frames = index['provenance'][record['sha256']]['aligned_frames']
                anchor = 20 if i%2==0 else (20+frames-41)//2
                plan[group].append({'record':record, 'anchor':anchor})
    ids = [r['record']['sha256'] for group in plan.values() for r in group]
    if len(ids)!=36 or len(set(ids))!=36:
        raise ValueError('source adaptation/retention/validation episode overlap')
    return plan


def calibrated_semantic_weight(norms):
    """Equal mean forecast-gradient norms on six source-training windows only.

    raw is already multiplied by the fixed native .1 objective multiplier.
    Bounds are declared engineering guards, not selected on target outcomes.
    """
    if len(norms)!=6 or any(set(row)!={'raw','semantic'} or
            any(not math.isfinite(v) or v<=0 for v in row.values()) for row in norms):
        raise ValueError('six finite positive source gradient pairs required')
    ratio = sum(r['raw'] for r in norms)/sum(r['semantic'] for r in norms)
    lower,upper=CONFIG['semantic_weight_bounds']
    return {'unclipped_ratio':ratio, 'weight':min(upper,max(lower,ratio)), 'bounds':[lower,upper]}
