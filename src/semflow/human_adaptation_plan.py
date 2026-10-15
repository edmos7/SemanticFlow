"""Deterministic whole-clip fitting plan; no held-out human-data claim."""

from .adaptation_plan import CONFIG
from .config import ContractError
from .human_batching import human_indices


HUMAN_CONFIG = {**CONFIG, 'windows_per_clip':12, 'independent_resets':True,
                'human_evaluation':'training-window fit only', 'rollouts':0}


def human_plan(manifest):
    clips = manifest['human_clips']
    if len(clips) != 3 or len({c['target_id'] for c in clips}) != 3:
        raise ValueError('requires three distinct one-clip tasks')
    result = {}
    for clip in sorted(clips, key=lambda c:c['target_id']):
        valid = []
        for anchor in range(len(clip['timestamps_seconds'])):
            try:
                human_indices(clip['timestamps_seconds'], anchor)
            except ContractError:
                continue
            valid.append(anchor)
        count = HUMAN_CONFIG['windows_per_clip']
        if len(valid) < count:
            raise ValueError('insufficient causal human windows')
        anchors = [valid[round(i*(len(valid)-1)/(count-1))] for i in range(count)]
        result[clip['target_id']] = {'clip_sha256':clip['sha256'], 'anchors':anchors}
    return result
