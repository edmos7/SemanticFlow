#!/usr/bin/env python3
"""Build labeled target-policy previews; require matched conditions for pairing."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np
from PIL import Image, ImageDraw
from build_demo_showcase import decode, gif


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(root, condition):
    report = json.loads((root / 'report.json').read_text())
    if report['status'] != 'completed' or report['condition'] != condition or report['optimizer_steps'] != 0:
        raise ValueError('requires completed inference-only ' + condition + ' run')
    rows = {r['target_id']: r for r in report['rollouts']}
    if set(rows) != set(report['planned_targets']) or len(rows) != 3:
        raise ValueError('requires exactly three target attempts')
    videos = {}
    for target, row in rows.items():
        path = root / target / row['video']
        if sha(path) != row['video_sha256'] or row['status'] != 'completed':
            raise ValueError('incomplete or modified video')
        frames, times, _ = decode(path)
        if len(frames) != row['steps'] + 1 or any(abs(float(t) - i / 20) > 1e-7 for i, t in enumerate(times)):
            raise ValueError('video coverage/timing differs')
        videos[target] = (frames, times)
    return report, rows, videos


def validate_pair(before, after, before_rows, after_rows):
    for key in ('config', 'source_request', 'artifact_pins', 'planned_targets', 'sampler_clock'):
        if before[key] != after[key]:
            raise ValueError('before/after contract differs: ' + key)
    for target in after_rows:
        for key in ('task', 'seed', 'initial_state_id', 'initial_state_sha256', 'bddl_sha256', 'image_convention'):
            if before_rows[target][key] != after_rows[target][key]:
                raise ValueError('before/after initial conditions differ: ' + key)
        if not before_rows[target]['paired_with_adapted']:
            raise ValueError('frozen run did not verify pairing')


def labeled(frame, text):
    image = Image.new('RGB', (frame.shape[1], frame.shape[0] + 28), 'white')
    image.paste(Image.fromarray(frame), (0, 28))
    ImageDraw.Draw(image).text((6, 8), text, fill='black')
    return np.asarray(image)


def encode(path, frames):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=20)
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        for i, array in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(array, format='rgb24')
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-run', type=Path, required=True)
    p.add_argument('--before-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    after, after_rows, after_videos = load_run(args.after_run, 'semflow')
    before = before_rows = before_videos = None
    if args.before_run:
        before, before_rows, before_videos = load_run(args.before_run, 'frozen')
        validate_pair(before, after, before_rows, after_rows)
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'purpose': 'Qualitative learned-policy target attempts; not a benchmark rate',
                  'paired': before is not None, 'presentation_fps': 6, 'mp4_fps': 20, 'targets': {}, 'files': {}}
    for target in after['planned_targets']:
        frames, times = after_videos[target]
        shutil.copyfile(args.after_run / target / after_rows[target]['video'], args.output / (target + '-after.mp4'))
        shown = [labeled(f, 'After human adaptation (SemFlow)') for f in frames]
        if before is not None:
            earlier, earlier_times = before_videos[target]
            # Fixed-step attempts should match duration. Never stretch time or silently hold a terminal frame.
            if times != earlier_times:
                raise ValueError('paired video durations differ; explicit terminal-frame handling required')
            shown = [np.concatenate((labeled(b, 'Before human adaptation (Base600)'), a), axis=1) for b, a in zip(earlier, shown)]
            encode(args.output / (target + '-comparison.mp4'), shown)
            shutil.copyfile(args.before_run / target / before_rows[target]['video'], args.output / (target + '-before.mp4'))
        gif(shown, times, args.output / (target + ('-comparison.gif' if before else '-after.gif')), width=720 if before else 360)
        selected = [shown[i] for i in (0, 100, 200, 300, 400) if i < len(shown)]
        sheet = Image.new('RGB', (shown[0].shape[1] * len(selected), shown[0].shape[0] + 24), 'white')
        for i, frame in enumerate(selected):
            sheet.paste(Image.fromarray(frame), (i * frame.shape[1], 0))
            ImageDraw.Draw(sheet).text((i * frame.shape[1] + 6, frame.shape[0] + 6), f't={i*5}s', fill='black')
        sheet.save(args.output / (target + '-contact-sheet.png'))
        keys = ('task', 'seed', 'initial_state_id', 'checkpoint_sha256', 'adapter_sha256', 'video_sha256', 'steps', 'termination', 'simulator_success')
        provenance['targets'][target] = {'after': {k: after_rows[target][k] for k in keys}}
        if before:
            provenance['targets'][target]['before'] = {k: before_rows[target][k] for k in keys}
    for path in sorted(args.output.iterdir()):
        provenance['files'][path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
