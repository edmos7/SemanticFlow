#!/usr/bin/env python3
"""Package human video and source-component previews; never render policy results."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path):
    import av
    import numpy as np
    frames = []
    times = []
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            array = frame.to_ndarray(format='rgb24')
            time = Fraction(frame.pts) * frame.time_base
            frames.append(array)
            times.append(time)
            digest.update(np.asarray(array.shape, dtype='<i8').tobytes())
            digest.update(array.tobytes())
            digest.update(str(time).encode() + b'\n')
    return frames, times, digest.hexdigest()


def remux(source, dest):
    import av
    with av.open(str(source)) as inp, av.open(str(dest), 'w') as out:
        stream = inp.streams.video[0]
        target = out.add_stream_from_template(stream)
        target.time_base = stream.time_base
        target.metadata.clear()
        for packet in inp.demux(stream):
            if packet.dts is None:
                continue
            packet.stream = target
            out.mux(packet)


def gif(frames, times, dest, width=360):
    """Presentation-only 6 Hz preview; preserve duration with GIF centisecond rounding."""
    import bisect
    from PIL import Image, ImageOps
    end = float(times[-1])
    start = float(times[0])
    sample = [start + i / 6 for i in range(int((end - start) * 6) + 1)]
    sample_times = [float(t) for t in times]
    images = []
    for t in sample:
        frame = frames[max(0, bisect.bisect_right(sample_times, t) - 1)]
        images.append(ImageOps.contain(Image.fromarray(frame), (width, 260)))
    duration = [170 if i % 3 else 160 for i in range(len(images))]
    images[0].save(dest, save_all=True, append_images=images[1:], duration=duration, loop=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('assets/demos'))
    args = p.parse_args()
    root = args.data_root.resolve(strict=True)
    if not any(d.is_mount() and d != Path('/') for d in (root, *root.parents)):
        raise ValueError('input data must remain on a mounted data volume')
    import h5py
    from PIL import Image, ImageOps, ImageDraw, ImageFont
    from semflow.source_data import decode_episode
    tasks = json.loads(Path('configs/tasks.json').read_text())
    manifest = json.loads((root / 'datasets/human-ego-v1/ingestion-2026-10-14/manifest.json').read_text())
    index = json.loads((root / 'datasets/libero90-source-v1/index.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {'schema_version': 1, 'purpose': 'Human inputs and LIBERO-90 source demonstration previews, not adapted policy rollouts', 'tasks': [], 'files': {}}
    public = {'schema_version': 1, 'episodes': [], 'human_clips': []}
    for task in tasks['targets']:
        ident = task['id']
        record = next(c for c in manifest['human_clips'] if c['target_id'] == ident)
        source = root / record['path']
        if sha(source) != record['sha256']:
            raise ValueError('original human video hash mismatch')
        dest = args.output / f'human-{ident}.mp4'
        remux(source, dest)
        frames, times, digest = decode(source)
        new_frames, new_times, new_digest = decode(dest)
        if digest != new_digest or times != new_times or len(frames) != len(new_frames):
            raise ValueError('remux altered decoded RGB or presentation timestamps')
        del new_frames
        gif(frames, times, args.output / f'human-{ident}.gif')
        public['human_clips'].append({'id': f'human_{ident}', 'target_id': ident, 'path': dest.as_posix(), 'sha256': sha(dest), 'instruction': record['instruction'], 'viewpoint': 'egocentric', 'timestamps_seconds': [float(t) for t in times]})
        entry = {'id': ident, 'target_task': task['task'], 'human_instruction': record['instruction'], 'source_video_sha256': record['sha256'], 'packaged_video_sha256': sha(dest), 'decoded_rgb_and_pts_sha256': digest, 'frames': len(frames), 'first_to_last_pts_seconds': float(times[-1] - times[0]), 'rgb_and_pts_unchanged': True, 'audio_included': False, 'components': []}
        tiles = [ImageOps.pad(Image.fromarray(frames[int((len(frames) - 1) * f)]), (300, 220)) for f in (0, .5, 1)]
        for n, name in enumerate(task['source_tasks'], 1):
            episode = next(e for e in index['manifest']['episodes'] if e['task'] == name and e['episode_key'] == 'data/demo_0')
            with h5py.File(root / episode['path'], 'r') as handle:
                sample = decode_episode(handle, index, episode)
            rgb = sample['rgb']['agentview_rgb']
            gif(rgb, sample['timestamps_seconds'], args.output / f'libero-{ident}-component-{n}.gif', width=256)
            entry['components'].append({'task': name, 'episode_key': episode['episode_key'], 'episode_sha256': episode['sha256'], 'instruction': sample['instruction'], 'frames': len(rgb), 'fps': 20, 'view': 'agentview', 'image_orientation': 'opengl vertically corrected to upright'})
            tiles.append(ImageOps.pad(Image.fromarray(rgb[len(rgb) // 2]), (300, 220)))
        sheet = Image.new('RGB', (1500, 290), 'white')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=16)
        for n, tile in enumerate(tiles):
            sheet.paste(tile, (n * 300, 30))
            draw.text((n * 300 + 8, 8), ['Human: start', 'Human: middle', 'Human: end', 'LIBERO component 1', 'LIBERO component 2'][n], fill='black', font=font)
        draw.text((8, 264), 'Human input + source dataset examples. Not human-adapted target rollouts.', fill='black', font=font)
        sheet.save(args.output / f'{ident}-contact-sheet.png')
        provenance['tasks'].append(entry)
        print(json.dumps({'task': ident, 'human_frames_verified': len(frames), 'source_components': 2}), flush=True)
    for path in sorted(args.output.iterdir()):
        if path.is_file() and path.name != 'provenance.json':
            provenance['files'][path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    Path('configs/human_clips.json').write_text(json.dumps(public, indent=2) + '\n')


if __name__ == '__main__':
    main()
