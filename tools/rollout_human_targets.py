#!/usr/bin/env python3
"""Bounded qualitative LIBERO-10 videos from recovered task-specific adapters."""
import argparse
import base64
import json
from pathlib import Path
import select
import signal
import subprocess
import time

import numpy as np
import probe_egosteer as native_probe
from probe_human_actions import validate_inputs, restore_task
from adapt_human import reset_adapters
from rehearse_adaptation import load_common, sha
from train_source_calibration import tensor_digest
from semflow.source_batching import HISTORY_OFFSETS, panda_observation, controller_actions

CONFIG = {'seed': 10000, 'initial_state_id': 0, 'max_steps': 400, 'flow_steps': 10,
          'control_hz': 20, 'trial_timeout_seconds': 600, 'process_timeout_seconds': 2400}


def observed_window(response, history, instruction, normalization, convention):
    """Only measured cameras/proprioception and ordinary language enter policy."""
    step = response['observations'][-1]['step']
    frames = []
    for frame in response['observations']:
        rgb = np.frombuffer(base64.b64decode(frame['rgb'], validate=True), dtype=np.uint8).reshape(2, 128, 128, 3)
        observation = {k: frame[k] for k in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')}
        observation.update(agentview_image=rgb[0], robot0_eye_in_hand_image=rgb[1])
        measured = panda_observation(observation, normalization, image_convention=convention)
        history[frame['step']] = measured
        frames.append((frame['step'], measured['rgb']))
    ids = [max(0, step + int(offset)) for offset in HISTORY_OFFSETS]
    window = {'history_rgb': np.stack([history[i]['rgb'] for i in ids], axis=1),
              'current_rgb': history[step]['rgb'],
              'panda_states': np.stack([history[i]['state'] for i in ids]),
              'observation_times': (np.asarray(ids, dtype=np.float32) - step) / 20,
              'instruction': instruction}
    return window, frames


def validate_transition(response, previous_step, requested_steps):
    executed = response['step'] - previous_step
    frames = [f['step'] for f in response['observations']]
    if (not 1 <= executed <= requested_steps or response['step'] > CONFIG['max_steps']
            or frames != list(range(previous_step + 1, response['step'] + 1))
            or not np.isclose(response['simulated_seconds'], executed / 20, rtol=0, atol=1e-7)):
        raise ValueError('target RPC omitted frames or violated controller timing/step budget')


class Video:
    """Stream both upright cameras at simulator rate, retaining partial attempts."""
    def __init__(self, path):
        import av
        self.av = av
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=20)
        self.stream.width, self.stream.height = 256, 128
        self.stream.pix_fmt = 'yuv420p'
        self.count = 0

    def add(self, frames):
        for step, rgb in frames:
            if step != self.count:
                raise ValueError('video frames must cover every observed simulator step')
            frame = self.av.VideoFrame.from_ndarray(np.concatenate(rgb, axis=1), format='rgb24')
            frame.pts = step
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            self.count += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()


class Simulator:
    def __init__(self, args, directory):
        self.log = (directory / 'simulator.log').open('w')
        try:
            self.worker = subprocess.Popen([str(args.simulator_python), str(Path(__file__).with_name('source_simulator_worker.py')),
                '--libero-checkout', str(args.libero_checkout), '--config-dir', str(directory / 'libero_config'),
                '--tasks', str(args.pilot.parent / 'tasks.json'), '--target-mode',
                '--target-initial-states', str(args.target_initial_states)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        except BaseException:
            self.log.close()
            raise

    def call(self, request):
        self.worker.stdin.write(json.dumps(request) + '\n')
        self.worker.stdin.flush()
        if not select.select([self.worker.stdout], [], [], 60)[0]:
            raise TimeoutError('target simulator reply exceeded60s')
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('target simulator exited; inspect simulator.log')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['error'])
        return response

    def close(self):
        try:
            if self.worker.poll() is None:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait(timeout=15)
        finally:
            self.log.close()


def trial(args, policy, processor, index, target, directory, row, save):
    import torch
    simulator, video = None, None
    started = time.monotonic()
    previous_alarm = signal.alarm(max(1, min(CONFIG['trial_timeout_seconds'], int(args.process_deadline - started))))
    history = {}
    path = directory / 'rollout.mp4'
    try:
        simulator = Simulator(args, directory)
        reset = {'operation': 'reset', 'task': target['task'], 'seed': CONFIG['seed'], 'initial_state_id': 0}
        response = simulator.call(reset)
        repeated = simulator.call(reset)
        if response != repeated:
            raise ValueError('fixed target reset changed observations or state identity')
        row.update(initial_state_sha256=response['initial_state_sha256'], bddl_sha256=response['bddl_sha256'],
                   reset_reproduced=True, image_convention=response['image_convention'])
        convention = response['image_convention']
        lower, upper = response['lower'], response['upper']
        noise_rng = torch.Generator(device='cuda').manual_seed(CONFIG['seed'])
        video = Video(path)
        window, frames = observed_window(response, history, target['instruction'], index['normalization'], convention)
        video.add(frames)
        while row['steps'] < CONFIG['max_steps']:
            if time.monotonic() - started >= CONFIG['trial_timeout_seconds']:
                raise TimeoutError('qualitative target trial exceeded600s')
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                encoded = policy.encode_observations(**processor([window], 'cuda'))
                prediction = policy.sample_actions(encoded, torch.randn((1, 32, 7), device='cuda', generator=noise_rng), num_steps=10)
            count = min(8, CONFIG['max_steps'] - row['steps'])
            actions, clipped = controller_actions(prediction[0, :count].float().cpu().numpy(), index['normalization'], lower, upper)
            row['clipped_action_values'] += clipped
            row['attempted_steps'] += len(actions)
            save()
            response = simulator.call({'operation': 'step', 'actions': actions.tolist()})
            validate_transition(response, row['steps'], len(actions))
            row['steps'], row['simulator_success'] = response['step'], response['success']
            window, frames = observed_window(response, history, target['instruction'], index['normalization'], convention)
            video.add(frames)
            save()
            print(json.dumps({'event': 'target_rollout_progress', 'target': target['id'], 'steps': row['steps']}), flush=True)
            if response['success'] or response['done']:
                row['termination'] = 'simulator_success' if response['success'] else 'environment_done'
                break
        else:
            row['termination'] = 'step_limit'
        row['status'] = 'completed'
    except Exception as error:
        row.update(status='incomplete', termination='timeout' if isinstance(error, TimeoutError) else 'error',
                   error=str(error).split('\n')[0][:600])
    finally:
        try:
            if video is not None:
                row['video_frames'] = video.count
                video.close()
                row.update(video=path.name, video_sha256=sha(path), video_fps=20, presentation_speed=1)
        finally:
            if simulator is not None:
                simulator.close()
            row['seconds'] = time.monotonic() - started
            signal.alarm(max(1, previous_alarm - int(row['seconds'])) if previous_alarm else 0)
            save()


def prepare_initial_states(checkout, tasks, output):
    """Decode trusted pinned LIBERO files outside the torch-free simulator."""
    import torch
    from run_libero_cloud import LIBERO_SHA
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != LIBERO_SHA:
        raise ValueError('target initial states require the pinned LIBERO revision')
    output.mkdir(exist_ok=False)
    pins = {}
    for target in tasks:
        source = checkout / 'libero/libero/init_files/libero_10' / (target['task'] + '.pruned_init')
        relative = source.relative_to(checkout).as_posix()
        tracked = subprocess.check_output(['git', '-C', str(checkout), 'show', 'HEAD:' + relative])
        if source.read_bytes() != tracked:
            raise ValueError('benchmark initial-state file differs from pinned checkout')
        # Upstream files contain NumPy objects, requiring trusted pickle recovery.
        states = torch.load(source, map_location='cpu', weights_only=False)
        state = np.asarray(states[0], dtype='<f8')
        if state.ndim != 1 or not np.isfinite(state).all():
            raise ValueError('invalid fixed benchmark state')
        dest = output / (target['task'] + '.npy')
        np.save(dest, state, allow_pickle=False)
        pins[target['id']] = {'source_sha256': sha(source), 'npy_sha256': sha(dest),
                              'initial_state_id': 0, 'libero_revision': revision}
    return pins


def validate_after_reference(args, request, spec, selected):
    if args.condition != 'frozen':
        return None
    if args.after_report is None or args.after_receipt is None:
        raise ValueError('frozen comparison requires exported adapted report and receipt')
    receipt = json.loads(args.after_receipt.read_text())
    if sha(args.after_report) != receipt['comparison_report_sha256'] or receipt['stage'] != 'target-rollouts':
        raise ValueError('adapted reference receipt differs')
    after = json.loads(args.after_report.read_text())
    if (after['status'] != 'completed' or after['condition'] != 'semflow'
            or after['source_request'] != request or after['artifact_pins'] != spec['artifacts']
            or after['config'] != CONFIG or after['planned_targets'] != selected
            or after['sampler_clock'] != 'historical indexed'
            or len(after['rollouts']) != len(selected)):
        raise ValueError('adapted reference scope differs')
    rows = {r['target_id']: r for r in after['rollouts']}
    if set(rows) != set(selected) or any(r['status'] != 'completed' or not r['parameters_buffers_unchanged']
            or r['seed'] != CONFIG['seed'] or r['initial_state_id'] != 0 for r in rows.values()):
        raise ValueError('adapted trials are incomplete or differently seeded')
    return rows


def run(args, report):
    import torch
    request, index, tasks, _, spec, training, targets = validate_inputs(args)
    selected = targets if args.target == 'all' else [args.target]
    if any(t not in targets for t in selected):
        raise ValueError('unknown target')
    after = validate_after_reference(args, request, spec, selected)
    if not torch.cuda.is_available():
        raise RuntimeError('verified native GPU environment required')
    report.update(config=CONFIG, condition=args.condition, source_request=request, artifact_pins=spec['artifacts'],
                  planned_targets=selected, sampler_clock='historical indexed', rollouts=[])
    policy, processor, entries, _ = load_common(args, index, request, report)
    if tensor_digest({n:p for n,p in entries}) != training['semflow']['initial_adapter_sha256']:
        raise ValueError('common initialization differs from human training')
    initial = {n:p.detach().cpu().clone() for n,p in entries}
    saved = torch.load(args.input_root / 'semflow.pt', map_location='cpu', weights_only=True) if args.condition == 'semflow' else None
    if after is not None:
        report['adapted_reference_sha256'] = sha(args.after_report)
    args.target_initial_states = args.output.parent / 'initial-states'
    report['benchmark_initial_states'] = prepare_initial_states(args.libero_checkout,
        [t for t in tasks['targets'] if t['id'] in selected], args.target_initial_states)
    save = lambda: native_probe.write_report(report, args.output)
    for ident in selected:
        if time.monotonic() >= args.process_deadline:
            raise TimeoutError('target process budget exhausted; remaining targets unrun')
        if args.condition == 'semflow':
            restore_task(entries, saved, training['semflow'], ident)
        else:
            reset_adapters(entries, initial)
        policy.requires_grad_(False).eval()
        before = tensor_digest(dict(policy.named_parameters()) | dict(policy.named_buffers()))
        target = next(t for t in tasks['targets'] if t['id'] == ident)
        directory = args.output.parent / ident
        directory.mkdir(exist_ok=False)
        row = {'target_id': ident, 'task': target['task'], 'seed': CONFIG['seed'], 'initial_state_id': 0,
               'checkpoint_sha256': spec['artifacts']['semflow']['checkpoint'] if args.condition == 'semflow' else request['checkpoint_sha256'],
               'adapter_sha256': tensor_digest({n:p for n,p in entries}),
               'status': 'running', 'steps': 0, 'attempted_steps': 0, 'clipped_action_values': 0,
               'simulator_success': None}
        report['rollouts'].append(row)
        save()
        trial(args, policy, processor, index, target, directory, row, save)
        if tensor_digest(dict(policy.named_parameters()) | dict(policy.named_buffers())) != before:
            raise ValueError('target inference mutated model parameters or buffers')
        row['parameters_buffers_unchanged'] = True
        if after is not None:
            paired = all(row.get(k) == after[ident].get(k) for k in ('task', 'seed', 'initial_state_id', 'initial_state_sha256', 'bddl_sha256', 'image_convention'))
            row['paired_with_adapted'] = paired
            if not paired:
                raise ValueError('frozen and adapted initial conditions differ')
        save()
    report['status'] = 'completed' if all(r['status'] == 'completed' for r in report['rollouts']) else 'incomplete'
    report['benchmark_success_rate_claimed'] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('request', 'panel-request', 'input-root', 'index', 'checkpoint', 'checkpoint-receipt',
                 'upstream-root', 'simulator-python', 'libero-checkout', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--condition', choices=('semflow', 'frozen'), default='semflow')
    parser.add_argument('--after-report', type=Path)
    parser.add_argument('--after-receipt', type=Path)
    parser.add_argument('--target', default='all', choices=('all', 'basket_two_foods', 'bowl_drawer_close', 'mugs_two_plates'))
    parser.add_argument('--pilot', type=Path, default=Path('configs/pilot.json'))
    parser.add_argument('--calibration', type=Path, default=Path('configs/source_calibration.json'))
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/semflow-model-cache/hub'))
    parser.add_argument('--teacher-path', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous target attempt')
    args.output.parent.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'optimizer_steps': 0, 'scope': 'Qualitative target videos; no training or outcome-based tuning.'}
    started = time.monotonic()
    args.process_deadline = started + CONFIG['process_timeout_seconds']
    def timeout(_signum, _frame):
        raise TimeoutError('target rollout process exceeded2400s')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(CONFIG['process_timeout_seconds'])
    try:
        run(args, report)
    except Exception as error:
        report.update(status='failed', error=str(error).split('\n')[0][:600])
        raise
    finally:
        signal.alarm(0)
        report['seconds'] = time.monotonic() - started
        native_probe.write_report(report, args.output)
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
