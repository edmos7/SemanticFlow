"""Verified single-view human RGB, with causal PTS-based history and targets."""

import bisect
import hashlib
from pathlib import Path

import numpy as np

from .config import ContractError
from .data import forecast_window, validate_timestamps


HISTORY_AGES = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
OBSERVED_FIELDS = frozenset({'history_rgb', 'observation_times', 'instruction'})


def _file_digest(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024*1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def human_indices(timestamps, anchor):
    """Six distinct past/current frames and three strictly future targets."""
    future = forecast_window(timestamps, anchor)
    origin = timestamps[anchor]
    history = tuple(bisect.bisect_right(timestamps, origin - age) - 1 for age in HISTORY_AGES)
    if min(history) < 0 or len(set(history)) != 6 or history[-1] != anchor:
        raise ContractError('human window requires six distinct observations over one second')
    return history, future.target_indices


def letterbox(frame, image_size):
    """Preserve the whole scene; bilinear resize, centered black padding."""
    from PIL import Image
    height, width = image_size
    if (type(height) is not int or type(width) is not int or min(height, width) < 1
            or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3):
        raise ContractError('expected uint8 RGB and positive output dimensions')
    h, w = frame.shape[:2]
    scale = min(height / h, width / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    image = Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR)
    output = Image.new('RGB', (width, height), (0, 0, 0))
    output.paste(image, ((width-size[0])//2, (height-size[1])//2))
    return np.asarray(output).copy()


def read_human_window(data_root, record, anchor, *, image_size):
    """Hash and fully decode the file; retain only the nine selected RGB frames.

    Caller validates the episode manifest. This reader independently verifies
    path containment, the original bytes and every decoded timestamp. Storage
    policy is enforced by the host caller; cloud copies use an explicit root.
    """
    import av
    root = Path(data_root).resolve(strict=True)
    relative = Path(record['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ContractError('human video path must be data-root-relative')
    path = root / relative
    if (any(p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents)
            or not path.resolve(strict=True).is_relative_to(root)):
        raise ContractError('human video path escapes data root or uses symlinks')
    timestamps = record['timestamps_seconds']
    history, future = human_indices(timestamps, anchor)
    selected = set(history + future)
    frames, decoded = {}, []
    # Hash and decode the same open file, then rehash it to detect modification.
    with path.open('rb') as handle:
        digest = _file_digest(handle)
        if digest != record['sha256']:
            raise ContractError('human video hash differs from manifest')
        handle.seek(0)
        with av.open(handle) as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ContractError('human video requires one video stream')
            for index, frame in enumerate(container.decode(streams[0])):
                if frame.pts is None:
                    raise ContractError('human frame lacks presentation timestamp')
                decoded.append(float(frame.pts * frame.time_base))
                if index in selected:
                    frames[index] = letterbox(frame.to_ndarray(format='rgb24'), image_size)
        handle.seek(0)
        if _file_digest(handle) != digest:
            raise ContractError('human video changed while decoding')
    validate_timestamps(decoded)
    if decoded != timestamps:
        raise ContractError('decoded timestamps differ from manifest')
    origin = timestamps[anchor]
    observations = {
        'history_rgb': np.stack([frames[i] for i in history]),
        'observation_times': np.array([timestamps[i]-origin for i in history], dtype=np.float64),
        'instruction': record['instruction'],
    }
    return {'observations': observations,
            'loss_targets': {'future_rgb': np.stack([frames[i] for i in future])},
            'provenance': {'clip_sha256': digest, 'target_id': record['target_id'],
                           'anchor': anchor, 'history_indices': history, 'target_indices': future,
                           'future_offsets_seconds': [timestamps[i]-origin for i in future],
                           'preprocessing': 'bilinear whole-scene letterbox, centered black padding',
                           'image_size': list(image_size)}}


def timestamp_metadata(times):
    """Encode measured times on a microsecond grid for Qwen's index/fps API.

    fps here is a metadata tick rate, NOT an asserted camera frame rate.
    Sampling is disabled: exactly six supplied images are consumed. Qwen's
    timestamp calculation indices/fps recovers PTS to within 0.5 microseconds.
    """
    times = np.asarray(times)
    if times.shape != (6,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ContractError('six increasing measured observation times required')
    ticks = np.rint((times-times[0])*1_000_000).astype(np.int64)
    if not np.all(np.diff(ticks) > 0):
        raise ContractError('history timestamps collapse on microsecond grid')
    return {'total_num_frames': int(ticks[-1])+1, 'fps': 1_000_000.0,
            'frames_indices': ticks.tolist()}


class MeasuredHumanVisionInputs:
    """Mixin for the pinned upstream processor; no assumed constant video FPS."""

    def build_vision_inputs(self, samples):
        return {'videos': [sample['chest_images'] for sample in samples],
                'video_metadata': [timestamp_metadata(sample['observation_times']) for sample in samples],
                'do_sample_frames': False}


class HumanBatchProcessor:
    def __init__(self, native, cfg, *, processor=None):
        self.image_size = tuple(map(int, cfg.data.target_image_size))
        self.state_token = native.state_token_index
        self.action_token = native.action_token_index
        if processor is None:
            from src.dataset.qwen3_vl_batching import Qwen3VLBatchProcessor

            class MeasuredProcessor(MeasuredHumanVisionInputs, Qwen3VLBatchProcessor):
                pass

            h, w = self.image_size
            processor = MeasuredProcessor(
                model_name_or_path=str(cfg.pretrained.model_name_or_path), processor=native.backbone.processor,
                processor_call_kwargs={'min_pixels': h*w, 'max_pixels': h*w,
                    'text_kwargs': {'padding': 'longest', 'truncation': False, 'return_tensors': 'pt'}})
        self.processor = processor

    def __call__(self, observations, device):
        import torch
        from .models.native_policy import OBSERVATION_KEYS
        if not observations:
            raise ContractError('human batch cannot be empty')
        samples, messages = [], []
        for observed in observations:
            if set(observed) != OBSERVED_FIELDS:
                raise ContractError('human processor accepts observed RGB, times and instruction only')
            rgb, times = observed['history_rgb'], np.asarray(observed['observation_times'])
            if rgb.dtype != np.uint8 or rgb.shape != (6, *self.image_size, 3):
                raise ContractError('human RGB shape differs from six-frame preprocessing contract')
            timestamp_metadata(times)
            if times[-1] != 0 or np.any(times > 0):
                raise ContractError('history must end at the current forecast origin')
            instruction = observed['instruction']
            if not isinstance(instruction, str) or not instruction.strip() or '<' in instruction or '>' in instruction:
                raise ContractError('ordinary instruction cannot be empty or contain model control tokens')
            samples.append({'vision_type': 'video', 'active_views': ['chest'],
                            'chest_images': torch.from_numpy(rgb), 'observation_times': times})
            messages.append([{'role': 'user', 'content': [{'type': 'video'}, {'type': 'text', 'text':
                f'Task: {instruction} The video is from the chest viewpoint. Predict how the scene will change.'}]}])
        batch = self.processor.encode_messages(messages, samples, add_generation_prompt=True)
        batch['answer_start_idx'] = self.processor.find_answer_start_idx(batch['input_ids'])
        if set(batch) - OBSERVATION_KEYS:
            raise ContractError('native processor returned unreviewed observation fields')
        ids, mask = batch['input_ids'], batch['attention_mask'].bool()
        if (((ids == self.state_token) | (ids == self.action_token)) & mask).any().item():
            raise ContractError('human prompt contains state/action slots')
        lengths = mask.sum(-1)
        if (not torch.equal(batch['answer_start_idx'], lengths)
                or not torch.equal(mask, torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None])):
            raise ContractError('human prompt must be complete with contiguous right padding')
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key,value in batch.items()}
        current = torch.from_numpy(np.stack([o['history_rgb'][-1] for o in observations])).to(device)
        return {'batch': batch, 'current_rgb': torch.stack((torch.zeros_like(current), current), dim=1),
                'view_valid': torch.tensor([[False, True]]*len(observations), device=device),
                'observation_times': torch.from_numpy(np.stack([o['observation_times'] for o in observations])).to(device),
                'forecast_origin': torch.zeros(len(observations), device=device)}
