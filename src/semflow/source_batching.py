"""Source-only causal windows and measured Panda controller conversions."""

from collections import OrderedDict
import copy

import numpy as np

from .config import ContractError
from .data import validate_episode_manifest
from .source_acquisition import validate_source_lock
from .source_data import decode_episode, normalize


HISTORY_OFFSETS = np.array([-20, -16, -12, -8, -4, 0])
FUTURE_OFFSETS = np.array([10, 20, 40])
CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


class SourceWindowSampler:
    """Task-balanced, episode-balanced sampling with resumable RNG.

    `open_source(task)` must return a handle backed by the pinned file bytes.
    Only measured RGB/state enters observation windows. Future RGB and action
    labels are separate loss inputs and never enter the native prompt.
    """

    def __init__(self, index, tasks, open_source, *, split="train", seed=42, cache_size=8):
        if split not in ("train", "validation"):
            raise ContractError("source sampler accepts train/validation only")
        validate_source_lock(index["source_lock"], tasks)
        validate_episode_manifest(index["manifest"], tasks, split_seed=index["split"]["seed"],
                                  validation_fraction=index["split"]["validation_fraction"])
        if index["manifest"]["human_clips"] or index["normalization"]["fit_split"] != "train":
            raise ContractError("source sampler cannot include human/target calibration")
        expected_fit = sorted(e["sha256"] for e in index["manifest"]["episodes"] if e["split"] == "train")
        if index["normalization"]["fit_episode_sha256"] != expected_fit:
            raise ContractError("normalization provenance differs from source training split")
        self.index, self.open_source, self.split = index, open_source, split
        self.rng = np.random.default_rng(seed)
        self.cache, self.cache_size = OrderedDict(), cache_size
        self.tasks = sorted(index["coverage"])
        self.episodes = {task: [e for e in index["manifest"]["episodes"]
                               if e["task"] == task and e["split"] == split
                               and index["provenance"][e["sha256"]]["aligned_frames"] >= 61]
                         for task in self.tasks}
        if len(self.tasks) != 6 or any(not episodes for episodes in self.episodes.values()):
            raise ContractError("each source needs a complete causal training window")

    def state_dict(self):
        return {"split": self.split, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        if state["split"] != self.split:
            raise ContractError("cannot restore sampler across splits")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])

    def episode(self, record):
        key = record["sha256"]
        if key not in self.cache:
            self.cache[key] = decode_episode(self.open_source(record["task"]), self.index, record)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def sample(self, batch_size=2):
        if type(batch_size) is not int or batch_size < 1:
            raise ContractError("batch size must be positive")
        result = []
        for _ in range(batch_size):
            task = self.tasks[int(self.rng.integers(6))]
            options = self.episodes[task]
            record = options[int(self.rng.integers(len(options)))]
            episode = self.episode(record)
            anchor = int(self.rng.integers(20, len(episode["actions"]) - 40))
            result.append(source_window(episode, anchor))
        return result


def source_window(episode, anchor):
    if type(anchor) is not int or anchor < 20 or anchor + 40 >= len(episode["actions"]):
        raise ContractError("source window lacks causal history or complete future coverage")
    history, future = anchor + HISTORY_OFFSETS, anchor + FUTURE_OFFSETS
    return {"history_rgb": np.stack([episode["rgb"][key][history] for key in CAMERAS]),
            "panda_states": episode["panda_state"][history].copy(),
            "current_rgb": np.stack([episode["rgb"][key][anchor] for key in CAMERAS]),
            "observation_times": HISTORY_OFFSETS.astype(np.float32) / 20,
            "forecast_origin": 0., "instruction": episode["instruction"],
            "loss_targets": {"future_rgb": np.stack([episode["rgb"][key][future] for key in CAMERAS]),
                             "actions": episode["actions"][anchor:anchor + 32].copy()},
            "provenance": {"episode_sha256": episode["sha256"], "anchor": anchor,
                           "split": episode["split"]}}


def panda_observation(observation, statistics, *, image_convention):
    """Convert actual simulator observations, never simulator future states."""
    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float64)
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,) or gripper.shape != (2,):
        raise ContractError("Panda observation must contain 3D position, xyzw quaternion and two gripper joints")
    if not np.isfinite(quaternion).all() or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
        raise ContractError("Panda quaternion must be a finite unit quaternion")
    quaternion = -quaternion if quaternion[3] < 0 else quaternion
    state = normalize(np.concatenate((position, quaternion, gripper)), statistics["state"])
    if image_convention not in ("opengl", "opencv"):
        raise ContractError("unknown simulator image convention")
    rgb = np.stack([observation[key] for key in ("agentview_image", "robot0_eye_in_hand_image")])
    if rgb.shape != (2, 128, 128, 3) or rgb.dtype != np.uint8:
        raise ContractError("Panda cameras must return dual raw uint8 128px RGB")
    return {"rgb": rgb[:, ::-1].copy() if image_convention == "opengl" else rgb.copy(), "state": state}


def controller_actions(normalized, statistics, lower, upper):
    raw = normalize(normalized, statistics["action"], inverse=True)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (raw.ndim != 2 or raw.shape[1] != 7 or lower.shape != (7,) or upper.shape != (7,)
            or not np.array_equal(lower, -np.ones(7)) or not np.array_equal(upper, np.ones(7))):
        raise ContractError("controller conversion requires the verified seven-dimensional [-1,1] bounds")
    return np.clip(raw, lower, upper), int(np.count_nonzero((raw < lower) | (raw > upper)))


def source_success_gate(task_successes, task_trials, expected_tasks):
    if set(task_successes) != set(expected_tasks) or set(task_trials) != set(expected_tasks) or len(expected_tasks) != 6:
        raise ContractError("source gate requires exactly all six components")
    for task in expected_tasks:
        if (type(task_trials[task]) is not int or task_trials[task] < 10 or type(task_successes[task]) is not int
                or not 0 <= task_successes[task] <= task_trials[task]):
            raise ContractError("source gate needs at least ten completed trials per component")
    rates = {task: task_successes[task] / task_trials[task] for task in sorted(expected_tasks)}
    macro = sum(rates.values()) / 6
    return {"passed": macro >= .6 and min(rates.values()) >= .3, "macro": macro,
            "minimum_task": min(rates.values()), "rates": rates, "trials": dict(task_trials)}
