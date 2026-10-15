"""Observed-only EgoSteer forecasts connected to a learned Panda flow interface.

New interfaces require source calibration. Inputs/outputs use source-normalized
Panda coordinates; this module does not convert an untrained output into robot
commands. Load native weights before wrapping, then insert the shared adapters.

Native flow calling order follows EgoSteer (5e646d9), Copyright 2026 The
EgoSteer Authors, Apache-2.0; see licenses/egosteer/{LICENSE,NOTICE}.
Changes add observed-only validation, forecast conditioning and Panda maps.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ._validation import check_float, check_rows, check_valid
from .forecast import ForecastActionConditioner
from .native_forecast import NativeForecastReadout
from .scene import SceneResampler


OBSERVATION_KEYS = frozenset({
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw",
    "pixel_values_videos", "video_grid_thw", "mm_token_type_ids",
    "answer_start_idx", "camera_intrinsic",
})


@dataclass(frozen=True)
class EncodedObservation:
    batch: dict
    backbone: Any
    raw_forecasts: Tensor
    scene_forecasts: Tensor
    current_scene: Tensor
    view_valid: Tensor
    task_tokens: Tensor
    task_valid: Tensor


class NativeSemFlowPolicy(nn.Module):
    """Reuse native experts and add source-trained 9D state / 7D action maps.

    Panda state convention is eef position(3), quaternion xyzw(4), gripper(2).
    Normalize using source training statistics before this interface. Human
    videos omit state entirely. All three experimental arms use this same
    architecture and must start from the same source-calibrated checkpoint.
    """

    def __init__(self, native: nn.Module, scene_dim: int = 256):
        super().__init__()
        if native.action_horizon != 32 or native.num_state_tokens != 6:
            raise ValueError("the pinned native interface requires 32 actions and 6 history frames")
        if any(expert.detach_prefix_kv for expert in (native.flow_expert, native.world_model_expert)):
            raise ValueError("native experts must preserve prediction-side prefix gradients")
        self.native = native
        feature_dim = native.world_model_expert.hidden_size
        self.forecast = NativeForecastReadout(feature_dim)
        self.scene = SceneResampler(feature_dim, native.vlm_hidden_size, dim=scene_dim, num_queries=16)
        self.conditioner = ForecastActionConditioner(native.action_hidden_size, scene_dim, scene_dim)
        self.panda_state = nn.Linear(9, native.state_dim)
        self.panda_action_in = nn.Linear(7, native.action_dim)
        self.panda_action_out = nn.Linear(native.action_dim, 7)
        native.frozen_teacher.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.native.frozen_teacher.eval()
        return self

    def _validate_observations(self, batch: dict, observation_times: Tensor,
                               forecast_origin: Tensor, panda_states: Tensor | None) -> None:
        unexpected = set(batch) - OBSERVATION_KEYS
        if unexpected:
            raise ValueError(f"runtime batch accepts observed inputs only; unexpected fields: {sorted(unexpected)}")
        ids, mask = batch["input_ids"], batch["attention_mask"]
        if ids.ndim != 2 or mask.shape != ids.shape or ids.dtype != torch.long:
            raise ValueError("input_ids and attention_mask must share [B,L] shape")
        if not ((mask == 0) | (mask == 1)).all().item():
            raise ValueError("attention_mask must be binary")
        valid = mask.bool()
        check_rows("attention_mask", valid)
        lengths = valid.sum(-1)
        # Native prefix lengths assume right padding, not internal gaps.
        expected = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        if not torch.equal(valid, expected) or not torch.equal(batch["answer_start_idx"], lengths):
            raise ValueError("runtime requires a complete prompt only, with contiguous right padding")
        if ((ids == self.native.action_token_index) & valid).any().item():
            raise ValueError("action target slots cannot enter the observed prefix")
        check_float("observation_times", observation_times, 2)
        check_float("forecast_origin", forecast_origin, 1)
        if observation_times.shape != (ids.shape[0], 6) or forecast_origin.shape != (ids.shape[0],):
            raise ValueError("exactly six causal observation timestamps are required")
        if observation_times.device != ids.device or forecast_origin.device != ids.device:
            raise ValueError("observation timestamps must share the prefix device")
        if (observation_times > forecast_origin[:, None]).any().item() or (observation_times[:, 1:] < observation_times[:, :-1]).any().item():
            raise ValueError("observation timestamps must be ordered and never in the future")
        if not torch.equal(observation_times[:, -1], forecast_origin):
            raise ValueError("current RGB must correspond to the final observation at forecast origin")
        state_slots = ((ids == self.native.state_token_index) & valid).sum(-1)
        if panda_states is None:
            if state_slots.any().item():
                raise ValueError("state slots require actual Panda states; missing human state stays absent")
        else:
            check_float("panda_states", panda_states, 3)
            if panda_states.shape != (ids.shape[0], 6, 9) or panda_states.device != ids.device:
                raise ValueError("Panda states must be source-normalized [B,6,9] on the prefix device")
            if not (state_slots == 6).all().item():
                raise ValueError("Panda observations need exactly six state slots in the prompt")

    def pool_raw(self, raw: Tensor, view_valid: Tensor, task_tokens: Tensor,
                 task_valid: Tensor) -> Tensor:
        """Pool all valid scene patches; preserve raw scale outside this readout.

        Input [B,2,H,P,D], output [B,H,16,scene_dim]. For loss targets call under
        no_grad with detached context; prediction calls must remain differentiable.
        """
        check_float("raw features", raw, 5)
        batch, views, horizons, patches, dim = raw.shape
        if views != 2:
            raise ValueError("native raw features must retain both view slots")
        check_valid("view_valid", view_valid, (batch, 2), raw.device)
        check_rows("view_valid", view_valid)
        valid = view_valid[:, None, :, None].expand(-1, horizons, -1, patches).reshape(batch * horizons, -1)
        features = raw.permute(0, 2, 1, 3, 4).reshape(batch * horizons, views * patches, dim)
        task = task_tokens[:, None].expand(-1, horizons, -1, -1).flatten(0, 1)
        mask = task_valid[:, None].expand(-1, horizons, -1).flatten(0, 1)
        pooled = self.scene(features.to(task.dtype), task, patch_valid=valid, task_valid=mask)
        return pooled.reshape(batch, horizons, self.scene.num_queries, self.scene.dim)

    def encode_observations(self, batch: dict, current_rgb: Tensor, view_valid: Tensor, *,
                            observation_times: Tensor, forecast_origin: Tensor,
                            panda_states: Tensor | None = None) -> EncodedObservation:
        """Encode six past/current frames; current_rgb is their final raw frame.

        Future frames/labels and their validity cannot be supplied to this path.
        Current-image DINO extraction is frozen; pooling forecasts is not detached.
        """
        self._validate_observations(batch, observation_times, forecast_origin, panda_states)
        batch_size = batch["input_ids"].shape[0]
        if current_rgb.ndim != 5 or current_rgb.shape[:2] != (batch_size, 2) or current_rgb.shape[-1] != 3 or current_rgb.dtype != torch.uint8:
            raise ValueError("current_rgb must be raw uint8 [B,2,H,W,3]")
        if current_rgb.device != batch["input_ids"].device:
            raise ValueError("current RGB must share the prefix device")
        check_valid("view_valid", view_valid, (batch_size, 2), current_rgb.device)
        check_rows("view_valid", view_valid)
        working = dict(batch)
        slots = self.native.build_slot_embeddings(working)
        if panda_states is not None:
            slots["state"] = self.native.state_encoder(self.panda_state(panda_states))
        backbone = self.native.forward_backbone_stream(working, slots)
        task = backbone.last_hidden_states
        # Task context consists only of observed prompt tokens, including text.
        task_valid = batch["attention_mask"].bool()
        forecast = self.forecast(self.native, working, backbone, view_valid)
        predicted_scene = self.pool_raw(forecast["pred"], view_valid, task, task_valid)
        with torch.no_grad():
            current = self.native.frozen_teacher(current_rgb.flatten(0, 1).unsqueeze(1))
        current = current.reshape(batch_size, 2, 1, *current.shape[-2:])
        current_scene = self.pool_raw(current, view_valid, task, task_valid)[:, 0]
        return EncodedObservation(working, backbone, forecast["pred"], predicted_scene,
                                  current_scene, view_valid, task, task_valid)

    @staticmethod
    def intervened_forecasts(observation: EncodedObservation, intervention: str) -> Tensor:
        if intervention == "none":
            return observation.scene_forecasts
        if intervention == "zero":
            return torch.zeros_like(observation.scene_forecasts)
        if intervention == "static":
            return observation.current_scene[:, None].expand_as(observation.scene_forecasts)
        if intervention == "shuffle":
            return observation.scene_forecasts[:, [2, 0, 1]]
        raise ValueError("forecast intervention must be none, zero, static or shuffle")

    def predict_velocity(self, observation: EncodedObservation, noisy_actions: Tensor,
                         flow_time: Tensor, *, forecast_intervention: str = "none") -> Tensor:
        """Native flow expert, driven by predicted scenes, returns [B,32,7]."""
        check_float("noisy Panda actions", noisy_actions, 3)
        batch, count, dim = noisy_actions.shape
        if (count, dim) != (32, 7) or batch != observation.raw_forecasts.shape[0]:
            raise ValueError("Panda flow input must be [B,32,7]")
        check_float("flow_time", flow_time, 2)
        if flow_time.shape != (batch, count) or flow_time.device != noisy_actions.device:
            raise ValueError("flow_time must share [B,32] shape and device with actions")
        if ((flow_time < 0) | (flow_time > 1)).any().item():
            raise ValueError("flow_time must be within [0,1]")
        native = self.native
        action = native.action_encoder(self.panda_action_in(noisy_actions))
        futures = self.intervened_forecasts(observation, forecast_intervention)
        action = self.conditioner(action, observation.current_scene.to(action.dtype), futures.to(action.dtype),
            current_valid=torch.ones(observation.current_scene.shape[:2], dtype=torch.bool, device=action.device),
            future_valid=torch.ones((batch, 3), dtype=torch.bool, device=action.device))
        position = native.build_action_position_ids(observation.batch, observation.backbone, noisy_actions)
        time = native.time_embedding(flow_time.flatten()).reshape(batch, count, -1)
        hidden = native.flow_expert(suffix_embeds=action, prefix_cache=observation.backbone.prefix_cache,
            suffix_position_ids=position, cond=time,
            suffix_mask=torch.ones((batch, count), dtype=torch.bool, device=action.device), num_parallel_chunks=1)
        result = self.panda_action_out(native.action_decoder(hidden))
        check_float("Panda velocity", result, 3)
        return result

    @torch.no_grad()
    def sample_actions(self, observation: EncodedObservation, initial_noise: Tensor, *,
                       num_steps: int = 10, forecast_intervention: str = "none") -> Tensor:
        """Return a normalized 32-step chunk using paired caller-supplied noise.

        Source normalization/controller conversion is required before executing
        the first eight actions. This function performs no simulator interaction.
        """
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or not 1 <= num_steps <= 100:
            raise ValueError("num_steps must be an integer from 1 to 100")
        actions = initial_noise.clone()
        for index in range(num_steps):
            time = torch.full(actions.shape[:2], index / num_steps, device=actions.device, dtype=actions.dtype)
            velocity = self.predict_velocity(observation, actions, time, forecast_intervention=forecast_intervention)
            actions = actions + velocity / num_steps
        check_float("sampled Panda actions", actions, 3)
        return actions
