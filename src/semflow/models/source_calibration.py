"""Source-only action utility, spatial reconstruction, and non-collapse losses."""

import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapters import named_shared_adapter_parameters
from .losses import raw_future_feature_loss


class SpatialReconstruction(nn.Module):
    """Reconstruct DINO patches from scene tokens plus position/view queries.

    Queries contain no patch targets. This prevents an identity bypass of the
    16-token scene bottleneck. Targets retain their original DINO feature scale.
    """

    def __init__(self, scene_dim, patch_dim, patches=576):
        super().__init__()
        self.patches = patches
        self.queries = nn.Parameter(torch.randn(2 * patches, scene_dim) * scene_dim**-0.5)
        self.attention = nn.MultiheadAttention(scene_dim, 4, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(scene_dim), nn.Linear(scene_dim, patch_dim))

    def forward(self, scene):
        query = self.queries[None].expand(len(scene), -1, -1)
        decoded, _ = self.attention(query, scene, scene, need_weights=False)
        return self.head(decoded + query)


def noncollapse_loss(scene, epsilon=1e-4):
    """Variance across observations after query averaging, not across query IDs."""
    if scene.ndim != 3 or len(scene) < 2:
        raise ValueError("non-collapse objective requires at least two source observations")
    values = scene.float().mean(1)
    centered = values - values.mean(0)
    std = (centered.square().mean(0) + epsilon).sqrt()
    variance = F.relu(1 - std).mean()
    covariance = centered.T @ centered / max(1, len(values) - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diag())
    return variance + 0.01 * off_diagonal.square().sum() / values.shape[-1]


def enable_source_training(policy, reconstruction):
    """Freeze the pinned native base; train all added interfaces and shared LoRA."""
    policy.requires_grad_(False).eval()
    names = ("forecast", "scene", "conditioner", "panda_state", "panda_action_in", "panda_action_out")
    for name in names:
        getattr(policy, name).requires_grad_(True)
    for _, parameter in named_shared_adapter_parameters(policy.native):
        parameter.requires_grad_(True)
    reconstruction.requires_grad_(True).train()
    policy.native.frozen_teacher.requires_grad_(False).eval()
    return [p for p in policy.parameters() if p.requires_grad] + list(reconstruction.parameters())


def calibration_losses(policy, reconstruction, observation, future_rgb, actions, *, weights, flow_inputs=None):
    if actions.shape != (len(future_rgb), 32, 7) or future_rgb.shape[1:3] != (2, 3):
        raise ValueError("source losses require full 32x7 actions and two-view three-horizon RGB")
    if not torch.isfinite(actions).all() or future_rgb.dtype != torch.uint8:
        raise ValueError("invalid source loss targets")
    if flow_inputs is None:
        noise = torch.randn_like(actions)
        time = torch.rand((len(actions), 1, 1), device=actions.device)
    else:
        noise, time = flow_inputs
        if (noise.shape != actions.shape or time.shape != (len(actions), 1, 1)
                or noise.device != actions.device or time.device != actions.device
                or noise.dtype != actions.dtype or time.dtype != actions.dtype
                or not torch.isfinite(noise).all() or not torch.isfinite(time).all()
                or ((time < 0) | (time > 1)).any() or noise.requires_grad or time.requires_grad):
            raise ValueError("invalid paired flow noise/time")
    noisy = (1 - time) * noise + time * actions
    prediction = policy.predict_velocity(observation, noisy, time[:, :, 0].expand(-1, 32))
    action_loss = F.mse_loss(prediction.float(), (actions - noise).float())
    with torch.no_grad():
        teacher = policy.native.frozen_teacher(future_rgb.flatten(0, 1))
        teacher = teacher.reshape_as(observation.raw_forecasts).detach()
    feature_valid = observation.view_valid[:, :, None].expand(-1, -1, 3).flatten(0, 1)
    raw_loss = raw_future_feature_loss(observation.raw_forecasts.flatten(0, 1), teacher.flatten(0, 1), feature_valid)
    # Source pooling learns from both observed and future source DINO frames.
    # Future target scenes never enter predict_velocity or the runtime encoder.
    future_scene = policy.pool_raw(teacher, observation.view_valid, observation.task_tokens.detach(), observation.task_valid)
    scene = future_scene.flatten(0, 1)
    raw = teacher.permute(0, 2, 1, 3, 4).flatten(0, 1).flatten(1, 2)
    reconstructed = reconstruction(scene)
    reconstruction_loss = F.mse_loss(reconstructed.float(), raw.float())
    scenes = torch.cat((observation.current_scene, scene), dim=0)
    collapse_loss = noncollapse_loss(scenes)
    components = {"action": action_loss, "raw_future": raw_loss,
                  "reconstruction": reconstruction_loss, "noncollapse": collapse_loss}
    if set(weights) != set(components) or any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("source objective requires positive weights for all four components")
    total = sum(weights[name] * value for name, value in components.items())
    if not torch.isfinite(total):
        raise ValueError("source calibration produced nonfinite loss")
    return total, components


def trainable_delta(policy):
    """Pinned frozen base plus this complete delta reconstruct the common model."""
    parameters = dict(policy.named_parameters())
    selected = {name for name, p in parameters.items() if p.requires_grad}
    # Capture all newly added module state, even if later frozen for adaptation.
    return {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()
            if name in selected or not name.startswith("native.") or ".lora_" in name}


def checkpoint_payload(policy, reconstruction, optimizer, sampler, *, step, contract):
    return {"schema_version": 1, "step": step, "contract": copy.deepcopy(contract),
            "model_delta": trainable_delta(policy), "reconstruction": copy.deepcopy(reconstruction.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "sampler": sampler.state_dict(),
            "trainable_names": [n for n, p in policy.named_parameters() if p.requires_grad],
            "rng": {"python": random.getstate(), "numpy": np.random.get_state()[1].tolist(),
                    "numpy_meta": [np.random.get_state()[0], *np.random.get_state()[2:]],
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore_checkpoint(payload, policy, reconstruction, optimizer, sampler, *, contract):
    if payload["schema_version"] != 1 or payload["contract"] != contract:
        raise ValueError("checkpoint source/configuration/base contract differs")
    expected_names = [n for n, p in policy.named_parameters() if p.requires_grad]
    if payload["trainable_names"] != expected_names:
        raise ValueError("checkpoint trainable parameter set differs")
    state = policy.state_dict()
    expected_delta = set(trainable_delta(policy))
    if set(payload["model_delta"]) != expected_delta:
        raise ValueError("checkpoint delta does not completely cover new interfaces and shared adapters")
    for name, tensor in payload["model_delta"].items():
        if tensor.shape != state[name].shape or tensor.dtype != state[name].dtype or not torch.isfinite(tensor).all():
            raise ValueError("invalid checkpoint tensor: " + name)
    with torch.no_grad():
        for name, tensor in payload["model_delta"].items():
            state[name].copy_(tensor)
    reconstruction.load_state_dict(payload["reconstruction"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    sampler.load_state_dict(payload["sampler"])
    rng = payload["rng"]
    random.setstate(rng["python"])
    meta = rng["numpy_meta"]
    np.random.set_state((meta[0], np.asarray(rng["numpy"], dtype=np.uint32), *meta[1:]))
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"]:
        if len(rng["cuda"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA device count differs")
        torch.cuda.set_rng_state_all(rng["cuda"])
    return payload["step"]
