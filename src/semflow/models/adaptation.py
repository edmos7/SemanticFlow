"""Matched forecast adaptation with immutable common-checkpoint targets."""

import torch
from torch.nn import functional as F

from .losses import raw_future_feature_loss, semantic_flow_loss


@torch.no_grad()
def fixed_forecast_targets(policy, observation, future_rgb):
    """Call before any update. Cache outputs, including fixed task conditioning.

    Freezing pooling weights alone does not fix semantic labels if the task
    context comes from the adapting backbone. These detached copies bind both
    feature and context targets to the common initialization.
    """
    if (future_rgb.dtype != torch.uint8 or future_rgb.ndim != 6
            or future_rgb.shape[:3] != (len(observation.raw_forecasts), 2, 3)):
        raise ValueError('target RGB requires two view slots and three horizons')
    raw = policy.native.frozen_teacher(future_rgb.flatten(0, 1)).reshape_as(observation.raw_forecasts)
    scene = policy.pool_raw(raw, observation.view_valid, observation.task_tokens.detach(), observation.task_valid)
    return {'raw': raw.detach().cpu().clone(), 'scene': scene.detach().cpu().clone(),
            'view_valid': observation.view_valid.detach().cpu().clone()}


def forecast_objective(observation, targets, statistics, *, condition, semantic_weight):
    if condition not in ('standard', 'semflow') or not 0 < semantic_weight <= 1:
        raise ValueError('invalid matched adaptation condition or calibrated weight')
    if set(targets) != {'raw', 'scene', 'view_valid'} or any(t.requires_grad for t in targets.values()):
        raise ValueError('forecast teacher cache must contain detached fixed targets')
    device = observation.raw_forecasts.device
    if not torch.equal(targets['view_valid'].to(device), observation.view_valid):
        raise ValueError('teacher view availability differs from observed input')
    valid = observation.view_valid[:, :, None].expand(-1, -1, 3).flatten(0, 1)
    raw = raw_future_feature_loss(observation.raw_forecasts.flatten(0, 1), targets['raw'].to(device).flatten(0, 1), valid)
    semantic = semantic_flow_loss(observation.scene_forecasts, targets['scene'].to(device),
        torch.ones(observation.scene_forecasts.shape[:2], dtype=torch.bool, device=device), statistics)
    total = .1*raw + (semantic_weight*semantic.total if condition == 'semflow' else 0)
    return total, {'raw':raw, 'semantic':semantic.total, 'state':semantic.state, 'transition':semantic.transition}


def source_action_retention(policy, observation, actions, *, seed):
    """Measured source actions only; shared paired noise/time across arms."""
    if actions.shape != (len(observation.raw_forecasts),32,7) or not torch.isfinite(actions).all():
        raise ValueError('source retention requires finite measured 32x7 action chunks')
    # HDF5 source normalization can produce float64 arrays. Keep regression
    # inputs in float32 so autocast can select the model's bfloat16 compute
    # dtype instead of feeding double tensors directly to native projections.
    actions = actions.float()
    generator = torch.Generator(device=actions.device).manual_seed(seed)
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype, generator=generator)
    time = torch.rand((len(actions),1,1), device=actions.device, dtype=actions.dtype, generator=generator)
    noisy = (1-time)*noise + time*actions
    prediction = policy.predict_velocity(observation, noisy, time[:, :, 0].expand(-1,32))
    result = F.mse_loss(prediction.float(), (actions-noise).float())
    if not torch.isfinite(result):
        raise ValueError('nonfinite source retention loss')
    return result
