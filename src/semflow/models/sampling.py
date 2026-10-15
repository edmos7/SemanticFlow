"""Numerically matched native solver for parity audits only."""


def accumulated_sample(policy, observation, noise, steps, *, native_update=True):
    import torch
    actions = noise.clone()
    time = torch.zeros(len(noise), device=noise.device, dtype=noise.dtype)
    delta = 1. / steps
    for _ in range(steps):
        velocity = policy.predict_velocity(observation, actions, time[:, None].expand(-1, 32))
        actions = actions + (delta * velocity if native_update else velocity / steps)
        time = time + delta
    return actions
