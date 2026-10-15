"""Small, shared shape and finite-value checks for research interfaces."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor

DEFAULT_HORIZONS = (0.5, 1.0, 2.0)


def check_float(name: str, value: Tensor, ndim: int) -> None:
    if value.ndim != ndim or not value.is_floating_point():
        raise ValueError(f"{name} must be a {ndim}-dimensional floating tensor")
    if any(size == 0 for size in value.shape):
        raise ValueError(f"{name} must not have empty dimensions")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains NaN or infinity, including padding")


def check_valid(name: str, value: Tensor, shape: tuple[int, ...], device: torch.device) -> None:
    if value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a boolean tensor with shape {shape}")
    if value.device != device:
        raise ValueError(f"{name} must be on the input device")


def check_rows(name: str, valid: Tensor) -> None:
    if not valid.any(dim=-1).all().item():
        raise ValueError(f"{name} must contain at least one valid item per example")


def check_horizons(horizons: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in horizons)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("horizons must be finite, positive seconds")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("horizons must be strictly increasing")
    return values
