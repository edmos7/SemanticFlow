"""Matched-horizon raw-feature and source-normalized semantic objectives."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import DEFAULT_HORIZONS, check_float, check_horizons, check_valid


@dataclass(frozen=True)
class SourceFeatureStatistics:
    """Per-channel scene statistics, fitted once on source training examples.

    The source manifest identifies the exact dataset/split provenance. Callers
    must persist these tensors with the common source checkpoint and reuse them
    unchanged across conditions; target clips must never refit these statistics.
    """

    mean: Tensor
    scale: Tensor
    source_manifest: str

    def __post_init__(self) -> None:
        check_float("source mean", self.mean, 1)
        check_float("source scale", self.scale, 1)
        if self.mean.shape != self.scale.shape or self.mean.device != self.scale.device:
            raise ValueError("source mean and scale must have matching shapes and devices")
        if (self.scale <= 0).any().item():
            raise ValueError("source scales must be strictly positive")
        if not self.source_manifest.strip():
            raise ValueError("source_manifest is required for calibration provenance")
        # Clone to avoid sharing a mutable buffer with the caller or teacher.
        object.__setattr__(self, "mean", self.mean.detach().clone())
        object.__setattr__(self, "scale", self.scale.detach().clone())

    @classmethod
    def fit(cls, source_features: Tensor, source_valid: Tensor, *,
            source_manifest: str, split: str = "source_train", min_scale: float = 1e-6) -> "SourceFeatureStatistics":
        """Fit [D] statistics from source scene features [N,Q,D] only.

        `split` is an explicit caller contract, not an automatic leakage test.
        The manifest/data loader must independently exclude all target data.
        """
        if split != "source_train":
            raise ValueError("calibration must use source_train data only")
        if not math.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        check_float("source_features", source_features, 3)
        check_valid("source_valid", source_valid, (source_features.shape[0],), source_features.device)
        if not source_valid.any().item():
            raise ValueError("source calibration requires valid source examples")
        selected = source_features.detach().float()[source_valid].flatten(0, 1)
        return cls(selected.mean(dim=0), selected.std(dim=0, unbiased=False).clamp_min(min_scale), source_manifest)

    def normalize(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.mean.numel():
            raise ValueError("feature dimension does not match source calibration")
        # Compute losses in float32 without detaching prediction gradients.
        return (features.float() - self.mean.to(device=features.device, dtype=torch.float32)) / self.scale.to(device=features.device, dtype=torch.float32)


@dataclass(frozen=True)
class SemanticFlowLoss:
    total: Tensor
    state: Tensor
    transition: Tensor
    valid_states: int
    valid_transitions: int


def _check_pair(prediction: Tensor, teacher: Tensor, valid: Tensor,
                horizons: Sequence[float]) -> None:
    values = check_horizons(horizons)
    check_float("prediction", prediction, 4)
    check_float("teacher", teacher, 4)
    if prediction.shape != teacher.shape or prediction.device != teacher.device:
        raise ValueError("prediction and teacher must share shape [B,H,Q,D] and device")
    if prediction.shape[1] != len(values):
        raise ValueError("feature horizon dimension must match configured horizons")
    check_valid("valid", valid, tuple(prediction.shape[:2]), prediction.device)
    if not valid.any().item():
        raise ValueError("future supervision requires at least one valid horizon")


def _masked_mse(prediction: Tensor, teacher: Tensor, valid: Tensor) -> Tensor:
    # Select before subtraction so finite extreme values in masked padding do
    # not overflow a squared residual and contaminate backward with 0 * inf.
    difference = prediction[valid].float() - teacher[valid].float()
    loss = difference.square().mean()
    if not torch.isfinite(loss).item():
        raise ValueError("masked MSE overflowed or became nonfinite")
    return loss


def raw_future_feature_loss(prediction: Tensor, teacher: Tensor, valid: Tensor,
                            *, horizons: Sequence[float] = DEFAULT_HORIZONS) -> Tensor:
    """Raw DINO feature MSE with no normalization or implicit rescaling.

    Inputs are [B,H,P,D]. This is a raw MSE component, not the complete native
    EgoSteer training objective: retain its configured external loss multiplier.
    """
    _check_pair(prediction, teacher, valid, horizons)
    return _masked_mse(prediction, teacher.detach(), valid)


def semantic_flow_loss(prediction: Tensor, teacher: Tensor, valid: Tensor,
                       statistics: SourceFeatureStatistics, *, state_weight: float = 1.0,
                       transition_weight: float = 1.0,
                       horizons: Sequence[float] = DEFAULT_HORIZONS) -> SemanticFlowLoss:
    """Source-normalized state and successive-horizon delta MSE.

    Features are [B,H,Q,D]. A transition is supervised only when both adjacent
    horizons are valid. Deltas are feature changes, not changes divided by time.
    Targets are detached; prediction/readout/backbone gradients remain intact.
    """
    _check_pair(prediction, teacher, valid, horizons)
    weights = (state_weight, transition_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights) or not any(weights):
        raise ValueError("loss weights must be finite, nonnegative, and not both zero")
    # Replace invalid finite padding before normalization or differencing.
    pred = statistics.normalize(prediction.masked_fill(~valid[..., None, None], 0))
    target = statistics.normalize(teacher.detach().masked_fill(~valid[..., None, None], 0))
    state = _masked_mse(pred, target, valid)
    transition_valid = valid[:, 1:] & valid[:, :-1]
    transition_count = int(transition_valid.sum().item())
    if transition_weight and not transition_count:
        raise ValueError("transition supervision requires at least one adjacent valid horizon pair")
    if transition_count:
        transition = _masked_mse(pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], transition_valid)
    else:
        transition = state * 0
    total = state_weight * state + transition_weight * transition
    if not torch.isfinite(total).item():
        raise ValueError("semantic flow loss became nonfinite")
    return SemanticFlowLoss(total, state, transition, int(valid.sum().item()), transition_count)
