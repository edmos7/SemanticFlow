"""Shared Qwen language adapters for the matched rank-8 adaptation arms.

Load the native checkpoint before injection. Subsequent ``state_dict`` saves
include frozen base weights and adapters; inject the same topology before load.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SHARED_LORA_RANK = 8
SHARED_LORA_PROJECTIONS = ("q_proj", "v_proj")


class LoRALinear(nn.Module):
    """Frozen linear map plus ``B(A(x))`` with unit scaling and no dropout.

    The zero-initialized B preserves the pretrained output. A and B inherit the
    base's dtype/device. Freezing the base does not detach its input gradients.
    """

    def __init__(self, base: nn.Linear, rank: int = SHARED_LORA_RANK) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear requires an unwrapped nn.Linear")
        if type(rank) is not int or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must be a positive integer no larger than either linear dimension")
        if not base.weight.is_floating_point() or base.weight.device.type == "meta":
            raise ValueError("LoRALinear requires materialized floating-point base weights")
        if base.bias is not None and (base.bias.device != base.weight.device or base.bias.dtype != base.weight.dtype):
            raise ValueError("base weight and bias must share dtype and device")
        self.base = base
        self.rank = rank
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.train(base.training)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, inputs: Tensor) -> Tensor:
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_A), self.lora_B)


def _shared_projections(policy: nn.Module) -> tuple[tuple[str, nn.Module, str, nn.Module], ...]:
    if not isinstance(policy, nn.Module):
        raise TypeError("policy must be an nn.Module")
    backbone = getattr(policy, "backbone", None)
    language = getattr(backbone, "language_model", None)
    layers = getattr(language, "layers", None)
    if not isinstance(layers, nn.ModuleList) or not len(layers):
        raise ValueError("policy requires nonempty backbone.language_model.layers as nn.ModuleList")
    result = []
    for index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        if not isinstance(attention, nn.Module):
            raise ValueError(f"shared language layer {index} has no self_attn module")
        for name in SHARED_LORA_PROJECTIONS:
            projection = getattr(attention, name, None)
            if not isinstance(projection, (nn.Linear, LoRALinear)):
                raise ValueError(f"shared language layer {index}.{name} is incompatible")
            path = f"backbone.language_model.layers.{index}.self_attn.{name}"
            result.append((path, attention, name, projection))
        if result[-2][3].in_features != result[-1][3].in_features:
            raise ValueError(f"shared language layer {index} q/v input dimensions differ")
    return tuple(result)


def _unique_registered_parameters(module: nn.Module, selected: set[int]) -> tuple[tuple[str, nn.Parameter], ...]:
    entries = tuple((name, parameter) for name, parameter in module.named_parameters(remove_duplicate=False)
                    if id(parameter) in selected)
    if len(entries) != len(selected) or len({id(parameter) for _, parameter in entries}) != len(selected):
        raise ValueError("selected parameters must each have exactly one registered module path")
    return entries


def inject_shared_lora(policy: nn.Module, rank: int = SHARED_LORA_RANK) -> tuple[str, ...]:
    """Insert q/v adapters in every native shared language layer exactly once.

    This fixed projection list/rank is identical in standard and SemFlow arms.
    Other policy modules are untouched; select the adaptation trainable set
    with ``enable_shared_adapter_training`` after adding the remaining modules.
    """
    if type(rank) is not int or rank != SHARED_LORA_RANK:
        raise ValueError("the matched pilot requires rank 8")
    projections = _shared_projections(policy)
    selected = []
    # Validate the entire target set before replacing or freezing any module.
    for path, _, _, projection in projections:
        if isinstance(projection, LoRALinear):
            raise ValueError(f"shared projection is already wrapped: {path}")
        if rank > min(projection.in_features, projection.out_features):
            raise ValueError(f"rank exceeds a shared projection dimension: {path}")
        if not projection.weight.is_floating_point() or projection.weight.device.type == "meta":
            raise ValueError(f"shared projection lacks materialized floating-point weights: {path}")
        if projection.bias is not None and (projection.bias.device != projection.weight.device or projection.bias.dtype != projection.weight.dtype):
            raise ValueError(f"shared projection weight/bias dtype or device differs: {path}")
        selected.extend(projection.parameters())
    identities = {id(parameter) for parameter in selected}
    if len(identities) != len(selected):
        raise ValueError("shared q/v projections must not alias parameters")
    _unique_registered_parameters(policy, identities)
    for _, attention, name, projection in projections:
        setattr(attention, name, LoRALinear(projection, rank))
    return tuple(path for path, _, _, _ in projections)


def named_shared_adapter_parameters(policy: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    """Return audited unique q/v adapter parameters, excluding all base weights."""
    entries = []
    for path, _, _, projection in _shared_projections(policy):
        if not isinstance(projection, LoRALinear) or projection.rank != SHARED_LORA_RANK:
            raise ValueError(f"expected an inserted rank-8 shared adapter: {path}")
        entries.extend(((f"{path}.lora_A", projection.lora_A), (f"{path}.lora_B", projection.lora_B)))
    identities = {id(parameter) for _, parameter in entries}
    if len(identities) != len(entries):
        raise ValueError("shared adapters must not alias parameters")
    registered = _unique_registered_parameters(policy, identities)
    if {name for name, _ in registered} != {name for name, _ in entries}:
        raise ValueError("shared adapter registration differs from the pinned language paths")
    return tuple(entries)


def enable_shared_adapter_training(module: nn.Module, *, policy: nn.Module | None = None) -> tuple[tuple[str, nn.Parameter], ...]:
    """Freeze/evaluate the whole controller, then enable exactly shared LoRA.

    ``module`` may contain the native ``policy`` plus frozen forecast/control
    readouts. Returned names are relative to ``module`` for optimizer audits.
    Callers must clone the same source checkpoint independently for each arm.
    """
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    policy = module if policy is None else policy
    if not any(child is policy for child in module.modules()):
        raise ValueError("the native policy must be registered inside the module being frozen")
    entries = named_shared_adapter_parameters(policy)
    identities = {id(parameter) for _, parameter in entries}
    registered = _unique_registered_parameters(module, identities)
    module.requires_grad_(False).eval()
    module.zero_grad(set_to_none=True)
    for _, parameter in registered:
        parameter.requires_grad_(True)
    if {id(parameter) for parameter in module.parameters() if parameter.requires_grad} != identities:
        raise RuntimeError("adaptation trainable set differs from shared adapters")
    return registered
