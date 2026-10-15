"""Semantic components and native integration interfaces awaiting source training.

Importing this optional subpackage requires PyTorch. Core data/CLI tools do not.
"""

from .forecast import CausalForecastReadout, ForecastActionConditioner
from .losses import SourceFeatureStatistics, SemanticFlowLoss, raw_future_feature_loss, semantic_flow_loss
from .scene import SceneResampler
from .adapters import inject_shared_lora, named_shared_adapter_parameters, enable_shared_adapter_training
from .native_forecast import NativeForecastReadout
from .native_policy import NativeSemFlowPolicy

__all__ = [
    "CausalForecastReadout",
    "ForecastActionConditioner",
    "SceneResampler",
    "SemanticFlowLoss",
    "SourceFeatureStatistics",
    "raw_future_feature_loss",
    "semantic_flow_loss",
    "NativeForecastReadout",
    "NativeSemFlowPolicy",
    "inject_shared_lora",
    "named_shared_adapter_parameters",
    "enable_shared_adapter_training",
]
