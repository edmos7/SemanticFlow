"""Small causal forecast and action interfaces for a pretrained backbone.

These modules are untrained components. They do not load EgoSteer, implement
its native action expert, or establish that its hidden states are causal.
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._validation import DEFAULT_HORIZONS, check_float, check_horizons, check_rows, check_valid


class CausalForecastReadout(nn.Module):
    """Read multi-horizon features from exclusively observed hidden states.

    Integration must also ensure the upstream hidden states were computed
    without future images/actions. Timestamp checks cannot detect a leak that
    has already entered a backbone's hidden representation.
    """

    def __init__(self, history_dim: int, feature_dim: int, num_tokens: int = 16,
                 dim: int = 256, num_heads: int = 4,
                 horizons: Sequence[float] = DEFAULT_HORIZONS) -> None:
        super().__init__()
        if min(history_dim, feature_dim, num_tokens, dim, num_heads) <= 0 or dim % num_heads:
            raise ValueError("dimensions must be positive and dim divisible by num_heads")
        self.horizons = check_horizons(horizons)
        self.history_dim = history_dim
        self.feature_dim = feature_dim
        self.num_tokens = num_tokens
        self.register_buffer("horizons_seconds", torch.tensor(self.horizons))
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * dim ** -0.5)
        self.history_projection = nn.Linear(history_dim, dim)
        self.time_projection = nn.Linear(1, dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, feature_dim)

    def forward(self, history_hidden: Tensor, *, observation_times: Tensor,
                forecast_origin: Tensor, history_valid: Tensor) -> Tensor:
        """Return [B,H,num_tokens,feature_dim] for causal [B,T,history_dim].

        Times are in seconds in the same clock. Nondecreasing equal timestamps
        are permitted for multiple tokens from a single observation.
        """
        check_float("history_hidden", history_hidden, 3)
        batch, count, hidden_dim = history_hidden.shape
        if hidden_dim != self.history_dim:
            raise ValueError("history feature dimension does not match the readout")
        check_float("observation_times", observation_times, 2)
        check_float("forecast_origin", forecast_origin, 1)
        if tuple(observation_times.shape) != (batch, count) or tuple(forecast_origin.shape) != (batch,):
            raise ValueError("observation_times and forecast_origin shapes do not match history")
        if observation_times.device != history_hidden.device or forecast_origin.device != history_hidden.device:
            raise ValueError("timestamps must be on the history device")
        check_valid("history_valid", history_valid, (batch, count), history_hidden.device)
        check_rows("history_valid", history_valid)
        if ((observation_times > forecast_origin[:, None]) & history_valid).any().item():
            raise ValueError("future observations cannot enter a causal forecast")
        # Compare consecutive valid times even when padding separates them.
        earlier = observation_times.masked_fill(~history_valid, -torch.inf).cummax(dim=1).values
        if ((observation_times < earlier) & history_valid).any().item():
            raise ValueError("valid observation times must be nondecreasing")
        ages = (observation_times - forecast_origin[:, None]).masked_fill(~history_valid, 0).to(history_hidden.dtype)
        memory = self.history_projection(history_hidden.masked_fill(~history_valid[..., None], 0)) + self.time_projection(ages[..., None])
        horizon = self.time_projection(self.horizons_seconds[:, None].to(history_hidden.dtype))
        queries = (self.queries[None, :, :] + horizon[:, None, :]).flatten(0, 1)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        update, _ = self.attention(queries, memory, memory, key_padding_mask=~history_valid, need_weights=False)
        output = self.output_projection(self.norm(queries + update))
        output = output.reshape(batch, len(self.horizons), self.num_tokens, self.feature_dim)
        check_float("forecast output", output, 4)
        return output


class ForecastActionConditioner(nn.Module):
    """Residual action-token attention to current tokens and predicted futures.

    Pass native action-expert tokens here and retain the native action readout
    downstream. Inputs must be forecasts, never ground-truth future features.
    This architectural interface cannot distinguish forecasts from labels;
    rollout/training integration must enforce that contract.
    """

    def __init__(self, action_dim: int, current_dim: int, forecast_dim: int,
                 dim: int = 256, num_heads: int = 4,
                 horizons: Sequence[float] = DEFAULT_HORIZONS) -> None:
        super().__init__()
        if min(action_dim, current_dim, forecast_dim, dim, num_heads) <= 0 or dim % num_heads:
            raise ValueError("dimensions must be positive and dim divisible by num_heads")
        self.horizons = check_horizons(horizons)
        self.action_dim = action_dim
        self.current_dim = current_dim
        self.forecast_dim = forecast_dim
        self.register_buffer("horizons_seconds", torch.tensor(self.horizons))
        self.action_projection = nn.Linear(action_dim, dim)
        self.current_projection = nn.Linear(current_dim, dim)
        self.forecast_projection = nn.Linear(forecast_dim, dim)
        self.time_projection = nn.Linear(1, dim)
        self.current_tag = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.future_tag = nn.Parameter(torch.randn(dim) * dim ** -0.5)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.output_projection = nn.Linear(dim, action_dim)

    def forward(self, action_tokens: Tensor, current_tokens: Tensor,
                predicted_futures: Tensor, *, current_valid: Tensor,
                future_valid: Tensor, action_valid: Tensor | None = None) -> Tensor:
        """Return residual [B,A,action_dim], consuming futures [B,H,Q,D]."""
        check_float("action_tokens", action_tokens, 3)
        check_float("current_tokens", current_tokens, 3)
        check_float("predicted_futures", predicted_futures, 4)
        batch, action_count, action_dim = action_tokens.shape
        if action_dim != self.action_dim or current_tokens.shape[-1] != self.current_dim or predicted_futures.shape[-1] != self.forecast_dim:
            raise ValueError("input feature dimensions do not match the action conditioner")
        if current_tokens.shape[0] != batch or predicted_futures.shape[0] != batch:
            raise ValueError("all inputs must have the same batch size")
        if predicted_futures.shape[1] != len(self.horizons):
            raise ValueError("future features must match the configured horizons")
        if any(value.device != action_tokens.device or value.dtype != action_tokens.dtype
               for value in (current_tokens, predicted_futures)):
            raise ValueError("all action conditioner inputs must share device and dtype")
        check_valid("current_valid", current_valid, tuple(current_tokens.shape[:2]), action_tokens.device)
        check_valid("future_valid", future_valid, tuple(predicted_futures.shape[:2]), action_tokens.device)
        check_rows("current_valid", current_valid)
        check_rows("future_valid", future_valid)
        if action_valid is not None:
            check_valid("action_valid", action_valid, (batch, action_count), action_tokens.device)
        current = self.current_projection(current_tokens.masked_fill(~current_valid[..., None], 0)) + self.current_tag
        time = self.time_projection(self.horizons_seconds[:, None].to(predicted_futures.dtype))
        future = self.forecast_projection(predicted_futures.masked_fill(~future_valid[..., None, None], 0)) + time[None, :, None, :] + self.future_tag
        future_mask = future_valid[..., None].expand(-1, -1, predicted_futures.shape[2]).flatten(1, 2)
        memory = torch.cat((current, future.flatten(1, 2)), dim=1)
        valid = torch.cat((current_valid, future_mask), dim=1)
        action_input = action_tokens if action_valid is None else action_tokens.masked_fill(~action_valid[..., None], 0)
        query = self.action_projection(action_input)
        update, _ = self.attention(query, memory, memory, key_padding_mask=~valid, need_weights=False)
        residual = self.output_projection(self.norm(update))
        if action_valid is not None:
            residual = residual.masked_fill(~action_valid[..., None], 0)
        output = action_tokens + residual
        check_float("conditioned action tokens", output, 3)
        return output
