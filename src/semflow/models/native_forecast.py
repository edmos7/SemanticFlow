"""Prediction-only extension of EgoSteer's native one-second world readout.

The query layout, MRoPE positions and patch rearrangement follow the pinned
EgoSteer world stream. No teacher or privileged conditioning is invoked here.
The caller must build the supplied prefix exclusively from observations.

Adapted from EgoSteer (5e646d9), Copyright 2026 The EgoSteer Authors,
Apache-2.0; see licenses/egosteer/{LICENSE,NOTICE}. Changes separate teacher
extraction, extend horizons and validate observed-only inputs.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from ._validation import DEFAULT_HORIZONS, check_float, check_horizons, check_rows, check_valid


class NativeForecastReadout(nn.Module):
    """Reuse the native world expert at 0.5, 1 and 2 seconds.

    Only a new horizon embedding belongs to this module. Pass the policy into
    ``forward`` to avoid registering its backbone and heads a second time.
    Source calibration learns the horizon extension; it is not pretrained.
    """

    def __init__(self, hidden_size: int,
                 horizons: Sequence[float] = DEFAULT_HORIZONS) -> None:
        super().__init__()
        if not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.horizons = check_horizons(horizons)
        if self.horizons != DEFAULT_HORIZONS:
            raise ValueError("native pilot horizons must be (0.5, 1.0, 2.0) seconds")
        self.hidden_size = hidden_size
        self.horizon_embedding = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, policy: nn.Module, batch: Mapping[str, Any],
                backbone_output: Any, view_valid: Tensor) -> dict[str, Tensor]:
        """Return ``pred`` [B,2,3,P,D] and ``valid`` [B,2,3].

        Native view order is preserved. Every horizon is predicted for each
        observed view, irrespective of teacher availability. Missing views
        have masked queries and zero outputs. Frozen native modules remain
        differentiable with respect to their inputs and the prefix cache.
        """
        forbidden = sorted(key for key in batch if (
            any(word in key.lower() for word in ("future", "teacher", "target"))
            or key.lower() in {"actions", "actions_valid_mask", "labels", "motion",
                               "head_motion", "chest_motion", "camera_motion"}
        ))
        if forbidden:
            raise ValueError(f"privileged inputs cannot enter native forecasts: {forbidden}")
        prefix_hidden = backbone_output.last_hidden_states
        check_float("backbone last_hidden_states", prefix_hidden, 3)
        batch_size, _, _ = prefix_hidden.shape
        device = prefix_hidden.device
        check_valid("view_valid", view_valid, (batch_size, 2), device)
        check_rows("view_valid", view_valid)
        if backbone_output.prefix_cache is None:
            raise ValueError("native forecasts require an observed prefix cache")
        input_ids = batch.get("input_ids")
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
            raise ValueError("input_ids must have shape [B,S] matching the prefix")
        if input_ids.device != device:
            raise ValueError("input_ids must be on the prefix device")
        cfg = policy.world_model_config
        if cfg.action_conditioning or cfg.motion_conditioning:
            raise ValueError("native forecasts require action and motion conditioning disabled")
        if policy.wm_num_future_frames != 1:
            raise ValueError("native forecasts require the one-frame pretrained readout")
        gh, gw, uf = policy.wm_grid_h, policy.wm_grid_w, policy.wm_upsample_factor
        if any(not isinstance(size, int) or size <= 0 for size in (gh, gw, uf)):
            raise ValueError("native world grid and upsample dimensions must be positive integers")
        dim = self.hidden_size
        query_count = gh * gw
        head = policy.wm_head
        if policy.world_model_expert.hidden_size != dim:
            raise ValueError("native world hidden size does not match the readout")
        if tuple(head.query_embed.shape) != (query_count, dim) or tuple(head.view_embed.shape) != (2, dim):
            raise ValueError("native query/view embedding shapes do not match the world grid")
        if head.query_embed.device != device or head.view_embed.device != device or self.horizon_embedding.device != device:
            raise ValueError("native readout and embeddings must be on the prefix device")

        # One native query block per view, decoded separately for each horizon.
        # Adding all horizons to a single suffix would change the 1s attention
        # context and positions relative to the pretrained model.
        base_queries = head.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        native_suffix = torch.cat([base_queries + head.view_embed[view] for view in range(2)], dim=1)
        suffix_mask = view_valid[:, :, None].expand(-1, -1, query_count).reshape(batch_size, 2 * query_count)
        relative_positions = []
        text_cursor = rope_cursor = 0
        for _ in range(2):
            positions, text_cursor, rope_cursor = policy._build_vision_mrope_position_ids(
                text_start=text_cursor, rope_start=rope_cursor, num_frames=1,
                grid_h=gh, grid_w=gw, device=device,
            )
            relative_positions.append(positions)
        position_ids = policy.build_suffix_position_ids_from_relative(
            batch, backbone_output, torch.cat(relative_positions, dim=1), device=device,
        )
        if tuple(position_ids.shape) != (4, batch_size, 2 * query_count) or position_ids.device != device:
            raise ValueError("native world position ids must have shape [4,B,2*queries] on the prefix device")

        predictions = []
        for horizon in self.horizons:
            # Avoid even an extra arithmetic operation on the native baseline.
            suffix = native_suffix if horizon == 1.0 else (
                native_suffix + (horizon - 1.0) * self.horizon_embedding.to(dtype=native_suffix.dtype)
            )
            hidden = policy.world_model_expert(
                suffix_embeds=suffix, prefix_cache=backbone_output.prefix_cache,
                suffix_position_ids=position_ids, suffix_mask=suffix_mask,
            )
            if tuple(hidden.shape) != (batch_size, 2 * query_count, dim):
                raise ValueError("native world expert returned an unexpected hidden shape")
            query_hidden = hidden.reshape(batch_size, 2, query_count, dim)
            projected = head.output_proj(query_hidden)
            if tuple(projected.shape) != (batch_size, 2, query_count, uf * uf * dim):
                raise ValueError("native world output projection has an unexpected shape")
            # Preserve the pretrained subpatch ordering (grid row, subpatch
            # row, grid column, subpatch column), without feature normalization.
            patches = projected.reshape(batch_size * 2, gh, gw, uf, uf, dim)
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, 2, gh * gw * uf * uf, dim)
            predictions.append(patches.masked_fill(~view_valid[:, :, None, None], 0))
        prediction = torch.stack(predictions, dim=2)
        check_float("native forecasts", prediction, 5)
        valid = view_valid[:, :, None].expand(-1, -1, len(self.horizons)).clone()
        return {"pred": prediction, "valid": valid}
