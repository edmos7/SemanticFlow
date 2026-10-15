"""Task-conditioned whole-scene token resampling; no object or hand labels."""

import torch
from torch import Tensor, nn

from ._validation import check_float, check_rows, check_valid


class SceneResampler(nn.Module):
    """Learn scene queries from frozen spatially encoded image patch features.

    This module is a trainable source-data interface, not a pretrained semantic
    teacher. Train and validate it on source tasks, then freeze its parameters
    for target adaptation. Frozen parameters must still permit gradients from
    predicted features to upstream adapters; do not wrap that path in no_grad.
    """

    def __init__(self, patch_dim: int, task_dim: int, dim: int = 256,
                 num_queries: int = 16, num_heads: int = 4) -> None:
        super().__init__()
        if min(patch_dim, task_dim, dim, num_queries, num_heads) <= 0 or dim % num_heads:
            raise ValueError("dimensions must be positive and dim divisible by num_heads")
        self.patch_dim = patch_dim
        self.task_dim = task_dim
        self.num_queries = num_queries
        self.dim = dim
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * dim ** -0.5)
        self.patch_projection = nn.Linear(patch_dim, dim)
        self.task_projection = nn.Linear(task_dim, dim)
        self.task_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.patch_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.task_norm = nn.LayerNorm(dim)
        self.scene_norm = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, patches: Tensor, task_tokens: Tensor, *,
                patch_valid: Tensor, task_valid: Tensor) -> Tensor:
        """Map [B,P,patch_dim] and [B,L,task_dim] to [B,num_queries,dim].

        Masks are True for usable tokens. Inputs, including masked padding,
        must be finite. No hardcoded object identities enter the queries.
        """
        check_float("patches", patches, 3)
        check_float("task_tokens", task_tokens, 3)
        batch, patches_count, patch_dim = patches.shape
        if patch_dim != self.patch_dim or task_tokens.shape[0] != batch or task_tokens.shape[-1] != self.task_dim:
            raise ValueError("patch/task feature dimensions or batch sizes do not match the resampler")
        if patches.device != task_tokens.device or patches.dtype != task_tokens.dtype:
            raise ValueError("patch and task tokens must share device and dtype")
        check_valid("patch_valid", patch_valid, (batch, patches_count), patches.device)
        check_valid("task_valid", task_valid, tuple(task_tokens.shape[:2]), patches.device)
        check_rows("patch_valid", patch_valid)
        check_rows("task_valid", task_valid)
        query = self.queries.unsqueeze(0).expand(batch, -1, -1)
        task = self.task_projection(task_tokens.masked_fill(~task_valid[..., None], 0))
        update, _ = self.task_attention(query, task, task, key_padding_mask=~task_valid, need_weights=False)
        query = self.task_norm(query + update)
        patch = self.patch_projection(patches.masked_fill(~patch_valid[..., None], 0))
        update, _ = self.patch_attention(query, patch, patch, key_padding_mask=~patch_valid, need_weights=False)
        scene = self.scene_norm(query + update)
        output = self.output_norm(scene + self.feed_forward(scene))
        check_float("scene output", output, 3)
        return output
