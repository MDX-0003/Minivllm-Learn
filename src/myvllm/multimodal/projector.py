from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VisionProjector(nn.Module):
    """Project patch features into the LLM hidden size and align token count."""

    def __init__(
        self,
        *,
        vision_dim: int,
        hidden_size: int,
        seed: int = 1,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(int(vision_dim))
        self.proj = nn.Linear(int(vision_dim), int(hidden_size), bias=False)

        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            nn.init.xavier_uniform_(self.proj.weight)
    # input features shape [patch_num, vis_dim]
    def _align_num_tokens(self, features: torch.Tensor, num_vision_tokens: int) -> torch.Tensor:
        target_tokens = int(num_vision_tokens)
        if target_tokens <= 0:
            return features.new_empty((0, features.shape[-1]))
        if features.shape[0] == target_tokens:
            return features
        
        # if patch_num != target_tokens ,need interpolate to target_tokens
        # interpolate will change last dim , so output [1,vis_dim,target_tokens]
        resized = F.interpolate(
            features.transpose(0, 1).unsqueeze(0),# [1,vis_dim,patch_num]
            size=target_tokens,
            mode="linear",
            align_corners=False,
        )
        return resized.squeeze(0).transpose(0, 1)# [target_tokens , vis_dim]

    @torch.inference_mode()
    def forward(
        self,
        features: torch.Tensor,# [batch , patch_num , vis_dim]
        *,
        num_vision_tokens: int,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # place holder length may not equal to patch_num from vision_encoder
        if features.numel() == 0 or int(num_vision_tokens) <= 0:
            out_dtype = dtype or self.proj.weight.dtype
            return torch.empty((0, self.proj.out_features), device=self.proj.weight.device, dtype=out_dtype)

        features = self._align_num_tokens(features, num_vision_tokens)
        projected = self.proj(self.input_norm(features))
        if dtype is not None:
            projected = projected.to(dtype=dtype)
        return projected
