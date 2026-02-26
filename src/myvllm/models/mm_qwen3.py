from __future__ import annotations

import torch

from myvllm.models.qwen3 import Qwen3ForCausalLM


class MMQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 with an 'inputs_embeds' forward path for Milestone 1.

    Decode path remains unchanged (still uses token ids).
    Prefill can provide `inputs_embeds` that already includes vision prefix.
    """

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Delegate to underlying model; `Qwen3Model` supports input_ids or inputs_embeds.
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds)
