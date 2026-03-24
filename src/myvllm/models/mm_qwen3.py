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
        vis_embeds: torch.Tensor | None = None,
        vis_masks: torch.Tensor | None = None,#和inpute_emds等长的mask，每段seq前一部分为1，表示需要替换为对应的vis_token
    ) -> torch.Tensor:
        # Delegate to underlying model; Qwen3Model supports input_ids or inputs_embeds.
        if inputs_embeds is None and input_ids is not None:
            # 1. 获取纯文本的 embedding
            inputs_embeds = self.model.embed_tokens(input_ids)
            
            # 2. 如果提供了多模态特征，将其替换到指定的 placeholder 位置
            if vis_embeds is not None and vis_masks is not None:
                inputs_embeds[vis_masks] = vis_embeds
                
        # 3. 此时已经融合完毕，直接以 inputs_embeds 的形式传入底层 Qwen3Model
        return self.model(input_ids=None, inputs_embeds=inputs_embeds)
