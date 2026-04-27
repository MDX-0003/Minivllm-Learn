from __future__ import annotations

import torch

from myvllm.models.qwen3 import Qwen3ForCausalLM


class MMQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 with a model-owned multimodal merge contract for prefill."""

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        multimodal_embeddings: torch.Tensor | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids must be provided when building input embeddings")

        inputs_embeds = self.model.embed_tokens(input_ids)
        if multimodal_embeddings is None and is_multimodal is None:
            return inputs_embeds
        if multimodal_embeddings is None or is_multimodal is None:
            raise ValueError("multimodal_embeddings and is_multimodal must be provided together")
        if is_multimodal.dtype != torch.bool:
            raise ValueError("is_multimodal must be a boolean mask")
        if is_multimodal.ndim != 1:
            raise ValueError("is_multimodal must be a flat boolean mask")
        if is_multimodal.numel() != input_ids.numel():
            raise ValueError("is_multimodal length must match input_ids length")
        if multimodal_embeddings.ndim != 2:
            raise ValueError("multimodal_embeddings must have shape [num_mm_tokens, hidden_size]")

        expected_rows = int(is_multimodal.sum().item())
        if expected_rows != multimodal_embeddings.shape[0]:
            raise ValueError(
                "is_multimodal.sum() must match multimodal_embeddings.shape[0]"
            )

        # Clone before replacement so the merged tensor is an explicit product of
        # this contract. That makes the method safe to reuse in tests and keeps
        # the merge rule local to the model instead of hidden inside runner code.
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[is_multimodal] = multimodal_embeddings.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        multimodal_embeddings: torch.Tensor | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            if input_ids is not None:
                raise ValueError("Provide either input_ids or inputs_embeds, not both")
            if multimodal_embeddings is not None or is_multimodal is not None:
                raise ValueError(
                    "Provide either premerged inputs_embeds or multimodal merge arguments, not both"
                )
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_input_ids(
                input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )

        # The backbone only needs one fully merged embedding stream. Keeping that
        # handoff narrow makes later VL-chat alignment changes much easier,
        # because the protocol logic stays above this line.
        return self.model(input_ids=None, inputs_embeds=inputs_embeds)
