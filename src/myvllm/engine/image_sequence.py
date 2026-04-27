from __future__ import annotations

from myvllm.engine.sequence import Sequence
from myvllm.multimodal.processor import MultimodalPayload
from myvllm.sampling_parameters import SamplingParams


class ImageSequence(Sequence):
    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        *,
        multimodal: MultimodalPayload | None = None,
    ):
        super().__init__(token_ids=token_ids, sampling_params=sampling_params)

        # Store the payload as one structured object so every consumer sees the
        # same multimodal contract. This avoids scattering related fields across
        # the sequence object and later having to reconstruct which values
        # belong together during prefill.
        self.multimodal = multimodal
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = self.num_tokens

    @property
    def image_path(self) -> str | None:
        return self.multimodal.image_path if self.multimodal is not None else None

    @property
    def num_vision_tokens(self) -> int:
        return self.multimodal.num_vision_tokens if self.multimodal is not None else 0

    @property
    def is_multimodal(self) -> list[bool]:
        if self.multimodal is None:
            return [False] * len(self.token_ids)
        return self.multimodal.is_multimodal

    @property
    def placeholder_length(self) -> int:
        if self.multimodal is None:
            return 0
        return self.multimodal.placeholder_length

    def append_token(self, token_id):
        super().append_token(token_id)

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def __getstate__(self):
        base = super().__getstate__()
        return (
            self.multimodal,
            *base,
        )

    def __setstate__(self, state):
        self.multimodal = state[0]
        super().__setstate__(state[1:])
