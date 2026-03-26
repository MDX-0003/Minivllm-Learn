from __future__ import annotations

from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams

# llm_engine.add_prompt + model_runner.make_warmup_sequences
class ImageSequence(Sequence):
    def __init__(
        self,
        text_token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        *,
        image_path: str | None = None,
        num_vision_tokens: int = 0,
        placeholder_token_ids: list[int] | None = None,
        placeholder_mask: list[bool] | None = None,
    ):
        if placeholder_token_ids is None:
            placeholder_token_ids = []
        if placeholder_mask is None:
            placeholder_mask = [False] * len(placeholder_token_ids)
        if len(placeholder_mask) != len(placeholder_token_ids):
            raise ValueError("placeholder_mask must align with placeholder_token_ids")
        full_token_ids = placeholder_token_ids + text_token_ids
        super().__init__(token_ids=full_token_ids, sampling_params=sampling_params)

        self.image_path = image_path
        self.num_vision_tokens = int(num_vision_tokens) if num_vision_tokens else 0
        self.placeholder_length = len(placeholder_token_ids)
        self.placeholder_mask = list(placeholder_mask)

        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = self.num_tokens
        
    def append_token(self, token_id):
        super().append_token(token_id)

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def __getstate__(self):
        base = super().__getstate__()
        return (
            self.image_path,
            self.num_vision_tokens,
            self.placeholder_length,
            self.placeholder_mask,
            *base,
        )

    def __setstate__(self, state):
        self.image_path = state[0]
        self.num_vision_tokens = state[1]
        self.placeholder_length = state[2]
        self.placeholder_mask = state[3]
        super().__setstate__(state[4:])
