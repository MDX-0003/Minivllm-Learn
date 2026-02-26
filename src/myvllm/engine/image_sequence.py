from __future__ import annotations

from myvllm.engine.sequence import Sequence
from myvllm.sampling_parameters import SamplingParams


class ImageSequence(Sequence):
    """Sequence with an image prefix (Milestone 1).

    We model the image as `num_vision_tokens` prefix tokens that only exist as embeddings.
    They count toward KV-cache length and positions, but do not exist in `token_ids`.

    IMPORTANT: we keep the parent `Sequence` implementation untouched.

    Notes about accounting:
    - `Sequence.token_ids` keeps *text* tokens only.
    - `ImageSequence.num_tokens` includes vision + text.
    - We treat vision tokens as part of the prompt, so `num_prompt_tokens = num_tokens`.
    """

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        *,
        image_path: str | None = None,
        num_vision_tokens: int = 0,
    ):
        super().__init__(token_ids=token_ids, sampling_params=sampling_params)
        self.image_path = image_path
        self.num_vision_tokens = int(num_vision_tokens) if num_vision_tokens else 0

        # Override length accounting to include vision prefix.
        self.num_tokens = self.num_vision_tokens + len(self.token_ids)
        self.num_prompt_tokens = self.num_tokens

    def append_token(self, token_id):
        # Parent updates token_ids + last_token + num_tokens, which is correct because
        # vision prefix is constant.
        super().append_token(token_id)

    @property
    def last_block_num_tokens(self):
        """Number of tokens in the last KV-cache block.

        The base `Sequence.last_block_num_tokens` derives this value by slicing
        `token_ids`, which only contains *text* tokens.

        For `ImageSequence`, the effective stream length is (vision + text), so the
        last block token count must be computed from `num_tokens` directly.
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def __getstate__(self):
        # Extend base serialization so multiprocessing/shm path can carry image info.
        base = super().__getstate__()
        return (self.image_path, self.num_vision_tokens, *base)

    def __setstate__(self, state):
        # state = (image_path, num_vision_tokens, <base_state...>)
        self.image_path = state[0]
        self.num_vision_tokens = state[1]
        super().__setstate__(state[2:])
