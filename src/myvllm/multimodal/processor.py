from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MultimodalPayload:
    image_path: str | None
    num_vision_tokens: int
    placeholder_token_ids: list[int]
    is_multimodal: list[bool]

    @property
    def placeholder_length(self) -> int:
        return len(self.placeholder_token_ids)


@dataclass
class PreparedMultimodalInput:
    token_ids: list[int]
    multimodal: MultimodalPayload | None


class Processor:
    def __init__(self, config: dict, tokenizer: Any):
        self.num_vision_tokens = int(config.get("num_vision_tokens", 128))
        self.tokenizer = tokenizer

        # Resolve the special token ids once here so the rest of the engine can
        # treat the multimodal prefix as a stable protocol instead of repeatedly
        # querying tokenizer state in different layers.
        self.vision_start_token_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        self.image_pad_token_id = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        self.vision_end_token_id = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))

    def process(self, prompt: str, image_path: str | None) -> PreparedMultimodalInput:
        text_token_ids = self.tokenizer.encode(prompt)
        if not image_path or self.num_vision_tokens <= 0:
            return PreparedMultimodalInput(token_ids=text_token_ids, multimodal=None)

        # The processor is now the single prompt-side source of truth for how a
        # multimodal request is laid out in token space. That keeps the engine
        # and runner focused on orchestration instead of protocol details.
        placeholder_token_ids = (
            [self.vision_start_token_id]
            + [self.image_pad_token_id] * self.num_vision_tokens
            + [self.vision_end_token_id]
        )
        token_ids = placeholder_token_ids + text_token_ids

        # Only the image_pad span should be replaced by projected image
        # embeddings. We mark the full flattened sequence here so later stages
        # can consume one explicit mask rather than reconstructing protocol
        # semantics from placeholder lengths and ad-hoc conditionals.
        is_multimodal = (
            [False]
            + [True] * self.num_vision_tokens
            + [False]
            + [False] * len(text_token_ids)
        )

        multimodal = MultimodalPayload(
            image_path=image_path,
            num_vision_tokens=self.num_vision_tokens,
            placeholder_token_ids=placeholder_token_ids,
            is_multimodal=is_multimodal,
        )
        return PreparedMultimodalInput(token_ids=token_ids, multimodal=multimodal)
