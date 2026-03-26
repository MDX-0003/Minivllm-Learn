from dataclasses import dataclass


@dataclass
class MultimodalInput:
    placeholder_token_ids: list[int]
    placeholder_mask: list[bool]
    num_vision_tokens: int
    image_meta: dict

# use in "mm_llm_engine.add_prompt",paser "image_path" and config to generate vis place holder
# the output will set into image sequence
class Processor:
    def __init__(self, config: dict):
        self.num_vision_tokens = int(config.get("num_vision_tokens", 128))
        self.vision_start_token_id = int(config["vision_start_token_id"])
        self.image_pad_token_id = int(config["image_pad_token_id"])
        self.vision_end_token_id = int(config["vision_end_token_id"])

    def process(self, image_path: str) -> MultimodalInput:
        # Keep the placeholder protocol explicit:
        # <vision_start> + N * <image_pad> + <vision_end>
        placeholder_token_ids = (
            [self.vision_start_token_id]
            + [self.image_pad_token_id] * self.num_vision_tokens
            + [self.vision_end_token_id]
        )
        # Only image_pad positions should be replaced by vision embeddings.
        placeholder_mask = (
            [False]
            + [True] * self.num_vision_tokens
            + [False]
        )
        return MultimodalInput(
            placeholder_token_ids=placeholder_token_ids,
            placeholder_mask=placeholder_mask,
            num_vision_tokens=self.num_vision_tokens,
            image_meta={
                "image_path": image_path,
                "placeholder_length": len(placeholder_token_ids),
            },
        )
