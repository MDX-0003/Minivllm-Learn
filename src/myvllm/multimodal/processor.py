from dataclasses import dataclass
@dataclass
class MultimodalInput:
    placeholder_token_ids: list[int]   # e.g. [0] * 128
    num_vision_tokens: int              # e.g. 128
    image_meta: dict                    # e.g. {"image_path": "...", "batch_id": 0}


class Processor:
    def __init__(self, config: dict):
        self.num_vision_tokens = config.get("num_vision_tokens", 128)  # 暂时用配置
        self.placeholder_id = 0                                        # 暂时用假 id

    def process(self, image_path: str) -> MultimodalInput:
        return MultimodalInput(
            placeholder_token_ids=[self.placeholder_id] * self.num_vision_tokens,
            num_vision_tokens=self.num_vision_tokens,
            image_meta={"image_path": image_path},
        )