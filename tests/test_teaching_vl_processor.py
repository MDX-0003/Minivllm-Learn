import torch

from myvllm.multimodal.vl_processor import (
    VLContentItem,
    VLMessage,
    VLProcessor,
    VLRequest,
    VisionSpan,
)


class TeachingTokenizer:
    def __init__(self):
        self.special_ids = {
            "<|im_start|>": 10,
            "<|im_end|>": 11,
            "<|vision_start|>": 20,
            "<|image_pad|>": 21,
            "<|vision_end|>": 22,
        }
        self.text_ids = {
            "system": 100,
            "user": 101,
            "assistant": 102,
            "You": 103,
            "are": 104,
            "a": 105,
            "helpful": 106,
            "assistant.": 107,
            "Describe": 108,
            "this": 109,
            "image": 110,
            "briefly.": 111,
        }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_ids[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        i = 0
        specials = sorted(self.special_ids, key=len, reverse=True)
        while i < len(text):
            matched = False
            for token in specials:
                if text.startswith(token, i):
                    ids.append(self.special_ids[token])
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace() and not any(
                text.startswith(token, j) for token in specials
            ):
                j += 1
            piece = text[i:j]
            ids.append(self.text_ids[piece])
            i = j
        return ids


class TeachingImageProcessor:
    def __call__(self, image_paths: list[str]):
        assert image_paths == ["demo.png"]
        return {
            "pixel_values": torch.arange(64 * 3, dtype=torch.float32).view(64, 3),
            "image_grid_thw": torch.tensor([[1, 8, 8]], dtype=torch.long),
        }


def make_request() -> VLRequest:
    return VLRequest(
        messages=[
            VLMessage(
                role="user",
                content=[
                    VLContentItem(type="image", image="demo.png"),
                    VLContentItem(type="text", text="Describe this image briefly."),
                ],
            )
        ]
    )


def test_vl_processor_expands_image_placeholder_from_grid_and_builds_mask():
    processor = VLProcessor(
        tokenizer=TeachingTokenizer(),
        image_processor=TeachingImageProcessor(),
        spatial_merge_size=2,
    )

    output = processor.process(make_request())

    assert output.messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "demo.png"},
                {"type": "text", "text": "Describe this image briefly."},
            ],
        }
    ]
    assert "<|vision_start|><|image_pad|><|vision_end|>" in output.prompt_text
    assert "<|image_pad|>" * 16 in output.expanded_prompt_text
    assert output.image_token_counts == [16]
    image_start = output.input_ids.index(21)
    assert output.image_token_spans == [
        VisionSpan(start=image_start, end=image_start + 16, token_id=21, length=16)
    ]
    assert sum(output.is_multimodal) == 16
    assert all(output.is_multimodal[index] for index in range(image_start, image_start + 16))
    assert not output.is_multimodal[image_start - 1]
    assert not output.is_multimodal[image_start + 16]
    assert output.pixel_values.shape == (64, 3)
    assert output.image_grid_thw.tolist() == [[1, 8, 8]]


def test_vl_processor_converts_output_to_model_inputs():
    processor = VLProcessor(
        tokenizer=TeachingTokenizer(),
        image_processor=TeachingImageProcessor(),
        spatial_merge_size=2,
    )

    output = processor.process(make_request())
    model_inputs = processor.to_model_inputs(output)

    assert set(model_inputs) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
    assert model_inputs["input_ids"].shape == (1, len(output.input_ids))
    assert model_inputs["attention_mask"].shape == (1, len(output.input_ids))
    assert torch.equal(model_inputs["pixel_values"], output.pixel_values)
    assert torch.equal(model_inputs["image_grid_thw"], output.image_grid_thw)

test_vl_processor_converts_output_to_model_inputs()