from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from pprint import pprint

@dataclass
class VLContentItem:
    type: str
    image: str | None = None
    text: str | None = None


@dataclass
class VLMessage:
    role: str
    content: list[VLContentItem]


@dataclass
class VLRequest:
    messages: list[VLMessage]
    add_generation_prompt: bool = True


@dataclass
class SpecialTokenIds:
    im_start_token_id: int | None
    im_end_token_id: int | None
    vision_start_token_id: int
    vision_end_token_id: int
    image_token_id: int


@dataclass
class VisionSpan:
    start: int
    end: int
    token_id: int
    length: int


@dataclass
class VisionInputs:
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    def shape(self):
        return {
            "pixel_shape":self.pixel_values.shape if self.pixel_values.shape  else 0,
            "image_grid_shape" : self.image_grid_thw.shape if self.image_grid_thw.shape  else 0
            }

@dataclass
class VLProcessorOutput:
    messages: list[dict[str, Any]]
    prompt_text: str
    expanded_prompt_text: str
    input_ids: list[int]
    attention_mask: list[int]
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    special_token_ids: SpecialTokenIds
    image_token_counts: list[int]
    image_token_spans: list[VisionSpan]
    is_multimodal: list[bool]


class VLProcessor:
    # Build a teaching Qwen-VL processor around tokenizer and image processor dependencies.
    # Input keys: tokenizer maps rendered prompt text to ids; image_processor maps image paths
    # to pixel_values and image_grid_thw; spatial_merge_size controls image token expansion.
    # Output: an object that can process structured VLRequest values into VLProcessorOutput.
    def __init__(
        self,
        *,
        tokenizer: Any,
        image_processor: Any | None = None,
        spatial_merge_size: int = 2,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.spatial_merge_size = int(spatial_merge_size)
        self.system_prompt = system_prompt
        self.special_token_ids = self._resolve_special_token_ids()
        print("===special_token_ids===")
        pprint(self.special_token_ids)

    # Create a teaching processor from a Hugging Face AutoProcessor.
    # Input key: model_name_or_path identifies the Qwen2.5-VL processor assets.
    # Output: a VLProcessor using the official tokenizer/image processor internals.
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        spatial_merge_size: int = 2,
        system_prompt: str = "You are a helpful assistant.",
        **kwargs: Any,
    ) -> "VLProcessor":
        from transformers import AutoProcessor

        official_processor = AutoProcessor.from_pretrained(model_name_or_path, **kwargs)
        return cls(
            tokenizer=official_processor.tokenizer,
            image_processor=official_processor.image_processor,
            spatial_merge_size=spatial_merge_size,
            system_prompt=system_prompt,
        )

    # Run the full teaching processor contract from structured request to model-ready fields.
    # Input key: request.messages contains ordered role/content items with image and text entries.
    # Output key: VLProcessorOutput stores prompt text, ids, visual tensors, spans, and masks.
    def process(self, request: VLRequest) -> VLProcessorOutput:
        messages = self.build_messages(request)
        prompt_text = self.render_prompt(messages, add_generation_prompt=False)
        vision_inputs = self.preprocess_images(messages)
        image_token_counts = self.compute_image_token_counts(vision_inputs.image_grid_thw)
        expanded_prompt_text = self.expand_image_placeholders(prompt_text, image_token_counts)
        input_ids = self.tokenize(expanded_prompt_text)
        attention_mask = [1] * len(input_ids)
        image_token_spans = self.find_image_token_spans(input_ids)
        is_multimodal = self.build_multimodal_mask(input_ids)
        return VLProcessorOutput(
            messages=messages,
            prompt_text=prompt_text,
            expanded_prompt_text=expanded_prompt_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=vision_inputs.pixel_values,
            image_grid_thw=vision_inputs.image_grid_thw,
            special_token_ids=self.special_token_ids,
            image_token_counts=image_token_counts,
            image_token_spans=image_token_spans,
            is_multimodal=is_multimodal,
        )

    # Convert local dataclass messages into the official Qwen-style messages shape.
    # Input key: request.messages is typed local data; content items carry image path or text.
    # Output key: list[dict] preserves role plus ordered image/text content dictionaries.
    def build_messages(self, request: VLRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            content = []
            for item in message.content:
                if item.type == "image":
                    content.append({"type": "image", "image": item.image})
                elif item.type == "text":
                    content.append({"type": "text", "text": item.text})
                else:
                    raise ValueError(f"Unsupported VL content type: {item.type}")
            messages.append({"role": message.role, "content": content})
        return messages

    # Render structured messages into Qwen chat-template prompt text.
    # Input key: messages are official-shaped role/content dicts; add_generation_prompt appends
    # the assistant prefix for generation. Output key: prompt text still contains one image_pad
    # marker per image before later expansion.
    def render_prompt(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        chunks = [
            "<|im_start|>system\n",
            self.system_prompt,
            "<|im_end|>\n",
        ]
        for message in messages:
            chunks.append(f"<|im_start|>{message['role']}\n")
            for item in message["content"]:
                if item["type"] == "image":
                    chunks.append("<|vision_start|><|image_pad|><|vision_end|>")
                elif item["type"] == "text":
                    chunks.append(str(item["text"]))
                else:
                    raise ValueError(f"Unsupported VL content type: {item['type']}")
            chunks.append("<|im_end|>\n")
        if add_generation_prompt:
            chunks.append("<|im_start|>assistant\n")
        return "".join(chunks)

    # Preprocess image paths into visual model inputs.
    # Input key: messages contain image content dictionaries with image paths.
    # Output keys: pixel_values is the processed visual tensor; image_grid_thw stores grid metadata.
    def preprocess_images(self, messages: list[dict[str, Any]]) -> VisionInputs:
        image_paths = self.extract_image_paths(messages)
        if not image_paths:
            return VisionInputs(pixel_values=None, image_grid_thw=None)
        if self.image_processor is None:
            raise ValueError("image_processor is required when messages contain image items")

        processed = self.image_processor(image_paths)
        return VisionInputs(
            pixel_values=processed.get("pixel_values"),
            image_grid_thw=processed.get("image_grid_thw"),
        )

    # Extract image paths from official-shaped messages in content order.
    # Input key: messages contain content items; image items carry the image path under "image".
    # Output key: list[str] is ordered exactly as images appear in the conversation.
    def extract_image_paths(self, messages: list[dict[str, Any]]) -> list[str]:
        image_paths: list[str] = []
        for message in messages:
            for item in message["content"]:
                if item["type"] == "image":
                    image = item.get("image")
                    if image is None:
                        raise ValueError("Image content item must include an image path")
                    image_paths.append(str(image))
        return image_paths

    # Compute how many image placeholder tokens each visual grid needs.
    # Input key: image_grid_thw rows are [time, height, width] after visual preprocessing.
    # Output key: counts are expanded image_pad token counts after spatial merge.
    def compute_image_token_counts(self, image_grid_thw: torch.Tensor | None) -> list[int]:
        if image_grid_thw is None:
            return []
        counts = []
        merge_area = self.spatial_merge_size * self.spatial_merge_size

        for t, h, w in image_grid_thw.tolist():
            grid_tokens = int(t) * int(h) * int(w)
            if grid_tokens % merge_area != 0:
                raise ValueError("image_grid_thw token count must be divisible by merge area")
            counts.append(grid_tokens // merge_area)

        return counts

    # Expand one textual image placeholder into the exact number of image_pad markers.
    # Input keys: prompt_text contains one <|image_pad|> per image; image_token_counts gives
    # the expanded token count for each image. Output key: expanded prompt text is tokenizable.
    def expand_image_placeholders(self, prompt_text: str, image_token_counts: list[int]) -> str:
        expanded = prompt_text
        placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        for count in image_token_counts:
            replacement = "<|vision_start|>" + ("<|image_pad|>" * count) + "<|vision_end|>"
            if placeholder not in expanded:
                raise ValueError("Prompt text has fewer image placeholders than image inputs")
            expanded = expanded.replace(placeholder, replacement, 1)
        if placeholder in expanded:
            raise ValueError("Prompt text has more image placeholders than image inputs")
        return expanded

    # Tokenize expanded prompt text into integer token ids.
    # Input key: prompt_text has already-expanded image_pad markers.
    # Output key: list[int] is the flattened model prompt token sequence.
    def tokenize(self, prompt_text: str) -> list[int]:
        return list(self.tokenizer.encode(prompt_text, add_special_tokens=False))

    # Find contiguous image token spans inside input ids.
    # Input key: input_ids is the flattened prompt sequence.
    # Output key: VisionSpan entries mark [start, end) ranges occupied by image_token_id.
    def find_image_token_spans(self, input_ids: list[int]) -> list[VisionSpan]:
        spans: list[VisionSpan] = []
        image_token_id = self.special_token_ids.image_token_id
        index = 0
        while index < len(input_ids):
            if input_ids[index] != image_token_id:
                index += 1
                continue
            start = index
            while index < len(input_ids) and input_ids[index] == image_token_id:
                index += 1
            spans.append(
                VisionSpan(
                    start=start,
                    end=index,
                    token_id=image_token_id,
                    length=index - start,
                )
            )
        return spans

    # Build the boolean mask used by embedding-merge code.
    # Input key: input_ids is the flattened prompt sequence.
    # Output key: True values mark image_token_id positions only, not vision_start/end.
    def build_multimodal_mask(self, input_ids: list[int]) -> list[bool]:
        image_token_id = self.special_token_ids.image_token_id
        return [token_id == image_token_id for token_id in input_ids]

    # Convert processor output into a Hugging Face-style model input dictionary.
    # Input key: output carries flat lists and optional visual tensors.
    # Output keys: tensors are batched as input_ids/attention_mask plus visual fields when present.
    def to_model_inputs(self, output: VLProcessorOutput) -> dict[str, torch.Tensor]:
        model_inputs: dict[str, torch.Tensor] = {
            "input_ids": torch.tensor([output.input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([output.attention_mask], dtype=torch.long),
        }
        if output.pixel_values is not None:
            model_inputs["pixel_values"] = output.pixel_values
        if output.image_grid_thw is not None:
            model_inputs["image_grid_thw"] = output.image_grid_thw
        return model_inputs

    # Resolve Qwen special token ids from the tokenizer once.
    # Input key: tokenizer.convert_tokens_to_ids maps marker strings to ids.
    # Output key: SpecialTokenIds centralizes text and vision marker ids.
    def _resolve_special_token_ids(self) -> SpecialTokenIds:
        return SpecialTokenIds(
            im_start_token_id=self._safe_token_id("<|im_start|>"),
            im_end_token_id=self._safe_token_id("<|im_end|>"),
            vision_start_token_id=int(self.tokenizer.convert_tokens_to_ids("<|vision_start|>")),
            vision_end_token_id=int(self.tokenizer.convert_tokens_to_ids("<|vision_end|>")),
            image_token_id=int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>")),
        )

    # Resolve an optional special token id without failing older or minimal tokenizers.
    # Input key: token is a marker string. Output key: int id or None when unavailable.
    def _safe_token_id(self, token: str) -> int | None:
        try:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
        except KeyError:
            return None
        if token_id is None:
            return None
        return int(token_id)
