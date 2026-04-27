import torch
import torch.nn as nn
import pytest

from myvllm.engine.image_sequence import ImageSequence
from myvllm.engine.mm_model_runner import MMModelRunner
from myvllm.multimodal.processor import MultimodalPayload, Processor
from myvllm.models.mm_qwen3 import MMQwen3ForCausalLM
from myvllm.sampling_parameters import SamplingParams


class DummyTokenizer:
    def __init__(self):
        self._special_ids = {
            "<|vision_start|>": 1001,
            "<|image_pad|>": 1002,
            "<|vision_end|>": 1003,
        }

    def encode(self, prompt: str) -> list[int]:
        # Keep tokenization deterministic so the processor test can assert exact layouts.
        return [200 + idx for idx, _ in enumerate(prompt.split(), start=1)]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special_ids[token]


class EchoBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, embedding_dim)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.arange(32 * embedding_dim, dtype=torch.float32).view(32, embedding_dim)
            )

    def forward(self, input_ids=None, *, inputs_embeds=None):
        # The multimodal wrapper should hand the merged embeddings to the backbone unchanged.
        return inputs_embeds


class CapturingMMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        input_ids=None,
        *,
        multimodal_embeddings=None,
        is_multimodal=None,
        inputs_embeds=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids,
                "multimodal_embeddings": multimodal_embeddings,
                "is_multimodal": is_multimodal,
                "inputs_embeds": inputs_embeds,
            }
        )
        return torch.zeros((input_ids.numel(), 4), dtype=torch.float32)

    def compute_logits(self, hidden_states):
        return hidden_states


def make_mm_model() -> MMQwen3ForCausalLM:
    model = MMQwen3ForCausalLM.__new__(MMQwen3ForCausalLM)
    nn.Module.__init__(model)
    model.model = EchoBackbone()
    return model


def test_embed_input_ids_replaces_only_multimodal_positions():
    model = make_mm_model()
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    is_multimodal = torch.tensor([False, True, True, False], dtype=torch.bool)
    multimodal_embeddings = torch.tensor(
        [
            [101.0, 102.0, 103.0, 104.0],
            [201.0, 202.0, 203.0, 204.0],
        ]
    )

    merged = model.embed_input_ids(
        input_ids,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )

    baseline = model.model.embed_tokens(input_ids)
    assert torch.equal(merged[0], baseline[0])
    assert torch.equal(merged[3], baseline[3])
    assert torch.equal(merged[1], multimodal_embeddings[0])
    assert torch.equal(merged[2], multimodal_embeddings[1])


def test_embed_input_ids_rejects_mismatched_multimodal_lengths():
    model = make_mm_model()

    with pytest.raises(ValueError, match="must match"):
        model.embed_input_ids(
            torch.tensor([1, 2, 3], dtype=torch.long),
            multimodal_embeddings=torch.ones(1, 4),
            is_multimodal=torch.tensor([False, True, True], dtype=torch.bool),
        )


def test_processor_builds_token_ids_and_mask_from_prompt_and_image():
    processor = Processor(
        config={"num_vision_tokens": 3},
        tokenizer=DummyTokenizer(),
    )

    prepared = processor.process("hello world", image_path="demo.png")

    assert prepared.token_ids == [1001, 1002, 1002, 1002, 1003, 201, 202]
    assert prepared.multimodal == MultimodalPayload(
        image_path="demo.png",
        num_vision_tokens=3,
        is_multimodal=[False, True, True, True, False, False, False],
        placeholder_token_ids=[1001, 1002, 1002, 1002, 1003],
    )


def test_image_sequence_round_trips_multimodal_payload_state():
    payload = MultimodalPayload(
        image_path="demo.png",
        num_vision_tokens=2,
        is_multimodal=[False, True, True, False],
        placeholder_token_ids=[1001, 1002, 1002, 1003],
    )
    seq = ImageSequence(
        token_ids=[1001, 1002, 1002, 1003, 201, 202],
        sampling_params=SamplingParams(),
        multimodal=payload,
    )

    restored = ImageSequence.__new__(ImageSequence)
    restored.__setstate__(seq.__getstate__())

    assert restored.token_ids == seq.token_ids
    assert restored.multimodal == payload


def test_runner_prefill_drops_empty_multimodal_arguments_for_text_only_batches():
    runner = MMModelRunner.__new__(MMModelRunner)
    runner.model = CapturingMMModel()
    runner._last_multimodal_embeddings = None
    runner._last_is_multimodal = torch.tensor([False, False, False], dtype=torch.bool)

    logits = runner.run_model(torch.tensor([1, 2, 3], dtype=torch.long), is_prefill=True)

    assert logits.shape == (3, 4)
    assert runner.model.calls[0]["multimodal_embeddings"] is None
    assert runner.model.calls[0]["is_multimodal"] is None
