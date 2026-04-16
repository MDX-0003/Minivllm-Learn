from pathlib import Path

import torch
from PIL import Image

from myvllm.multimodal.projector import VisionProjector
from myvllm.multimodal.vision_encoder import VisionEncoder


def test_real_vision_pipeline_is_image_conditioned_and_shape_aligned(tmp_path: Path):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"

    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(image_a)
    Image.new("RGB", (64, 64), color=(0, 255, 0)).save(image_b)

    encoder = VisionEncoder(image_size=64, patch_size=16, vision_dim=32, seed=7)
    projector = VisionProjector(vision_dim=32, hidden_size=48, seed=11)

    feats_a_1 = encoder.encode_image(str(image_a))
    feats_a_2 = encoder.encode_image(str(image_a))
    feats_b = encoder.encode_image(str(image_b))

    assert feats_a_1.shape == (16, 32)
    assert torch.allclose(feats_a_1, feats_a_2)
    assert not torch.allclose(feats_a_1, feats_b)

    embeds = projector(feats_a_1, num_vision_tokens=10)
    assert embeds.shape == (10, 48)
