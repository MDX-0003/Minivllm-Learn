from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch import nn


class VisionEncoder(nn.Module):
    """A minimal image-conditioned vision tower that emits patch features."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 14,
        vision_dim: int = 512,
        seed: int = 0,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.vision_dim = int(vision_dim)

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.vision_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
        self.post_norm = nn.LayerNorm(self.vision_dim)

        self.register_buffer("pixel_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        with torch.random.fork_rng():
            torch.manual_seed(int(seed))
            nn.init.xavier_uniform_(self.patch_embed.weight)
            nn.init.zeros_(self.patch_embed.bias)

    def _preprocess(self, image_path: str) -> torch.Tensor:
        #resize to image_size
        image = Image.open(Path(image_path)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        # tobytes()-> bytes(read only )
        # bytearray() -> bytearray(can write) , frombuffer() need a writable buffer
        # memoryview: shader buffer without copy 
        pixel_values = torch.frombuffer(memoryview(bytearray(image.tobytes())), dtype=torch.uint8)
        pixel_values = pixel_values.view(self.image_size, self.image_size, 3).permute(2, 0, 1).unsqueeze(0)
        # [1,3,224,224]
        pixel_values = pixel_values.to(dtype=torch.float32)
        pixel_values = pixel_values / 255.0
        return (pixel_values - self.pixel_mean.cpu()) / self.pixel_std.cpu()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)# [1,3,224,224] -> [1,512,16,16] , because 224 / 14 = 16
        x = x.flatten(2).transpose(1, 2)# [1,512,16,16] -> [1,512,256] -> [1,256,512] = (batch,patch_num,vision_dim)
        return self.post_norm(x) # LN(512) , normlize in each feature

    @torch.inference_mode()
    def encode_image(self, image_path: str | None) -> torch.Tensor:
        if not image_path:
            return torch.empty((0, self.vision_dim), device=self.patch_embed.weight.device, dtype=self.patch_embed.weight.dtype)

        pixel_values = self._preprocess(image_path).to(device=self.patch_embed.weight.device, dtype=self.patch_embed.weight.dtype)
        features = self.forward(pixel_values) # [1,256,512] = (batch,patch_num,vision_dim)
        return features.squeeze(0) # out = [patch_num,vision_dim]
