"""Milestone 1: fake vision prefix embeddings.

Dependency-free placeholder vision encoder.

Contract:
- Input: optional `image_path`.
- Output: `vision_embeds` with shape (T_vis, hidden_size).

Properties:
- Deterministic per image bytes (different images -> different prefix).
- Small magnitude for stability.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def _seed_from_bytes(data: bytes) -> int:
    h = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(h, "little", signed=False)


@torch.inference_mode()
def fake_vision_embeds(
    *,
    image_path: str | None,
    num_vision_tokens: int,
    hidden_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = int(num_vision_tokens)
    if t <= 0:
        return torch.empty((0, hidden_size), device=device, dtype=dtype)

    if not image_path:
        return torch.zeros((t, hidden_size), device=device, dtype=dtype)

    data = Path(image_path).read_bytes()
    seed = _seed_from_bytes(data)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x = torch.randn((t, hidden_size), generator=g, device="cpu", dtype=torch.float32) * 0.02
    return x.to(device=device, dtype=dtype)
