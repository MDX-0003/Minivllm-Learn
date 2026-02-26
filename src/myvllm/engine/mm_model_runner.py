from __future__ import annotations

import torch

import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event

from myvllm.engine.model_runner import ModelRunner
from myvllm.engine.image_sequence import ImageSequence
from myvllm.models.mm_qwen3 import MMQwen3ForCausalLM
from myvllm.utils import set_context
from myvllm.utils.fake_vision import fake_vision_embeds


class MMModelRunner(ModelRunner):
    """Multimodal ModelRunner (Milestone 1).

    Changes vs base:
    - Prefill: treat each sequence as (vision_prefix + text tokens).
    - Prefill: build `inputs_embeds` = concat(vision_embeds, text_embeds) and forward via inputs_embeds.
    - Decode: unchanged (still uses token ids; vision is already in KV cache).
    """

    def __init__(self, config: dict, rank: int, event):
        self._init_from_base_with_mm_model(config=config, rank=rank, event=event)

    def _init_from_base_with_mm_model(self, *, config: dict, rank: int, event):
        """Initialize MMModelRunner by mirroring ModelRunner.__init__.

        We deliberately DO NOT call `ModelRunner.__init__()`.

        What we *copied* from `ModelRunner.__init__` (same logic, same order):
        - distributed initialization (`dist.init_process_group`, `torch.cuda.set_device`)
        - sampler creation
        - warmup -> KV cache allocation -> optional CUDA graph capture
        - default device/dtype setup
        - shared-memory setup for multi-process RPC

        What we *changed* vs `ModelRunner.__init__` and why:
        - **Model construction**: base builds `Qwen3ForCausalLM` (text-only).
          We build `MMQwen3ForCausalLM` so prefill can accept `inputs_embeds`.
        - **Warmup timing**: base warmup runs immediately with the text-only model.
          For multimodal, warmup must run after the MM-capable model exists;
          otherwise prefill would need to pass `inputs_embeds` into a model that
          doesn't accept it.
        - **Sequence type during warmup**: base warmup constructs `Sequence`.
          Our overridden `prepare_prefill` expects `ImageSequence`, so warmup must
          construct `ImageSequence` (with no image / no vision prefix) to keep the
          execution contract consistent.
        """

        self.config = config
        self.event = event

        # set distributed config
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # set model (MM-enabled)
        self.model = MMQwen3ForCausalLM(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            num_heads=config['num_heads'],
            head_dim=config['head_dim'],
            scale=config['scale'],
            num_kv_heads=config['num_kv_heads'],
            rms_norm_epsilon=config['rms_norm_epsilon'],
            qkv_bias=config['qkv_bias'],
            base=config['base'],
            max_position=config['max_position'],
            intermediate_size=config['intermediate_size'],
            ffn_bias=config['ffn_bias'],
            num_layers=config['num_layers'],
            tie_word_embeddings=config['tie_word_embeddings'],
            block_size=self.block_size,
        )

        # Load weights in GPU (model moved to GPU before loading weights)
        self.model = self.model.cuda(rank)

        # Load pretrained weights if model_name_or_path is provided
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        from myvllm.layers.sampler import SamplerLayer
        self.sampler = SamplerLayer()

        # Store default dtype before it's needed in allocate_kv_cache
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step
        self._first_decode = False

        # Small cache for the last prefill embeds (used immediately by run_model).
        self._last_prefill_inputs_embeds: torch.Tensor | None = None

        # warm up model so that we know peak memory usage
        self.warmup_model()
        # allocate kv cache
        self.allocate_kv_cache()
        # capture cuda graph for decoding
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        # shared memory and barrier
        if self.world_size > 1:
            dist.barrier()
            if self.rank == 0:
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')

    def warmup_model(self):
        """Warm up with ImageSequence to match MM prefill path.

        Base ModelRunner.warmup_model creates plain `Sequence`, but MMModelRunner
        overrides `prepare_prefill` and expects `ImageSequence`.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        # Warmup should behave like text-only: no image and no vision prefix.
        seqs = [
            ImageSequence(token_ids=[0] * max_model_length, image_path=None, num_vision_tokens=0)
            for _ in range(batch_size)
        ]
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def prepare_prefill(self, seqs: list[ImageSequence]) -> torch.Tensor:
        # Similar to base implementation, but lengths include vision prefix.
        input_ids: list[int] = []
        slot_mappings: list[int] = []
        seqlens_q: list[int] = []
        seqlens_k: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        block_tables = []

        # For Milestone 1: disable prefix caching for multimodal sequences.
        # So we always prefill from scratch.
        for seq in seqs:
            if not isinstance(seq, ImageSequence):
                raise TypeError("MMModelRunner expects ImageSequence")
            seq.num_cached_tokens = 0

            t_vis = int(seq.num_vision_tokens)
            t_text = len(seq.token_ids)
            t_total = t_vis + t_text

            # We still only have text token ids.
            input_ids.extend(seq.token_ids)

            # q = tokens computed in this prefill step
            seqlens_q.append(t_total)
            # k = total context length after prefill
            seqlens_k.append(t_total)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])

            # slot mapping must cover BOTH vision + text tokens.
            if seq.block_table:
                # seq.block_table is sized for total tokens because Scheduler/BlockManager uses len(seq)
                # (and ImageSequence.__len__ includes vision prefix).
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id + 1) * self.block_size)))
                    else:
                        slot_mappings.extend(
                            list(
                                range(
                                    block_id * self.block_size,
                                    block_id * self.block_size + seq.last_block_num_tokens,
                                )
                            )
                        )

        # Pad block_tables (required when using cached K/V in attention kernels)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables) if all_block_tables else 0
        for seq in seqs:
            block_table = seq.block_table + [-1] * (max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)

        # Move text token ids to GPU for embedding lookup.
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q) if seqlens_q else 0,
            max_seqlen_k=max(seqlens_k) if seqlens_k else 0,
            slot_mapping=slot_mapping_t,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            if block_tables
            else None,
        )

        # Build inputs_embeds for the full (vision+text) concatenated stream.
        # 1) get text embeds for concatenated text token ids
        text_embeds = self.model.model.embed_tokens(input_ids_t)

        # 2) split per-seq and prefix vision embeds
        device = text_embeds.device
        dtype = text_embeds.dtype
        hidden = text_embeds.size(-1)

        pieces = []
        offset = 0
        for seq in seqs:
            t_vis = int(seq.num_vision_tokens)
            t_text = len(seq.token_ids)
            vis = fake_vision_embeds(
                image_path=seq.image_path,
                num_vision_tokens=t_vis,
                hidden_size=hidden,
                device=device,
                dtype=dtype,
            )
            txt = text_embeds[offset : offset + t_text]
            offset += t_text
            pieces.append(torch.cat([vis, txt], dim=0))

        self._last_prefill_inputs_embeds = torch.cat(pieces, dim=0)

        # Return FULL-LENGTH placeholder ids (vision + text) for compatibility.
        # Even though prefill forward uses `inputs_embeds`, the downstream attention
        # path still infers token count from the tensor passed into `run_model`.
        # Therefore its length must match context/slot_mapping length exactly.
        full_token_count = cu_seqlens_q[-1]
        # NOTE: do NOT use pin_memory here.
        # Some PyTorch builds/devices reject pinning for certain factory-created tensors.
        # This tensor is just a shape/length placeholder (prefill uses `inputs_embeds`),
        # so pageable CPU memory is fine.
        input_ids_full = torch.zeros(full_token_count, dtype=torch.long).cuda(non_blocking=True)
        return input_ids_full

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill:
            if self._last_prefill_inputs_embeds is None:
                raise RuntimeError("Prefill inputs_embeds not prepared. Call prepare_prefill first.")
            hidden_states = self.model(inputs_embeds=self._last_prefill_inputs_embeds)
            logits = self.model.compute_logits(hidden_states)
            # clear to avoid accidentally reusing
            self._last_prefill_inputs_embeds = None
            return logits

        # decode: fall back to parent (CUDA graph etc.)
        return super().run_model(input_ids, is_prefill=is_prefill)
