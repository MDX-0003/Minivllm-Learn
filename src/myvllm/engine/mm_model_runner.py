from __future__ import annotations

import torch

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
        self._last_prefill_inputs_embeds: torch.Tensor | None = None
        super().__init__(config=config, rank=rank, event=event)

    def build_model(self):
        return MMQwen3ForCausalLM(
            vocab_size=self.config['vocab_size'],
            hidden_size=self.config['hidden_size'],
            num_heads=self.config['num_heads'],
            head_dim=self.config['head_dim'],
            scale=self.config['scale'],
            num_kv_heads=self.config['num_kv_heads'],
            rms_norm_epsilon=self.config['rms_norm_epsilon'],
            qkv_bias=self.config['qkv_bias'],
            base=self.config['base'],
            max_position=self.config['max_position'],
            intermediate_size=self.config['intermediate_size'],
            ffn_bias=self.config['ffn_bias'],
            num_layers=self.config['num_layers'],
            tie_word_embeddings=self.config['tie_word_embeddings'],
            block_size=self.block_size,
        )

    def make_warmup_sequences(self, batch_size: int, max_model_length: int) -> list[ImageSequence]:
        return [
            ImageSequence(token_ids=[0] * max_model_length, image_path=None, num_vision_tokens=0)
            for _ in range(batch_size)
        ]

    def prepare_prefill(self, seqs: list[ImageSequence]) -> torch.Tensor:
        # Similar to base implementation, but lengths include vision prefix.
        input_ids: list[int] = []
        slot_mappings: list[int] = []
        seqlens_q: list[int] = []# length: num_seqs
        seqlens_k: list[int] = []
        cu_seqlens_q = [0]# length: num_seqs + 1
        cu_seqlens_k = [0]
        block_tables = []#num_seqs x num_blocks

        # For Milestone 1: disable prefix caching for multimodal sequences.
        # So we always prefill from scratch.
        for seq in seqs:
            if not isinstance(seq, ImageSequence):
                raise TypeError("MMModelRunner expects ImageSequence")
            seq.num_cached_tokens = 0

            t_vis = int(seq.num_vision_tokens)
            t_text = len(seq.token_ids)
            t_total = t_vis + t_text
            #seq.token_ids依然只代表文本长度，但构建kv input时，长度会算上视觉token
            #所以cu_seqlens_q、cu_seqlens_k、slot_mappings等都要以total长度为准，而不是文本长度。

            # We still only have text token ids.
            input_ids.extend(seq.token_ids)

            # q = tokens computed in this prefill step，包含视觉token和文本token
            seqlens_q.append(t_total)
            # k = total context length after prefill
            seqlens_k.append(t_total)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])#考虑了vis长度的q前缀长度
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
        #input_id即seq.token（文本），这里将其转换到文本embed，视觉embed在下文fake_vision_embeds里补充
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

            #(T_vis, hidden_size)伪造的视觉embed,目前不存在视觉token - 视觉embed这个步骤
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
            #这里完成了图片+文本以embed形式的拼接，每段seq的text前都接上同一份vis占位
        
        #这里把拼接好的embed存到成员，run时作为input进入model
        self._last_prefill_inputs_embeds = torch.cat(pieces, dim=0)

        # Return FULL-LENGTH placeholder ids (vision + text) for compatibility.
        # Even though prefill forward uses `inputs_embeds`, the downstream attention
        # path still infers token count from the tensor passed into `run_model`.
        # Therefore its length must match context/slot_mapping length exactly.
        # 记录下q的前缀长度，注意这是考虑到vis长度的
        full_token_count = cu_seqlens_q[-1]
        # NOTE: do NOT use pin_memory here.
        # Some PyTorch builds/devices reject pinning for certain factory-created tensors.
        # This tensor is just a shape/length placeholder (prefill uses `inputs_embeds`),
        # so pageable CPU memory is fine.
        # input_ids_full将在modelRunner.run里被返回并传入run_model
        # 但实际推理的目标是_last_prefill_inputs_embeds所以只返回一个全零作为占位
        # 注意text only里不是这样，原版不依靠_last_prefill_inputs_embeds这个成员，
        # 只是我们为了接入vis做了这个别扭的改动
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
