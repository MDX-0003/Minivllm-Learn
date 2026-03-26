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
        self._last_multimodal_embeddings: torch.Tensor | None = None
        self._last_is_multimodal: torch.Tensor | None = None
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
            ImageSequence(text_token_ids=[0] * max_model_length, image_path=None, num_vision_tokens=0)
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

            t_total = len(seq.token_ids)
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

        # Collect multimodal embeddings and masks.
        device = input_ids_t.device
        dtype = self.model.model.embed_tokens.weight.dtype
        hidden = self.config['hidden_size']

        vision_embeds_list = []
        seq_masks = []

        for seq in seqs:
            t_vis = int(seq.num_vision_tokens)
            t_total = len(seq.token_ids)

            # The sequence already contains:
            # <vision_start> + image_pad * T_vis + <vision_end> + text_tokens
            # Only image_pad positions should be replaced by vision embeddings.
            seq_mask = torch.zeros(t_total, dtype=torch.bool, device=device)
            if seq.placeholder_length:
                seq_mask[:seq.placeholder_length] = torch.tensor(
                    seq.placeholder_mask,
                    dtype=torch.bool,
                    device=device,
                )
                tmp_sum = sum(seq.placeholder_mask)
                print(f"placeholder_length = {seq.placeholder_length}")
                print(f"num_vision_tokens = {seq.num_vision_tokens}")
                print(f"sum of mask = {tmp_sum}")
            if t_vis > 0:
                vis = fake_vision_embeds(
                    image_path=seq.image_path,
                    num_vision_tokens=t_vis,
                    hidden_size=hidden,
                    device=device,
                    dtype=dtype,
                )
                vision_embeds_list.append(vis)
                
            seq_masks.append(seq_mask)

        # 拼装全局 mask 和 所有视觉特征到一维
        is_multimodal = torch.cat(seq_masks)
        
        if vision_embeds_list:
            multimodal_embeddings = torch.cat(vision_embeds_list, dim=0)
        else:
            multimodal_embeddings = None

        # 把这些多模态信息存下来，给之后的 run_model 用
        self._last_multimodal_embeddings = multimodal_embeddings
        self._last_is_multimodal = is_multimodal

        # 我们直接把包含了 vision placeholder 的 input_ids_t 返回给基类！
        # 这样在 run_model 接收到的 input_ids 长度和实际 token 长度一致。
        return input_ids_t

    #is_prefill来自scheduler.schedule，函数自身在scheduler.step -> 父类run 里被调用
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        if is_prefill:
            # 传递 input_ids 、多模态特征、input_ids里的多模态mask，让model内部自己去做 多模态的token merge
            hidden_states = self.model(
                input_ids=input_ids,
                vis_embeds=self._last_multimodal_embeddings,
                vis_masks=self._last_is_multimodal,
            )
            logits = self.model.compute_logits(hidden_states)
            
            # clear to avoid accidentally reusing
            self._last_multimodal_embeddings = None
            self._last_is_multimodal = None
            return logits

        # decode: fall back to parent (CUDA graph etc.)
        return super().run_model(input_ids, is_prefill=is_prefill)
