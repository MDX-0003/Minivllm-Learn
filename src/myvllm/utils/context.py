from dataclasses import dataclass 
import torch 


@dataclass
class Context:
    is_prefill: bool = False#prefill时对seq整段做attention，decode时只针对单一token

    # prefill的多seq长度前缀和，如3段seq长[3,2,4],前缀为[0,3,5,9]
    #q是除去缓存命中的长度，k是完整长度,max前缀的成员即数组内的最大值
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None#
    context_lens: torch.Tensor | None = None#decode时每个seq的len，末尾token即decode的本次输入
    block_tables: torch.Tensor | None = None
    #block_table存储了seq【已缓存】的prefix对应的block id,prefill时存下，decode时访问并取kv

_context = Context()

def get_context() -> Context:
    return _context

def reset_context():
    global _context
    _context = Context()
#见prepare_prefill
def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
