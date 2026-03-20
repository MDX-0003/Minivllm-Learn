import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        # block_size: number of tokens per block
        self.block_size: int = block_size
        # list of all blocks
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash to block id: this is for prefix caching
        self.hash_to_block_id: dict[int, int] = {}
        # free block ids
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used block ids
        self.used_block_ids: set[int] = set()

    # given token_ids, compute the hash value
    # use prefix_hash_value to compute the hash in a context-sensitive way
    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    # move this block to used list
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # whether we can allocate a block for this sequence
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        h = -1
        #Sequence.block_table：“我（这条 seq）用到了哪些 block”（局部视角）
        #BlockManager.blocks：“每个 block 现在被谁用、内容是什么、能不能复用、能不能回收”（全局视角）
        #见Sequence 类，num_blocks = ceil(len(seq)/block_size)）
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)#从整段seq里切分出block对应的长度
            # only compute hash for full blocks, always -1 for partial blocks
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)#给当前seq的一段block分配hash id
            
            # if cache miss or hash collision
            #如果当前seq切片 在 manager.blocks存在记录（hash），则【缓存命中】
            #命中时，no_cache_found = none
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # update sequence information
                #缓存命中后，更新seq的参数，后续在model_runner里 prefill时，会跳过此变量代表的token
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # update block information, considering the edge case that the block is not allocated yet but with hash code
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                    #缓存命中，但block目前没人用，则更新其状态
                else:
                    # update block information
                    #缓存命中，block目前有人用，则直接更新引用计数
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache miss
                #未命中，则分配新的vlock，登记到manager.blocks里，并把hash和block id的对应关系登记到manager.hash_to_block_id里
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            #将block id记录到seq的table
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        # update block information
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # update sequence information
        seq.block_table = []
        seq.num_cached_tokens = 0

    # this is to check whether we can append tokens to this sequence
    # when that token would require allocating a new block.
    def can_append(self, seq: Sequence) -> bool:
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    # this is the actual work to append tokens to this sequence
    # this is called when the new token has been added to the seq information
    # but no block in gpu has yet allocate for it
    def append(self, seq: Sequence) -> None:
        #只在seq被append_token以后，seq.num_tokens增加了1，且seq.block_table还没有更新的情况下调用
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # if the last block is now full, compute hash
        if seq.num_tokens % self.block_size == 0:#新增的token刚好填满block,可以用于计算hash
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))#把新算的hash存进manager
            self.hash_to_block_id[h] = block.block_id
        # if one new block is needed
        #新token需要单开一个新block
        elif seq.num_tokens % self.block_size == 1:
            # Previous block should be finalized
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # else, do nothing
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
