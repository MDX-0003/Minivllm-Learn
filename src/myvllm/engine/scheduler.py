from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # block manager
        self.block_manager = BlockManager(max_cached_blocks, block_size)
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_sequences = max_num_sequences
        # sequence queue
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.eos = eos


    def is_finished(self):
        return len(self.waiting) == 0 and len(self.running) == 0
    #任何seq首次进入schedule时，先进waiting，schedule会尝试从waiting里
    # 调度出满足条件的seq，prefill后进入running，running里不再触发prefill
    def add_sequence(self, sequence: Sequence):
        self.waiting.append(sequence)

    #llmEngine.step()会调用scheduler.schedule()来调度下一批次的序列进行生成
    #内部通过self.waiting和running来判定当前是prefill阶段还是decode阶段
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_sequences = []
        current_scheduled_tokens = 0
        # current_scheduled_tokens在当前schedule调用过程中，
        # 记录“本次已经调度了多少token”提交给LLM,提交数量不会超出max_num_batched_tokens
        # 在prefill阶段，current_scheduled_tokens会随着每个seq的调度而增加，
        # 在completion阶段，current_scheduled_tokens只会随着每个seq被调度而增加1

        # try schedule for prefilling from waiting queue if not exceeding limits
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                seq = self.waiting.popleft() # remove from waiting
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # try schedule for completion from running queue
        while self.running:
            seq = self.running.popleft()
            # use can_append to check whether still space can we append one more token
            if not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    break
                # append one token
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1 # only one token for completion

        # re-add to running queue in the same order
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    # postprocess after generation to check whether sequences are finished
    # if finished, deallocate blocks
    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            # Check stopping conditions:
            # EOS token
            # Reached max_tokens limit (number of completion tokens)
            # Reached max_model_length limit (total sequence length including prompt)
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = 1 + seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)