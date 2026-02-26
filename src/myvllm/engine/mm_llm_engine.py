from __future__ import annotations

import atexit
import time
import torch.multiprocessing as mp

from transformers import AutoTokenizer

from myvllm.engine.image_sequence import ImageSequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.mm_model_runner import MMModelRunner
from myvllm.sampling_parameters import SamplingParams


def worker_process(config, rank, event):
    model_runner = MMModelRunner(config, rank, event)
    model_runner.loop()


class MMLLMEngine:
    """Milestone-1 multimodal engine.

    It mirrors `LLMEngine` but instantiates `ImageSequence` and `MMModelRunner`.
    Only prefill uses the image prefix; decode stays text-only.
    """

    def __init__(self, config: dict):
        self.config = config
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256),
        )

        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()

        self.model_runner = MMModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.processes:
            p.join()

    def step(self):
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            return [], 0, is_prefill
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        self.scheduler.postprocess(scheduled_sequences, outputs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)
        return outputs, num_processed_tokens, is_prefill

    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        image_path = self.config.get("image_path")
        num_vision_tokens = int(self.config.get("num_vision_tokens", 0) or 0)
        seq = ImageSequence(
            token_ids=self.tokenizer.encode(prompt),
            sampling_params=sampling_params,
            image_path=image_path,
            num_vision_tokens=num_vision_tokens,
        )
        self.scheduler.add_sequence(seq)

    def generate(self, prompts: list[str], sampling_params: SamplingParams):
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)

        generated_tokens = {}
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(
                    num_processed_tokens,
                    "number of processed tokens",
                    num_processed_tokens / running_time,
                    "tokens/sec during prefilling",
                )
            else:
                print(
                    num_processed_tokens,
                    "number of processed tokens",
                    num_processed_tokens / running_time,
                    "tokens/sec during decoding",
                )
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {
            "text": [self.tokenizer.decode(tokens) for tokens in generated_tokens],
            "token_ids": generated_tokens,
        }
        return output
