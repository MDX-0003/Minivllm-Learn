import torch 
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    A custom sampler layer that selects elements from the input tensor
    based on provided indices.
    """

    def __init__(self):
        super().__init__()
        self.seed = 1234
        self._rng = None
        self._rng_device = None
        
    #SamplerLayer被runner持有，run求得logit以后用来完成采样，并交给上层调度器（这里是谁在管理？）append到seq
    #采样策略可以是多样的，但目标都是根据logits和temperature计算出下一个token id
    # 这里实现了一个简单的基于softmax的采样方法
    @torch.compile
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        #logits要理解为长 vocal_size的一个向量，元素代表着词表所有token的分数，分数数值不固定，也不会内积为1
        #正因为是原始分数，所以可以用temperature来缩放softmax以后，分数之间的差异

        #logit一定要经过softmax才会转变为概率分布
        logits/= temperature.unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)
        device = probs.device
        if self._rng is None or self._rng_device != device:
            self._rng = torch.Generator(device=device)
            self._rng.manual_seed(self.seed)
            self._rng_device = device
        noise = torch.empty_like(probs).exponential_(1, generator=self._rng).clamp_min_(1e-10)
        sample_tokens = probs.div_(noise).argmax(dim=-1)
        return sample_tokens