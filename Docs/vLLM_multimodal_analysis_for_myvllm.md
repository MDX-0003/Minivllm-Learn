# vLLM 多模态实现分析：如何给当前 `myvllm` 做一个“学习版可落地”的图片读取能力

> 目标：参考当前虚拟环境里已安装的官方 `vllm==0.15.0` 源码，分析它是如何支持多模态输入的，并结合你当前仓库的 `Milestone 1`，给出一条**尽量少改现有架构**、适合学习项目推进的落地路线。

---

## 1. 先说结论

如果只看“让模型能看到图片”这件事，官方 `vllm` 的多模态设计可以浓缩成 4 个核心动作：

1. **在 prompt/token 序列里预留 placeholder 位置**
	- 也就是先把图片对应的“token 位子”占出来。
	- 这样模型的 `positions`、`slot_mapping`、KV-cache 长度从一开始就对齐。

2. **把原始图片走一条独立的 processor/vision tower 路径，产出视觉 embeddings**
	- 图片不会直接变成普通 `input_ids`。
	- 它会先经过 preprocess、vision encoder、projector，最后变成 `(T_vis, hidden_size)`。

3. **在模型内部把 placeholder 对应的文本 embedding 替换成视觉 embedding**
	- 不是 runner 外面简单 `concat([vision, text])` 就结束。
	- 而是让输入 token 序列仍然完整保留，然后用 mask 把某些 token 位子的 embedding 改成视觉 embedding。

4. **decode 阶段继续走标准语言模型路径**
	- 因为视觉信息已经在 prefill 写进 KV-cache 了。
	- decode 不需要再次处理图片，这点和你 `Milestone 1` 的思路是一致的。

对你这个仓库来说，**最值得借鉴的不是官方实现有多复杂，而是它的“边界划分”非常清楚**：

- 输入处理归 `processor`
- 视觉特征提取归 `vision tower + projector`
- 文本/多模态融合归 `model.embed_input_ids(...)`
- 调度/KV/cache 仍然只看“最终总 token 数”

这意味着：

**你完全可以保留现在的 engine/scheduler/block manager 主体不动，只额外增加一层“placeholder + merge embeddings”的轻量多模态协议。**

---

## 2. 你当前仓库的现状：已经做对了什么

结合 `MileStone-note1.md` 和当前代码，`Milestone 1` 已经完成了一个很宝贵的验证：

### 2.1 你已经证明了这条大方向是对的

你当前做法是：

- 使用 `ImageSequence` 把总长度改成：

$$
T_{total} = T_{vis} + T_{text}
$$

- 在 `MMModelRunner.prepare_prefill()` 里构造：
  - `slot_mapping`
  - `cu_seqlens_q/k`
  - `inputs_embeds = concat(vision_embeds, text_embeds)`

- decode 阶段保持 text-only

这已经抓住了多模态推理里最关键的那一半：

> **只要 prefill 的长度账和 KV-cache 写入账是对的，decode 就可以继续复用原来的高性能路径。**

这点和官方 `vllm` 是一致的。

### 2.2 你当前方案的优点

1. **侵入小**：尽量通过新增 `MMModelRunner`、`MMLLMEngine`、`MMQwen3ForCausalLM` 扩展，而不是直接重写整套 text-only 流程。
2. **工程上可控**：把多模态限制在 prefill，避免破坏 decode/CUDA graph。
3. **很适合学习项目**：先把“长度 / KV / runner / model forward”链路跑通，再谈真实视觉语义。

### 2.3 你当前方案的主要瓶颈

但它也有明显上限：

1. **视觉 token 不存在于 token 序列本身，只存在于 runner 临时拼出来的 `inputs_embeds` 里**
2. **没有明确的 placeholder 协议**
	- 模型层看不到“哪些位置是视觉位”
3. **没有 processor 层**
	- `image_path -> image tensor -> vision features -> projected embeddings` 这条链尚未模块化
4. **和未来真实 VL 权重不容易对齐**
	- 官方 VL 模型通常依赖明确的 placeholder token / image token protocol

所以：

> 你的 `Milestone 1` 很适合作为“KV 与调度打通”的第一步，但如果要继续往“真正可读图”的方向走，下一步最值得补的不是大改 engine，而是**学习官方的 placeholder + processor + embed merge 机制**。

---

## 3. 官方 `vllm` 是怎么做多模态的

下面基于本机已安装的 `vllm==0.15.0` 源码分析。

### 3.1 本机环境信息

通过当前虚拟环境查询到：

- 包名：`vllm`
- 版本：`0.15.0`
- 安装路径：

`/home/dministrator/Minivllm-Learn/.venv/lib/python3.11/site-packages`

重点阅读了这些文件：

- `vllm/multimodal/inputs.py`
- `vllm/multimodal/registry.py`
- `vllm/multimodal/processing/processor.py`
- `vllm/model_executor/models/interfaces.py`
- `vllm/model_executor/models/utils.py`
- `vllm/model_executor/models/llava.py`
- `vllm/model_executor/models/qwen3_vl.py`

---

## 4. 官方设计的关键分层

### 4.1 第一层：多模态输入抽象 `multimodal/inputs.py`

这里定义了一个通用输入字典，例如：

- `image`
- `video`
- `audio`

最重要的不是字段名，而是这里引入了一个很关键的概念：

### `PlaceholderRange`

它记录：

- placeholder 在 prompt 里的偏移 `offset`
- placeholder 长度 `length`
- 哪些位置真正需要被 embedding 覆盖 `is_embed`

这说明官方设计从最开始就不是“图片额外插在 prompt 前面就完事”，而是：

> **把图片视作 prompt 中某一段 token 区间，这个区间先由 placeholder token 占位，之后再用视觉 embedding 覆盖。**

这一步的意义非常大：

1. prompt 长度天然包含视觉 token
2. positions、attention、KV-cache 账本天然对齐
3. 多图片时也能知道每张图对应 prompt 的哪一段

这比单纯在 runner 外拼 `concat([vision, text])` 更稳定。

---

### 4.2 第二层：`MultiModalRegistry` + 模型专属 processor

在 `vllm/multimodal/registry.py` 中，官方把“多模态输入如何处理”注册到具体模型类上。

也就是说，**不是所有模型共享一个通用图片处理逻辑**，而是：

- LLaVA 一套 processor
- Qwen2-VL 一套 processor
- Qwen3-VL 一套 processor

这样做的原因很现实：

- 每个 VL 模型的 placeholder token 不一样
- 图像预处理不一样
- vision tower 不一样
- projector 不一样
- 图像 token 数计算方式也不一样

对学习项目来说，这里最值得借鉴的是这种“职责拆分”：

### 处理图片，不应该直接写死在 runner 里

更合理的结构是：

- `processor`：负责图片预处理、决定 placeholder 长度、返回多模态 kwargs
- `model`：负责把多模态 kwargs 变成 embeddings，并 merge 到 text embeddings 中
- `runner`：负责像以前一样跑 prefill/decode 和 KV-cache

---

### 4.3 第三层：processor 负责把 prompt 改造成“带 placeholder 的 prompt”

在 `vllm/multimodal/processing/processor.py` 和 `llava.py` 里，官方做了一件非常关键的事：

### 它会改写 prompt / token ids

以 `LLaVA` 为例，大意是：

- 用户 prompt 里原本可能只有一个 `<image>`
- 但真正送入模型前，这个 `<image>` 会被替换成很多个 image placeholder token
- 数量正好等于视觉塔输出的 token 数

这和你当前 Milestone 1 的本质差异在于：

- 你现在是：**视觉长度存在于 `ImageSequence.num_tokens`，但 `token_ids` 本身没有对应的 placeholder token**
- 官方是：**视觉长度同时存在于 token 序列里和 embedding merge 协议里**

这会带来两个好处：

1. **总长度来自一套统一真相来源**
	- 不再需要特别依赖 runner 里“占位 `input_ids_full` 长度要凑齐”这种兼容技巧
2. **模型侧天然知道哪些 token 是多模态 placeholder**
	- 后续 merge 更清晰

---

### 4.4 第四层：模型实现 `embed_multimodal(...)`

在 `vllm/model_executor/models/interfaces.py` 中，所有支持多模态的模型都要实现：

- `get_placeholder_str(...)`
- `embed_multimodal(...)`
- `get_language_model()`

这里的核心思想是：

### 多模态模型 = 视觉塔 + 语言模型 的组合

语言模型还是语言模型，只是多了一层：

- `embed_multimodal(**kwargs)`：把图片变成视觉 embeddings
- `embed_input_ids(...)`：把 text embeddings 和 multimodal embeddings 合并

这是一种非常适合学习项目的拆法，因为它允许你：

- 不重写整个 transformer
- 只在 embedding 入口加一个多模态替换逻辑

---

### 4.5 第五层：真正的“融合”发生在 `_merge_multimodal_embeddings(...)`

这个函数在：

`vllm/model_executor/models/utils.py`

核心逻辑是：

```python
def _merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal):
	 inputs_embeds.masked_scatter_(
		  is_multimodal.unsqueeze(-1),
		  mm_embeds_flat.to(dtype=input_dtype)
	 )
```

它的意思非常直白：

1. 先用普通 `input_ids` 做文本 embedding
2. 再把 `is_multimodal == True` 的位置替换成视觉 embedding

也就是说，官方并不是把多模态完全变成另一条独立前向路径，而是：

> **仍然保留标准 token 序列，只是在 embedding 层做“局部替换”。**

这是我认为你接下来最值得迁移的设计。

---

## 5. 以 `LLaVA` 为例看官方完整调用链

`vllm/model_executor/models/llava.py` 特别适合作为学习参考，因为逻辑相对直观。

### 5.1 `get_placeholder_str(...)`

它返回：

```python
"<image>"
```

这说明 prompt 里有明确图片占位符。

### 5.2 processor 扩展占位符

processor 会根据图像大小和视觉塔输出 token 数，把 `<image>` 替换成足够多的 placeholder token。

### 5.3 `embed_multimodal(...)`

模型内部调用 vision tower + projector：

- 图像 -> image features
- image features -> projected image embeddings

### 5.4 `forward(...)`

`llava.py` 的注释写得很清楚：

> `input_ids` 已经包含了预留给 image embeddings 的位置。

这句话非常重要。

它意味着：

- KV-cache 的 token 数不是 runner 猜出来的
- 而是 prompt/token 本身就已经对齐了

然后 forward 里再把这些 placeholder 位置替换成视觉 embedding。

---

## 6. 以 `Qwen3-VL` 为例看更贴近你的目标模型的实现

`vllm/model_executor/models/qwen3_vl.py` 更接近你现在的 `Qwen3` 学习仓库。

### 6.1 placeholder 形式

它使用：

- `"<|vision_start|><|image_pad|><|vision_end|>"`
- `"<|vision_start|><|video_pad|><|vision_end|>"`

也就是说，Qwen 系 VL 模型通常不是单个 `<image>`，而是一组专门的视觉 token 协议。

### 6.2 `embed_multimodal(...)`

会：

1. 解析多模态输入
2. 分 modality 处理 `image` / `video`
3. 经过 vision tower 得到 embeddings
4. 返回一个按出现顺序排列好的 embedding 列表/元组

### 6.3 `embed_input_ids(...)`

这里是最值得记下来的：

```python
inputs_embeds = self._embed_text_input_ids(...)
inputs_embeds = _merge_multimodal_embeddings(...)
```

这说明 Qwen3-VL 最终也是沿用同样思路：

> **先 embed 文本，再按 `is_multimodal` mask 原位覆盖视觉 embedding。**

所以从架构思想上看，官方 `vllm` 的 VL 路线并不是“完全特殊处理”，而是：

- token 序列上：placeholder 占位
- embedding 层：局部替换
- transformer 主体：尽量复用原文本模型

这跟你“时间有限、尽量少动已有模型结构”的目标非常契合。

---

## 7. 你应该从官方方案里借什么，不借什么

### 7.1 建议直接借鉴的部分

#### A. 借“placeholder token + embedding merge”机制

这是最关键的。

理由：

- 它能让 `token_ids`、`num_tokens`、`positions`、`slot_mapping` 统一起来
- 对 KV-cache 很友好
- 对未来接真实 VL 权重也更自然

#### B. 借“processor 层”的职责划分

哪怕你不实现完整 registry，也建议单独做：

- `src/myvllm/multimodal/processor.py`

负责：

- 读图
- resize/normalize
- 生成 `pixel_values`
- 计算视觉 token 数
- 生成 placeholder token ids

#### C. 借“embed_multimodal(...)` + `embed_input_ids(...)`”接口风格

你不需要照搬官方所有协议，但建议把模型侧接口变成：

- `embed_multimodal(image_inputs) -> Tensor[(T_vis, hidden)]`
- `embed_input_ids(input_ids, multimodal_embeddings=None, is_multimodal=None)`

这会让“文本 embedding”和“多模态覆盖”边界非常清楚。

---

### 7.2 不建议你现在就照搬的部分

#### A. 不建议现在就照搬完整 `MultiModalRegistry`

原因很简单：

- 这是官方为了支持几十种模型做的基础设施
- 对你的学习仓库来说太重了

你当前只需要一个非常小的版本：

- 一个 `processor`
- 一个 `vision tower`
- 一个 `MMQwen3ForCausalLM`

就足够。

#### B. 不建议现在就照搬所有 video / audio / pruning / deepstack 逻辑

这些都是后续复杂化的方向。

你现在应该聚焦在：

> **单张图片 + 单轮 prompt + prefill 写入有效视觉信息 + decode 能稳定生成。**

#### C. 不建议一开始就强行兼容官方 Qwen3-VL 权重

因为这会把你马上拖进：

- 特殊 token 协议
- 专用 processor
- MRoPE / vision positional encoding
- projector 权重映射

这对学习项目会一下子过陡。

更现实的是：

先做一个**结构相似、但实现简化**的学习版。

---

## 8. 给你项目的“学习版落地方案”

下面这套方案的目标是：

> 在保留你现有 `Milestone 1` 大方向的前提下，把它从“runner 直接拼 fake embeddings”升级成“更接近官方 vLLM 的 placeholder + vision tower + merge 方案”。

我建议分三步走。

---

## 9. 阶段一：把当前 `Milestone 1` 从“前缀拼接”升级成“placeholder 替换”

### 9.1 目标

先不接真实 vision tower。

只做这件事：

- prompt/token 里显式插入若干 image placeholder token
- 模型 embedding 阶段把这些 placeholder token 的 embeddings 替换成 `fake_vision_embeds`

也就是说：

- **保留 fake vision**
- **改掉 runner 手工拼接 `[vision; text]` 的方式**

### 9.2 为什么这一步很值

因为这是一个高性价比升级：

1. 你不需要立刻实现真实视觉塔
2. 但你已经能把架构从“临时拼 tensor”升级成更像官方的协议
3. 后面把 fake vision 换成 real vision 时，runner 可以几乎不动

### 9.3 建议改法

#### 新增一个专用 image placeholder token id

例如在 config 里定义一个保留 token id：

- `image_token_id`

注意：

你的 tokenizer 和词表目前未必真的有这个 special token。

在“学习版最小改动”下，可以先这么做：

- 选一个不会正常出现在 prompt 中的特殊 id 作为内部占位
- 在 embedding 时遇到这个 id，不走普通 `embed_tokens`，而是由视觉 embedding 覆盖

#### 改 `ImageSequence`

让它不只是 `num_tokens = num_vision_tokens + len(text_token_ids)`。

而是显式保存：

- `prompt_token_ids_with_placeholders`
- `image_placeholder_mask`

或者最简单：

- `token_ids` 里直接包含 `num_vision_tokens` 个 `image_token_id`
- 然后再跟上文本 token ids

这样 `Sequence.num_tokens`、`block_table`、`slot_mapping` 就自然对齐了。

#### 改模型 embedding 入口

建议在 `MMQwen3ForCausalLM` 或 `Qwen3Model` 外围加一个新方法：

- `embed_input_ids(input_ids, multimodal_embeddings=None, is_multimodal=None)`

逻辑类似官方：

1. 正常 text embedding
2. 生成 `is_multimodal = input_ids == image_token_id`
3. 用 `masked_scatter` 或索引赋值把这些位置替换成视觉 embeddings

#### runner 只负责准备 mask/embeddings，不再负责 concat

`MMModelRunner.prepare_prefill()` 可以改为：

- 返回包含 placeholder 的 `input_ids`
- 额外缓存：
  - `multimodal_embeddings`
  - `is_multimodal`

然后在 prefill forward 调用：

- `self.model(input_ids=input_ids, multimodal_embeddings=..., is_multimodal=...)`

这比现在把 `inputs_embeds` 作为独立路径塞进去更贴近官方设计。

### 9.4 这一阶段做完后，你会得到什么

你会把当前架构从：

- `runner concat embeddings`

升级成：

- `token placeholders + model-side embedding merge`

这会是一个非常重要的“架构转正”节点。

---

## 10. 阶段二：把 `fake_vision_embeds` 换成轻量真实视觉塔

### 10.1 目标

在不追求完全复现 Qwen3-VL 官方权重的前提下，让模型确实接收到来自图片内容的非随机语义特征。

### 10.2 建议最小路线

对学习项目，我建议优先这两种策略之一：

#### 方案 A：接一个现成视觉 backbone，自己加 projector

例如：

- CLIP vision encoder
- SigLIP vision encoder

然后加一层：

- `nn.Linear(vision_dim, hidden_size)` 或两层 MLP projector

输出：

$$
vision\_embeds \in \mathbb{R}^{T_{vis} \times hidden\_size}
$$

优点：

- 结构简单
- 容易理解
- 非常符合学习项目

缺点：

- 不一定和你加载的语言模型权重完全对齐
- 生成质量有限

但这一步的重点不是“答得多准”，而是：

> **让图片信息经过真实视觉特征链路进入语言模型。**

#### 方案 B：先用 HuggingFace 已有图像模型输出 patch embeddings

如果你不想自己搭 vision transformer，可以先借一个现成模型输出中间层特征，再做简单 projector。

### 10.3 模块建议

建议新增：

- `src/myvllm/multimodal/processor.py`
  - 负责 PIL 读图、resize、normalize、tensor 化
- `src/myvllm/multimodal/vision_encoder.py`
  - 负责真实视觉特征提取
- `src/myvllm/multimodal/projector.py`
  - 负责映射到 `hidden_size`

然后保留：

- `src/myvllm/models/mm_qwen3.py`
  - 负责 merge embeddings

这样后面替换视觉塔时，runner 不需要乱动。

---

## 11. 阶段三：如果未来要更贴近官方 VL 权重，再补协议细节

这一步不是现在必须做，但可以作为路线图。

后续如果你想靠近真正的 `Qwen3-VL`，需要逐步补：

1. 专用 placeholder token 协议
2. 与视觉塔一致的 image preprocess
3. 与官方一致的 projector 结构
4. 更精确的 vision position / MRoPE 处理
5. 多图片输入顺序及占位范围管理
6. image hash 参与 prefix cache key

这会复杂很多，但前提是你已经把阶段一、阶段二的边界划清楚。

---

## 12. 结合你的仓库，我建议的具体文件级改动

下面是我认为最适合你当前项目节奏的文件级计划。

### 12.1 建议保留的已有文件

- `src/myvllm/engine/mm_llm_engine.py`
- `src/myvllm/engine/mm_model_runner.py`
- `src/myvllm/engine/image_sequence.py`
- `src/myvllm/models/mm_qwen3.py`

这些都可以继续沿用，但职责建议微调。

### 12.2 建议新增的目录与文件

建议新增一个独立子目录：

- `src/myvllm/multimodal/`

其中放：

#### `processor.py`

负责：

- 读图片
- 生成 placeholder token ids
- 计算 `num_vision_tokens`
- 返回：
  - `input_ids_with_placeholders`
  - `image_path` / `pixel_values` / `image_tensor`
  - `is_multimodal_mask`

#### `vision_encoder.py`

负责：

- 图像张量 -> 视觉特征

#### `projector.py`

负责：

- 视觉特征 -> `hidden_size`

### 12.3 `ImageSequence` 的建议改造

当前它主要是“长度补偿器”。

建议把它升级成“多模态 prompt 容器”：

至少保存：

- `image_path`
- `num_vision_tokens`
- `token_ids`（含 placeholder）
- `image_token_id`

最好再加：

- `multimodal_start_idx`
- `multimodal_end_idx`

这样调试 `slot_mapping` 和 stop condition 会方便很多。

### 12.4 `MMModelRunner` 的建议改造

当前 `prepare_prefill()` 的最大职责是：

- 负责长度账
- 负责拼 `inputs_embeds`

建议改成：

#### 保留：

- 构造 `slot_mapping`
- 构造 `cu_seqlens_q/k`
- 设置全局 attention context

#### 去掉：

- 手工 `concat([vision_embeds, text_embeds])`

#### 改为：

- 准备包含 placeholder 的 `input_ids`
- 生成 `is_multimodal`
- 生成 `multimodal_embeddings`

然后把真正 merge 的动作放到模型侧。

### 12.5 `MMQwen3ForCausalLM` 的建议改造

建议扩展成：

- `embed_multimodal(...)`
- `embed_input_ids(...)`
- `forward(input_ids, ..., multimodal_embeddings=None, is_multimodal=None)`

伪代码：

```python
def embed_input_ids(self, input_ids, multimodal_embeddings=None, is_multimodal=None):
	 inputs_embeds = self.model.embed_tokens(input_ids)
	 if multimodal_embeddings is not None:
		  inputs_embeds = merge(inputs_embeds, multimodal_embeddings, is_multimodal)
	 return inputs_embeds
```

然后 `forward(...)` 内部还是调用 `Qwen3Model`。

这样对现有 transformer 层影响很小。

---

## 13. 一条最小可行实施顺序

如果你准备继续开发，我建议按这个顺序，不要一口气改太多：

### Step 1：先完成 placeholder 版本的 fake vision

目标：

- 不用真实视觉塔
- 让 token 序列里真的出现 image placeholder
- 模型 embedding 层完成 merge

验收点：

- 不再依赖 `concat([vision; text])`
- `slot_mapping` 完全由 `token_ids` 长度自然驱动

### Step 2：补 stop reason / debug 日志

因为你当前已经遇到“prefill 后直接 finished”的问题。

建议至少打印：

- `num_tokens`
- `num_prompt_tokens`
- `num_completion_tokens`
- `max_model_length`
- `stop_due_to_max_length`
- `stop_due_to_max_tokens`
- `stop_due_to_eos`

### Step 3：替换 `fake_vision_embeds` 为真实视觉特征

目标：

- 图片内容真的影响 embeddings

### Step 4：再考虑和官方 Qwen-VL 协议更贴近的问题

例如：

- 特殊 token 协议
- 更精确的视觉 token 数
- projector / preprocess 对齐

---

## 14. 为什么我认为这条路线适合你这个仓库

你的项目是学习项目，不是工业级兼容框架。

所以最佳策略不是“照抄官方 `vllm` 全家桶”，而是：

### 学它的边界设计，不学它的全部复杂度

你真正需要吸收的有三件事：

1. **placeholder 让 KV/cache/positions 对齐**
2. **processor 让图片处理从 runner 里剥离出去**
3. **model-side merge 让 transformer 主体尽量保持不变**

这三件事已经足够把你的 `Milestone 1` 推进到一个更健康、可持续演化的结构。

---

## 15. 我对“能不能落地”的判断

答案是：**能，而且非常值得落地，但建议先落地“学习版 placeholder + merge”，不要直接上完整官方 VL。**

更具体地说：

### 能落地的部分

- 图片预处理模块化
- placeholder token 协议
- 模型侧 embedding merge
- 真实 vision tower + projector 的简化版
- decode 保持原样

### 暂时不建议硬上的部分

- 完整 registry 体系
- 所有官方 VL 模型协议兼容
- 视频/音频/多图混合
- 官方 Qwen3-VL 全权重级别兼容

---

## 16. 最后的建议：你现在最应该做的不是“继续堆功能”，而是先把协议固定住

如果让我给你的下一步只提一个最重要建议，那就是：

> **先把“图片如何在 token 序列里占位、如何在 embedding 层被替换”这套协议固定住。**

因为一旦这层稳定了：

- engine 不用老改
- slot_mapping 不用老炸
- KV-cache 契约就会稳定
- 后面换 fake vision / real vision 只是模块替换

反过来，如果 placeholder 协议没固定、图片长度账还散落在很多地方，那么后面每引入一个新视觉模块，都会把 runner / scheduler / attention context 再搅乱一次。

这也是官方 `vllm` 最值得你借鉴的地方。

---

## 17. 这份分析对应到你当前代码的一句话总结

你当前 `Milestone 1` 已经成功证明了：

- **“视觉信息只进 prefill、decode 不变”** 这条路线是成立的。

而官方 `vllm` 告诉你的下一步应该是：

- **把视觉前缀从“runner 临时拼接的 embeddings”升级成“token placeholder + model-side embedding merge”。**

这会是你这个小 `vllm` 框架从“能跑多模态 demo”走向“结构上像一个真正 VL 推理引擎”的关键一步。

