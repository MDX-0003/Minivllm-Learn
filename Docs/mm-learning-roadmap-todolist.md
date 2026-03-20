# 多模态学习版落地 TodoList（面向当前 `myvllm`）

## 文档目标

这份文档用于把当前仓库的多模态目标、已有基础、参考官方 `vllm` 的最小实现路线，整理成一份**可继续施工的 todolist**。

它重点回答 4 个问题：

1. 当前多模态能力已经做到哪里了
2. 现阶段真正的目标是什么
3. 参考官方 `vllm` 时，最小应该借哪些框架思想
4. 接下来每一步应该改什么、怎么验证、完成后得到什么

这不是纯分析笔记，也不是最终实现说明书；它是后续继续落地时可直接执行的路线文档。

---

## 先说当前总目标

当前目标不是一步做成完整的官方 `Qwen3-VL` 兼容推理器，而是做一个：

> **尽量少改现有 `myvllm` 架构、但在边界设计上向官方 `vllm` 靠拢的学习版多模态实现。**

更具体地说，当前目标可以拆成 3 层：

### 目标 1：保住现有主链

必须继续满足：

- text-only 路径尽量不受影响
- decode 快路径尽量不受影响
- 多模态影响优先限制在 prefill

### 目标 2：把当前 Milestone 1 从“临时跑通”升级成“结构更稳”

当前 Milestone 1 已经证明：

- 视觉信息可以只进 prefill
- 只要 KV-cache 长度账对齐，decode 可以不动

但当前方案仍然偏“runner 手工拼接 embeddings”，还不够像一个稳定的框架协议。

### 目标 3：向官方 `vllm` 学“边界”，而不是学“全家桶复杂度”

最值得借的不是完整 registry，而是这三层边界：

- `processor`：负责把图片和 prompt 变成模型需要的结构
- `model`：负责把视觉 embedding 合并到文本 embedding
- `runner`：继续只负责 prefill/decode/KV-cache 账本

---

## 当前已经完成了什么

结合已有代码与分析，可认为当前已经完成：

### 已完成 1：多模态 prefill 已打通

当前已有：

- `ImageSequence`
- `MMModelRunner`
- `MMLLMEngine`
- `MMQwen3ForCausalLM`
- `fake_vision_embeds`

已经能做到：

- 序列总长度按 $T_{vis}+T_{text}$ 处理
- `slot_mapping` / `cu_seqlens` 按总长度构造
- prefill 阶段把图片前缀和文本 embedding 拼起来喂给模型
- decode 维持原 text-only 路径

### 已完成 2：关键工程结论已经验证出来

已经验证出的正确方向：

1. 图片信息可以只写进 prefill
2. decode 不需要重新处理图片
3. 多模态最脆弱的部分不是“模型答得对不对”，而是：
   - token 数对不对
   - `slot_mapping` 长度对不对
   - KV-cache 写入账是否与模型实际处理长度一致

---

## 当前主要问题 / 为什么还要继续改

当前实现虽然能跑通，但还有几个明显问题。

### 问题 1：视觉 token 不在 token 序列里

当前做法里：

- `token_ids` 主要还是文本 token
- 图片前缀只存在于 `inputs_embeds = concat([vision, text])`

这意味着：

- 长度账有一部分在 `ImageSequence.num_tokens`
- 另一部分在 runner 临时拼 embeddings
- 还要依赖一个“全零 placeholder tensor”去补齐长度契约

这说明当前“总长度的真相来源”还没有完全统一。

### 问题 2：模型层没有明确占位协议

当前模型只是在 prefill 时接受 `inputs_embeds`，但并不知道：

- 哪些位置本来是视觉位
- 哪些位置是文本位
- 多模态替换是怎么发生的

这会使后续接真实视觉塔时，runner 和 model 的边界仍然容易混。

### 问题 3：图片处理还没有从 runner 中剥离

目前图片相关逻辑主要散落在：

- `MMModelRunner.prepare_prefill()`
- `fake_vision_embeds(...)`

这对学习阶段可以接受，但如果以后加：

- placeholder token
- 真正的 image preprocess
- 真实 vision encoder
- projector

继续把逻辑堆在 runner 里会越来越乱。

---

## 参考官方 `vllm` 时，最小要借什么

对当前仓库，最小应该借的不是完整 `MultiModalRegistry`，而是下面这套最小框架。

### 1. placeholder token 机制

官方 `vllm` 的核心启发是：

- 视觉长度应该体现在 token 序列本身里
- prompt 里先预留 placeholder 位置
- 后面再把这些位置的 embedding 替换成视觉 embedding

这比现在的“runner 外拼 concat embeddings”更稳定。

### 2. processor / model / runner 分层

建议明确职责：

- `processor`：决定视觉 token 数、构造 placeholder token ids、准备图片输入
- `model`：把 placeholder 位替换成视觉 embeddings
- `runner`：只管总 token 数、prefill/decode、KV-cache

### 3. 先 merge embedding，再复用原 transformer 主体

也就是：

1. 先按正常 `input_ids` 做文本 embedding
2. 再按 mask 把视觉位替换掉
3. 再把合并后的 `inputs_embeds` 送进原 transformer

这是最适合当前仓库的“最小像官方”的实现。

---

## Todo 总览

建议后续按下面 6 步推进。

1. 固定当前多模态最小契约
2. 引入图片 placeholder token 协议
3. 把 embedding merge 从 runner 挪到 model
4. 新增最小 `processor` 层
5. 补一轮验证 / 日志 / stop reason
6. 再考虑替换 fake vision 为 real vision

下面分别展开。

---

## TODO 1：固定当前多模态最小契约

### 目标

先把当前 Milestone 1 的约束写死，避免后续边改边漂移。

本步完成后，应明确当前学习版多模态的最小 contract：

- 视觉信息只进 prefill
- decode 保持 text-only
- 总 token 数一律按 $T_{vis}+T_{text}$ 计算
- `slot_mapping` 长度必须等于模型真实处理的 token 数

### 参考官方 vLLM 的哪一点

参考的是官方“先把长度和占位协议说清楚”的做法，而不是具体代码结构。

### 建议改动

优先不是改代码，而是补一份短文档或把现有文档里的 contract 提炼清楚，明确：

- 输入：文本 token + 图片
- prefill：处理总长度 = $T_{vis}+T_{text}$
- decode：只处理新增 token
- 当前版本仍使用 fake vision
- 当前版本仍不兼容官方 VL 权重

### 相关文件

- `Docs/MileStone-note1.md`
- `Docs/vLLM_multimodal_analysis_for_myvllm.md`
- 可新增一个更短的 contract note（可选）

### 验证方式

验收点不是跑分，而是“以后改代码时不会再混淆当前设计边界”。

### 完成后得到什么

得到一个稳定的工程边界，避免后续每一步都重新讨论“decode 要不要看图片”。

---

## TODO 2：引入图片 placeholder token 协议

### 目标

把当前“视觉长度只存在于 `num_tokens` 和 `inputs_embeds` 拼接里”的方式，升级成：

- `token_ids` 本身也包含视觉占位位子

这是整个路线里最关键的一步。

### 参考官方 vLLM 的哪一点

直接参考官方：

- prompt/token 序列中预留 placeholder 位置
- 总长度直接来自 token 序列

### 建议改动

#### 2.1 增加内部专用 placeholder token id

在 config 中增加一个内部约定字段，例如：

- `image_token_id`

它不一定要立刻成为 tokenizer 真正 special token，但要作为框架内部保留占位 id 使用。

#### 2.2 改造 `ImageSequence`

让 `token_ids` 直接变成：

- `[image_token_id] * num_vision_tokens + text_token_ids`

或者保存一份新的：

- `prompt_token_ids_with_placeholders`

推荐优先直接让 `token_ids` 含 placeholder，这样长度账最统一。

#### 2.3 `num_tokens` 由 token 序列自然决定

这一步做完后，理想状态应是：

- `len(seq.token_ids)` 本身就等于 $T_{vis}+T_{text}$
- 不再需要仅靠 `num_vision_tokens + len(text_token_ids)` 去“补长度”

### 相关文件

- `src/myvllm/engine/image_sequence.py`
- `src/myvllm/engine/mm_llm_engine.py`
- 可能补一点 `main.py` / config wiring

### 验证方式

1. 打印一条多模态序列的：
   - `len(seq.token_ids)`
   - `seq.num_tokens`
   - `seq.num_vision_tokens`
2. 确认三者关系清楚
3. 确认 `slot_mapping` 长度直接由 token 序列长度驱动

### 完成后得到什么

- 总长度的真相来源统一了
- 后面可以逐步删掉当前的“全零长度占位 tensor”式兼容技巧

---

## TODO 3：把 embedding merge 从 runner 挪到 model

### 目标

把当前：

- runner 手动 `concat([vision_embeds, text_embeds])`

升级成：

- model 先 embed 文本
- 再在 placeholder 位置替换成视觉 embeddings

### 参考官方 vLLM 的哪一点

直接参考：

- `embed_input_ids(...)`
- `_merge_multimodal_embeddings(...)`

也就是“token 序列完整保留，embedding 层局部替换”。

### 建议改动

#### 3.1 为 `MMQwen3ForCausalLM` 新增 embedding merge 入口

建议增加：

- `embed_multimodal(...)`
- `embed_input_ids(...)`
- `forward(..., multimodal_embeddings=None, is_multimodal=None)`

基本逻辑：

1. `inputs_embeds = embed_tokens(input_ids)`
2. `is_multimodal = (input_ids == image_token_id)`
3. 用 mask / 索引替换这些位置的 embedding
4. 将合并后的 embeddings 继续送入原模型

#### 3.2 `MMModelRunner.prepare_prefill()` 不再手工 concat

runner 改为只准备：

- `input_ids`（含 placeholder）
- `multimodal_embeddings`
- `is_multimodal`

然后在 prefill forward 调用：

- `self.model(input_ids=..., multimodal_embeddings=..., is_multimodal=...)`

### 相关文件

- `src/myvllm/models/mm_qwen3.py`
- `src/myvllm/models/qwen3.py`（如果需要最小入口改动）
- `src/myvllm/engine/mm_model_runner.py`

### 验证方式

1. 在 prefill 时打印：
   - `input_ids.shape[0]`
   - `multimodal_embeddings.shape[0]`
   - `is_multimodal.sum()`
2. 确认：

$$
\text{is\_multimodal.sum()} = T_{vis}
$$

3. 确认模型 forward 不再依赖 runner 侧的 `concat([vision, text])`

### 完成后得到什么

- runner 和 model 的职责边界更像官方
- 后面换真实视觉塔时，runner 几乎不用再碰融合逻辑

---

## TODO 4：新增最小 `processor` 层

### 目标

把“图片如何变成模型输入信息”从 runner 中剥出来，形成一个独立层。

### 参考官方 vLLM 的哪一点

参考的是官方的 `processor` 思想，而不是完整 registry。

### 建议改动

新增目录：

- `src/myvllm/multimodal/`

建议至少新增：

#### `processor.py`

负责：

- 接收 `image_path`
- 读取图片
- 决定 `num_vision_tokens`
- 构造 placeholder token ids
- 返回多模态输入结构

在 fake vision 阶段，processor 甚至可以先不做复杂 preprocess，只做：

- 图片路径检查
- placeholder token 构造
- 返回结构化结果

#### 如果一步不想加太多文件

也可以先只建：

- `src/myvllm/multimodal/processor.py`

后续再拆 `vision_encoder.py` / `projector.py`。

### 相关文件

- 新增 `src/myvllm/multimodal/processor.py`
- `src/myvllm/engine/mm_llm_engine.py`
- `src/myvllm/engine/image_sequence.py`

### 验证方式

1. 给定 `image_path` 和文本 prompt
2. 检查 processor 输出：
   - placeholder token 数是否正确
   - token 序列长度是否正确
   - image 元数据是否正确带到后续流程

### 完成后得到什么

- 图片处理不再散落在 runner 里
- 后续接真实视觉塔时有明确插槽

---

## TODO 5：补最小验证与 stop reason 日志

### 目标

把当前“prefill 后直接 finished、却不容易看出原因”的问题补上观测能力。

### 参考官方 vLLM 的哪一点

不是直接抄官方代码，而是学它“每层边界都有明确状态”的工程思路。

### 建议改动

优先补以下日志：

- `num_tokens`
- `num_prompt_tokens`
- `num_completion_tokens`
- `max_model_length`
- `stop_due_to_eos`
- `stop_due_to_max_tokens`
- `stop_due_to_max_length`

以及多模态 prefill 相关一致性检查：

- `len(input_ids) == slot_mapping.numel()`
- `is_multimodal.sum() == num_vision_tokens`
- `multimodal_embeddings.shape[0] == num_vision_tokens`

### 相关文件

- `src/myvllm/engine/scheduler.py`
- `src/myvllm/engine/mm_model_runner.py`
- 可能补少量 debug print / assert 到 `main.py`

### 验证方式

1. 跑一条单图 prompt
2. 能明确看到 sequence 是因为什么结束
3. 如果异常，能第一时间定位是：
   - 长度账错
   - stop 条件触发
   - embedding merge 数量不一致

### 完成后得到什么

- 后续每改一步都更容易定位问题
- 不会再只看到“Completion 空”却不知道是哪一步终止了

---

## TODO 6：将 fake vision 替换为 real vision（后续阶段）

### 目标

在框架边界稳定后，再把：

- `fake_vision_embeds`

替换成：

- 真实视觉特征提取 + projector

### 参考官方 vLLM 的哪一点

参考的是：

- 图片走独立 vision tower / projector 路径
- model 侧统一 merge

### 建议改动

后续新增：

- `src/myvllm/multimodal/vision_encoder.py`
- `src/myvllm/multimodal/projector.py`

第一版优先做“能通”的版本：

- 现成 vision backbone
- 简单 linear / MLP projector

先不要追求：

- 官方 Qwen3-VL 权重兼容
- 完整特殊 token 协议
- MRoPE 对齐

### 相关文件

- `src/myvllm/multimodal/vision_encoder.py`
- `src/myvllm/multimodal/projector.py`
- `src/myvllm/models/mm_qwen3.py`
- `src/myvllm/multimodal/processor.py`

### 验证方式

1. 两张不同图片输入时，视觉 embedding 不同
2. 模型输入的多模态 embedding 不再是随机占位
3. 不破坏前面已经稳定的 placeholder + merge 协议

### 完成后得到什么

- 多模态能力从“结构打通”升级到“真正接入图像语义”

---

## 推荐实施顺序（最小安全顺序）

建议严格按下面顺序，不要跳步：

1. 先固定 contract
2. 再让 token 序列里出现 placeholder
3. 再把 merge 移到 model
4. 再剥离 processor
5. 再补日志和验证
6. 最后才接真实视觉塔

原因很简单：

- placeholder 协议没固定之前，接视觉塔只会继续放大混乱
- runner / model 边界没理清之前，加 preprocess 也会越堆越乱

---

## 每一步的验收原则

后续每一步完成后，都至少检查下面 3 件事。

### 1. 长度账是否统一

必须始终满足：

$$
\text{模型真实处理 token 数} = \text{slot\_mapping 长度} = \text{KV-cache 实际写入 token 数}
$$

### 2. decode 是否仍然保持原路径

必须确认：

- decode 仍然只处理新增 token
- decode 不重新接图片
- decode CUDA graph 路径不被新的多模态逻辑污染

### 3. text-only 是否继续可用

必须确认：

- 不带图片时，旧 text-only 行为不应被新逻辑破坏

---

## 一句话版本总结

后续路线不要再继续强化“runner 临时拼 `inputs_embeds`”这条路，而应该尽快转向：

> **让图片先在 token 序列里占位，再在模型 embedding 层做替换；runner 只继续负责 token 数、prefill/decode 和 KV-cache 账本。**

这就是当前仓库参考官方 `vllm` 时，最小、最值得、也最可落地的实现路线。
