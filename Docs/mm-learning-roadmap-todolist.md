# 多模态学习版落地 TodoList（服务于后续自行编码）

## 文档目标

这份文档不再只是“方向说明”，而是给你后续自己写代码时直接对照的施工单。

它重点回答 4 件事：

1. 现在这条链路已经做到什么程度
2. 为什么当前还没有视觉理解能力
3. 接下来应该按什么顺序改
4. 每一步要改哪些关键内容、为什么改、做完怎么验收

---

## 当前现状：你已经打通的是“多模态工程链路”，不是“视觉理解能力”

当前仓库已经具备这些能力：

- `ImageSequence` 能把视觉前缀长度计入总长度
- `MMModelRunner` 能在 prefill 阶段携带视觉前缀
- `MMQwen3ForCausalLM` 能把视觉 embedding 替换进输入 embedding
- decode 仍然沿用 text-only 路径
- 增大 `max_model_length` 后，序列不会因为长度上限过早终止

这说明你已经证明了两件很重要的事：

1. 视觉信息可以只进入 prefill
2. KV-cache / slot mapping / decode 主链可以在多模态场景下继续工作

但当前还没有视觉理解能力，原因也很明确：

1. 当前模型是 `Qwen/Qwen3-0.6B` 文本模型，不是 VL 模型
2. 当前视觉输入是 `fake_vision_embeds(...)` 随机向量，不是训练对齐过的视觉特征
3. 当前 placeholder 协议仍是内部假协议，不是真正的 Qwen-VL 输入协议
4. prompt 模板仍是文本 chat template，不是 vision chat template

所以现在出现“能输出一些 token，但内容是一串数字”的现象，不是矛盾，而是当前阶段的正常退化表现。

---

## 为什么新的实施顺序必须调整

你当前的目标已经不是“先跑通”，而是“接入视觉理解能力”。

这意味着后续顺序不能再围绕：

- 增加更多 fake 视觉 token
- 调更大的 `max_model_length`
- 在 runner 里继续堆临时逻辑

而要改成下面这个顺序：

1. 先固定多模态输入协议
2. 再接入真实视觉特征
3. 再让模型真正消费这些视觉特征
4. 最后再做和具体 VL 权重更紧的对齐

原因很直接：

- 输入协议没定，后续每加一层都会继续漂
- 视觉特征不真实，模型永远只是在处理分布外噪声
- 模型和 tokenizer / special token 协议不对齐，哪怕能 decode，也很难得到可解释输出

---

## 推荐施工顺序总览

后续建议按下面 6 步推进，不要跳步。

1. 固定多模态输入协议与边界
2. 把 placeholder 机制改成“接近真实 VL 协议”的版本
3. 引入真实视觉塔和 projector，替换 fake vision
4. 规范化 model 侧 merge 接口
5. 让 engine / processor / prompt 形成统一入口
6. 最后再考虑对齐具体 VL checkpoint 或权重映射

下面每一步都写成“目标 / 关键改动 / 为什么改 / 验收目标”的形式。

---

## TODO 1：固定多模态输入协议与当前边界

### 目标

先把当前版本明确成一个稳定的 Milestone：

- 视觉信息只进入 prefill
- decode 不重新处理图片
- 总长度以“实际送入模型的 token 序列长度”为准
- model 负责 merge embedding
- runner 只负责调度、长度账、KV-cache 写入账

### 为什么先做这一步

如果这一步不先固定，后面你在 `processor`、`runner`、`model` 三层都会重复发明协议，导致：

- 同一个信息在多个类里重复保存
- 长度账的“真相来源”不唯一
- 真实视觉塔接进来后很难判断错误是在协议层还是语义层

### 关键改动

这一步主要更新文档和注释，不以大改代码为主。

建议明确以下几个契约，并在相关文件里补简短注释：

- `src/myvllm/engine/image_sequence.py`
  - `token_ids` 表示真实送入模型的 prompt token 序列
  - 其中应包含视觉 placeholder 位
- `src/myvllm/engine/mm_model_runner.py`
  - prefill 构造长度账、slot mapping、KV-cache 写入位置
  - 不负责解释图片语义
- `src/myvllm/models/mm_qwen3.py`
  - 负责把视觉 embedding 写入 placeholder 对应位置
- `src/myvllm/multimodal/processor.py`
  - 负责构造多模态输入结构，不负责调度

### 验收目标

满足下面 3 条即可认为通过：

1. 你能用一句话说明每个模块的职责，不再混淆
2. 你能明确说出“视觉信息在哪一层第一次变成 embedding”
3. 你能明确说出“decode 为什么不需要重新处理图片”

---

## TODO 2：把 placeholder 机制改成接近真实 VL 协议

### 目标

把当前“前面塞 `[0] * num_vision_tokens`”的方案，升级成一个明确的、可扩展的视觉占位协议。

这一步的重点不是“马上完全兼容 Qwen-VL”，而是先停止使用语义上错误的 placeholder。

### 为什么这一步优先级最高

你现在的 `placeholder_id = 0` 只是一个占坑值，见 `src/myvllm/multimodal/processor.py`。

这会带来两个问题：

1. 它不是视觉 special token，只是普通词表 token
2. 你的代码虽然在 prefill 时把这些位置替换成了 `vis_embeds`，但整个输入协议仍然和真实 VL 模型严重脱节

如果不先修这个协议，后面接真实 vision encoder 仍然会建立在错误输入格式上。

### 关键改动

建议把这一阶段拆成 3 个小点。

#### 2.1 引入明确的视觉 special token 配置

在 config 或 processor 层明确这些 token id：

- `vision_start_token_id`
- `image_pad_token_id`
- `vision_end_token_id`

如果你暂时还不想真正改 tokenizer 词表，至少也要在代码层保留这些字段，并把 placeholder 逻辑从单个 `0` 改成更明确的结构。

更推荐的做法是直接复用 Qwen tokenizer 已有 special tokens：

- `<|vision_start|>`
- `<|image_pad|>`
- `<|vision_end|>`

### 2.2 改造 `Processor.process()`

当前 `Processor.process()` 只返回：

- `placeholder_token_ids=[0] * num_vision_tokens`

建议改成返回：

- `placeholder_token_ids = [vision_start] + [image_pad] * num_vision_tokens + [vision_end]`
- `num_vision_tokens`
- `image_meta`
- 可选：`vision_token_span`

注意这里有一个实现选择：

1. `num_vision_tokens` 只表示纯视觉 patch token 数，不含 start/end
2. prompt 真实额外长度则是 `num_vision_tokens + 2`

这件事必须在文档和代码里说清楚，否则长度账会再次混乱。

### 2.3 改造 `ImageSequence`

当前 `ImageSequence` 已经把 `placeholder_token_ids + text_token_ids` 合成到 `token_ids` 中，这是对的。

这一步要补的不是大方向，而是把语义写清楚：

- `token_ids` 现在不再只是“文本 + 假占位”
- 而是“完整 prompt token 序列，其中视觉位也以真实占位 token 出现”

### 相关文件

- [processor.py](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/processor.py)
- [image_sequence.py](/home/dministrator/Minivllm-Learn/src/myvllm/engine/image_sequence.py)
- [mm_llm_engine.py](/home/dministrator/Minivllm-Learn/src/myvllm/engine/mm_llm_engine.py)
- [main.py](/home/dministrator/Minivllm-Learn/main.py)

### 验收目标

这一步通过的标准应当是：

1. 打印 `seq.token_ids[:20]` 时，前面不再是一串 `0`
2. `len(seq.token_ids)` 与 `seq.num_tokens` 一致
3. 你能明确区分：
   - 视觉 patch token 数
   - 真实 prompt 中多出来的视觉占位 token 数
4. prefill 后 `slot_mapping.numel()` 仍与真实输入长度一致

---

## TODO 3：引入真实视觉塔和 projector，替换 fake vision

### 目标

把当前的：

- `fake_vision_embeds(image_path, num_vision_tokens, hidden_size, ...)`

替换成：

- `vision_encoder(image) -> image_features`
- `projector(image_features) -> vision_embeds`

这是“接入视觉理解能力”的第一步实质性改动。

### 为什么这是当前真正决定语义质量的一步

现在输出一串数字，不是因为 decode 没跑通，而是因为你给模型的“视觉 embedding”本质上是随机噪声。

只要还在用 [`fake_vision.py`](/home/dministrator/Minivllm-Learn/src/myvllm/utils/fake_vision.py)，模型就不可能学会“图像内容 -> 语言回答”。

所以这一步的地位高于继续调：

- `max_model_length`
- `num_vision_tokens`
- decode 停止条件

### 关键改动

建议新增目录和模块：

- `src/myvllm/multimodal/vision_encoder.py`
- `src/myvllm/multimodal/projector.py`

推荐最小实现如下。

#### 3.1 `vision_encoder.py`

职责：

- 读入图片
- 做 resize / normalize
- 送入现成 vision backbone
- 输出 patch-level 或 sequence-level visual features

你当前阶段不必自己写 ViT。

优先选一个现成 backbone，例如：

- CLIP vision model
- SigLIP vision model

#### 3.2 `projector.py`

职责：

- 把视觉 backbone 输出维度映射到 LLM hidden size

最小可行版本：

- 一个 `nn.Linear(vision_dim, hidden_size)`

稍稳一点的版本：

- `Linear -> GELU/SiLU -> Linear`

#### 3.3 把 `fake_vision_embeds` 的调用点替换掉

当前调用点在 [`mm_model_runner.py`](/home/dministrator/Minivllm-Learn/src/myvllm/engine/mm_model_runner.py)。

建议替换思路是：

1. 先保留接口形状不变
2. 只把返回值来源从 fake 改成 real

也就是尽量让 runner 继续只关心：

- `vis_embeds.shape == (T_vis, hidden_size)`

而不关心这些 embedding 是怎么来的。

### 相关文件

- [mm_model_runner.py](/home/dministrator/Minivllm-Learn/src/myvllm/engine/mm_model_runner.py)
- 新增 [vision_encoder.py](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/vision_encoder.py)
- 新增 [projector.py](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/projector.py)
- [processor.py](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/processor.py)

### 验收目标

这一步的验收不要用“回答是否完全正确”来卡死，而要分层验收：

1. 同一张图，多次运行得到的 `vision_embeds` 基本一致
2. 不同图片的 `vision_embeds` 明显不同
3. `vision_embeds.shape[0]` 与视觉占位位数一致
4. 模型输出不再稳定退化为大段数字或纯符号
5. 同一个问题换图后，输出开始发生与图像相关的变化

---

## TODO 4：规范化 model 侧 merge 接口

### 目标

把“模型如何吃掉视觉 embedding”这件事固定成长期接口，而不是继续在 runner 里堆逻辑。

### 为什么这一步放在真实视觉塔之后

你的 `mm_qwen3.py` 现在已经在做基础 merge 了，所以这一步不是从 0 到 1。

但在接入真实 vision encoder 之后，你需要一个更稳定、更清晰的接口，不然以后：

- 视觉 token 数变化
- start/end token 引入
- 多图输入

都会让现有接口逐渐变乱。

### 关键改动

建议把 [`mm_qwen3.py`](/home/dministrator/Minivllm-Learn/src/myvllm/models/mm_qwen3.py) 整理成明确的两层接口：

1. `embed_input_ids(input_ids, multimodal_embeddings=None, is_multimodal=None)`
2. `forward(input_ids, multimodal_embeddings=None, is_multimodal=None)`

建议逻辑：

1. 先做 `inputs_embeds = embed_tokens(input_ids)`
2. 再用 `is_multimodal` 把视觉位替换成 `multimodal_embeddings`
3. 替换完成后把 `inputs_embeds` 交给原模型

如果你保留 `vis_embeds` / `vis_masks` 命名，也可以，但建议尽量朝：

- `multimodal_embeddings`
- `is_multimodal`

这种更通用的名字统一。

### 额外建议

如果你在 TODO 2 引入了：

- `vision_start`
- `image_pad`
- `vision_end`

那么这里要明确到底哪些位置参与替换。

推荐规则：

- 只替换中间的 `image_pad` 位置
- `vision_start` / `vision_end` 保留文本 embedding 或单独处理

不要把这个规则写在 runner 里，应该写在 model merge 逻辑或 processor 返回结构里。

### 相关文件

- [mm_qwen3.py](/home/dministrator/Minivllm-Learn/src/myvllm/models/mm_qwen3.py)
- 如有需要，少量改动 [qwen3.py](/home/dministrator/Minivllm-Learn/src/myvllm/models/qwen3.py)
- [mm_model_runner.py](/home/dministrator/Minivllm-Learn/src/myvllm/engine/mm_model_runner.py)

### 验收目标

1. runner 不再需要知道视觉 merge 的细节
2. `is_multimodal.sum()` 与 `multimodal_embeddings.shape[0]` 始终一致
3. 你可以单独测试 model merge，而不需要把整个 engine 都跑起来

---

## TODO 5：统一 engine / processor / prompt 入口

### 目标

让“图片 + prompt -> 最终模型输入”的构造过程集中到 `processor` 和 engine 入口，而不是继续零散分布。

### 为什么这一步必须做

你现在虽然已经有 `Processor`，但它还很薄，主要只是返回：

- placeholder token ids
- image path

如果后面继续加：

- 图片预处理
- 多张图
- 不同视觉 token 数
- 不同 prompt 模式

没有统一入口会很快失控。

### 关键改动

#### 5.1 扩充 `MultimodalInput`

建议在 [`processor.py`](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/processor.py) 的 `MultimodalInput` 中加入更明确字段，例如：

- `placeholder_token_ids`
- `num_vision_tokens`
- `vision_placeholder_mask` 或 `vision_token_spans`
- `pixel_values` 或预处理后的图像张量
- `image_meta`

#### 5.2 让 `MMLLMEngine.add_prompt()` 只做组装，不做推断

当前 `MMLLMEngine.add_prompt()` 里还会自己从 config 拿 `num_vision_tokens`。

建议改成：

1. `processor` 决定视觉相关输入
2. engine 只负责把 text tokens 和 processor 结果拼成 `ImageSequence`

#### 5.3 明确 prompt 入口协议

后面如果你打算向真实 VL checkpoint 靠拢，就不要一直沿用纯文本：

- `<|im_start|>user ...`

你至少要在文档里固定：

- 什么时候由 prompt 文本显式包含视觉标记
- 什么时候由 processor 自动注入视觉 token

学习版最简单的策略是：

- 文本 prompt 保持干净
- 视觉 special token 由 processor 注入到 token 序列，而不是让用户手写

### 相关文件

- [processor.py](/home/dministrator/Minivllm-Learn/src/myvllm/multimodal/processor.py)
- [mm_llm_engine.py](/home/dministrator/Minivllm-Learn/src/myvllm/engine/mm_llm_engine.py)
- [main.py](/home/dministrator/Minivllm-Learn/main.py)

### 验收目标

1. engine 不再自己决定视觉 token 细节
2. `processor` 输出的数据结构足够支撑后续真实视觉塔
3. 你能从单个 `MultimodalInput` 看清最终 prompt 的多模态组成

---

## TODO 6：最后再对齐具体 VL checkpoint 或权重映射

### 目标

在前面协议、视觉特征、merge 接口都稳定后，再去考虑：

- 是否换成真正的 VL checkpoint
- 是否对齐 Qwen-VL 的 special token / prompt 规范
- 是否支持官方风格的权重装载

### 为什么这一步必须最后做

如果前面的地基不稳，你现在直接上官方 VL 权重，只会让排错维度爆炸：

- 是权重没映射对？
- 是视觉塔维度不对？
- 是 placeholder 位置不对？
- 是 tokenizer special token 协议不对？

所以这一步必须建立在前 5 步已经清楚的前提下。

### 关键改动

这一阶段可能涉及：

- 更新 `model_name_or_path`
- 对齐真实 VL tokenizer / processor / config
- 扩展 `loader.py` 的权重映射
- 对齐 vision tower、projector、LLM hidden size

如果你要兼容现成 VL 权重，这里才是最可能需要较大改动的地方。

### 相关文件

- [main.py](/home/dministrator/Minivllm-Learn/main.py)
- [loader.py](/home/dministrator/Minivllm-Learn/src/myvllm/utils/loader.py)
- [mm_qwen3.py](/home/dministrator/Minivllm-Learn/src/myvllm/models/mm_qwen3.py)
- `multimodal/` 下新增的真实视觉模块

### 验收目标

这一步通过的标准应当更严格：

1. 同一问句对不同图片能给出可解释差异
2. 简单图像问题开始出现基本正确答案
3. 输出不再长期退化成数字串、标点串或空串
4. text-only 路径仍可正常工作

---

## 每一步都要检查的通用验收项

无论你做到哪一步，每次改完至少检查下面 5 项。

### 1. 长度账是否统一

始终要满足：

`模型真实处理 token 数 == slot_mapping 长度 == KV-cache 实际写入 token 数`

### 2. decode 是否保持独立

必须确认：

- decode 仍然只处理新增 token
- decode 不重新看图片
- decode 路径没有被多模态逻辑污染

### 3. text-only 是否仍可运行

必须确认：

- 不传图片时，旧链路还能正常生成

### 4. multimodal merge 数量是否一致

至少检查：

- `is_multimodal.sum()`
- `multimodal_embeddings.shape[0]`
- 真实视觉占位位数

这三者要一致。

### 5. 结束原因是否可见

建议始终保留 stop reason 观测：

- `stop_due_to_eos`
- `stop_due_to_max_tokens`
- `stop_due_to_max_length`

否则你会再次遇到“结果不对，但不知道是逻辑错还是提前停了”。

---

## 当前最推荐的最近三步

如果你现在就要继续写代码，我建议你先做下面三步，不要分散：

1. 先把 `placeholder_id = 0` 升级成明确的视觉 special token 协议
2. 再新增 `vision_encoder.py` 和 `projector.py`，把 fake vision 替换掉
3. 最后整理 `mm_qwen3.py` 的 merge 接口，把视觉替换规则固定下来

这三步做完之前，不建议再花很多时间调：

- `max_model_length`
- `num_vision_tokens`
- decode 采样细节

因为这些都不会从根本上解决“模型看不懂图像”的问题。

---

## 一句话总结

你下一阶段的主线不应该再是“如何让 fake vision 输出更长”，而应该变成：

> 先把视觉占位协议做正确，再把真实视觉特征接进来，再让模型以稳定接口消费这些特征。

只有这条路线，才是在你当前代码基础上真正朝“视觉理解能力”前进。
