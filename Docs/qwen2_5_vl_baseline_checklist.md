# MyvLLM 的 Qwen2.5-VL 基线检查清单

## 文档目标

本文档说明在 `run_qwen2_5_vl_baseline.py` 已经可以正常运行之后，**接下来应该做什么**。

到这个阶段，基线脚本就不再只是一个 demo。
它会变成一个**逆向分析工具**，用来学习真实的 Qwen2.5-VL 推理契约。

这份清单的目的，是回答三个问题：

1. 应该**如何**一步一步检查官方 Qwen2.5-VL 基线？
2. 每一步检查**为什么**有必要？
3. 哪些**可复用的信息**应该回迁到本地的类 vLLM 推理框架中？

本文档聚焦于：

- 官方 Qwen2.5-VL 基线推理
- 理解输入/输出契约
- 为本地服务框架提炼正确的经验

本文档**不**涉及：

- 训练
- 预训练
- 跑分
- 完整复现官方 vLLM 内部实现

---

## 当前情况

你已经有了一个可运行的基线脚本：

- [run_qwen2_5_vl_baseline.py](/home/dministrator/Minivllm-Learn/run_qwen2_5_vl_baseline.py)

这意味着有一件重要的事已经被证明：

- 真实的 Qwen2.5-VL checkpoint 能够在当前机器上正确分析至少一张图片

这是正确的起点。

从现在开始，目标不再是：

- “让脚本跑起来”

目标变成了：

- “利用这个脚本学习真实的模型契约”

---

## 这个基线应该教会你什么

只有当基线能清楚回答下列问题时，它才真正有价值。

### 1. 真实的上游用户输入形态是什么？

从概念上讲，Qwen2.5-VL 并不是从一个普通的纯字符串 prompt 开始的。
它是从一个结构化的 `messages` 对象开始的，其中包含：

- role 信息
- 有序的多模态内容
- 图像项
- 文本项

这很重要，因为本地框架最终应该让它的 processor 输入与这种结构保持一致。

### 2. 官方 Processor 实际生成的 Prompt 文本是什么？

本地框架不应该去猜测视觉占位符的格式。
它应该研究官方 processor 产出的精确 prompt 字符串。

这会告诉你：

- 图像占位符在 prompt 中是如何出现的
- 视觉占位符相对于文本放在什么位置
- assistant generation prompt 是如何被追加进去的

### 3. 实际传入模型的 Tensor 是什么？

本地框架目前主要围绕 token ids 和合并后的 embeddings 来思考。
而官方 Qwen2.5-VL 路径证明，模型并不只依赖 token ids。

它还会消费与图像相关的 tensor，例如：

- `pixel_values`
- `image_grid_thw`

这是这份基线能够带来的最重要经验之一。

### 4. 服务引擎应该如何切分职责？

这份基线能帮助你分清以下边界：

- prompt 构造
- 图像预处理
- 模型输入打包
- 模型调用
- completion 解码

这些边界，正是你的本地框架在 TODO5 以及后续真实模型适配阶段中需要澄清的内容。

---

## 修改脚本前，建议的阅读顺序

在你把基线脚本改造成一个检查工具之前，先按下面的顺序阅读官方参考资料。

### 1. Hugging Face Qwen2.5-VL 模型文档

用它来理解：

- 预期的模型输入
- processor 的行为
- 与 vision 相关的模型字段
- special token 配置

参考：

- https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl

### 2. Qwen 官方使用示例

用它来理解：

- 官方多模态 `messages` 格式
- 预期的推理流程
- 更接近真实使用方式的示例

参考：

- https://github.com/QwenLM/Qwen2.5-VL

### 3. vLLM 多模态输入文档

用它来理解：

- 服务框架应如何打包多模态请求
- `prompt` 与 `multi_modal_data` 之间的分工

参考：

- https://docs.vllm.ai/en/latest/features/multimodal_inputs.html

---

## 清单第 1 部分：检查上游输入契约

### 步骤 1：打印原始 `messages`

#### 要做什么

在基线脚本中，在进行任何处理之前打印 `messages` 对象。

建议的检查代码：

```python
from pprint import pprint
print("\n=== Raw messages ===")
pprint(messages)
```
实际结果
```python
[{'content': [{'image': '/home/dministrator/Minivllm-Learn/images/01/num.png',
               'type': 'image'},
              {'text': 'Describe this image briefly.', 'type': 'text'}],
  'role': 'user'}]
["The image shows a grid with numbers and letters arranged in a pattern. The numbers are 0, 8, 2, 0, 1, 2, 0, 1, 3. The letters are 'T', 'R', 'E', 'E', 'A', 'T"]
```
#### 为什么

这能展示 Qwen2.5-VL 的真实上游接口：

- 用户请求是结构化的
- 多模态内容是有顺序的
- 图像和文本不会过早被压平成单一字符串

#### 你应该学到什么

你应该记录下来：

- `messages` 才是概念上的输入，而不只是一个 prompt 字符串
- `content` 是一个列表，而不是一段普通句子
- 图像和文本的顺序具有语义意义

#### 应该向 MyvLLM 借鉴什么

这应该直接影响本地 `Processor` 的长期演进方向。

目标经验：

- processor 最终应该接受一种 `messages` 风格的结构
- 而不只是 `prompt: str` 加 `image_path`

---

## 清单第 2 部分：检查官方 Prompt 契约

### 步骤 2：打印 `apply_chat_template(..., tokenize=False)`

#### 要做什么

增加第二条检查路径，把 prompt 渲染成纯文本：

```python
prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print("\n=== Rendered prompt text ===")
print(prompt_text)
```
结果
```python
{'input_ids': tensor([[151644,   8948,    198,   2610,    525,    264,  10950,  17847,     13,
         151645,    198, 151644,    872,    198, 151652, 151655, 151655, 151655,
         151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
         151655, 151655, 151655, 151655, 151653,  74785,    419,   2168,  26753,
             13, 151645,    198, 151644,  77091,    198]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0'), 'pixel_values': tensor([[-1.1499, -1.1791, -1.3981,  ..., -1.3380, -0.4279,  1.1363],
        [-1.7923, -1.7923, -1.7923,  ...,  0.1977,  1.9468,  1.8188],
        [-0.9893, -1.0039, -1.1061,  ..., -1.4660, -1.2385, -0.6981],
        ...,
        [-0.5368, -1.5003, -1.5879,  ..., -1.4660, -1.4660, -0.9683],
        [-1.7193, -1.7485, -1.5295,  ..., -1.2385, -1.2385, -1.2385],
        [ 1.2296, -0.8142, -1.7923,  ..., -1.2385, -1.2385, -0.9825]],
       device='cuda:0'), 'image_grid_thw': tensor([[1, 8, 8]], device='cuda:0')}
```


#### 为什么

这是最重要的一步检查。

它会准确告诉你，官方 processor 是如何把多模态 messages 转换成 prompt 文本的。

你需要看到：

- 视觉占位符出现在哪里
- 它们相对于用户文本的顺序是什么
- assistant generation prompt 是如何插入的

#### 你应该学到什么

你应该观察并记录：

- 渲染后的文本里是否包含视觉占位符
- 这些占位符出现的位置
- tokenization 之前的 prompt 长什么样

#### 应该向 MyvLLM 借鉴什么

这是 TODO5 中 prompt 对齐的权威参考。

目标经验：

- 本地 processor 应该模仿官方 prompt 协议
- 除非出于明确的学习目的，否则不要另外发明一套占位符协议

---

### 步骤 3：打印分词后的 Prompt 与 Special Token IDs

#### 要做什么

检查 model config，以及 tokenizer 侧的分词结果。

建议的检查代码：

```python
print("\n=== Special token ids ===")
print("vision_start_token_id:", model.config.vision_start_token_id)
print("vision_end_token_id:", model.config.vision_end_token_id)
print("image_token_id:", model.config.image_token_id)

tokenized = processor(
    text=prompt_text,
    return_tensors="pt",
)
print("\n=== Prompt-only tokenization ===")
print("input_ids shape:", tokenized["input_ids"].shape)
print("first 64 ids:", tokenized["input_ids"][0, :64].tolist())
```

#### 为什么

只有 prompt 文本还不够。
你还需要知道 prompt 是如何变成 token ids 的。

这有助于回答：

- 哪些 id 代表视觉占位符
- 它们在 token 流中出现在哪里
- 你的本地占位符逻辑是否与真实 tokenizer/模型契约一致

#### 你应该学到什么

你应该记录：

- 精确的视觉 special token ids
- 图像占位符是否会折叠成重复的 image token ids
- 这些 token 在最终 prompt token 序列中的位置

#### 应该向 MyvLLM 借鉴什么

目标经验：

- 视觉占位符 id 应该来自真实模型/processor/config
- 而不只是来自本地约定

---

## 清单第 3 部分：检查真实的视觉 Tensor 路径

### 步骤 4：把 Prompt 渲染与图像处理拆开

#### 要做什么

当前基线脚本使用的是：

- `apply_chat_template(..., tokenize=True, return_dict=True, return_tensors="pt")`

这对于第一次跑通来说很方便，但不利于学习。

请创建第二版基线路径，把下面三件事显式拆开：

1. 渲染后的 prompt 文本
2. 图像预处理
3. 最终模型输入

你不需要立刻替换当前可运行的路径。
你只需要一个用于检查的版本。

建议的目标结构：

```python
prompt_text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

# Depending on the exact helper you use, collect processed image inputs here.
# The purpose is to separate text rendering from image preprocessing, not to
# optimize for brevity.
```

#### 为什么

这是官方 Transformers 用法与后续类 vLLM 服务之间的桥梁。

它能让你清楚看见：

- 文本 prompt 的形成
- 图像预处理
- 最终 tensor 打包

#### 你应该学到什么

你最终应该明确知道：

- prompt 文本和视觉 tensor 是两个不同的产物
- 模型同时受到二者的条件约束
- 你的服务框架应该保留这种拆分

#### 应该向 MyvLLM 借鉴什么

目标经验：

- 本地框架应该朝着这样的拆分方式演进：
  - prompt 侧契约
  - visual-data 侧契约

这会自然对齐到 vLLM 关于以下内容的思路：

- `prompt`
- `multi_modal_data`

---

### 步骤 5：打印所有最终模型输入的 Key 与 Shape

#### 要做什么

当你拿到最终传入 `model.generate(...)` 的 `inputs` 对象后，打印其中的每一个 key。

建议的检查代码：

```python
print("\n=== Final model inputs ===")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    else:
        print(f"{k}: type={type(v)}")
```

#### 为什么

这一步会揭示**真实的模型契约**。

你需要知道 checkpoint 实际消费的是什么，而不是本地框架当前“以为”它消费的是什么。

对于 Qwen2.5-VL，你应该特别关注这些字段：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`

#### 你应该学到什么

你应该记录：

- 传给模型的每一个输入 key
- 每个 tensor 的 shape
- 每个 tensor 的 dtype
- 哪些字段与文本相关
- 哪些字段与图像相关

#### 应该向 MyvLLM 借鉴什么

这一步应该直接指导未来的 `ModelAdapter` 接口设计。

目标经验：

- adapter 必须接受的不只是 token ids
- processor 最终必须产出 checkpoint 所需的真实视觉 tensor

---

## 清单第 4 部分：检查输出语义

### 步骤 6：比较输入 Token 长度与输出 Token 长度

#### 要做什么

在 generation 前后增加明确的长度打印：

```python
print("\n=== Generation lengths ===")
print("input_ids shape:", tuple(inputs["input_ids"].shape))
print("generated_ids shape:", tuple(generated_ids.shape))
print("trimmed completion lengths:", [len(x) for x in generated_ids_trimmed])
```

#### 为什么

这会帮助你理解，官方 `generate(...)` 的结果与服务框架里的以下概念之间是什么关系：

- prompt tokens
- completion tokens

官方 generation 输出通常会包含 prompt 前缀以及新增 token。
而你的本地框架已经是在 completion 语义下思考问题。

你必须清楚理解这两者的差异。

#### 你应该学到什么

你应该记录：

- prompt token 长度
- 总生成 token 长度
- trimming 之后的 completion token 长度

#### 应该向 MyvLLM 借鉴什么

目标经验：

- 未来的真实模型 adapter 应该保留框架现有的 completion 语义
- 裁掉 prompt 前缀 token 不是可有可无的收尾工作，而是正确输出处理的一部分

---

### 步骤 7：以人类可读的形式保存一个已知正确的样例

#### 要做什么

针对一张图片和一个问题，记录以下内容：

- 图像路径
- 原始 messages
- 渲染后的 prompt 文本
- special token ids
- 最终输入的 keys 和 shapes
- 回答文本

把这些内容保存到下面任意一种地方：

- `Docs/` 里的一个小型 markdown 说明
- 或者一个带注释的检查脚本中

#### 为什么

这会成为你之后做比对时的固定基线。

如果没有记录一个已知正确的样例，那么未来每一次改动都会更难验证。

#### 你应该学到什么

你是在为本地框架迁移创建一个“黄金样本”。

#### 应该向 MyvLLM 借鉴什么

目标经验：

- 后续 adapter 工作应该始终拿这个记录下来的官方基线做比较

---

## 在再次改动 MyvLLM 之前，你必须提取出的信息

在回到本地框架之前，请确保你能非常有把握地回答下列问题。

### A. Processor 层问题

- 官方栈接受的精确 `messages` 结构是什么？
- 官方 processor 渲染出的精确 prompt 字符串是什么？
- 图像占位符是由 processor、tokenizer，还是用户编写的 prompt 结构插入的？

### B. Token 层问题

- 哪些 token ids 代表视觉标记？
- 它们在最终 prompt 里出现在哪里？
- 最终 token 序列与纯文本 chat prompt 相比有什么不同？

### C. Tensor 层问题

- 哪些视觉 tensor 会被传给模型？
- 它们的 shapes 和 dtypes 是什么？
- 哪些字段是图像条件推理所严格必需的？

### D. 服务边界问题

- 什么属于 processor？
- 什么属于 model adapter？
- 什么属于 serving engine？

如果这些问题里有任何一个答案仍然模糊，说明这个基线还没有教会你足够多的东西。

---

## 应该借鉴到本地类 vLLM 框架中的内容

这是整份文档里最重要的一部分。

只有当它真的改变了本地框架的演进方式时，这份基线才算有价值。

### 1. 借鉴官方的上游输入形态

当前本地的简化方式：

- `prompt: str`
- `image_path: str`

应该借鉴的方向：

- `messages`

这并不意味着框架必须立刻暴露完全相同的公共 API。
它的意思是，本地 processor 在内部应该朝着相同的概念表示方式演进。

### 2. 借鉴官方的 Prompt 契约

当前本地的简化方式：

- 本地 processor 手动在前面拼接视觉占位符

应该借鉴的方向：

- prompt 构造应以官方 `apply_chat_template(...)` 为指导

这会降低你发明一套“本地写起来方便、但语义上错误”的视觉 prompt 布局的风险。

### 3. 借鉴 Prompt / Visual-Data 拆分

当前本地的简化方式：

- 项目主要围绕 token 序列和合并后的 embeddings 来思考

应该借鉴的方向：

- 明确保留以下两者之间的分离：
  - prompt 侧信息
  - visual tensor 侧信息

这是你未来演进到类 vLLM `prompt + multi_modal_data` 接口时，最直接可以带走的经验。

### 4. 借鉴真实的模型输入表面

当前本地的简化方式：

- 本地模型路径主要消费 token ids，加上本地投影后的多模态 embeddings

应该借鉴的方向：

- 未来的真实模型 adapter 必须接受真实的 Qwen2.5-VL 输入

这意味着框架应朝着承载真实 VL checkpoint 的方向演进，而不只是给一个纯文本模型外挂扩展能力。

### 5. 借鉴评估方法

当前本地的风险：

- 先改本地代码，然后再猜输出看起来是否还算合理

应该借鉴的方向：

- 始终把本地框架的行为与一个官方已知正确样例进行比较

这是下一阶段最安全、也是最快的调试策略。

---

## 建议加入到基线脚本中的注释标签

你的脚本已经证明了端到端正确性。
下一步改进应该面向“检查日志”，而不是增加更多功能。

我建议加入这些带标签的输出段落：

- `=== Raw messages ===`
- `=== Rendered prompt text ===`
- `=== Special token ids ===`
- `=== Prompt-only tokenization ===`
- `=== Final model inputs ===`
- `=== Generation lengths ===`
- `=== Final decoded answer ===`

原因很简单：

- 这些标签会把脚本从一个 demo 变成一个分析工具

---

## 建议的停止条件

只有在满足以下条件时，你才可以停止基线分析并回到本地框架开发：

1. 官方基线可以稳定运行
2. prompt 字符串已经被完整检查
3. 视觉 tensor 输入已经被完整检查
4. 已记录一个已知正确样例
5. 你可以把官方概念映射到本地框架职责上

只要其中任何一项缺失，本地集成阶段仍然会包含过多猜测。

---

## 最终总结

运行官方 Qwen2.5-VL 基线，不只是一次基本可用性的 sanity check。
它也是发现你要服务的这一模型家族真实推理契约的最干净方式。

最核心的经验是：

- 不要把官方模型视为“一个文本模型，再加上一些额外的图像 embeddings”

而应该把它视为一个具有以下特征的模型：

- 结构化的上游输入格式
- 官方 processor 契约
- 真实图像 tensor
- 特定的 prompt 协议

现在越忠实地从基线中学会这四件事，之后把它们适配进你自己的类 vLLM 推理框架时就会越容易。
