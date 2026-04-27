# Qwen2.5-VL 编码前知识补全指南

## 文档定位

这份文档写给一个已经走到“快要开始写真实 Qwen2.5-VL 代码”的学习者。

你现在不是从零开始。

你已经有：

- 一个官方 Qwen2.5-VL 基线脚本：[run_qwen2_5_vl_baseline.py](./../run_qwen2_5_vl_baseline.py)
- 一份基线检查清单：[qwen2_5_vl_baseline_checklist.md](./qwen2_5_vl_baseline_checklist.md)
- 一份整体教学路线图：[qwen2_5_vl_teaching_roadmap.md](./qwen2_5_vl_teaching_roadmap.md)
- 一条本地 fake multimodal engine 路径：
  - [processor.py](./../src/myvllm/multimodal/processor.py)
  - [image_sequence.py](./../src/myvllm/engine/image_sequence.py)
  - [mm_llm_engine.py](./../src/myvllm/engine/mm_llm_engine.py)
  - [mm_model_runner.py](./../src/myvllm/engine/mm_model_runner.py)
  - [mm_qwen3.py](./../src/myvllm/models/mm_qwen3.py)

这说明你已经完成了一个重要阶段：

- 你知道多模态不是简单把图片路径传给模型
- 你已经让 token 序列里出现了视觉占位符
- 你已经让 prefill 阶段可以把视觉 embedding 合并进输入流
- 你已经开始用 scheduler、block manager、KV cache 的方式思考生成式推理

但你卡住的原因也很明确：

你现在要从“本地可解释的 fake multimodal 机制”走向“官方 Qwen2.5-VL 真实契约”。

这一步最容易错的不是代码语法，而是知识边界。
如果知识边界不稳，代码会出现一种很危险的状态：

- 每个模块都能写一点
- 每个模块都知道一点别人的细节
- 但整体协议不和官方模型真正对齐

所以，这份文档不急着给代码方案。
它要回答一个更靠前的问题：

- 在开始写真实 `VLProcessor`、`VLModelAdapter` 和 engine 接入代码之前，你到底还缺哪些知识？

---

## 先给结论：你现在缺的不是“会不会写类”，而是五个知识闭环

如果把当前阶段压缩成一句话，就是：

你已经知道“多模态信息应该进入 prefill”，但还没有完全掌握“官方 Qwen2.5-VL 要求这个多模态输入长什么样”。

开始写代码前，建议先补齐五个闭环：

1. `messages -> prompt text` 的 processor 契约
2. `prompt text -> input_ids` 的特殊 token 契约
3. `image -> pixel_values + image_grid_thw -> image token count` 的视觉长度契约
4. `VLProcessor -> VLModelAdapter -> Engine` 的职责边界契约
5. `official baseline -> local teaching path` 的对照验证契约

这五个闭环决定了你写代码时每个字段应该放在哪里。

如果它们不清楚，你会反复遇到类似问题：

- 这个 placeholder 应该由谁生成？
- `input_ids` 里到底应该有几个 image token？
- `pixel_values` 应该放在 processor 输出里，还是 runner 临时读图？
- engine 是否应该知道 `<|image_pad|>` 是什么意思？
- `generate()` 返回的 token 是否需要裁掉 prompt 前缀？
- fake vision 路径还能不能继续作为真实 VL 路径的一部分？

这些问题不是“写着写着自然会清楚”的问题。
它们应该在写代码前先被讲清楚。

---

## 当前代码状态：你已经建好了什么

先把当前本地路径讲清楚。
只有知道自己已经有什么，才知道下一步缺什么。

### 1. 当前入口仍然是文本 prompt 加全局 image_path

在 [main.py](./../main.py) 里，当前配置仍然是：

```python
'model_name_or_path': 'Qwen/Qwen3-0.6B',
'image_path': ".../images/01/num.png",
'num_vision_tokens': 256,
```

这说明当前运行路径本质上仍然是：

```text
prompt: str
image_path: str
num_vision_tokens: int
```

它还不是官方 Qwen2.5-VL 的输入形态。

官方路径更接近：

```text
messages
  -> processor.apply_chat_template(...)
  -> input_ids + attention_mask + pixel_values + image_grid_thw
```

### 问：为什么这很重要？

答：

因为 `prompt: str + image_path: str` 丢掉了多模态输入里最重要的一层结构。

官方 `messages` 里，图像和文本是有顺序的。
例如：

```python
[
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
```

这里表达的不只是“有一张图和一句话”，还表达：

- 谁说的话
- 图像在文本之前还是之后
- 图像和文本同属一次用户消息
- 后续 assistant generation prompt 应该追加在哪里

如果最外层只剩 `prompt: str + image_path: str`，processor 就失去了很多上下文。

### 2. 当前本地 Processor 已经做了 placeholder 拼接

在 [processor.py](./../src/myvllm/multimodal/processor.py) 里，本地 `Processor` 会生成：

```text
[vision_start] + [image_pad] * num_vision_tokens + [vision_end] + text_token_ids
```

并同时生成 `is_multimodal` mask：

```text
False, True, True, ..., True, False, False, ...
```

这一步很有价值。

它说明你已经从早期的“视觉长度只存在于额外字段里”前进到了更合理的状态：

- token 序列里真的有视觉占位符
- scheduler 和 block manager 看到的是完整序列长度
- model 侧能按 mask 替换 embedding

### 问：这是不是已经等于官方 Qwen2.5-VL processor？

答：

不是。

它只是“结构思想上接近”官方 VL serving 机制。

当前本地 processor 仍然有几个明显差异：

- 它接受的是已经渲染好的 `prompt: str`，不是结构化 `messages`
- 它手动把视觉 placeholder 放在整个文本前面
- `num_vision_tokens` 来自配置，而不是来自真实图像预处理后的 `image_grid_thw`
- 它不产出 `pixel_values`
- 它不产出 `image_grid_thw`
- 它使用的是 Qwen3 文本模型 tokenizer 和 fake/local vision path，而不是 Qwen2.5-VL 官方 processor 契约

所以它适合作为学习脚手架，但不能直接当作真实 Qwen2.5-VL 输入协议。

### 3. 当前 ModelRunner 已经理解 prefill 和 decode 的区别

在 [mm_model_runner.py](./../src/myvllm/engine/mm_model_runner.py) 里，当前逻辑是：

- prefill 阶段：
  - 收集完整 token 序列
  - 构建 slot mapping
  - 构建 attention 上下文
  - 读取图像
  - 得到视觉特征
  - 投影成 hidden size
  - 交给模型做 embedding merge

- decode 阶段：
  - 回到父类文本 decode 路径
  - 每步只追加新的文本 token
  - 依赖 prefill 已经写入 KV cache 的上下文

这是正确方向。

### 问：为什么多模态图像通常只进入 prefill？

答：

因为图像是 prompt 上下文的一部分。

prefill 的任务是把已有上下文一次性送进模型，建立 KV cache。
只要图像对应的视觉 token 已经在 prefill 阶段占据了上下文位置，decode 阶段就不需要每生成一个 token 都重新编码图像。

decode 阶段真正追加的是：

```text
上一步生成的文本 token
```

而不是：

```text
整张图片 + 历史文本 + 新 token
```

这也是当前本地 engine 值得保留的学习成果。

### 4. 当前模型侧 merge contract 已经比较清楚

在 [mm_qwen3.py](./../src/myvllm/models/mm_qwen3.py) 里，`MMQwen3ForCausalLM` 做了一件很干净的事：

```text
input_ids
  -> text embeddings
  -> 用 is_multimodal mask 把视觉位置替换成 multimodal_embeddings
  -> 交给 backbone
```

这说明你已经掌握了一个很重要的 serving 思想：

视觉信息可以先在 token 序列里占位，再在 embedding 层完成替换。

### 问：这个思想能不能迁移到真实 Qwen2.5-VL？

答：

能迁移一部分，但不能原封不动照搬。

可以迁移的是工程思想：

- placeholder token 在序列中占位
- 视觉特征数量必须和 placeholder 数量对齐
- prefill 阶段建立包含视觉上下文的 KV cache
- decode 阶段继续文本生成

不能直接照搬的是具体实现：

- 真实 Qwen2.5-VL 的视觉特征来自官方视觉塔和 projector
- 真实输入包含 `pixel_values`、`image_grid_thw` 等字段
- image token 数量不是手写配置出来的
- special token id 必须和 Qwen2.5-VL checkpoint/config/tokenizer 对齐
- prompt text 必须遵循官方 chat template

这就是你现在卡住的核心：

你已经理解了“像 vLLM 一样服务多模态”的工程骨架，但还没有把这个骨架绑定到 Qwen2.5-VL 的真实输入契约上。

---

## 第一块知识：Processor 不是 tokenizer 包装器

这是里程碑 2 最重要的一句话。

`tokenizer` 只负责把文本转成 token ids。

但 Qwen2.5-VL 的 `processor` 至少负责四类事情：

1. 理解结构化 `messages`
2. 应用 chat template，生成 prompt text
3. 处理图像，生成视觉 tensor
4. 打包模型需要的完整输入字段

也就是说，processor 的输出不是单一的 `input_ids`。

它更接近：

```text
{
    input_ids,
    attention_mask,
    pixel_values,
    image_grid_thw,
    ...
}
```

### 问：为什么不能说 processor 只是 tokenizer 外面包了一层？

答：

因为 tokenizer 不知道图像。

tokenizer 可以回答：

- 这段字符串对应哪些 token ids？
- 某个 special token 对应哪个 id？

但 tokenizer 不能独立回答：

- `messages` 里的图片应该放到 prompt 的哪个位置？
- 图像 placeholder 应该如何插入？
- 图像应该 resize / patch / normalize 成什么 tensor？
- `image_grid_thw` 是多少？
- image token 数量应该是多少？
- 最终模型 forward 需要哪些视觉字段？

这些都是 processor 契约的一部分。

### 问：`apply_chat_template(tokenize=False)` 看什么？

答：

它让你看到“结构化消息被渲染成了什么 prompt text”。

这一阶段你还没有进入 token ids。
你看到的是：

- role 如何被渲染
- user 内容如何被渲染
- 图像占位符在文本里出现在哪里
- assistant generation prompt 如何追加

它回答的是：

```text
messages -> prompt text
```

### 问：`apply_chat_template(tokenize=True)` 看什么？

答：

它让你看到“最终模型输入长什么样”。

这一阶段你看到的不只是 token ids。
在 Qwen2.5-VL 的官方路径里，它还可能返回：

- `input_ids`
- `attention_mask`
- `pixel_values`
- `image_grid_thw`

它回答的是：

```text
messages -> token ids + visual inputs
```

### 问：为什么要把 `tokenize=False` 和 `tokenize=True` 拆开观察？

答：

因为它们暴露的是两层不同的真相。

`tokenize=False` 适合学习 prompt 协议。

你可以观察：

- prompt text 长什么样
- 占位符在文本中位于哪里
- assistant 开始生成的位置在哪里

`tokenize=True` 适合学习模型输入协议。

你可以观察：

- special token ids 是多少
- image token 重复了多少次
- `input_ids` shape 是多少
- 视觉 tensor 是否被一起打包

如果只看 `tokenize=True`，你会直接看到 tensor，但不容易理解这些 tensor 从哪里来。
如果只看 `tokenize=False`，你能理解 prompt，但看不到模型真正消费的字段。

两者必须并排看。

---

## 第二块知识：placeholder 是结构信号，不是图像语义本身

在 VL 模型里，placeholder 的作用容易被误解。

placeholder 不是在表达“这张图里有什么”。
它只是在文本/token 序列里告诉模型：

```text
这里有视觉内容的位置。
```

真正的图像语义来自视觉 tensor 经过视觉塔、projector 之后得到的视觉特征。

### 问：如果 placeholder 不是图像语义，为什么它还这么重要？

答：

因为它决定视觉特征应该放进上下文的哪个位置。

可以把它理解成一个座位表：

- placeholder 负责预留座位
- image features 负责坐到这些座位上
- attention / position / KV cache 都按这些座位计算

如果座位数错了，或者位置错了，视觉特征即使本身正确，也会进入错误的上下文位置。

### 问：当前本地代码在 placeholder 上已经做对了什么？

答：

它至少做对了三件事：

- token 序列里显式插入了视觉 token
- `is_multimodal` mask 只标记中间的 image pad 区间
- `multimodal_embeddings.shape[0]` 必须等于 `is_multimodal.sum()`

这些都是非常好的工程基础。

### 问：当前本地代码在 placeholder 上还缺什么？

答：

它还缺官方长度来源。

当前代码用：

```python
num_vision_tokens = config["num_vision_tokens"]
```

这适合教学阶段，但真实 Qwen2.5-VL 里，image token 数量应该从图像预处理后的视觉网格和模型配置推导出来。

路线图中已经提示了关键字段：

- `image_grid_thw`
- `patch_size`
- `spatial_merge_size`
- `image_token_id`

所以你接下来不能只问：

```text
我要插入几个 <|image_pad|>？
```

而要问：

```text
官方 processor 对这张图片产生了怎样的 image_grid_thw？
这个 grid 对应多少个最终 image token？
input_ids 里 image_token_id 实际重复了多少次？
这个数量是否和模型侧 image features 数量一致？
```

### 问：为什么 `image_token_id` 的数值和它重复的次数不是一回事？

答：

`image_token_id` 是“哪一个 token 表示图像占位”。

重复次数是“这张图片需要多少个视觉位置”。

例如：

```text
image_token_id = 151655
```

只说明某个 token id 代表 image placeholder。

但 input_ids 里可能出现：

```text
151655, 151655, 151655, ...
```

重复多少次，取决于图像经过预处理后形成的视觉网格，以及模型的 merge 规则。

这也是为什么“知道 special token id”不等于“理解视觉长度契约”。

---

## 第三块知识：`image_grid_thw` 是桥，不是装饰字段

`image_grid_thw` 很容易被初学者当成一个普通 metadata。

但对 Qwen2.5-VL 来说，它是连接图像 tensor 和 token 占位长度的重要字段。

它描述的不是原图分辨率。

它描述的是图像经过 processor 处理后，在视觉侧形成的网格结构：

- `t`：时间维度，单图通常是 1
- `h`：视觉网格高度
- `w`：视觉网格宽度

### 问：为什么不能用原图宽高直接推 image token 数量？

答：

因为模型看到的不是原始图片像素。

中间还有：

- resize
- normalize
- patch
- spatial merge
- 视觉塔编码

原图宽高只是输入材料。
`image_grid_thw` 才是 processor 处理后交给模型的重要结构信息。

### 问：本地 `VLProcessorOutput` 为什么必须保留 `image_grid_thw`？

答：

因为 adapter 需要它。

如果你未来写：

```text
VLProcessor -> VLModelAdapter -> real Qwen2.5-VL model
```

那么 adapter 不能只拿到 `input_ids`。

它还需要知道视觉侧输入，例如：

- `pixel_values`
- `image_grid_thw`

否则 adapter 无法忠实调用真实 checkpoint。

### 问：如果本地 processor 只返回 `input_ids`，会造成什么后果？

答：

你会把视觉信息的处理责任推迟到别的模块。

最常见的后果是：

- runner 开始负责读图
- model adapter 开始猜图像 tensor shape
- engine 开始知道 placeholder 协议
- processor 退化成 tokenizer helper

这样职责会散掉。

正确方向是：

```text
processor 负责把上游多模态输入装配成模型输入契约
adapter 负责把这个契约映射给真实模型
engine 负责调度、prefill/decode、KV cache、采样
```

---

## 第四块知识：三层边界必须先画清楚

你现在最需要的一张图不是模型结构图，而是职责边界图。

建议先采用下面这条链路：

```text
VLRequest
  -> VLProcessor
  -> VLProcessorOutput
  -> VLModelAdapter
  -> VLEnginePath
```

### 1. `VLRequest` 应该表达什么

`VLRequest` 表达上游请求。

它应该更接近：

```text
messages
```

而不是：

```text
prompt + image_path
```

第一版可以很小，只支持：

- 单轮
- 单图
- user message
- text question

但概念上最好保留结构。

### 问：为什么第一版也要保留 `messages` 思维？

答：

因为这会影响 processor 的职责。

如果输入已经是扁平 prompt，processor 就只能在字符串外面补东西。
如果输入是 messages，processor 才能负责：

- role 渲染
- 图像和文本顺序
- chat template
- generation prompt
- 视觉占位符插入

### 2. `VLProcessor` 应该负责什么

`VLProcessor` 负责输入装配。

它应该知道：

- 怎么接受本地 `VLRequest`
- 怎么转成官方风格 `messages`
- 怎么调用或模仿 `apply_chat_template`
- 怎么得到 prompt text
- 怎么得到 `input_ids`
- 怎么得到 `pixel_values`
- 怎么得到 `image_grid_thw`
- 怎么记录 special token 区间

它不应该负责：

- KV cache 分配
- scheduler 策略
- decode loop
- logits sampling
- 手写真实模型 forward 细节

### 3. `VLProcessorOutput` 至少应该包含什么

一个教学版输出可以先包含：

```text
prompt_text
input_ids
attention_mask
pixel_values
image_grid_thw
special_token_ids
image_token_span 或 is_multimodal
raw_messages
```

其中 `raw_messages` 和 `prompt_text` 不一定是模型 forward 必需字段。
但它们对教学项目很重要，因为你要能对照官方基线。

### 问：为什么教学版输出要保留一些“调试字段”？

答：

因为你的目标不是只跑通，而是理解。

当本地输出不对时，你需要知道问题发生在哪一层：

- `messages` 构造错了？
- prompt text 错了？
- token ids 错了？
- image token 数量错了？
- `pixel_values` shape 错了？
- adapter 映射错了？

如果 processor output 只保留模型最小输入，debug 会困难很多。

### 4. `VLModelAdapter` 应该负责什么

`VLModelAdapter` 负责把本地框架的 processor output 交给真实 Qwen2.5-VL checkpoint。

它应该知道：

- 真实模型 forward / generate 需要哪些 key
- `VLProcessorOutput` 的字段如何映射到模型输入
- 模型返回值里哪里能拿到 logits 或 generated ids
- 如何把输出转换回本地框架需要的形式

它不应该负责：

- 构造用户 messages
- 决定 chat template
- 读取原始图片路径并自行预处理
- scheduler 的 block table

### 5. `VLEnginePath` 应该负责什么

engine 负责服务框架语义。

它应该知道：

- prefill 阶段处理完整 prompt
- decode 阶段追加新 token
- 序列长度如何计入 block manager
- 什么时候停止生成
- 如何裁剪 completion

它最好不要知道：

- `<|vision_start|>` 的具体 id
- `image_grid_thw` 怎么由图片算出来
- image token 数量的模型专属规则
- Qwen2.5-VL processor 内部细节

### 问：如果 engine 知道了 `<|image_pad|>` 的含义，是不是一定错？

答：

不一定。

教学项目里，engine 有时会暂时接触这些字段，方便你验证长度统计。

但长期边界应该是：

- engine 可以知道“这段 token 是多模态占位区间”
- engine 不应该知道“这个模型为什么要这样展开图像”

也就是说，engine 可以消费抽象后的 span / mask / length。
但不要让 engine 成为 Qwen2.5-VL prompt 协议的拥有者。

---

## 第五块知识：当前 fake path 和真实 Qwen2.5-VL path 的差异

下面这张表是你开始写代码前最应该反复看的。

| 维度 | 当前本地 fake path | 真实 Qwen2.5-VL path | 开始编码前要补的知识 |
| --- | --- | --- | --- |
| 上游输入 | `prompt: str + image_path` | 结构化 `messages` | `messages` 如何表达 role、图像、文本顺序 |
| prompt 构造 | 先用 Qwen3 text chat template，再手动加视觉占位符 | 官方 processor 应用 VL chat template | `apply_chat_template(tokenize=False)` 的真实输出 |
| placeholder 长度 | `config["num_vision_tokens"]` | 由图像预处理和模型配置决定 | `image_grid_thw`、merge、image token count |
| 视觉输入 | 本地 `VisionEncoder` + `VisionProjector` | `pixel_values` + Qwen2.5-VL 视觉塔 | processor 输出哪些视觉 tensor |
| 模型 | Qwen3 text backbone 包一层 merge | Qwen2.5-VL checkpoint | adapter 如何调用真实模型 |
| 语义能力 | 工程上可验证，不保证真实看图 | checkpoint 具备图文对齐能力 | 区分“能生成”和“真的视觉条件生效” |
| engine 价值 | 验证 prefill/decode 和长度账 | 承载真实 VL 推理 | 哪些 engine 假设还能保留 |

### 问：当前 fake path 是不是要废掉？

答：

不是。

它的学习价值很大。

它已经帮你验证了：

- placeholder 应该进入 token 序列
- multimodal mask 应该和 embedding 行数对齐
- prefill 可以合并视觉 embedding
- decode 可以继续文本路径
- scheduler 和 block manager 可以按完整长度工作

但它不能继续被误认为真实 VL path。

更准确的定位是：

```text
fake path = 学习 engine 多模态机制的脚手架
real path = 对齐 Qwen2.5-VL checkpoint 的目标路径
```

### 问：为什么“能输出一些 token”不等于“视觉理解成功”？

答：

因为语言模型只要输入合法，通常就能生成文本。

但生成文本可能来自：

- 文本 prompt 本身
- 模型先验
- 随机或未对齐视觉 embedding 的扰动
- 错误 placeholder 协议下的偶然输出

真正的视觉理解至少需要：

- 官方或对齐的视觉 encoder/projector
- 正确的 prompt placeholder 协议
- 正确的 image token 长度
- `pixel_values` 与 `image_grid_thw` 被真实模型消费
- 输出能随图像内容合理变化

所以不要用“有 completion”作为真实 VL 成功标准。

---

## 编码前应该完成的一张三列对照表

里程碑 2 建议产出一张三列对照表。
这里把它具体化。

你应该对固定样例记录：

| 层级 | 你要记录什么 | 你要回答的问题 |
| --- | --- | --- |
| `messages` | 原始 role/content/image/text 结构 | 图像和文本的顺序在哪里表达？ |
| prompt text | `apply_chat_template(tokenize=False)` 输出 | processor 生成了什么占位符和 generation prompt？ |
| token ids / special token 区间 | `input_ids`、special token ids、image token span | placeholder 被编码成哪些 id？重复了多少次？ |
| visual inputs | `pixel_values.shape`、`image_grid_thw` | 图像侧 tensor 如何进入模型？ |
| output trimming | `generated_ids`、trimmed ids、decoded answer | completion 为什么要裁掉 prompt 前缀？ |

### 建议记录模板

可以在文档或日志里固定下面这些标签：

```text
=== Raw messages ===
...

=== Rendered prompt text ===
...

=== Special token ids ===
vision_start_token_id = ...
vision_end_token_id = ...
image_token_id = ...

=== Final input ids ===
shape = ...
image token count = ...
image token span = ...

=== Visual inputs ===
pixel_values.shape = ...
image_grid_thw = ...

=== Generation ===
generated_ids.shape = ...
trimmed_completion_ids = ...
answer_text = ...
```

### 问：为什么这张表比直接写 `VLProcessor` 更重要？

答：

因为它会成为你后续所有实现的真值。

如果本地 processor 输出不对，你可以逐层比：

- messages 是否一致
- prompt text 是否一致
- special token ids 是否一致
- image token 数量是否一致
- visual tensor shape 是否一致
- output trimming 是否一致

没有这张表，你写代码时只能靠猜。

---

## 你现在最适合写的第一批代码是什么

在补完上面的知识后，第一批代码不应该直接大改 engine。

更合适的顺序是：

1. 先把官方 baseline 改造成更清晰的检查工具
2. 再实现一个教学版 `VLProcessorOutput` 数据结构
3. 再写一个包装官方 processor 的本地 `VLProcessor`
4. 再写对照日志，把官方输出和本地输出并排展示
5. 最后才开始考虑 `VLModelAdapter`

### 问：为什么第一版 `VLProcessor` 可以包装官方 processor？

答：

因为当前阶段的学习重点不是复刻官方图像预处理。

当前阶段更重要的是：

- 定义本地输入输出边界
- 看清 processor 负责哪些字段
- 把官方行为变成可观察、可对照的本地契约
- 防止 engine 和 adapter 各自偷偷处理输入

包装官方 processor 并不等于没有学习价值。

只要你自己定义了：

- `VLRequest`
- `VLProcessorOutput`
- 对照日志
- 字段职责说明

你就是在构建本地教学框架的边界。

### 问：什么时候才需要自己实现更多 processor 细节？

答：

当你已经能稳定回答下面问题之后：

- 官方 prompt text 为什么长这样？
- image token 数量为什么是这个数？
- `pixel_values` shape 为什么是这个 shape？
- `image_grid_thw` 和 placeholder 数量如何对应？
- adapter 需要哪些字段？

在那之前，硬写一份自制 processor 很容易只是“看起来像”，但行为不对。

---

## 动手前自测问答

下面这些问题建议你在写代码前逐个回答。
如果某个问题回答不出来，就回到官方 baseline 继续观察。

### Q1：官方 Qwen2.5-VL 的概念输入是什么？

答：

不是单个 prompt 字符串，而是结构化 `messages`。

`messages` 里包含 role，以及按顺序排列的多模态 content。

### Q2：prompt text 是谁生成的？

答：

由 processor 的 chat template 路径生成。

用户提供结构化 messages，不应该手写完整模型 prompt。

### Q3：图像占位符是在 `messages` 里直接写死的吗？

答：

通常不是。

`messages` 里表达的是“这里有一个 image 类型内容”。
具体渲染成什么视觉占位符、占位符放在哪里、tokenize 后变成哪些 id，是 processor/tokenizer/config 协同后的结果。

### Q4：tokenizer 能替代 processor 吗？

答：

不能。

tokenizer 只处理文本到 token ids 的映射。
processor 还要处理 messages、chat template、图像预处理、视觉 tensor 打包和模型输入字段。

### Q5：`VLProcessorOutput` 为什么不能只包含 `input_ids`？

答：

因为真实 Qwen2.5-VL 模型还需要视觉侧输入。

至少应该考虑：

- `attention_mask`
- `pixel_values`
- `image_grid_thw`
- prompt text
- image token span 或 mask
- special token ids

### Q6：当前本地 `Processor` 最大的教学价值是什么？

答：

它把视觉占位符真正放进了 token 序列，并生成了可用于 embedding merge 的 mask。

这让 scheduler、block manager、prefill 和模型 merge 能在同一条长度账上工作。

### Q7：当前本地 `Processor` 最大的局限是什么？

答：

它的视觉长度来自手写配置，输入来自扁平 prompt，视觉 tensor 不来自官方 Qwen2.5-VL processor。

所以它不是官方真实 processor 契约。

### Q8：`image_grid_thw` 描述的是原图尺寸吗？

答：

不是。

它描述图像经过 processor 处理后，在视觉侧形成的时间、高、宽网格。

### Q9：为什么 image token 数量必须和 image features 数量对齐？

答：

因为模型会在 image token 对应的位置放入视觉 features。

如果 token 占位数量和 features 行数不同，最轻会 shape mismatch，最重会位置语义错乱。

### Q10：engine 应该知道哪些多模态信息？

答：

engine 应该知道足够用于调度和长度统计的信息，例如：

- 序列总长度
- 哪些位置属于多模态占位
- prefill 是否包含视觉上下文

engine 不应该拥有模型专属的 prompt 渲染规则和图像预处理规则。

### Q11：adapter 为什么比重写完整模型更适合当前阶段？

答：

因为你的当前目标是让本地框架学会承载真实 Qwen2.5-VL checkpoint。

adapter 可以先复用 Hugging Face 的真实模型能力，让你专注于：

- 输入字段如何映射
- 输出 logits / ids 如何回到本地框架
- engine 哪些假设仍然成立

这比一开始重写完整视觉塔和 VL 模型结构更适合教学路线。

### Q12：什么时候可以说自己准备好开始写代码？

答：

当你能不看源码，画出这条链：

```text
messages
  -> prompt text
  -> input_ids + special token spans
  -> pixel_values + image_grid_thw
  -> model forward/generate
  -> prompt trimming
  -> completion text
```

并且能说清楚每一步由哪个模块负责。

---

## 编码前检查清单

在开始写真实路径代码之前，建议确认下面每一项。

### 官方基线层

- [ ] 已固定一张图片和一个问题
- [ ] 已打印 raw `messages`
- [ ] 已打印 `apply_chat_template(tokenize=False)` 的 prompt text
- [ ] 已打印 `apply_chat_template(tokenize=True)` 或 processor 输出的 `input_ids`
- [ ] 已记录 `vision_start_token_id`、`vision_end_token_id`、`image_token_id`
- [ ] 已记录 `pixel_values.shape`
- [ ] 已记录 `image_grid_thw`
- [ ] 已记录 generated ids 和 trimmed completion ids

### 知识理解层

- [ ] 能解释 processor 为什么不是 tokenizer
- [ ] 能解释 placeholder 为什么只是结构信号
- [ ] 能解释 image token 数量不是手写常量
- [ ] 能解释 `pixel_values` 和 `image_grid_thw` 的作用
- [ ] 能解释 prefill 和 decode 在 VL 推理中的分工

### 本地设计层

- [ ] 已决定 `VLRequest` 是否保留 messages 结构
- [ ] 已决定 `VLProcessorOutput` 包含哪些字段
- [ ] 已决定 processor、adapter、engine 的职责边界
- [ ] 已决定 fake path 和 real path 如何并存
- [ ] 已准备好用官方基线对照本地输出

如果这些项目还没有完成，不建议直接大改 engine。
更好的下一步是继续增强 baseline 检查和 processor 对照日志。

---

## 最小实现路线建议

如果要把这份知识转化成代码，建议按下面顺序走。

### 第一步：修正基线脚本的观察方式

当前 [run_qwen2_5_vl_baseline.py](./../run_qwen2_5_vl_baseline.py) 已经能跑官方模型。

下一步不是增加功能，而是让输出更适合学习：

- `tokenize=False` 时明确打印 prompt text
- `tokenize=True` 时明确打印 final inputs
- 分开打印 special token ids
- 统计 image token 出现次数
- 记录 prompt length 和 completion length

### 第二步：定义教学版数据结构

先定义结构，再写逻辑。

建议先有：

```text
VLRequest
VLProcessorOutput
```

`VLProcessorOutput` 第一版可以偏调试友好，而不是只保留最小 forward 参数。

### 第三步：实现包装官方 processor 的 `VLProcessor`

第一版可以内部使用：

```python
AutoProcessor.from_pretrained(...)
processor.apply_chat_template(...)
```

重点是把输出整理成本地稳定结构。

### 第四步：写 processor 对照脚本或测试

对同一个样例，并排输出：

- 官方 processor 原始输出
- 本地 `VLProcessorOutput`

确认关键字段一致。

### 第五步：再写 `VLModelAdapter`

当 processor output 稳定后，再写 adapter。

adapter 第一版可以先直接调用 Hugging Face Qwen2.5-VL。

目标不是性能，而是回答：

- 本地 processor output 能不能喂给真实模型？
- 模型输出能不能被本地框架理解？
- prompt trimming 后的 completion 是否合理？

### 第六步：最后接 engine

只有当 processor 和 adapter 都能独立工作后，再接入当前 engine。

这样如果出错，你才知道问题更可能在：

- 输入契约
- 模型 adapter
- engine 长度账
- prefill/decode

而不是所有问题混在一起。

---

## 最后总结

你现在卡在“即将写代码”的阶段，并不是坏事。

这恰好说明你已经走到了一个真正重要的边界：

从本地可控的 fake multimodal 机制，走向官方 Qwen2.5-VL 的真实输入契约。

当前最值得补的知识不是更多 PyTorch 写法，而是：

- processor 到底装配了什么
- placeholder 到底代表什么
- image token 数量从哪里来
- `pixel_values` 和 `image_grid_thw` 为什么必须保留
- adapter 和 engine 各自不应该知道什么

当这些问题能被你稳定回答时，代码反而会变简单。

因为你不会再问：

```text
这个字段应该塞到哪里？
```

你会自然知道：

```text
这是 processor 的职责。
这是 adapter 的职责。
这是 engine 的职责。
这个字段必须和官方基线对齐。
这个字段只是教学调试用。
```

到了那时，开始写 `VLProcessor` 就不是盲目开工，而是把已经理解清楚的契约落成代码。
