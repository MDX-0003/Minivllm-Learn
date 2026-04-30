# 教学版 VLProcessor 接入真实 Tokenizer 与 ImageProcessor 计划

## 目标

当前 [vl_processor.py](./../src/myvllm/multimodal/vl_processor.py) 已经把核心 processor 链路拆出来了：

```text
VLRequest
  -> messages
  -> prompt_text
  -> image_grid_thw
  -> image_token_counts
  -> expanded_prompt_text
  -> input_ids
  -> image_token_spans + is_multimodal
  -> model_inputs
```

但当前测试 [test_teaching_vl_processor.py](./../tests/test_teaching_vl_processor.py) 里的两个关键组件还是假的：

- `TeachingTokenizer`
- `TeachingImageProcessor`

这两个 fake 组件适合验证本地逻辑，但还不能证明教学版 `VLProcessor` 已经和真实 Qwen2.5-VL processor 契约对齐。

下一步目标是：

- 保留 fake 测试，用来稳定验证本地逻辑
- 新增真实组件接入路径，用官方 `AutoProcessor` 拿到真实 tokenizer 和 image processor
- 新增一个可选的集成检查脚本或慢测试，和 [run_qwen2_5_vl_baseline.py](./../run_qwen2_5_vl_baseline.py) 的输出对齐

---

## 先回答命名问题：Processor 和 ImageProcessor 官方也是这样分层吗？

是的。

Hugging Face 官方 `Qwen2_5_VLProcessor` 本身就是一个总装配器。

它内部包装：

- `image_processor`
- `tokenizer`
- 新版 Transformers 中还包括 `video_processor`
- `chat_template`

官方文档对 `Qwen2_5_VLProcessor` 的定位是：它把 Qwen2.5-VL 的 image processor 和 Qwen2 tokenizer 包装成一个 processor，并提供统一入口。

所以命名上确实会出现：

```text
Processor       = 总装配器，负责多模态输入契约
ImageProcessor  = 图像预处理器，只负责图像侧 tensor
Tokenizer       = 文本分词器，只负责文本/token ids
```

这不是本地代码命名混乱，而是官方多模态栈本来就这么分层。

为了避免本地阅读混淆，建议在教学文档和代码注释里固定使用下面称呼：

| 本地/官方对象 | 推荐称呼 | 职责 |
| --- | --- | --- |
| `VLProcessor` | 教学版总 processor | 组织 messages、prompt、token ids、visual inputs |
| `official_processor` | 官方总 processor | HF `AutoProcessor` 返回的 Qwen2.5-VL processor |
| `tokenizer` | 文本 tokenizer | special token id 解析和 prompt tokenization |
| `image_processor` | 图像 processor | 图片 resize/normalize/patch/grid，产出视觉 tensor |
| `video_processor` | 视频 processor | 暂不接入，后续视频路径再考虑 |

---

## 当前 fake 测试的价值

fake 测试不应该删除。

它验证的是教学版 `VLProcessor` 自己的逻辑：

- `messages` 是否被转换成官方风格 dict
- prompt 里是否出现 `<|vision_start|><|image_pad|><|vision_end|>`
- `image_grid_thw = [[1, 8, 8]]` 是否能算出 16 个 image token
- 一个 `<|image_pad|>` 是否能展开成 16 个
- `input_ids` 里是否能找到连续 image token span
- `is_multimodal` 是否只标记 image token 位置
- `to_model_inputs()` 是否能打包 tensor dict

fake 测试的重点是：

```text
本地逻辑可控、稳定、无网络、无模型下载。
```

真实接入测试的重点才是：

```text
本地教学版输出是否和官方 processor 输出对齐。
```

这两类测试要并存，不要互相替代。

---

## 真实接入的推荐结构

建议新增一个明确的真实组件构造入口。

当前已有：

```python
VLProcessor.from_pretrained(model_name_or_path)
```

但后续建议让它承担更清晰的职责：

```text
AutoProcessor.from_pretrained(...)
  -> official_processor
  -> official_processor.tokenizer
  -> official_processor.image_processor
  -> teaching VLProcessor
```

概念代码：

```python
official_processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

teaching_processor = VLProcessor(
    tokenizer=official_processor.tokenizer,
    image_processor=official_processor.image_processor,
    spatial_merge_size=2,
)
```

这里有一个重要点：

教学版 `VLProcessor` 不应该直接把官方 `Qwen2_5_VLProcessor` 当黑盒调用到底。

更适合的路线是：

1. 先借官方 tokenizer
2. 再借官方 image_processor
3. chat template 可以先由本地 `render_prompt()` 复写
4. 最终和官方 `official_processor.apply_chat_template(...)` 输出对照

这样你仍然能看到每一步发生了什么，而不是重新回到黑盒。

---

## 真实 tokenizer 接入计划

### 输入

来自官方总 processor：

```python
tokenizer = official_processor.tokenizer
```

### 本地使用位置

在 `VLProcessor` 中：

```python
self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
self.tokenizer.encode(expanded_prompt_text, add_special_tokens=False)
```

### 需要验证的内容

对当前样例，应该能对齐：

```text
vision_start_token_id = 151652
vision_end_token_id   = 151653
image_token_id        = 151655
```

并且 tokenized 后的 `input_ids` 中：

```text
image_token_id 连续出现 16 次
```

### 风险点

本地 `render_prompt()` 是手写的，不一定完全等价于官方 chat template。

所以第一阶段不要只比较最终 decoded text。
应该先比较：

- `prompt_text`
- `expanded_prompt_text`
- `input_ids`
- special token 区间

---

## 真实 image_processor 接入计划

### 输入

来自官方总 processor：

```python
image_processor = official_processor.image_processor
```

### 当前问题

官方 `Qwen2VLImageProcessor` 的调用方式不一定等价于当前 fake 的：

```python
image_processor(["demo.png"])
```

真实 image processor 往往需要接收已经加载的图片对象，或者通过官方 processor 的更高层入口处理 message 中的图片。

因此不要假设真实 image processor 可以永远直接吃字符串路径。

### 推荐做法

为真实图像路径新增一个很薄的 adapter：

```text
Local image path
  -> PIL.Image.open(path).convert("RGB")
  -> official image_processor(images=[image], return_tensors="pt")
  -> pixel_values + image_grid_thw
```

建议不要把这段逻辑直接塞进 `VLProcessor.preprocess_images()`。

更清楚的做法是新增一个小类：

```python
class HFQwenImageProcessorAdapter:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image_paths: list[str]) -> dict:
        ...
```

这样 `VLProcessor` 仍然只依赖一个简单契约：

```text
image_processor(image_paths) -> {"pixel_values": ..., "image_grid_thw": ...}
```

### 需要验证的内容

对当前样例，应该尽量对齐 [qwen_2.5_vl_ouput](./../qwen_2.5_vl_ouput)：

```text
pixel_values.shape = (64, 1176)
image_grid_thw     = [[1, 8, 8]]
```

注意：你的输出里已经出现过官方提示：

```text
Qwen2VLImageProcessor is now loaded as a fast processor by default...
```

这意味着 fast/slow image processor 可能导致细节略有差异。
第一阶段先对齐 shape、grid 和 token 数，不要要求 `pixel_values` 每个浮点值完全一致。

---

## 为什么不直接调用 official_processor.apply_chat_template(tokenize=True)

可以调用，但不建议把它作为教学版 `VLProcessor` 的内部主路径。

因为：

```python
official_processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

会一次性完成：

- chat template
- placeholder 展开
- tokenization
- image preprocessing
- tensor packaging

这对生产使用很方便，但对当前学习目标太黑盒。

教学版更好的方式是：

```text
官方 processor = 对照真值
教学 VLProcessor = 分步复写
```

也就是说：

- 可以用官方 processor 生成 baseline
- 不要让教学版 processor 直接把所有工作委托给官方 processor
- 每一步都要能单独打印和比较

---

## 建议新增的本地文件

### 1. 新增真实 image processor adapter

建议文件：

```text
src/myvllm/multimodal/hf_qwen_image_processor.py
```

职责：

```text
把本地 image path 转成官方 image_processor 可消费的图片对象，并返回 pixel_values/image_grid_thw。
```

建议接口：

```python
class HFQwenImageProcessorAdapter:
    def __init__(self, image_processor):
        self.image_processor = image_processor

    def __call__(self, image_paths: list[str]) -> dict[str, torch.Tensor]:
        ...
```

### 2. 新增真实 processor 检查脚本

建议文件：

```text
inspect_teaching_vl_processor_real.py
```

职责：

```text
用真实 tokenizer + image_processor 跑教学版 VLProcessor，并和官方 baseline 打印对照。
```

建议输出：

```text
=== Official processor ===
prompt_text = ...
input_ids.shape = ...
image_grid_thw = ...
pixel_values.shape = ...
image_token_count = ...

=== Teaching VLProcessor ===
prompt_text = ...
expanded_prompt_text = ...
input_ids length = ...
image_grid_thw = ...
pixel_values.shape = ...
image_token_spans = ...
sum(is_multimodal) = ...

=== Diff summary ===
prompt_text_equal = ...
input_ids_equal = ...
image_grid_thw_equal = ...
image_token_count_equal = ...
```

### 3. 新增可选慢测试

建议文件：

```text
tests/test_teaching_vl_processor_real_components.py
```

默认可以加 `pytest.mark.skip` 或通过环境变量开启。

原因：

- 真实 processor 依赖模型文件
- 可能访问 cache
- 运行更慢
- 不适合每次单元测试都跑

建议触发方式：

```bash
RUN_REAL_VL_PROCESSOR_TEST=1 uv run pytest tests/test_teaching_vl_processor_real_components.py
```

---

## 接入顺序

### 第一步：保留当前 fake 单元测试

不要动：

```text
tests/test_teaching_vl_processor.py
```

它是本地逻辑的快速保护网。

### 第二步：新增 `HFQwenImageProcessorAdapter`

目标是把真实 image path 接到官方 image processor。

验收：

```text
输入 images/01/num.png
输出包含 pixel_values 和 image_grid_thw
image_grid_thw 接近或等于 [[1, 8, 8]]
```

### 第三步：用真实 tokenizer 替换 fake tokenizer

通过：

```python
official_processor = AutoProcessor.from_pretrained(model_name)
tokenizer = official_processor.tokenizer
```

验收：

```text
special token ids 对齐官方输出
```

### 第四步：跑教学版 processor 的真实组件路径

概念调用：

```python
official_processor = AutoProcessor.from_pretrained(model_name)
image_adapter = HFQwenImageProcessorAdapter(official_processor.image_processor)

teaching_processor = VLProcessor(
    tokenizer=official_processor.tokenizer,
    image_processor=image_adapter,
    spatial_merge_size=2,
)

output = teaching_processor.process(request)
```

验收：

```text
output.image_grid_thw.tolist() == [[1, 8, 8]]
output.image_token_counts == [16]
sum(output.is_multimodal) == 16
```

### 第五步：和官方总 processor 对照

同一个 `messages`，跑：

```python
official_inputs = official_processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
```

对比：

```text
official input_ids
teaching input_ids
official image_grid_thw
teaching image_grid_thw
official pixel_values.shape
teaching pixel_values.shape
official image_token_id count
teaching image_token_id count
```

### 第六步：再考虑接入 engine

只有当上面对齐后，再接：

```text
MMLLMEngine.add_vl_request(...)
```

不要现在就把真实 image processor 塞进 engine。

原因：

```text
processor 对齐问题和 engine 长度统计问题应该分开 debug。
```

---

## 当前代码需要注意的小问题

这些不是本计划要立刻修的内容，但后续接真实组件前建议清理：

1. `vl_processor.py` 中存在调试打印：

```python
print("=== is_multimodal length ===")
print("=== input_ids length ===")
```

建议后续改成可选 debug flag。

2. `process()` 当前调用：

```python
prompt_text = self.render_prompt(messages, add_generation_prompt=False)
```

如果要对齐官方 generation baseline，应该确认是否要改回：

```python
add_generation_prompt=request.add_generation_prompt
```

3. `tests/test_teaching_vl_processor.py` 文件末尾目前有手动调用：

```python
test_vl_processor_converts_output_to_model_inputs()
```

正式测试文件里通常不需要手动调用，交给 pytest 收集即可。

4. `VisionInputs.shape()` 对 `None` 的处理后续要小心。

当前写法在 `pixel_values is None` 时会访问 `.shape` 出错。

---

## 最小验收标准

真实组件接入完成后，至少要能对当前图片样例说明：

```text
官方总 processor:
input_ids.shape       = (1, 42)
pixel_values.shape    = (64, 1176)
image_grid_thw        = [[1, 8, 8]]
image_token_id count  = 16

教学版 VLProcessor:
len(input_ids)        = 42
pixel_values.shape    = (64, 1176)
image_grid_thw        = [[1, 8, 8]]
sum(is_multimodal)    = 16
image_token_spans     = one span with length 16
```

如果这些能对齐，说明教学版 `VLProcessor` 已经从 fake 组件阶段进入真实 Qwen2.5-VL processor 契约阶段。

---

## 参考来源

- Hugging Face Transformers Qwen2.5-VL 文档：`Qwen2_5_VLProcessor` 包装 `image_processor`、`tokenizer`，新版还包含 `video_processor` 和 `chat_template`。
- Hugging Face Transformers Qwen2.5-VL forward 文档：模型输入包含 `input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 等字段。
- 当前仓库官方基线输出：[qwen_2.5_vl_ouput](./../qwen_2.5_vl_ouput)
