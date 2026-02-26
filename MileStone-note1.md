# Milestone-note 1 (Multimodal Milestone 1) — Implementation & Debug Notes

> 目标：在 **不破坏原 text-only decode/CUDA-Graph 路径** 的前提下，为仓库增加一个“Milestone 1 级别”的 **多模态(图片+文本) prefill** 演示。
>
> 核心约定（来自本仓库 `agent.md` 的工作原则）：
> - **尽量不改动已有类/函数**；优先通过 **继承 / 新建 class / 替换 wiring** 实现扩展。
> - 只有在“基类完全没有入口”时才做最小必要改动（并说明原因）。

---

## 1. Milestone 1 的范围与成功标准

### 1.1 范围（刻意限制）

- **只做 prefill** 的多模态注入：把图片转为一段“视觉前缀 token embeddings”，拼到文本 embeddings 前面。
- **decode 阶段完全不引入图片**：依赖 KV-cache（视觉前缀在 prefill 已写入 KV-cache），decode 仍然走原先的 token-id + CUDA Graph 快路径。
- Milestone 1 使用 **fake vision embeddings**：不引入额外视觉塔（不依赖 PIL/CLIP 等），只用图片字节做一个确定性的 embedding 生成作为占位。

### 1.2 成功标准（当时的验收点）

- `main.py` 中指定 `image_path` 后，可以跑通 generate 流程。
- Prefill 阶段可以写入 KV-cache（slot_mapping/token count 对齐），不触发 attention/KV-cache 相关断言。
- Decode 阶段不被多模态逻辑干扰（仍使用 base runner 的 decode 路径）。

---

## 2. 总体设计：prefill 注入、decode 保持不变

### 2.1 最小“契约”(contract)

对每条 sequence：
- 输入侧：
  - 文本 token ids：长度 $T_{text}$
  - 视觉前缀 token embeddings：长度 $T_{vis}$（不对应 token id）
- Prefill 实际送入模型：
  - `inputs_embeds` 形状：$(T_{vis}+T_{text}, hidden)$
- KV-cache 写入侧：
  - `context.slot_mapping` 长度必须严格等于 $T_{vis}+T_{text}$

### 2.2 为什么 prefill 用 `inputs_embeds`

因为 Milestone 1 的“视觉 token”在实现上 **没有 token id**，只能在 embeddings 层注入；因此需要模型 forward 支持 `inputs_embeds`。

我们遵守“尽量不改旧代码”的原则：
- runner/engine 层：完全通过新增类实现
- 模型层：提供一个 MM wrapper（`MMQwen3ForCausalLM`）把 `inputs_embeds` 路径打通
- 仅在旧代码没有入口时做了最小必要改动（见 3.4）

---

## 3. 代码改动清单（新增/修改的文件）

### 3.1 新增：`src/myvllm/engine/image_sequence.py`

- `ImageSequence(Sequence)`
- 新增字段：
  - `image_path: str | None`
  - `num_vision_tokens: int`
- 关键点：
  - `num_tokens = num_vision_tokens + len(token_ids)`：让调度器/块管理/positions/KV-cache 以“总 token 数”工作。
  - 覆盖 `last_block_num_tokens`：修复“base 用 text-only token_ids slice 推 last block token 数”的问题（详见 Bug #6）。

### 3.2 新增：`src/myvllm/utils/fake_vision.py`

- `fake_vision_embeds(image_path, num_vision_tokens, hidden_size, device, dtype)`
- 读取图片字节并哈希得到随机种子，生成确定性的 tensor：
  - 输出形状：$(T_{vis}, hidden)$

### 3.3 新增：`src/myvllm/models/mm_qwen3.py`

- `MMQwen3ForCausalLM(Qwen3ForCausalLM)`
- forward 支持：
  - `forward(input_ids=None, *, inputs_embeds=None)`
- Milestone 1 只在 prefill 使用 `inputs_embeds`，decode 仍由 runner 维持 token id 路径。

### 3.4 变更（最小必要）：`src/myvllm/models/qwen3.py`

> 这是 Milestone 1 唯一“不得不触碰”旧实现的点（原因：底层 `Qwen3Model` 没有 `inputs_embeds` 入口）。

- `Qwen3Model.forward` 增加对 `inputs_embeds` 的支持。
- 保持 text-only 兼容：原有调用仍传 `input_ids`，行为不变。

### 3.5 新增：`src/myvllm/engine/mm_model_runner.py`

- `MMModelRunner(ModelRunner)`
- 只改 prefill：
  - override `prepare_prefill()`：
    - 以 $T_{vis}+T_{text}$ 计算 `cu_seqlens_q/k`、`slot_mapping`、positions/KV-cache 写入映射
    - embeddings 侧：`inputs_embeds = concat(fake_vision_embeds, embed_tokens(text_ids))`
  - override `run_model()`：prefill 时走 `self.model(inputs_embeds=...)`
- decode：`super().run_model(...)`，保持 CUDA Graph 路径。

### 3.6 新增：`src/myvllm/engine/mm_llm_engine.py`

- `MMLLMEngine`：负责在 `add_prompt` 等位置创建 `ImageSequence`，并使用 `MMModelRunner`。
- 从 `config` 读取：
  - `image_path`
  - `num_vision_tokens`

---

## 4. 初始化顺序与“继承优先”原则的落地方式

`ModelRunner.__init__` 是一个“带强副作用”的初始化：会很早做 warmup / KV cache / cudagraph capture。

为保证 warmup 也走 MM-capable 的模型与 sequence：
- `MMModelRunner` 选择 **不调用** `ModelRunner.__init__`
- 采用“拷贝 parent init 逻辑”的方式，但把搬运出来的逻辑提取成子函数：
  - `MMModelRunner._init_from_base_with_mm_model(...)`
- 并在 docstring 里注明：删/改了哪些逻辑、原因是什么（例如：必须先构造 MM 模型再 warmup）。

---

## 5. Bug 回顾与修复思路（按时间线）

> 本章是本次 Milestone 1 的关键产出：把遇到的坑、定位方法、最终解决方案都记录下来，方便后续 Milestone 2/3 做真实视觉塔时复用。

### Bug #1：warmup 阶段构造了 `Sequence`，但 MM prefill 期望 `ImageSequence`

- **现象**：在 prefill 或 warmup 路径出现 `TypeError: MMModelRunner expects ImageSequence`（或类似类型不匹配）。
- **根因**：base `ModelRunner.warmup_model()` 内固定构造 `Sequence`，而我们在 `MMModelRunner.prepare_prefill()` 强依赖 `ImageSequence` 字段。
- **修复**：在 `MMModelRunner` override `warmup_model()`：
  - warmup 用 `ImageSequence(..., image_path=None, num_vision_tokens=0)`
  - 保持 warmup 与 text-only 行为一致但类型满足 MM prefill。

### Bug #2：init 过程中 warmup 太早执行，导致模型还不是 MM-capable

- **现象**：`forward()` unexpected keyword argument `inputs_embeds`（或输入路径不匹配）。
- **根因**：base `ModelRunner.__init__` 早期 warmup 依赖 `self.model`；如果先走了基类 init，会用 text-only 模型 warmup。
- **修复**：`MMModelRunner` 不调用 base init，而是自己复刻 init 顺序：
  - 先构造 `MMQwen3ForCausalLM` + load weights
  - 再 warmup / allocate KV / capture cuda graph

### Bug #3：`fake_vision.py` 在开发过程中被误编辑导致语法/缩进损坏

- **现象**：`IndentationError` / `SyntaxError`。
- **根因**：中途编辑残留无效代码块。
- **修复**：恢复为单一职责模块：只提供 `fake_vision_embeds`。

### Bug #4：`FileNotFoundError`（图片路径不存在）

- **现象**：读取图片字节时报错。
- **根因**：`config['image_path']` 指向不存在路径。
- **修复**：运行时提供真实存在的图片路径（这属于 demo 配置问题，不是框架 bug）。

### Bug #5：`RuntimeError: Only dense CPU tensors can be pinned`

- **现象**：`torch.zeros(..., pin_memory=True)` 报错。
- **根因**：某些 PyTorch 构建/设备组合下，对 factory 创建的特定 tensor 做 pin 会失败。
- **修复（MM 扩展层内）**：
  - 对“占位用途”的 `input_ids_full` 禁用 `pin_memory=True`
  - 真实 CPU->GPU token id 仍可保留 pin（可选优化）。

### Bug #6：slot_mapping/token count 不匹配（KV-cache 写入断言）

- **现象**：
  - `AssertionError: Slot mapping size must match number of tokens`
- **根因（关键）**：
  - `ImageSequence.token_ids` 只有 text tokens
  - base `Sequence.last_block_num_tokens` 通过 slice `token_ids` 来推最后一个 block token 数
  - 对于含 vision prefix 的序列，这会导致最后 block token 数计算偏小，从而 `slot_mapping` 构造缺 token。

- **修复（最小侵入，继承友好）**：
  1) `ImageSequence` 覆盖 `last_block_num_tokens`：使用 `num_tokens`（vision+text 总长）直接计算。
  2) 同时在 `MMModelRunner.prepare_prefill()` 返回“full-length placeholder ids”（长度 = $T_{vis}+T_{text}$），避免 attention 内部从输入长度得到的 token count 与 context 不一致。

---

## 6. 当前状态与后续建议

### 6.1 当前状态

- `uv run python main.py` 已可跑通，不再触发 pin_memory 与 slot_mapping 相关错误。
- 输出内容可能为空，这与 fake vision embedding 的“占位能力”有关，属于 Milestone 1 的预期限制。

### 6.2 后续建议

- 给 `MMModelRunner.run_model(is_prefill=True)` 增加轻量一致性断言/日志（只在 MM 扩展中）：
  - `inputs_embeds.shape[0] == context.slot_mapping.numel()`
  - `inputs_embeds.shape[-1] == hidden_size`
- Milestone 2/3 替换 fake vision 为真实视觉塔时，建议把“vision->embeds”接口固定为模块化组件，保持 `MMModelRunner` 的 prefill packing 逻辑稳定。

---

## 7. 复现与验证

使用 `uv` 环境运行：

```bash
cd /home/dministrator/Minivllm-Learn
uv run python main.py
```

验证点：
- 不出现 `Only dense CPU tensors can be pinned`
- 不出现 `slot_mapping size must match number of tokens`
- 能打印 prefill 的 processed tokens / tokens/sec
