# Milestone-note 1 (Multimodal Milestone 1) — Implementation & Debug Notes

> 目标：在 **不破坏原 text-only decode/CUDA-Graph 路径** 的前提下，为仓库增加一个“Milestone 1 级别”的 **多模态(图片+文本) prefill** 演示。
>
> 核心约定（来自本仓库 `agent.md` 的工作原则）：
> - **尽量不改动已有类/函数**；优先通过 **继承 / 新建 class / 替换 wiring** 实现扩展。
> - 只有在“基类完全没有入口”时才做最小必要改动（并说明原因）。

---

## 1. Milestone 1 的范围与成功标准

- **decode 阶段完全不引入图片**：依赖 KV-cache（视觉前缀在 prefill 已写入 KV-cache），decode 仍然走原先的 token-id + CUDA Graph 快路径。

**详细解释**

你可以把这次 Milestone 1 理解成：

- 我们先“假装”图片已经被变成了一段长度为 $T_{vis}$ 的向量序列（每个向量维度 = `hidden_size`），这段序列就像“图片的文字版前缀”。
- 这段“图片前缀”只在 **prefill** 被喂给模型一次，让它把图片信息写进 KV cache。
- 之后 **decode** 每次只喂 1 个新 token（跟原本 text-only 一样），因为 KV cache 里已经有“图片前缀”的历史了，所以 decode 不需要再看图片。

### 1.2 成功标准（当时的验收点）

- `main.py` 中指定 `image_path` 后，可以跑通 generate 流程。
- Decode 阶段不被多模态逻辑干扰（仍使用 base runner 的 decode 路径）。


1) **跑得通**：给一个真实存在的 `image_path`，整条 generate 流程不要在中途报错。

2) **KV-cache 写入长度对齐**：prefill 时，模型实际“处理了多少 token”，KV-cache 也必须“写入多少 token”的位置。这里的 token 数是 $T_{vis}+T_{text}$。

3) **decode 不受影响**：decode 是你这个仓库的性能核心（CUDA Graph + one-token-step）。Milestone 1 故意不改它，所以必须确认 decode 仍然走原先那套逻辑。

---

## 2. 总体设计：prefill 注入、decode 保持不变

### 2.1 最小“契约”(contract)
- 输入侧：
  - 文本 token ids：长度 $T_{text}$
  - 视觉前缀 token embeddings：长度 $T_{vis}$（不对应 token id）
- Prefill 实际送入模型：
  - `inputs_embeds` 形状：$(T_{vis}+T_{text}, hidden)$
- KV-cache 写入侧：
  **详细解释**

  这里的“契约”就是：

  - 你往模型里塞了多少个“时间步 token”（不管它来自图片还是文字），KV-cache 就必须对应写入多少个位置。
  - `slot_mapping` 就是“把第 i 个 token 的 KV 写到 KV-cache 的哪个格子里”。

  所以只要你看到类似报错：

  - `Slot mapping size must match number of tokens`

  你就可以条件反射：**“我喂给模型的 token 数” 和 “我给 slot_mapping 准备的长度” 不相等**。

  另外提醒一下：`slot_mapping` 这个东西不只在一个地方用。

  - 它在 `prepare_prefill()` 被构造。
  - 它在 attention/kernel 写 KV 的时候被读取（通过 `set_context(...)` 放进全局 context）。

  所以这玩意是“多处被调用的关键链路”，错一处就全炸。

### 2.2 为什么 prefill 用 `inputs_embeds`

因为 Milestone 1 的“视觉 token”在实现上 **没有 token id**，只能在 embeddings 层注入；因此需要模型 forward 支持 `inputs_embeds`。

我们遵守“尽量不改旧代码”的原则：
- runner/engine 层：完全通过新增类实现
- 模型层：提供一个 MM wrapper（`MMQwen3ForCausalLM`）把 `inputs_embeds` 路径打通
- 仅在旧代码没有入口时做了最小必要改动（见 3.4）

**详细解释**

text-only 情况下，模型输入是 `input_ids`（整数 token id），模型自己在内部做 embedding。

但图片前缀在 Milestone 1 根本没有“token id”，只有“我直接生成的一段向量”。所以我们只能走 `inputs_embeds`：

- `inputs_embeds` = 你已经把（图片前缀 + 文本 token）都变成向量了，然后直接喂给 transformer。

这也是为什么你一定要让底层 `Qwen3Model.forward(...)` 支持 `inputs_embeds`（见 3.4），否则模型根本没入口接这个向量。

再强调一次调用链（多次提醒）：

- `MMModelRunner.prepare_prefill()` 负责拼出 `inputs_embeds`
- `MMModelRunner.run_model()`（或它内部调用的路径）负责把 `inputs_embeds` 真正传进模型
- 模型侧 `MMQwen3ForCausalLM.forward()` 负责转发
- 最终落到 `Qwen3Model.forward(inputs_embeds=...)`

这条链任何一环不支持 `inputs_embeds`，就会出现 `unexpected keyword argument inputs_embeds` 之类的错误。

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

  **详细解释**

  `ImageSequence` 的作用一句话：**让系统“以为”这条序列长度变长了（多了图片前缀），但 `token_ids` 里仍然只装文本 token**。

  这很绕，所以我再拆开讲：

  - 调度器 / block manager / KV-cache 都只关心 “这条序列一共有多少 token”（也就是 `len(seq)` / `seq.num_tokens`）。
  - 但 `token_ids` 是一个 Python list，本来只存文字 token。
  - 多模态时，图片前缀没有 token_id，没法塞进 `token_ids`。

  所以我们用 `ImageSequence.num_tokens = num_vision_tokens + len(token_ids)` 来“骗过”上层：

  - 上层以为：这条序列真的有 $T_{vis}+T_{text}$ 个 token。
  - 实际 `token_ids` 还是只有 $T_{text}$ 个。

  为什么要 override `last_block_num_tokens`？因为 base 版用 `token_ids` slice 来算最后一个 block 的 token 数。

  - 但 `token_ids` 不包含图片前缀，所以会算少。
  - 算少会导致 `slot_mapping` 少一截，KV-cache 写入时立刻报错（Bug #6）。

  多次提醒：`ImageSequence` 不止在 `mm_llm_engine.py` 创建一次。

  - warmup 时也要创建 `ImageSequence`（`MMModelRunner.warmup_model()`）
  - 真实推理 add_prompt 时也创建 `ImageSequence`（`MMLLMEngine.add_prompt()`）

  你如果只改了一个地方，另一个地方还在用 `Sequence`，就会类型不匹配或长度账不一致。

### 3.2 新增：`src/myvllm/utils/fake_vision.py`

- `fake_vision_embeds(image_path, num_vision_tokens, hidden_size, device, dtype)`
- 读取图片字节并哈希得到随机种子，生成确定性的 tensor：
  - 输出形状：$(T_{vis}, hidden)$

  **详细解释**

  这个模块就是“假装的视觉塔”。它做的事很土但很有用：

  - 读图片文件的字节内容
  - 哈希一下当随机种子
  - 生成一个**确定性的**随机张量 `vision_embeds`（同一张图每次生成一样，不同图大概率不一样）

  它不需要 PIL/CLIP，所以 Milestone 1 依赖最少，主要价值是**把引擎的多模态线路打通**。

### 3.3 新增：`src/myvllm/models/mm_qwen3.py`

- `MMQwen3ForCausalLM(Qwen3ForCausalLM)`
- forward 支持：
  - `forward(input_ids=None, *, inputs_embeds=None)`
- Milestone 1 只在 prefill 使用 `inputs_embeds`，decode 仍由 runner 维持 token id 路径。

**详细解释**

这就是一个“转接头”。你可以把它理解成：

- 原来的 `Qwen3ForCausalLM` 只接受 `input_ids`
- 现在我们给它加一个“如果你给我 `inputs_embeds` 我也能吃”的入口

注意：decode 仍然用 `input_ids`（每步一个 token id）。所以 MM wrapper 的存在不会改变 decode 的形状假设。

多次提醒：这个 `MMQwen3ForCausalLM` 被 `MMModelRunner.__init__` 创建，且发生在 warmup 之前。

- 如果 warmup 之前模型还是 text-only，就会在 warmup/prefill 阶段因为不支持 `inputs_embeds` 直接炸（Bug #2）。

### 3.4 变更（最小必要）：`src/myvllm/models/qwen3.py`

> 这是 Milestone 1 唯一“不得不触碰”旧实现的点（原因：底层 `Qwen3Model` 没有 `inputs_embeds` 入口）。

- `Qwen3Model.forward` 增加对 `inputs_embeds` 的支持。
- 保持 text-only 兼容：原有调用仍传 `input_ids`，行为不变。

**详细解释**

你这里做的改动非常关键，而且确实是“没入口就只能改”的那种：

- 最底层真正跑 transformer 的是 `Qwen3Model.forward()`
- 如果它不支持 `inputs_embeds`，上面再怎么 wrapper 都没用

所以你加了：

- `inputs_embeds` 不为空：直接用它当输入
- 否则：用 `input_ids` 去过 `embed_tokens`

这保证了：

- text-only 代码完全不改也能跑（仍传 `input_ids`）
- multimodal prefill 才会走 `inputs_embeds`

### 3.5 新增：`src/myvllm/engine/mm_model_runner.py`

- `MMModelRunner(ModelRunner)`
- 只改 prefill：
  - override `prepare_prefill()`：
    - 以 $T_{vis}+T_{text}$ 计算 `cu_seqlens_q/k`、`slot_mapping`、positions/KV-cache 写入映射
    - embeddings 侧：`inputs_embeds = concat(fake_vision_embeds, embed_tokens(text_ids))`
  - override `run_model()`：prefill 时走 `self.model(inputs_embeds=...)`
- decode：`super().run_model(...)`，保持 CUDA Graph 路径。

**详细解释**

`MMModelRunner` 是 Milestone 1 的“真正核心”，因为**所有 KV-cache 写入账本都在这里算**。

prefill 时它做了三件核心事：

1) **长度账：把每个 seq 的长度从 $T_{text}$ 改成 $T_{vis}+T_{text}$**
  - 体现在 `seqlens_q/k`、`cu_seqlens_q/k` 这些边界数组

2) **KV 写入账：`slot_mapping` 必须覆盖图片前缀 + 文本 token**
  - 这就是 Bug #6 的高发地

3) **输入张量：拼 `inputs_embeds = [vision_embeds; text_embeds]`**
  - `vision_embeds` 来自 `fake_vision_embeds(...)`
  - `text_embeds` 来自 `self.model.model.embed_tokens(input_ids_t)`

然后它还做了一个“看起来奇怪但救命”的动作：

- 返回 `input_ids_full = torch.zeros(full_token_count)` 这个占位 tensor

原因是：虽然你真正 forward 用的是 `inputs_embeds`，但下游有些路径（attention 写 KV 的逻辑）会用“你传进 run_model 的 tensor 的长度”当 token count。于是你必须确保这个长度 = $T_{vis}+T_{text}$，才能和 `slot_mapping` 对齐。

多次提醒调用链（这是最容易断的地方）：

- `MMLLMEngine.step()` 会 call runner 的 `run(seqs, is_prefill)`
- `ModelRunner.run()` 内会调用 `prepare_prefill()` 或 `prepare_decode()`（prefill 这一步你 override 了）
- `prepare_prefill()` 会调用 `set_context(...)` 把 `slot_mapping/cu_seqlens/...` 塞进全局 context
- 然后 `run_model()` 才真正 forward，attention 层会从全局 context 读取这些东西

也就是说：**prepare_prefill() 给 context 写的东西，会被模型层（`Qwen3DecoderLayer`/`Attention`）在完全不同的文件里读出来用**。这就是“同一个变量跨多处被调用”的典型坑。

### 3.6 新增：`src/myvllm/engine/mm_llm_engine.py`

- `MMLLMEngine`：负责在 `add_prompt` 等位置创建 `ImageSequence`，并使用 `MMModelRunner`。
- 从 `config` 读取：
  - `image_path`
  - `num_vision_tokens`

**详细解释**

`MMLLMEngine` 你可以理解成“LLMEngine 的多模态分身”。它主要负责两件事：

1) **add_prompt 时不再创建 `Sequence`，而是创建 `ImageSequence`**（带 `image_path/num_vision_tokens`）

2) **把 runner 换成 `MMModelRunner`**（prefill 会拼图片前缀 embeddings）

多次提醒：如果你只换 engine 不换 runner，或者只换 runner 不换 sequence 类型，都会在运行时炸。

- engine 负责“造什么 Sequence”
- runner 负责“怎么把 Sequence pack 成模型输入”

这俩是一套，必须一起换。

---

## 4. 初始化顺序与“继承优先”原则的落地方式

`ModelRunner.__init__` 是一个“带强副作用”的初始化：会很早做 warmup / KV cache / cudagraph capture。

为保证 warmup 也走 MM-capable 的模型与 sequence：
- `MMModelRunner` 选择 **不调用** `ModelRunner.__init__`
- 采用“拷贝 parent init 逻辑”的方式，但把搬运出来的逻辑提取成子函数：
  - `MMModelRunner._init_from_base_with_mm_model(...)`
- 并在 docstring 里注明：删/改了哪些逻辑、原因是什么（例如：必须先构造 MM 模型再 warmup）。

**详细解释**

这里最核心的一句是：**`ModelRunner.__init__` 有很多副作用，而且顺序很重要**。

base runner 的顺序大概是：

- 建模型（text-only）→ warmup → 分配 KV-cache →（可选）capture CUDA Graph

但 multimodal 要求：warmup/prefill 可能会走 `inputs_embeds`，所以 warmup 之前模型必须已经是 MM 版本。

因此你做了一个工程上很常见的“不得已的做法”：

- 不调用父类 `__init__`
- 复制父类 init 的代码，但把“建模那一行”换成建 `MMQwen3ForCausalLM`

这虽然丑，但逻辑最直：**我只在必要的地方插一刀，避免动到其他稳定路径**。

多次提醒：warmup 不只是跑一下那么简单，它会触发：

- `prepare_prefill()` 的打包逻辑
- `set_context(...)` 写全局 context
- 模型跑 forward

所以 warmup 其实会经过“几乎完整的 prefill 链路”，这就是为什么 warmup 阶段也必须用 `ImageSequence`（Bug #1）。

---

## 5. Bug 回顾与修复思路（按时间线）

> 本章是本次 Milestone 1 的关键产出：把遇到的坑、定位方法、最终解决方案都记录下来，方便后续 Milestone 2/3 做真实视觉塔时复用。

### Bug #1：warmup 阶段构造了 `Sequence`，但 MM prefill 期望 `ImageSequence`

- **现象**：在 prefill 或 warmup 路径出现 `TypeError: MMModelRunner expects ImageSequence`（或类似类型不匹配）。
- **根因**：base `ModelRunner.warmup_model()` 内固定构造 `Sequence`，而我们在 `MMModelRunner.prepare_prefill()` 强依赖 `ImageSequence` 字段。
- **修复**：在 `MMModelRunner` override `warmup_model()`：
  - warmup 用 `ImageSequence(..., image_path=None, num_vision_tokens=0)`
  - 保持 warmup 与 text-only 行为一致但类型满足 MM prefill。

  **详细解释**

  warmup 不是“随便跑跑”。它会调用你 override 的 `prepare_prefill()`。

  - base warmup 造的是 `Sequence`
  - 但你 MM prefill 里会访问 `seq.num_vision_tokens/seq.image_path` 这种字段

  所以 warmup 造错类型，运行时就会：

  - 要么 `isinstance` 检查失败
  - 要么属性不存在报错

  修复方式很直白：warmup 也造 `ImageSequence`，但把 `num_vision_tokens=0`，等价于“纯文本”。

### Bug #2：init 过程中 warmup 太早执行，导致模型还不是 MM-capable

- **现象**：`forward()` unexpected keyword argument `inputs_embeds`（或输入路径不匹配）。
- **根因**：base `ModelRunner.__init__` 早期 warmup 依赖 `self.model`；如果先走了基类 init，会用 text-only 模型 warmup。
- **修复**：`MMModelRunner` 不调用 base init，而是自己复刻 init 顺序：
  - 先构造 `MMQwen3ForCausalLM` + load weights
  - 再 warmup / allocate KV / capture cuda graph

  **详细解释**

  这其实就是初始化顺序错误：

  - warmup 时你需要 `inputs_embeds` 这条路
  - 但模型还是旧的 text-only 模型

  所以报错是很符合逻辑的：模型不认识 `inputs_embeds`。

  解决方法：把“建 MM 模型”放到 warmup 之前。

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

  **详细解释**

  `pin_memory=True` 是一个性能优化：让 CPU->GPU 拷贝更快。但它不是功能必需。

  你这里的 `input_ids_full` 只是一个“占位 shape/长度”的 tensor，里面的值不会被真正用来 embed（prefill 走的是 `inputs_embeds`），所以你完全可以不用 pin。

  记住一句话：**遇到 pin_memory 报错，先把它当性能优化关掉，保证功能跑通**。

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

**详细解释**

这个 bug 的本质是“你明明有 $T_{vis}+T_{text}$ 个 token 要写 KV，但你只准备了 $T_{text}$ 个位置”。

为什么会这样？因为你有两套“长度来源”：

- 上层长度：`len(seq)`（你已经通过 `ImageSequence.num_tokens` 修正为 $T_{vis}+T_{text}$）
- 另一条长度：某些地方会看“输入张量长度”（例如传进 `run_model` 的 tensor）

如果这两条长度不一致：

- `slot_mapping` 按 $T_{vis}+T_{text}$ 造
- 但模型/attention 认为只有 $T_{text}$

就会在写 KV 时触发断言。

所以你的双重修复是对的：

1) `ImageSequence.last_block_num_tokens` 用 `num_tokens` 计算，保证 slot_mapping 不会少
2) `prepare_prefill()` 返回一个“长度刚好等于 $T_{vis}+T_{text}$ 的占位输入”，保证下游按同样长度工作

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

**详细解释**

Milestone 2/3 最容易做错的一点是：你一引入真实视觉塔，就开始到处改打包逻辑。这样非常容易把已经跑通的 KV-cache / cu_seqlens / slot_mapping 再次搞崩。

正确姿势是：

- prefill packing（长度账、slot_mapping、cu_seqlens）尽量别动
- 只把 `fake_vision_embeds(...)` 换成 `real_vision_embeds(...)`（比如 CLIP/Vit 输出 + projector）

也就是说：**把难的“引擎账本”冻结住，把变动集中在“图片到向量”那一小块**。

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

**详细解释**

你可以把验证点当成“两个老大难 bug 的回归测试”：

- 看到 pin_memory 那个报错：说明你又在某个“占位 tensor”上 pin 了（或者某些设备组合不支持）
- 看到 slot_mapping 报错：说明你又把“token 数”算错了（最常见就是忘了把 $T_{vis}$ 算进去）

如果你后面做 Milestone 2/3，任何一次改动只要触发这俩报错，都先回到本 Milestone 1 的契约：

- $T_{total} = T_{vis} + T_{text}$
- `slot_mapping` 长度必须等于 $T_{total}$

---

## 8. 我指定了图片路径，但 `Completion:` 为空；纯文本却能正常回答（更准确的代码原因）

你这个现象我已经用 `uv run python main.py` 复现到了：

- 控制台只出现一次 `prefilling` 的 tokens/sec
- 没有 `decoding` 的 tokens/sec
- 最终打印 `Completion:` 为空字符串

这通常不是“图片路径没读到”，而是 **prefill 之后 sequence 被立刻判 FINISHED**，所以 decode 根本没开始。

### 8.1 关键点：在你的引擎里，prefill 也会“采样 1 个 token 并 append 到序列里”

链路（多次提醒，别断）：

1) `MMLLMEngine.step()` → 调度到 prefill batch（从 waiting 拉到 running）
2) `MMModelRunner.run(..., is_prefill=True)` → 计算 logits（prefill 用 `inputs_embeds`）
3) `SamplerLayer.forward(logits, temperature)` → 采样出 token_id（**prefill 也采样**）
4) `Scheduler.postprocess(seqs, token_ids)` → `seq.append_token(token_id)`（**prefill 后 num_tokens 会 +1**）
5) `Scheduler.postprocess` 立刻检查停止条件：EOS / max_tokens / max_model_length

所以你“以为 prefill 不会产出回答”这点，在当前实现里并不成立：**prefill 会产出下一 token 并且会触发停止判定**。

### 8.2 为什么这会在多模态里更容易把你直接弄成空 completion？

因为多模态序列长度是：

$$T_{total} = T_{vis} + T_{text}$$

你现在又把 vision prefix 也算进 `ImageSequence.num_tokens` 了（这是对的），那么只要：

- `SamplingParams.max_model_length` 没设置得足够大（至少要容纳 $T_{vis}+T_{text}$，还要再留生成空间）

prefill 后 append 的那 1 个 token 就会让它更容易满足：

- `seq.num_tokens >= seq.max_model_length`

于是 sequence 直接 finished，decode 根本跑不起来。

**【傻子也能听懂的解释】**

你现在相当于：一上来就把“图片前缀”当成 256 个 token 塞进了队列，然后 prefill 又顺手生成了 1 个 token。

如果你给的“最大总长度上限”没有把这 257 个 token 算进去，系统就会说：长度超了，别生成了——于是你看见的就是空 `Completion`。

### 8.3 还有一个会让你更迷惑的点：你只在 seq finished 时才把 completion 取出来

你现在 `step()` 的 outputs 只收：

```python
[(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
```

所以只要 seq：

- 没进入正常 decode 循环
- 或者被 max_length 提前 finished

你就“看不到生成的过程”，最后只剩一个空 completion。

---

## 9. 当前多模态“还缺什么”（便于你排 Milestone 2/3 的 TODO）

你说得对：Milestone 1 目前缺少真实的图像语义数据流，所以即便它开始 decode，理论上也只能“瞎编”。缺失点主要是这些：

1) **真实 vision tower**：`image -> vision features`（例如 ViT / CLIP）
2) **projector**：把 vision features 映射到 LLM 的 `hidden_size`，得到 `vision_embeds (T_vis, hidden_size)`
3) **image preprocess**：resize/normalize/patchify，保证输入一致且 `T_vis` 可控
4) **与目标权重一致的多模态协议**：你走“软前缀 concat inputs_embeds”是可行路线，但要与未来加载的 VL 权重对齐
5) （后续性能）**multimodal prefix cache key**：要把图片 hash 加进 key，否则会错用 KV-cache

**【傻子也能听懂的解释】**

现在的 `fake_vision_embeds` 只是“把图片变成一堆随机向量”，模型不可能真的理解。

Milestone 2/3 的核心就是：把“随机向量”换成“有语义的向量”。

---

## 10. 下一步 TODO（建议按这个顺序做）

### 10.1 先把工程链路修到“多模态能稳定进入 decode 并吐 token”

1) **把 `SamplingParams.max_model_length` 调大**（必须能容纳 vision prefix）
  - 最低要求：`max_model_length > num_vision_tokens + prompt_len + 1`

2) **加 stop reason 日志**（建议在 `Scheduler.postprocess`，只在 FINISHED 时打印一次）
  - 打印：`num_tokens / num_prompt_tokens / num_completion_tokens / max_model_length`
  - 打印：`stop_due_to_eos / stop_due_to_max_tokens / stop_due_to_max_length`

3) **调试期让 outputs 可见**
  - 临时每步打印 `last_token` 或每 N 步打印一次，避免“只在 finished 才能看到”

### 10.2 再补齐真正的视觉语义数据流（Milestone 2/3）

4) **实现真实 vision tower + projector**，替换 `fake_vision_embeds`
5) **实现 preprocess**，保证 dtype/device/shape 稳定
6) **确认与目标 VL 权重一致的协议**（软前缀 or 特殊 token）
7) （可选）**为 multimodal 做 prefix cache key**（text + image hash）

**【傻子也能听懂的解释】**

先把“能生成”搞定，再搞“能看图”。否则你会同时被两类问题折磨：到底是引擎 stop 条件把你弄死了，还是视觉语义根本没进去。
