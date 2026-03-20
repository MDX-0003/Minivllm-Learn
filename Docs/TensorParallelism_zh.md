# 张量并行（Tensor Parallel, TP）原理速记（以 Linear 为例）

张量并行的目标是：把**同一层里的一个大矩阵乘法**拆到多张 GPU 上协同完成。

关键点：

- **切分的是参数张量（weight/bias 等）和相应的计算**，而不是把 `module` 这个“Python 对象”切开。
- 在常见的分布式实现中（PyTorch distributed / vLLM 风格），通常采用 **one process per GPU**：每张 GPU 对应一个进程（一个 rank）。

---

## 1. 多张 GPU 时，每张 GPU 分别持有什么？

在 TP 里，每个 GPU（更准确：每个 *rank/进程*）一般会持有：

- **该层参数的一个 shard（分片）**：例如线性层的 `weight` 只保存某些行或某些列。
- **与本 shard 对应的局部计算结果**：前向时每个 rank 只算自己负责的那一片输出（或中间结果）。
- **必要的通信张量**：为了把局部结果组装成下一层需要的形状，常见通信包括：
  - `all-reduce`（求和聚合）
  - `all-gather`（拼接聚合）
  - `reduce-scatter`（边聚合边分发）

在 LLM 推理里，注意力头（heads）和 KV cache 也常按 TP 切分：每张卡只存自己负责的那部分 head 的 K/V，从而降低单卡显存压力。

---

## 2. `module` 实例是每个 GPU 都有，还是 CPU 上唯一？

在实际的多 GPU 分布式运行中，通常是：

- **每张 GPU 对应一个进程（rank）**。
- **每个进程里都会各自构建一份模型的 module 实例**（Python 对象在不同进程内彼此独立）。
- 再把参数 `.to(device)` 放到该进程绑定的 GPU 上。

所以并不是“CPU 上只有唯一一份 module，然后被多个 GPU 共享”。在多进程架构下，Python 对象本身无法跨进程共享；各 rank 是通过通信原语协同完成计算。

---

## 3. 如果 GPU 上各有一个 Linear 层，那么切分的对象是什么？

切分对象是：**Linear 的参数张量以及输出/中间结果所对应的维度**。

设线性层为：

$$y = x W^T + b$$

其中 $W$ 的形状通常为 `[out, in]`。

TP 常见两种切法：

---

## 4. Column Parallel Linear（按输出维度切，常说按 `out_features` 切）

直觉：把输出特征维分块，每个 rank 负责输出的一段。

- `W: [out, in]` **沿着 `out` 维切分**（等价于切“行”）
- 每张 GPU 保存：
  - `W_i: [out/tp, in]`
  - `b_i: [out/tp]`
- 每张 GPU 本地计算：
  - `y_i = linear(x, W_i, b_i)`，得到 `[batch, out/tp]`
- 如果下一层需要完整的 `y`，就做一次 `all-gather`：
  - `y = concat(y_i, dim=-1)`

简化示意：

```
W (out,in) 按 out 切成 [W0; W1; W2; W3]
GPU0: W0 -> y0
GPU1: W1 -> y1
GPU2: W2 -> y2
GPU3: W3 -> y3
all-gather: y = [y0, y1, y2, y3]
```

### 与本仓库示例的对应

在 `src/myvllm/layers/linear_test.py` 的 `ColumnParallelLinear.weight_loader` 中：

- `loaded_weights.size(0)`：取 `out` 维大小
- `loaded_weights.narrow(0, start_index, shard_size)`：**沿 dim=0（out 维）切出行分片**
- `param_data.copy_(slided_weight)`：把该 rank 负责的 shard 拷贝进本 rank 的 `nn.Parameter`

这段代码体现的就是“每个 rank/进程的 Linear 层实例里，只保存全量权重的一部分”。

---

## 5. `MergedColumnParallelLinear`：它和父类有什么区别？

你现在的实现位于 `src/myvllm/layers/linear.py`：`MergedColumnParallelLinear(ColumnParallelLinear)`。

先给结论（功能差异）：

- **父类 `ColumnParallelLinear`**：面向“单个线性矩阵”的 column-parallel（按 `out` 维切）。加载 checkpoint 时，`weight_loader(param, loaded_weights)` **一次性**把“全量权重的本 rank 分片”拷贝到本地参数。
- **`MergedColumnParallelLinear`**：面向“多个线性矩阵沿 `out` 维拼接后的大矩阵”。它的前向仍是一次 `linear(x, W_merged, b)`（继承父类 `forward`，不变），但加载时支持你**按子矩阵逐段装载**到 merged 参数的正确区间。

换句话说：

- `ColumnParallelLinear` 更适合：checkpoint 里就是一个完整的大矩阵，你也一次传入一整块权重来加载。
- `MergedColumnParallelLinear` 更适合：checkpoint 里是多块独立矩阵（例如 `q_proj.weight / k_proj.weight / v_proj.weight`），但你部署/计算时想把它们合并成一个 fused 的大矩阵，用一次 GEMM 提升效率。

### 5.1 为什么要 merge？（动机）

以注意力里的 Q/K/V 投影为例：

- 朴素实现需要 3 次 GEMM：
  - `Q = X @ Wq^T`
  - `K = X @ Wk^T`
  - `V = X @ Wv^T`
- 如果把 `Wq/Wk/Wv` 沿 `out` 维拼起来：
  - `Wqkv = concat([Wq, Wk, Wv], dim=0)`，形状 `[out_q + out_k + out_v, in]`
  - 一次 GEMM：`Y = X @ Wqkv^T`
  - 再把 `Y` 沿输出特征维切回 `Q/K/V`

这样做的常见收益：更少的 kernel launch、更好的融合机会、吞吐更高。

### 5.2 它到底“比父类多做了什么”？（核心：分段装载）

`MergedColumnParallelLinear` 本质仍是 column-parallel：每个 rank 只持有 merged 大矩阵在 `out` 维上的一个 shard。

真正的新增点在 `weight_loader`：它多了一个参数 `loaded_weight_id`（表示你这次加载的是 merged 里的第几块子矩阵），从而允许你对 merged 参数“定位区间并填充”。

设：

- `output_sizes = [o0, o1, o2, ...]` 表示 merged 后每一块子矩阵的 `out` 大小（例如 Q/K/V 三块）。
- 总输出为 `O = sum(output_sizes)`。
- TP size 为 `tp`。
- merged 后本 rank 的本地参数形状为 `[O/tp, in]`。

当你要把第 `k` 块（大小 `ok`）加载进 merged 参数时，这段实现做了两件事：

1) **在“本地 shard 参数”中定位第 k 块的区间**（local offset）：

- `offset = sum(output_sizes[:k]) // tp`
- `shard_size = output_sizes[k] // tp`
- `param_slice = param_data.narrow(0, offset, shard_size)`

2) **从“全量子矩阵 loaded_weights（形状 [ok, in]）”中取出本 rank 对应的分片**：

- `start = tp_rank * shard_size`
- `weight_slice = loaded_weights.narrow(0, start, shard_size)`
- `param_slice.copy_(weight_slice)`

图示（以 `tp=4`、三块矩阵 Q/K/V 为例）：

```
全量（未切分）:
  Wq (oq,in)
  Wk (ok,in)
  Wv (ov,in)

合并后:
  Wqkv = [Wq; Wk; Wv]  (oq+ok+ov, in)  沿 dim0 拼接

每个 rank 的本地参数:
  Wqkv_rank 形状: ((oq+ok+ov)/tp, in)
  且内部逻辑区间为: [Q_shard_rows | K_shard_rows | V_shard_rows]

加载时（逐块调用 weight_loader）:
  load Q: 拿到 Wq 的本 rank shard -> 填到 Wqkv_rank 的 Q 区间
  load K: 拿到 Wk 的本 rank shard -> 填到 Wqkv_rank 的 K 区间
  load V: 拿到 Wv 的本 rank shard -> 填到 Wqkv_rank 的 V 区间
```

### 5.3 看代码时的几个注意点

- `MergedColumnParallelLinear` **没有重写 `forward`**：它只保证“参数形状是 merged 的”，前向就是一次 `linear`。它不会自动把输出再 split。
  - 通常 split 会发生在更上层逻辑里（例如 attention 层拿到 fused 输出后再切成 Q/K/V）。
- 它的 `weight_loader` 签名是三参：`(param, loaded_weights, loaded_weight_id)`。
  - 这意味着加载 checkpoint 的调用方需要知道这是 merged 层，并额外传入 `loaded_weight_id`。
- `output_sizes[k]` 必须能被 `tp` 整除，否则 `// tp` 会导致形状不匹配。

### 5.4 “单独加载某一块权重”的意义是什么？（结合应用场景）

你现在这个类的典型应用就是 **fused/merged 权重计算**：计算时希望把多个线性投影合成一次 GEMM，但 checkpoint 往往仍然按“原始层”分别存权重。

以 QKV 为例（checkpoint 常见三份参数）：

- checkpoint 里：`q_proj.weight`、`k_proj.weight`、`v_proj.weight` 三个独立张量，各自形状一般是 `[out_q, in]`、`[out_k, in]`、`[out_v, in]`。
- 部署时：你想用一个 `Wqkv = [Wq; Wk; Wv]` 做一次 `Y = X @ Wqkv^T`。

这就产生了“计算结构”和“权重存储结构”的不一致：

- 计算想要 1 个 merged 参数。
- 权重却是 3 份独立张量。

`MergedColumnParallelLinear.weight_loader(..., loaded_weight_id)` 的意义就是：

- **不需要先把三份全量权重 concat 成一个大张量再加载**。
- 而是可以把 `Wq/Wk/Wv` **逐块**读取、逐块切 shard、逐块写入 merged 参数对应区间。

这样做的实际收益通常有三类：

1) **避免额外的拼接开销与内存峰值**

- 如果用父类 `ColumnParallelLinear` 来模拟 merge，你往往要先构造 `Wqkv_full = cat([Wq_full, Wk_full, Wv_full], dim=0)`（在 CPU 或 GPU 上都可能产生一次额外的大分配）。
- 对大模型（比如 4096×4096 级别）来说，这会带来明显的瞬时内存峰值和拷贝开销。
- 逐块加载则是“读一块 -> 切一块 -> 写一块”，可以把峰值压到接近单块矩阵的大小。

2) **兼容 checkpoint 的命名与格式**

- 很多 checkpoint 就是按 `q_proj/k_proj/v_proj` 分开存的。
- `MergedColumnParallelLinear` 让你不必改变 checkpoint 格式（也不必离线重打包成 fused 权重），加载逻辑就能直接对接。

3) **更灵活地支持“块大小不完全相同”的合并**

- 例如 GQA/MQA 场景里，`K/V` 的 heads 数可能小于 `Q`，从而 `out_k/out_v` 可能与 `out_q` 不同。
- 逐块加载时你只要让 `output_sizes=[out_q,out_k,out_v]` 匹配即可；每块按自己的 `shard_size` 定位与加载。

一个非常直观的加载伪代码（示意）：

```python
# merged_qkv = MergedColumnParallelLinear(input_size, [out_q, out_k, out_v])
merged_qkv.weight_loader(merged_qkv.weight, checkpoint['q_proj.weight'], loaded_weight_id=0)
merged_qkv.weight_loader(merged_qkv.weight, checkpoint['k_proj.weight'], loaded_weight_id=1)
merged_qkv.weight_loader(merged_qkv.weight, checkpoint['v_proj.weight'], loaded_weight_id=2)
```

对比父类的思路（你不想要的流程）：

```python
Wqkv_full = torch.cat([Wq_full, Wk_full, Wv_full], dim=0)  # 额外大分配
column_parallel.weight_loader(column_parallel.weight, Wqkv_full)
```

因此，“单独加载某一块权重”不是为了改变 TP 的切分方式（它仍然是按 `out` 维 shard），而是为了**让 fused 计算结构能无缝对接非 fused 的 checkpoint**，并降低加载时的额外开销。

---

## 6. Row Parallel Linear（按输入维度切，常说按 `in_features` 切）

直觉：把输入特征维分块，每个 rank 负责输入的一段，并计算输出的部分和。

- `W: [out, in]` **沿着 `in` 维切分**（等价于切“列”）
- 每张 GPU 保存：
  - `W_i: [out, in/tp]`
- 输入也对应切分：
  - `x_i: [batch, in/tp]`
- 每张 GPU 本地算“部分和”：
  - `p_i = x_i @ W_i^T`，形状为 `[batch, out]`
- 然后用 `all-reduce(sum)` 聚合得到完整输出：
  - `y = sum_i p_i`

偏置 `b` 的处理：常见做法是只在某一处加一次，或者各处一致加但需保证语义一致（实现策略可因框架而异）。

简化示意：

```
x (batch,in) 按 in 切成 [x0 | x1 | x2 | x3]
W (out,in) 按 in 切成 [W0 | W1 | W2 | W3]
GPUk: pk = xk @ Wk^T  (都是 batch,out)
all-reduce(sum): y = p0 + p1 + p2 + p3
```

---

## 7. 一句话总结

- **每张 GPU / 每个 rank 都有一份完整的 module 结构（Python 模型对象）**；
- **但该 module 内的参数张量只装“全量权重的一个切片（shard）”**；
- **切分对象是权重/激活的某个维度（例如 Linear 的 `out` 或 `in`），并通过通信把结果拼回或求和**。
