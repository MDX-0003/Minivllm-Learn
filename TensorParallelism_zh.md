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

## 5. Row Parallel Linear（按输入维度切，常说按 `in_features` 切）

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

## 6. 一句话总结

- **每张 GPU / 每个 rank 都有一份完整的 module 结构（Python 模型对象）**；
- **但该 module 内的参数张量只装“全量权重的一个切片（shard）”**；
- **切分对象是权重/激活的某个维度（例如 Linear 的 `out` 或 `in`），并通过通信把结果拼回或求和**。
