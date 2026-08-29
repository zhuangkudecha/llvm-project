# 单 Transformer Block 前向（CUDA C++）

对应 24 周 GPU CodeGen 计划 [§16.1 扩展路线](../../Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md) 中的 attention/softmax/layernorm 练习。所有 kernel 从零手写，复用 Phase 1 的 tiled matmul、tree reduction、错误检查与测量纪律。

## 文件

```text
transformer.cu     全部代码：5 个 kernel + host 前向 + CPU reference + main
CMakeLists.txt     构建（C++17/CUDA17，-O3 -lineinfo 供 ncu 归因）
```

## 构建与运行（需有 CUDA 的环境，如 4090 实例）

```bash
cmake -S . -B build -G Ninja
cmake --build build
./build/transformer_block --seq 64 --d 256 --heads 4
```

参数：`--seq N --d D --heads H --ffn F --seed S`（默认 64/256/4/4D/42；要求 `D % H == 0` 且 `d_head = D/H <= 256`）。

期望输出：

```text
[correctness] max_abs_err=...  max_rel_err=...  -> PASS
[timing]      block forward: median=... us  p20=...  p80=...  (warmup=20, runs=100)
```

验收命令（计划纪律）：

```bash
compute-sanitizer --tool memcheck ./build/transformer_block
compute-sanitizer --tool memcheck ./build/transformer_block --seq 127 --d 96 --heads 6  # 非整除边界
ncu --set full --kernel-name regex:matmul ./build/transformer_block
```

## 计算图（GPT-2 风格 decoder block，无 causal mask）

```text
X ──┬─ LN1 ── QKV ── Attention ── O ── (+) ── H1 ── LN2 ── FF1(+b) ── GELU ── FF2(+b) ── (+) ── Out
    │                                   ↑                          ↑
    └───────────────────────────────────┘                          └──────────────┘
                           (residual)                              (residual)
```

权重布局（row-major）：`Wqkv [D,3D]`、`Wo [D,D]`、`W1 [D,FF] + b1 [FF]`、`W2 [FF,D] + b2 [D]`、两个 LayerNorm 的 `gamma/beta [D]`。Q/K/V 是 QKV 缓冲的列切片（偏移 0/D/2D），head h 的列起点是 `h*d_head`，无需拷贝。

## Kernel 一览

| kernel | block 映射 | 要点 |
|---|---|---|
| `matmulTiledKernel` | 16×16 线程算一个 16×16 输出 tile | 两次 `__syncthreads()`、M/N/K 三向 mask、可选 bias、coalesced tile load |
| `layernormKernel` | 一行一个 block | double 累加 + tree 归约，`E[x²]-E[x]²` 方差 clamp |
| `attentionKernel` | (query 行, head) 一个 block | 在线 softmax（flash attention 的 rescale 技巧），K/V 只扫一遍 |
| `geluKernel` | 逐元素 grid-stride | tanh 近似，与 CPU reference 同公式 |
| `residualAddKernel` | 逐元素 grid-stride | 残差连接 |

## 关键设计点

- **Attention 为什么要 fused**：naive 做法要把 `N×N` 的 scores 矩阵写到 global memory 再读回来（写+读 2 次 `N²` 浮点流量）；在线 softmax 只扫一遍 K/V，scores 只存在于 block 内部。这就是 flash attention 节省 global traffic 的核心思路。
- **在线 softmax**：`m = max(m, s)`，`alpha = exp(m_old - m_new)` 重缩放历史累加 `o`，`l = l*alpha + p` 维护 exp 和；最后 `o /= l`。
- **每 key 两次 block 同步**（s 归约 + m/l 发布），是当前版本的性能瓶颈；flash attention 用更大的 K/V tile 摊薄同步，这是下一步的优化方向。
- **Q/K/V 零拷贝切片**：attention 直接以 `dQKV + D`、`dQKV + 2D` 为 K/V 基址，不产生拷贝 kernel。
- **正确性纪律**：CPU double 累加 reference，逐元素比较 `max_rel_err <= 1e-3`；所有 CUDA 调用走 `CHECK_CUDA`；kernel launch 后立即 `cudaGetLastError`。
- **测量纪律**：warmup 20 次、测量 100 次、CUDA Event 只量 kernel 时间、输出 median/p20/p80；correctness 在计时之前。

## 已知取舍与下一步

- matmul 是 FP32 CUDA core 版（未用 Tensor Core / 未 double-buffer / 未 swizzle），性能目标是"正确可教学"，不是接近 cuBLAS。
- 无 causal mask、无 KV cache、无多头并行优化（每 head 一个 block 已够小模型）。
- 下一步建议：加 causal mask → 对比 naive 两遍 softmax 与在线版本的 ncu 指标 → 换 FP16 + `mma`/cuBLAS 对照 → 用 Triton 写同一 block 与 CUDA 版本对拍（对应计划 Phase 4 的 Triton 路线）。
