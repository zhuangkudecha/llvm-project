# RTX 4090 GPU Kernel/CodeGen Engineer Learning Plan

详细逐周执行文档：[RTX4090_GPU_CodeGen_24Week_Execution_Plan.md](./RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

## 1. 定位

本计划面向希望成长为中高级 AI 编译器 Kernel/CodeGen 工程师的学习者，执行周期为 2026 年 7 月中旬至 12 月底。

目标不是泛泛学习所有 AI 编译器组件，也不是只跑通 Toy Tutorial，而是围绕一个明确硬件、一个真实编译器栈和一个可测量 workload，建立完整的 GPU CodeGen 性能优化能力。

```text
方向：NVIDIA GPU Kernel/CodeGen
目标硬件：RTX 4090
MLIR 基础环境：upstream llvm-project/mlir
性能项目底座：Triton compiler
目标 workload：FP16 Linear + Bias + ReLU
```

核心能力目标：

```text
1. 能独立阅读和编写 MLIR pass、rewrite 与 transformation
2. 能理解 TTIR -> TTGIR -> LLVM IR -> PTX 的 CodeGen 路径
3. 能独立实现 CUDA 与 Triton GPU kernel
4. 能建立公平、可复现的 correctness 和 benchmark 基线
5. 能使用 ncu/nsys 从硬件指标定位性能瓶颈
6. 能修改一个真实 GPU CodeGen transformation 并完成 ablation
7. 能解释优化的适用范围、收益、代价和退化场景
```

核心主线：

```text
GPU programming model
  -> CUDA/Triton kernel
  -> MLIR transformation
  -> TTIR/TTGIR layout and mapping
  -> LLVM IR/PTX/SASS
  -> RTX 4090 profiling
  -> hypothesis-driven optimization
  -> regression and engineering delivery
```

---

## 2. 年底验收标准

### 2.1 硬验收

年底必须完成：

```text
1. 独立实现 CUDA 和 Triton 版本的 fused Linear + Bias + ReLU
2. 从源码构建 Triton，并能 dump/解释各级 IR 与 PTX
3. 在 Triton compiler 中修改至少一个 MLIR transformation
4. 为 transformation 编写 IR/FileCheck 或等价编译器测试
5. 在 RTX 4090 上建立稳定 benchmark 和 ncu 分析流程
6. 六组标准 shape 全部通过 correctness
7. 六组中至少四组达到原始 Triton baseline 的 90%
8. 任何一组不得比原始 Triton 慢 20% 以上
9. 输出包含成功和失败实验的完整性能报告
```

### 2.2 良好标准

```text
1. 六组 shape 的性能几何平均达到 Triton baseline 的 95%
2. 至少两组超过 Triton 5%
3. 能用 IR/PTX/SASS 和 ncu 指标解释收益来源
4. 有自动 correctness 和性能回归测试
```

### 2.3 挑战目标

```text
1. 六组 shape 的性能几何平均不低于 Triton baseline
2. 在一个明确 shape 区间稳定超过 Triton 5%-10%
3. 优势不能依赖更低精度、更少 fusion 或关闭 Triton autotune
4. 明确记录不适用场景和性能退化原因
```

“超过 Triton”是挑战目标，不是年底硬门槛。一次偶然快 2% 的结果不算优化成功；完整的假设、实现、测量和归因闭环比单点胜出更重要。

---

## 3. 范围控制

### 3.1 本阶段主线

```text
NVIDIA RTX 4090
CUDA
Triton DSL
TTIR / Triton Dialect
TTGIR / TritonGPU Dialect
LLVM/NVVM/PTX
Linalg / Transform / TilingInterface
Nsight Systems / Nsight Compute
FP16 Tensor Core GEMM
```

### 3.2 年底前暂缓

```text
AMD/Intel GPU 后端
通用 CPU 深度优化
LLVM SelectionDAG/GlobalISel/寄存器分配器开发
完整模型图编译器
分布式训练编译
专家 Kernel dispatch 系统
FP8/INT8/量化 CodeGen
通用 dynamic-shape 编译器
从零搭建完整 GPU runtime
IREE 全栈
```

这些内容不是不重要，而是不能与当前主线并行展开。年底项目完成后再按岗位需求扩展。

### 3.3 两个代码库的分工

```text
upstream MLIR：
  学习 ODS、Pass、Pattern、DialectConversion、TilingInterface 和测试方法

Triton：
  完成真实 GPU CodeGen、layout/mapping、benchmark 和性能优化项目
```

不要求在 upstream MLIR 中从零补齐一条达到 Triton 性能的完整 GPU pipeline。

---

## 4. 进度跟踪

| 阶段 | 当前状态 | 通过标准 |
|---|---|---|
| Phase 1：CUDA/Triton Kernel 基础 | 未开始 | 独立实现并解释 tiled matmul 和 fused kernel |
| Phase 2：MLIR Transformation 桥接 | 基础练习已完成，阶段未通过 | 参数化 matmul tiling + 完整测试 |
| Phase 3：Triton Compiler 内部 | 未开始 | 修改并测试一个真实 compiler pass |
| Phase 4：冻结 Baseline | 未开始 | 六组正确性 + 稳定 RTX 4090 数据 |
| Phase 5：性能优化实验 | 未开始 | hypothesis/change/measurement/ablation 闭环 |
| Phase 6：泛化与回归 | 未开始 | 几何平均、最坏退化和适用范围 |
| Phase 7：工程化交付 | 未开始 | 可复现代码、测试和性能报告 |

状态只能依据阶段门槛更新，不能因为“读完资料”而标记完成。

---

## 5. Capstone Workload 规范

### 5.1 算子语义

```text
A:    tensor<MxKxf16>
B:    tensor<KxNxf16>
bias: tensor<Nxf16>

acc = matmul(A, B)       // FP32 accumulation
Y   = relu(acc + bias)   // bias broadcast over M
out = cast<f16>(Y)
```

必须将 Matmul、Bias 和 ReLU 放在同一个 GPU kernel 中。不能用自己的 fused kernel 对比由三个独立 kernel 构成的 baseline。

### 5.2 第一版数据类型和布局

```text
A/B/bias：FP16
accumulator：FP32
output：FP16

A：row-major [M, K]
B：row-major [K, N]
bias：contiguous [N]
output：row-major [M, N]
shape：静态
```

第一版不加入 BF16、FP8、INT8、非连续输入、转置 B 和 dynamic shape。

### 5.3 标准 shape 集

| 场景 | M | N | K | 主要压力 |
|---|---:|---:|---:|---|
| 小批量 | 32 | 4096 | 4096 | latency、CTA 并行度 |
| 中小批量 | 128 | 4096 | 4096 | 常见 batch/token 聚合 |
| 中等 GEMM | 512 | 4096 | 4096 | Tensor Core 利用率 |
| MLP 扩展 | 512 | 16384 | 4096 | 大 N、权重流量 |
| MLP 收缩 | 512 | 4096 | 16384 | 大 K、reduction 压力 |
| 大型方阵 | 4096 | 4096 | 4096 | 峰值吞吐能力 |

`M=1` 更接近 GEMV/推理解码问题，需要不同的并行策略，放到第一版完成后的扩展阶段。

### 5.4 正确性标准

参考结果：

```python
ref = torch.relu(
    torch.matmul(a.float(), b.float()) + bias.float()
).half()

torch.testing.assert_close(
    actual,
    ref,
    rtol=1e-2,
    atol=1e-2,
)
```

还必须满足：

```text
1. 无 NaN/Inf
2. 六组标准 shape 全部通过
3. 至少十组固定随机种子输入
4. 覆盖正值、负值和 ReLU 零边界附近输入
5. 覆盖不能被 BLOCK_M/N/K 整除的边界 case
```

### 5.5 Triton baseline 公平性

baseline 必须：

```text
1. 使用同一个 fused Matmul + Bias + ReLU kernel
2. 对 M/N/K 开启 autotune
3. 搜索合理的 BLOCK_M/BLOCK_N/BLOCK_K
4. 搜索合理的 num_warps 和 num_stages
5. 使用合理的 program ordering / GROUP_M
6. 使用相同 dtype、layout、精度和 fusion 边界
7. 固定 Triton commit、CUDA、driver 和 benchmark harness
```

不能通过减少计算、降低精度、关闭 Triton autotune 或挑选一次偶然结果来获得优势。

### 5.6 Benchmark 方法

```text
预热：100 次
正式测量：500 次
完整测量轮数：至少 3 轮
延迟：median、P20、P80
吞吐：effective TFLOPS
steady-state latency：不包含编译和 autotune
编译时间与 autotune 时间：单独报告
```

每次记录：

```text
GPU 型号
GPU power limit
实际 GPU/memory clock
温度与 throttling 状态
driver/CUDA 版本
Triton commit
PyTorch 版本
输入 shape/dtype/layout
随机种子
benchmark 命令
```

### 5.7 Profiler 指标

```text
kernel duration
achieved occupancy
active warps
registers per thread
shared memory per CTA
DRAM throughput
L2 hit rate
global load/store efficiency
shared memory bank conflict
warp stall reasons
Tensor Core / MMA utilization
kernel launch count
```

---

## 6. 学习节奏

每周最低投入：

```text
工作日：5 天 × 1 小时 = 5 小时
周末：2 天 × 3 小时 = 6 小时
每周：约 11 小时
24 周：约 264 小时
```

建议按以下比例分配：

```text
40% 编码与调试
25% 阅读源码和文档
20% benchmark/profiling
15% 测试、总结和复盘
```

每周至少产生一个可检查 artifact：

```text
.cu/.py/.mlir case
FileCheck test
IR/PTX diff
benchmark CSV
ncu report
实验记录
源码调用链笔记
```

---

## 7. Phase 1：CUDA 与 Triton Kernel 基础

周期：第 1-4 周，约 44 小时

### 7.1 目标

先成为基础 GPU kernel 开发者，再进入 GPU kernel compiler。能够解释 kernel 的线程映射、访存、同步和资源使用，并在 RTX 4090 上完成第一次可复现测量。

RTX 4090 从第 2 周开始介入，不把真实 GPU 验证推迟到课程后半段。

第一次租用前必须确认实例允许运行 `ncu` 并访问所需 GPU performance counters。若服务商限制 profiler 权限，应更换实例或供应商，不能用“kernel 能运行”代替性能分析环境验收。

### 7.2 CUDA 必学内容

```text
grid / block / thread
warp execution
global/shared/register memory
coalesced access
shared memory bank conflict
__syncthreads
occupancy
register pressure
CUDA event timing
Tensor Core/MMA 基本概念
```

### 7.3 Triton 必学内容

```text
@triton.jit
tl.program_id
tl.arange
tl.load / tl.store
mask
tl.dot
blocked programming model
BLOCK_M/N/K
GROUP_M
num_warps
num_stages
triton.autotune
triton.testing.do_bench
```

### 7.4 参考资料

优先使用官方文档和随固定 Triton commit 提供的教程：

```text
NVIDIA CUDA C++ Programming Guide
NVIDIA CUDA C++ Best Practices Guide
NVIDIA Nsight Compute Documentation
NVIDIA Nsight Systems Documentation
Triton Programming Guide
Triton tutorials/01-vector-add.py
Triton tutorials/02-fused-softmax.py
Triton tutorials/03-matrix-multiplication.py
```

### 7.5 实践任务

```text
CUDA：
  1. vector add
  2. reduction
  3. naive matmul
  4. coalesced matmul
  5. shared-memory tiled matmul

Triton：
  1. vector add
  2. fused softmax
  3. matmul tutorial
  4. Linear + Bias + ReLU
  5. 修改 BLOCK_M/N/K、num_warps、num_stages 并测量
```

### 7.6 阶段门槛

进入下一阶段前必须能够回答：

```text
1. 一个 CUDA block 和一个 Triton program instance 分别是什么？
2. 为什么相邻线程访问连续地址通常更快？
3. shared-memory tiling 减少了哪些 global memory traffic？
4. tile 增大为什么可能同时提高 reuse 和 register pressure？
5. occupancy 为什么不是越高越好？
6. benchmark 为什么需要 warmup 和重复测量？
```

阶段产出：

```text
cuda-kernel-notes.md
cuda/vector_add.cu
cuda/tiled_matmul.cu
triton/vector_add.py
triton/fused_linear.py
first-rtx4090-benchmark.csv
```

---

## 8. Phase 2：MLIR Transformation 桥接

周期：第 5-7 周，约 33 小时

### 8.1 当前起点

已有练习：

```text
LearningSteps/rewrite-pattern-addi-zero/
```

该练习已经覆盖：

```text
OpRewritePattern
PatternRewriter::replaceOp
func.func pass anchor
pass registration
FileCheck 正例和负例
```

仍需补足：

```text
folding / canonicalization / pattern 的职责边界
多 op producer-consumer 匹配
参数化 transformation
TilingInterface
边界 tile
rewrite legality
分析与 preservation
```

### 8.2 必学内容

```text
Operation / Region / Block / Value / Use
SSA 和 dominance 基础
PatternBenefit
GreedyRewriteConfig
fold() 与 canonicalization patterns
DialectConversion
ConversionTarget / TypeConverter
DestinationStyleOpInterface
TilingInterface
ReifyRankedShapedTypeOpInterface
scf.for / scf.forall tiling
Transform Dialect 基础
```

### 8.3 参考文档

```text
docs/PatternRewriter.md
docs/Canonicalization.md
docs/DialectConversion.md
docs/Dialects/Linalg/_index.md
docs/Dialects/Transform.md
docs/Tutorials/transform/
include/mlir/Interfaces/TilingInterface.h
lib/Dialect/SCF/Transforms/TileUsingInterface.cpp
```

### 8.4 实践任务

```text
1. 给 AddZeroPattern 关闭 driver folding，隔离验证自定义 pattern
2. 写一个包含 producer-consumer 约束的多 op rewrite
3. 写参数化 linalg.matmul tiling transformation
4. 生成 scf.for 或 scf.forall loop nest
5. 覆盖整除和非整除 tile
6. 对错误 tile size 提供明确诊断
7. 写 FileCheck 正例、负例和边界测试
```

### 8.5 阶段门槛

```text
1. 能解释 rewriter 和 greedy driver 的职责差异
2. 能解释为什么 AddIOp 自带 fold 会抢先于自定义 pattern
3. 能解释 FuncOp pass 与 ModuleOp pass 的 anchor 和作用域差异
4. 能从空文件写出一个参数化 transformation
5. 能说明 transformation 的前置条件和失败方式
6. 测试能够证明目标 transformation 被执行，而非其他 fold 产生相同结果
```

阶段产出：

```text
linalg-matmul-tiling pass
完整 FileCheck 测试
tiling 前后 IR 对照
transformation legality 说明
```

---

## 9. Phase 3：Triton Compiler 内部

周期：第 8-10 周，约 33 小时

### 9.1 目标

从 Triton 使用者进入 Triton compiler 开发者角色，能够定位一个 DSL construct 在各层 IR 中的表示，并能修改、构建和测试一个真实 compiler pass。

### 9.2 环境要求

```text
1. clone Triton 源码
2. 固定并记录 commit
3. 使用 Triton 要求的 LLVM revision，不直接假设当前 upstream LLVM ABI 兼容
4. 从源码完成 editable build
5. 保存 compile_commands.json
6. 记录构建和测试命令
```

### 9.3 编译链路

```text
Python AST / Triton DSL
  -> Triton Dialect / TTIR
  -> TritonGPU Dialect / TTGIR
  -> LLVM Dialect
  -> LLVM IR
  -> PTX
  -> CUBIN
```

### 9.4 必学内容

```text
Triton Dialect operations
TritonGPU layout encodings
blocked / slice / dot operand layouts
CTA / warp / thread mapping
layout conversion
coalescing
matmul acceleration
software pipelining
shared-memory allocation
TritonGPU to LLVM lowering
NVIDIA backend stage construction
```

### 9.5 推荐源码顺序

```text
python/triton/compiler/compiler.py
python/triton/language/core.py
include/triton/Dialect/Triton/IR/
include/triton/Dialect/TritonGPU/IR/
lib/Dialect/Triton/Transforms/
lib/Dialect/TritonGPU/Transforms/
third_party/nvidia/backend/compiler.py
third_party/nvidia/lib/
lib/Conversion/
```

具体目录以固定 commit 为准；遇到路径变化时从 backend stage registration 和 pass name 反向定位，不照抄过期路径。

### 9.6 调试工具

```text
MLIR_ENABLE_DUMP
MLIR_DUMP_PATH
TRITON_REPRODUCER_PATH
TRITON_KERNEL_DUMP
TRITON_KERNEL_OVERRIDE
LLVM_IR_ENABLE_DUMP
MLIR_ENABLE_TIMING
TRITON_INTERPRET
triton-opt
lit/FileCheck
```

### 9.7 实践任务

```text
1. dump vector add 的 TTIR/TTGIR/LLVM IR/PTX
2. dump matmul 的 TTIR/TTGIR/LLVM IR/PTX
3. 对照 BLOCK_M/N/K 在 TTGIR 中的布局语义
4. 对照 num_warps 对 warp mapping 的影响
5. 选择一个简单 pass，修改行为并写 compiler test
6. 用 reproducer 独立运行该 pass
7. 恢复原行为，证明测试能够捕获差异
```

### 9.8 阶段门槛

```text
1. 能指出 tiling、layout、thread mapping 分别在哪一层表达
2. 能解释 TTIR 与 TTGIR 的职责差异
3. 能从 pass pipeline 定位具体 C++ implementation
4. 能编写并运行一个 Triton compiler regression test
5. 能将一个性能假设对应到候选 transformation
```

阶段产出：

```text
triton-compiler-pipeline.md
TTIR/TTGIR/LLVM/PTX 对照样例
一个带测试的 Triton compiler 小改动
```

---

## 10. Phase 4：冻结 Baseline

周期：第 11-14 周，约 44 小时

### 10.1 目标

在进行任何 compiler 优化前，先冻结 workload、正确性、autotune、环境和测量方法，避免通过移动基线制造收益。

### 10.2 实践任务

```text
1. 完成 fused Linear + Bias + ReLU Triton baseline
2. 实现相同语义的实验 compiler 路径
3. 跑通六组标准 shape
4. 建立 FP32 reference correctness
5. 固定 Triton commit 和 autotune configs
6. 固定 benchmark harness
7. 采集 baseline latency 和 ncu report
8. 保存 TTIR/TTGIR/LLVM/PTX/SASS
9. 建立结果 CSV schema
```

### 10.3 Baseline 报告必须回答

```text
1. 每组 shape 的计算量和主要性能区间是什么？
2. 哪些 shape 是 latency/parallelism 问题？
3. 哪些 shape 更接近 compute-bound？
4. Triton 为每组 shape 选择了什么配置？
5. 各配置的 register/shared-memory/occupancy 是多少？
6. 编译时间和 steady-state latency 分别是多少？
7. 测量噪声范围是多少？
```

### 10.4 阶段门槛

```text
1. baseline 在三轮完整测量中的 median 波动处于可解释范围
2. 六组 shape correctness 全部通过
3. baseline 不能在后续实验中随意修改
4. 如必须修改 baseline，要重新跑全部历史实验
```

阶段产出：

```text
benchmark.py
correctness.py
baseline-configs.py
baseline-results.csv
baseline-ncu/
baseline-report.md
```

---

## 11. Phase 5：性能优化实验

周期：第 15-19 周，约 55 小时

### 11.1 分析闭环

每个实验必须使用同一模板：

```text
Observation：观察到了什么指标或 IR 现象？
Hypothesis：为什么这可能是瓶颈？
Change：修改了哪个 transformation 或调度决策？
Expected：预计哪些 IR 和硬件指标发生变化？
Correctness：如何验证语义不变？
Measurement：延迟和 profiler 指标如何变化？
Ablation：关闭单项修改后结果是否恢复？
Conclusion：哪些 shape 获益或退化，为什么？
```

### 11.2 候选方向

只选一个主问题深入，不并行实现全部方向：

```text
小 M 下 CTA 并行度不足
BLOCK_M/N/K 选择不匹配 shape
program ordering 与 L2 reuse
layout conversion 开销
shared-memory traffic
bank conflict
register pressure 与 occupancy
software pipeline stage 数
边界 tile masking 开销
epilogue fusion 的寄存器生命周期
```

选择依据必须来自 baseline profiler，而不是预先猜测。

### 11.3 实现要求

```text
1. 修改必须位于 compiler/transformation 层，而不只是手调一个 kernel 常量
2. transformation 必须有 correctness test
3. transformation 必须能够开关，支持 ablation
4. 对生成 TTGIR/LLVM/PTX 做 before/after diff
5. 记录 compile-time 和 code-size 影响
6. 不隐藏退化 shape
```

### 11.4 阶段门槛

```text
1. 至少完成三次完整实验，其中允许两次失败
2. 每个实验都具有可证伪假设
3. 至少一个实验产生符合预期方向的硬件指标变化
4. 能解释性能收益不是测量噪声
5. 能解释失败实验为什么失败
```

阶段产出：

```text
compiler patch
compiler regression tests
experiments/NN-<name>.md
IR/PTX/SASS diffs
ncu before/after
ablation results
```

---

## 12. Phase 6：泛化与回归

周期：第 20-22 周，约 33 小时

### 12.1 目标

证明优化针对的是一个可描述的 shape 区间，而不是一个碰巧胜出的测试点。

### 12.2 实践任务

```text
1. 重跑六组冻结 shape
2. 增加邻近 shape，检查局部泛化
3. 检查非整除边界 shape
4. 重跑十组随机 correctness
5. 建立性能回归阈值
6. 统计几何平均和 worst regression
7. 分析获益区间和退化区间
8. 检查 compiler-time、cache 和代码体积变化
```

### 12.3 结果报告

必须同时报告：

```text
best improvement
geometric mean
worst regression
measurement variance
compile-time delta
适用 shape 条件
不适用条件
硬件和软件版本
```

阶段产出：

```text
regression-suite/
generalization-results.csv
optimization-scope.md
```

---

## 13. Phase 7：工程化交付

周期：第 23-24 周，约 22 小时

### 13.1 最终项目结构

```text
gpu-codegen-capstone/
  README.md
  environment.md
  docs/
    compiler_pipeline.md
    baseline_report.md
    optimization_design.md
    tuning_report.md
    failed_experiments.md
  kernels/
    cuda/
    triton/
  compiler/
    patches/
    tests/
    reproducers/
  benchmark/
    correctness.py
    benchmark.py
    configs.py
    results/
  profiling/
    ncu/
    nsys/
  ir/
    ttir/
    ttgir/
    llvm/
    ptx/
    sass/
```

### 13.2 README 必须回答

```text
1. workload 的精确定义是什么？
2. baseline 如何实现和 autotune？
3. compiler pipeline 每层 IR 表达什么？
4. baseline 瓶颈证据是什么？
5. 修改了哪个 transformation，为什么？
6. IR/PTX/SASS 发生了什么变化？
7. ncu 指标是否符合假设？
8. 六组 shape 的性能如何？
9. 哪些情况退化，为什么？
10. 如何复现构建、correctness、benchmark 和 profiling？
```

### 13.3 最终展示

```text
1. 一个可运行的 fused Linear + Bias + ReLU
2. 一条 TTIR -> TTGIR -> LLVM/PTX 的完整解释
3. 一个真实 Triton compiler transformation 修改
4. 完整 compiler test 和 correctness test
5. RTX 4090 baseline 与优化后数据
6. profiler 驱动的原因分析
7. 至少一个失败实验及复盘
8. 明确的优化适用范围
```

---

## 14. 24 周总表

| 周次 | 阶段 | 预算 | 主要门槛 |
|---|---|---:|---|
| 1-4 | CUDA/Triton Kernel 基础 | 44h | 独立实现并解释 tiled matmul 和 fused kernel |
| 5-7 | MLIR Transformation 桥接 | 33h | 参数化 matmul tiling + 完整测试 |
| 8-10 | Triton Compiler 内部 | 33h | 修改并测试一个真实 compiler pass |
| 11-14 | 冻结 Baseline | 44h | 六组正确性 + 稳定 RTX 4090 数据 |
| 15-19 | 性能优化实验 | 55h | hypothesis/change/measurement/ablation 闭环 |
| 20-22 | 泛化与回归 | 33h | 几何平均、最坏退化和适用范围 |
| 23-24 | 工程化交付 | 22h | 可复现代码、测试和性能报告 |
| 合计 |  | 264h |  |

### 14.1 每周复盘模板

```text
本周目标：
完成 artifact：
实际投入小时：
遇到的 correctness 问题：
遇到的 performance 问题：
新增源码调用链：
一个被证实的假设：
一个被否定的假设：
下周唯一主目标：
```

### 14.2 Stop/Go 规则

```text
1. 阶段门槛未通过，不以“看完文档”视为完成
2. correctness 未通过，不进入性能比较
3. baseline 未冻结，不开始宣称性能收益
4. 没有 profiler 证据，不修改 compiler transformation
5. 没有 ablation，不归因于单项优化
6. 单点收益但完整测试集严重退化，不视为完成
```

---

## 15. 常用工具

### 15.1 MLIR/Triton Compiler

| 工具/配置 | 用途 |
|---|---|
| `mlir-opt` | 运行和组合 upstream MLIR pass |
| `triton-opt` | 独立运行 Triton MLIR pass/reproducer |
| `FileCheck` | compiler transformation 测试 |
| `llvm-lit` | 执行 compiler regression tests |
| `MLIR_ENABLE_DUMP` | dump Triton 各 MLIR pass 前后 IR |
| `TRITON_REPRODUCER_PATH` | 捕获可独立复现的 MLIR |
| `TRITON_KERNEL_DUMP` | 保存 IR/PTX 等 kernel artifact |
| `LLVM_IR_ENABLE_DUMP` | dump LLVM IR |
| `MLIR_ENABLE_TIMING` | 分析 pass 编译时间 |
| `TRITON_INTERPRET` | 无 GPU 时调试部分 Triton 语义 |
| `lldb/gdb` | 调试 pass 和 lowering |

### 15.2 GPU 分析

| 工具 | 用途 |
|---|---|
| `ncu` | kernel 指标、occupancy、stall、memory、Tensor Core |
| `nsys` | launch timeline、同步和 host/device 行为 |
| `compute-sanitizer` | 内存越界、race 和同步问题 |
| `nvdisasm/cuobjdump` | 检查 SASS 和资源信息 |
| `nvidia-smi` | 记录设备、功耗、时钟和温度 |
| Triton benchmark utilities | warmup、重复测量和 quantile |

---

## 16. 年底后的扩展路线

完成本计划以后，再根据目标岗位选择一条扩展路线。

### 16.1 更深 GPU CodeGen

```text
persistent/stream-K scheduling
split-K reduction
dynamic shape dispatch
layout algebra
async copy/software pipeline
更多 fused epilogue
attention/softmax/layernorm
FP8/INT8
autotuning cost model
```

### 16.2 图编译与端到端系统

```text
PyTorch/FX/Inductor
StableHLO/TOSA
shape inference
graph fusion
IREE Flow/Stream/HAL
runtime dispatch
memory planning
```

### 16.3 LLVM 后端

```text
LLVM IR/MIR
SelectionDAG/GlobalISel
TableGen
instruction selection
register allocation
instruction scheduling
machine-level performance analysis
```

### 16.4 多硬件后端

```text
AMD ROCm/CDNA
Intel GPU
NPU/ASIC
CPU SIMD
vendor-specific matrix instructions
```

---

## 17. 最重要的学习原则

1. 不以阅读量衡量进度，以可运行 artifact 和阶段门槛衡量进度。
2. 不只看最终 IR，要保存每个关键 transformation 的 before/after。
3. 不在 correctness 通过以前讨论性能。
4. 不用自己刻意写慢的 baseline 证明优化有效。
5. 不只看 latency，要用 profiler 验证硬件行为是否符合假设。
6. 不隐藏失败实验和退化 shape。
7. 不为了练习 ODS 强行创造新 IR；只有现有 IR 无法保留必要 CodeGen 信息时才设计新抽象。
8. 不从零重建已有 GPU runtime；把有限时间投入 transformation 和性能分析。
9. 每次只改变一个主要变量，并进行 ablation。
10. 优化结论必须包含适用范围、代价和复现环境。

最终心智模型：

```text
中高级 GPU Kernel/CodeGen 工程师能力 =
  GPU kernel programming
  MLIR transformation engineering
  layout/tiling/mapping understanding
  compiler pipeline debugging
  hardware-aware performance analysis
  hypothesis-driven optimization
  correctness/regression discipline
  production-quality technical communication
```
