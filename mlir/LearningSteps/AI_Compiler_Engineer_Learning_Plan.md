# AI Compiler Engineer Learning Plan

## 目标

本计划面向希望参与 AI 编译器工程开发的学习者。目标不是只会跑 Toy tutorial，而是能够流畅参与真实 AI compiler pipeline 的开发、调试、分析和优化。

完成本计划后，应该具备以下能力：

```text
1. 能读懂 tensor/linalg/memref/scf/vector/gpu/llvm 多层 IR
2. 能编写 MLIR pass、rewrite pattern、conversion pattern
3. 能解释并实现 fusion、tiling、bufferization、vectorization、lowering
4. 能使用调试和分析工具定位性能瓶颈
5. 能理解专家 kernel、library call、compiler-generated kernel 的取舍
6. 能完成一个小型 AI kernel compiler 项目证明能力
```

核心主线：

```text
high-level tensor computation
  -> linalg structured ops
  -> fusion / tiling
  -> bufferization / memory planning
  -> vector / gpu mapping
  -> llvm lowering / runtime call
  -> profiling / tuning
```

---

## 常用分析与调优工具

### MLIR/LLVM 内部工具

| 工具/参数 | 用途 |
|----------|------|
| `mlir-opt` | 跑 pass pipeline，观察 IR lowering |
| `mlir-translate` | MLIR 与 LLVM IR 等格式互转 |
| `mlir-cpu-runner` / `mlir-runner` | 运行 lowered MLIR |
| `FileCheck` | 写 pass 测试 |
| `-mlir-print-ir-before-all` | 打印每个 pass 前 IR |
| `-mlir-print-ir-after-all` | 打印每个 pass 后 IR |
| `-mlir-disable-threading` | 调试时稳定输出和断点行为 |
| `-pass-pipeline=...` | 精确控制 pass 嵌套 |
| `-debug-only=...` | Debug build 下打开特定 debug 日志 |
| `lldb` | macOS 上调试 pass、pattern、lowering |
| `llvm-lit` | 跑 MLIR 测试 |
| `llvm-profdata` / `llvm-cov` | 编译器自身代码覆盖率分析 |

常用调试命令：

```bash
mlir-opt input.mlir \
  -pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
  -mlir-print-ir-before-all \
  -mlir-print-ir-after-all \
  -mlir-disable-threading
```

### CPU 性能分析

| 工具 | 用途 |
|------|------|
| `perf` | Linux CPU profiling，查看热点、cache miss、branch miss |
| `Instruments` | macOS 性能分析 |
| `time` / `hyperfine` | 粗粒度 benchmark |
| `valgrind/cachegrind` | cache 行为分析，Linux 常用 |
| `llvm-mca` | 分析 LLVM 生成汇编的吞吐、延迟、瓶颈 |

### GPU 性能分析

| 平台 | 常用工具 |
|------|----------|
| NVIDIA | Nsight Systems, Nsight Compute, `nvprof` legacy, `ncu`, `nsys` |
| AMD | rocprof, Omniperf, rocTracer |
| Intel GPU | Intel VTune, Level Zero tracing, vendor GPU tools |

GPU 调优重点指标：

```text
kernel launch count
kernel duration
occupancy
register usage
shared memory usage
global memory throughput
L2 hit rate
warp stall reason
memory coalescing
bank conflict
tensor core utilization
```

### AI Compiler 常见分析维度

```text
1. IR 层面：
   op 数量是否减少？
   fusion 是否发生？
   tensor 是否被不必要地 materialize？
   bufferization 是否产生额外 copy？

2. Loop 层面：
   loop nest 顺序是否合理？
   tile size 是否匹配 cache/shared memory？
   reduction 是否被正确分块？

3. Memory 层面：
   alloc/dealloc 是否过多？
   中间 tensor 是否落到 global memory？
   buffer 是否能原地复用？

4. Vector/GPU 层面：
   是否生成 vector.contract？
   是否映射到 warp/block？
   是否使用 mma/tensor core？
   访存是否 coalesced？

5. Runtime 层面：
   kernel launch 是否过多？
   library call 是否比 generated kernel 更合适？
   dynamic shape 是否导致重复编译或 fallback？
```

---

## Phase 0: MLIR 工程基础

建议时间：1 周

### 目标

能够编译、运行、调试 MLIR pass，并理解 `.td`、C++、CMake、测试之间的关系。

### 必学内容

```text
mlir-opt 基本使用
pass pipeline
ODS / TableGen
PatternRewriter
DialectConversion 基础
FileCheck 测试
lldb 调试 pass
```

### 参考文档

```text
docs/Tutorials/MlirOpt.md
docs/PassManagement.md
docs/PatternRewriter.md
docs/DialectConversion.md
docs/DefiningDialects/Operations.md
docs/DefiningDialects/_index.md
docs/Tutorials/QuickstartRewrites.md
```

### 参考代码

```text
examples/toy/Ch2/include/toy/Ops.td
examples/toy/Ch2/mlir/Dialect.cpp
examples/toy/Ch3/mlir/ToyCombine.td
examples/toy/Ch3/mlir/ToyCombine.cpp
include/mlir/Dialect/Arith/IR/ArithOps.td
lib/Dialect/Arith/IR/ArithOps.cpp
```

### 实践任务

```text
1. 用 mlir-opt 跑 canonicalize、cse
2. 用 lldb 断到某个 pass 的 runOnOperation
3. 写一个简单 rewrite pattern
4. 写一个 FileCheck 测试
```

### Quiz

1. `.td` 文件和 `.mlir` 文件的区别是什么？
td文件是mlir方言系统的生成脚本，可以通过td生成Operation的一些实现，mlir文件可以视作模型的表示。
2. `PatternRewriter::replaceOp` 和 `eraseOp` 分别适合什么场景？
3. `DialectConversion` 中 `ConversionTarget` 的作用是什么？
4. 为什么调试 pass 时常加 `-mlir-disable-threading`？
5. `mlir-opt -pass-pipeline='builtin.module(func.func(canonicalize))'` 中 pass 为什么嵌套在 `func.func` 下？

### 阶段产出

```text
一个小 pass 或 rewrite：
arith.addf(x, 0.0) -> x
或 toy.reshape(toy.reshape(x)) -> toy.reshape(x)
```

---

## Phase 1: AI Compiler IR 分层

建议时间：1-2 周

### 目标

看到一个混合 dialect 的 `.mlir` 文件，能判断每个 op 所在抽象层级，以及它在 AI compiler pipeline 中的角色。

### 必学 Dialect

```text
builtin / func / arith
tensor
linalg
memref
scf / affine
vector
gpu / nvgpu
llvm
```

### 核心心智模型

```text
tensor:
  高层值语义，适合图优化、融合、形状推导

linalg:
  结构化 tensor 计算，适合表达 matmul/conv/reduction/elementwise

memref:
  显式 buffer 和内存布局

scf / affine:
  结构化循环和可分析 loop nest

vector:
  SIMD / register tile / contraction

gpu / nvgpu:
  block/thread/warp/shared memory/tensor core 相关表示

llvm:
  接近后端代码生成的目标 dialect
```

### 参考文档

```text
docs/Dialects/Builtin.md
docs/Dialects/Func.md
docs/Dialects/TensorOps.md
docs/Dialects/MemRef.md
docs/Dialects/Linalg/_index.md
docs/Dialects/Affine.md
docs/Dialects/Vector.md
docs/Dialects/GPU.md
docs/Dialects/LLVM.md
docs/TargetLLVMIR.md
```

如果本地没有 `TensorOps.md`，以源码为准：

```text
include/mlir/Dialect/Tensor/IR/TensorOps.td
```

### 参考代码

```text
include/mlir/Dialect/Linalg/IR/LinalgOps.td
include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td
include/mlir/Dialect/Vector/IR/VectorOps.td
include/mlir/Dialect/GPU/IR/GPUOps.td
include/mlir/Dialect/MemRef/IR/MemRefOps.td
```

### 实践任务

```text
1. 手写一个 tensor + linalg.generic 的 elementwise add
2. 用 bufferization 把 tensor 转成 memref
3. 用 convert-linalg-to-loops 把 linalg 降到 scf
4. 用 finalize-memref-to-llvm 降到 llvm dialect
```

### Quiz

1. `tensor` 和 `memref` 的核心区别是什么？
2. 为什么 AI compiler 通常先在 tensor/linalg 层做 fusion？
3. `scf.for` 和 `affine.for` 的区别是什么？
4. `vector.contract` 通常对应哪类硬件行为？
5. 为什么 LLVM dialect 不是一开始就适合做 AI 图优化？

### 阶段产出

```text
一张 lowering map：
tensor/linalg -> memref/scf -> vector/gpu -> llvm
```

---

## Phase 2: Linalg 作为 AI 算子表达

建议时间：2 周

### 目标

能够用 `linalg` 表达常见 AI 计算，并理解 indexing maps、iterator types、destination-style op。

### 必学内容

```text
linalg.generic
linalg.matmul
linalg.fill
linalg.transpose
linalg.conv_2d_nhwc_hwcf
indexing_maps
iterator_types
ins / outs
parallel / reduction iterator
destination-style op
```

### 参考文档

```text
docs/Dialects/Linalg/_index.md
docs/Dialects/Linalg/OpDSL.md
docs/Rationale/RationaleLinalgDialect.md
docs/Bufferization.md
```

### 参考代码

```text
include/mlir/Dialect/Linalg/IR/LinalgOps.td
include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td
lib/Dialect/Linalg/IR/LinalgOps.cpp
test/Dialect/Linalg/loops.mlir
test/Dialect/Linalg/vectorization/linalg-ops.mlir
test/Dialect/Linalg/roundtrip-linalg-named-ops.mlir
```

### 实践任务

```text
1. 手写 linalg.matmul
2. 手写 linalg.generic 版 elementwise add
3. 手写 matmul + bias + relu
4. 手写 reduction sum
5. 对每个 case 跑 convert-linalg-to-loops
```

### Quiz

1. `linalg.generic` 的 `indexing_maps` 表示什么？
2. `iterator_types = ["parallel", "reduction"]` 分别意味着什么？
3. 为什么 `linalg` 使用 `ins` / `outs` 风格？
4. `linalg.matmul` 和手写三层循环相比有什么编译器优势？
5. 什么是 destination-style op？

### 阶段产出

```text
LearningSteps 或 examples 中保存一组可运行 .mlir：
matmul.mlir
matmul_bias_relu.mlir
reduction.mlir
```

---

## Phase 3: Fusion / Tiling / Bufferization

建议时间：3 周

### 目标

掌握 AI compiler 的关键优化：减少中间 tensor、提升 locality、控制 buffer 和 copy。

### 必学内容

```text
producer-consumer fusion
elementwise fusion
tile-and-fuse
linalg tiling
scf.for / scf.forall tiling
one-shot-bufferize
in-place analysis
copy insertion
buffer deallocation
promotion
packing / layout
```

### 参考文档

```text
docs/Bufferization.md
docs/Canonicalization.md
docs/Dialects/Transform.md
docs/Tutorials/transform/Ch0.md
docs/Tutorials/transform/Ch1.md
docs/Tutorials/transform/Ch2.md
docs/Tutorials/transform/Ch3.md
```

### 参考代码

```text
lib/Dialect/Linalg/Transforms/Tiling.cpp
lib/Dialect/Linalg/Transforms/Fusion.cpp
lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp
lib/Dialect/Linalg/Transforms/Promotion.cpp
lib/Dialect/Linalg/Transforms/PackAndUnpackPatterns.cpp
lib/Dialect/SCF/Transforms/TileUsingInterface.cpp
lib/Dialect/Bufferization/Transforms/Bufferize.cpp
lib/Dialect/Bufferization/Transforms/OneShotAnalysis.cpp
test/Dialect/Linalg/transform-op-tile.mlir
test/Dialect/Linalg/tile-and-fuse-tensors.mlir
test/Dialect/Linalg/one-shot-bufferize.mlir
```

### 调优与分析重点

```text
1. Fusion 是否减少了 op 数量？
2. 中间 tensor 是否还落地？
3. Tiling 后 loop nest 是否符合数据复用方向？
4. Bufferization 是否插入了额外 copy？
5. alloc/dealloc 数量是否过多？
6. tile size 是否导致过多边界处理？
```

常用命令：

```bash
mlir-opt input.mlir \
  -canonicalize \
  -cse \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-linalg-to-loops \
  -mlir-print-ir-after-all \
  -mlir-disable-threading
```

### 实践任务

```text
matmul + bias + relu:
  1. 保持为多个 linalg op
  2. 做 fusion
  3. 做 tiling
  4. 做 bufferization
  5. 观察 IR 中 alloc/copy/store 数量变化
```

### Quiz

1. 为什么 fusion 通常应该在 tensor/linalg 层做？
2. tiling 的 tile size 会影响哪些性能因素？
3. bufferization 中 in-place 分析想解决什么问题？
4. copy insertion 通常意味着什么性能风险？
5. tile-and-fuse 和先 fuse 再 tile 的区别是什么？

### 阶段产出

```text
一份 IR evolution 文档：
matmul_bias_relu 在 fusion、tiling、bufferization 前后的 IR 对比
```

---

## Phase 4: AI 硬件基础与性能模型

建议时间：2 周

### 目标

建立 AI compiler 工程师必须具备的硬件心智模型：能把 IR transformation、lowering 策略和真实硬件性能后果联系起来。

本阶段不是学习芯片设计，而是回答：

```text
为什么这个 fusion / tiling / layout / lowering 会更快或更慢？
```

### 必学内容

```text
CPU:
  cache hierarchy
  cache line
  SIMD / vector register
  memory bandwidth
  llvm-mca 基础

GPU execution model:
  grid
  block / CTA
  thread
  warp / wavefront
  SM / CU
  occupancy
  SIMT
  warp divergence
  barrier / synchronization

GPU memory hierarchy:
  register
  shared memory / LDS
  L1 / L2
  global memory / HBM
  local memory / spill
  memory coalescing
  shared memory bank conflict

Matrix hardware:
  tensor core
  MMA / WMMA / WGMMA
  MFMA
  DPAS
  mixed precision: f16 / bf16 / tf32 / int8

Performance model:
  FLOPs
  memory bandwidth
  arithmetic intensity
  roofline model
  compute-bound / memory-bound / launch-bound

Data movement:
  row-major / column-major
  NCHW / NHWC
  blocked layout
  packed layout
  alignment
  padding
  swizzle

Reduction:
  tree reduction
  warp-level reduction
  block-level reduction
  atomic
  numerical precision

Runtime:
  kernel launch overhead
  stream / queue
  async copy
  host-device synchronization
  dynamic shape dispatch
```

### 参考资料

硬件资料建议以 vendor 官方文档为主：

```text
NVIDIA:
  CUDA C++ Programming Guide
  CUDA C++ Best Practices Guide
  Nsight Compute Documentation
  Nsight Systems Documentation

AMD:
  ROCm Documentation
  rocprof / Omniperf Documentation
  AMD CDNA/RDNA architecture overview

Intel:
  Intel GPU / Xe architecture docs
  Intel VTune documentation

通用:
  Roofline model papers/tutorials
  LLVM llvm-mca documentation
```

MLIR 源码中可对应查看：

```text
docs/Dialects/GPU.md
docs/Dialects/Vector.md
include/mlir/Dialect/GPU/IR/GPUOps.td
include/mlir/Dialect/NVGPU/IR/NVGPU.td
include/mlir/Dialect/Vector/IR/VectorOps.td
test/Dialect/GPU/
test/Dialect/NVGPU/
test/Conversion/VectorToGPU/
```

### 和 MLIR 优化的对应关系

```text
fusion:
  减少 kernel launch
  减少中间 tensor 写回 global memory

tiling:
  提高 cache/shared memory/register reuse
  控制 parallelism 和 occupancy

vectorization:
  利用 SIMD / tensor core / matrix instruction

bufferization:
  控制 alloc/copy
  避免不必要 materialization

layout / packing:
  改善 memory coalescing
  减少 bank conflict
  匹配 tensor core tile shape

lowering:
  决定最终是否能生成目标硬件友好的指令形态
```

### 实践任务

```text
1. 对 elementwise chain 做 roofline 判断：为什么它通常 memory-bound？
2. 对 matmul 做 arithmetic intensity 估算：为什么它更可能 compute-bound？
3. 对比两个 tile size，分析 register/shared memory/occupancy 的影响。
4. 读一个 GPU profiler report，解释主要瓶颈指标。
5. 画出 GPU memory hierarchy，并标注 MLIR 中哪些 pass 会影响 data movement。
```

如果暂时没有 GPU，可以先做静态分析：

```text
1. 统计 IR 中 load/store 数量
2. 估算 bytes moved
3. 估算 FLOPs
4. 计算 arithmetic intensity
5. 解释 fusion/tiling 理论上应该改善什么指标
```

### Quiz

1. 为什么 elementwise op 通常是 memory-bound？
2. 为什么 matmul 的 K 维 tile size 对性能很关键？
3. occupancy 为什么不是越高越好？
4. register pressure 过高会造成什么后果？
5. shared memory bank conflict 是什么，它和 layout 有什么关系？
6. coalesced memory access 为什么对 GPU 很重要？
7. tensor core 对 dtype 和 tile shape 有哪些约束？
8. fusion 减少的是 compute 还是 memory traffic？为什么？
9. roofline model 如何帮助判断优化方向？
10. kernel launch-bound 通常应该用什么优化缓解？

### 阶段产出

```text
一份 hardware-performance-notes.md：
  - GPU 执行模型图
  - GPU 内存层级图
  - roofline 基础解释
  - fusion/tiling/vectorization/bufferization 与硬件指标的对应关系
  - 一个 matmul 或 elementwise 的 arithmetic intensity 分析
```

---

## Phase 5: Vector / GPU / LLVM Lowering

建议时间：3-4 周

### 目标

理解高层 tensor 计算如何逐步贴近硬件，掌握 vector 和 GPU lowering 的基本路径。

### 必学内容

```text
vector.transfer_read/write
vector.contract
vector.multi_reduction
vector lowering
gpu.launch
gpu.thread_id / block_id
gpu shared memory address space
nvgpu.ldmatrix
nvgpu.mma / wgmma concepts
llvm dialect
```

### 参考文档

```text
docs/Dialects/Vector.md
docs/Dialects/GPU.md
docs/Dialects/NVGPU.md
docs/Dialects/LLVM.md
docs/TargetLLVMIR.md
```

如果本地没有 `NVGPU.md`，以源码和测试为准。

### 参考代码

```text
lib/Dialect/Linalg/Transforms/Vectorization.cpp
lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp
lib/Conversion/VectorToSCF/VectorToSCF.cpp
lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp
lib/Conversion/GPUToROCDL/LowerGpuOpsToROCDLOps.cpp
test/Dialect/Linalg/vectorization/
test/Conversion/VectorToLLVM/
test/Conversion/VectorToGPU/
test/Dialect/GPU/
test/Dialect/NVGPU/
```

### 调优与分析重点

```text
1. 是否生成 vector.contract？
2. vector shape 是否匹配目标硬件寄存器/tensor core？
3. transfer_read/write 是否产生连续访存？
4. GPU block/thread mapping 是否合理？
5. shared memory 是否减少 global memory traffic？
6. register usage 是否过高影响 occupancy？
```

GPU profiler 关注：

```text
Nsight Systems:
  kernel launch timeline
  host/device synchronization
  memory copy

Nsight Compute:
  occupancy
  memory throughput
  warp stall reason
  tensor core utilization
  shared memory bank conflict
```

### 实践任务

```text
1. linalg.matmul -> vector.contract
2. vector.contract -> lower vector
3. 写一个简单 gpu.launch elementwise kernel
4. 对比 CPU LLVM path 和 GPU-oriented IR path
```

### Quiz

1. `vector.contract` 为什么适合表示 matmul 内核？
2. `vector.transfer_read` 和普通 `memref.load` 有什么区别？
3. GPU 上 block、thread、warp 的层级分别承担什么职责？
4. shared memory 通常用来优化什么？
5. 为什么 register usage 过高会降低 occupancy？

### 阶段产出

```text
一个最小 GPU-oriented lowering case：
linalg.matmul -> tiled/vectorized/gpu-ish IR
```

---

## Phase 6: 性能分析与调优方法

建议时间：2-3 周

### 目标

建立 AI compiler 工程师的性能分析闭环：从 IR 变化、运行时间、硬件指标三层判断优化是否有效。

### 必学内容

```text
benchmark 方法
IR diff 方法
pass pipeline ablation
kernel launch 分析
memory traffic 分析
roofline 思维
occupancy 分析
autotuning 搜索空间
regression test
```

### 分析闭环

```text
1. 建立 baseline
2. 添加一个优化
3. dump IR 对比变化
4. benchmark 运行时间
5. profiler 看硬件指标
6. 判断优化是否真的命中瓶颈
7. 固化测试和文档
```

### 常见调优方向

```text
Fusion:
  减少 kernel launch
  减少中间 tensor global memory traffic

Tiling:
  提升 cache/shared memory reuse
  控制并行粒度

Vectorization:
  提高 SIMD/tensor core 利用率

Memory planning:
  减少 alloc/free/copy
  提高 buffer reuse

Layout:
  改善 coalescing、cache locality、bank conflict

Library/custom call:
  对成熟热点算子复用专家 kernel
```

### 参考文档和代码

```text
docs/PassManagement.md
docs/Bufferization.md
docs/Dialects/Transform.md
test/Dialect/Linalg/transform-*.mlir
test/Integration/Dialect/Linalg/CPU/
test/Integration/GPU/
```

### 实践任务

```text
1. 对 matmul_bias_relu 做三个版本：
   baseline
   fused
   fused + tiled

2. 统计：
   IR op 数量
   alloc/copy 数量
   loop nest 结构
   执行时间

3. 写一份性能分析报告：
   为什么某个优化有效或无效？
```

### Quiz

1. 一个优化让 IR 更复杂，但运行更快，可能原因是什么？
2. 一个优化减少了 op 数量，但运行更慢，可能原因是什么？
3. 如何判断瓶颈是 compute-bound 还是 memory-bound？
4. 为什么 benchmark 必须有 warmup 和多次运行？
5. pass pipeline ablation 的意义是什么？

### 阶段产出

```text
一份 tuning report：
包含 baseline、优化版本、IR diff、性能数据、原因分析
```

---

## Phase 7: 专家 Kernel 与编译器结合

建议时间：1-2 周

### 目标

理解真实工业 AI compiler 如何结合自动生成代码与专家手写 kernel。

### 必学内容

```text
library call lowering
custom call
runtime dispatch
expert kernel fallback
shape/dtype/layout based dispatch
autotuning
```

### 常见工程策略

```text
核心大算子:
  cuBLAS / cuDNN / rocBLAS / oneDNN / CUTLASS / custom kernel

大量 elementwise:
  compiler fusion generated kernel

特殊组合:
  custom fused kernel 或 Triton kernel

fallback:
  generic MLIR lowering
```

### 参考文档和代码

```text
docs/DialectConversion.md
docs/TargetLLVMIR.md
lib/Dialect/Linalg/Transforms/Transforms.cpp
lib/Conversion/LinalgToStandard/LinalgToStandard.cpp
test/Dialect/Linalg/library-calls.mlir
test/Conversion/
```

### 实践任务

```text
1. 匹配 linalg.matmul
2. 如果 shape/dtype 满足条件，lower 成 mock runtime call:
   call @expert_matmul(...)
3. 否则保留 generic lowering path
4. 写测试覆盖两条路径
```

### Quiz

1. 为什么不是所有 matmul 都应该直接调用专家库？
2. 过早 lowering 到 library call 会损失哪些优化机会？
3. custom call 需要处理哪些 ABI 和 memref descriptor 问题？
4. autotuning 的搜索空间通常包括哪些参数？
5. compiler-generated kernel 和 expert kernel 如何共存？

### 阶段产出

```text
linalg.matmul -> expert_matmul call 的 conversion pattern
```

---

## Phase 8: Triton 编译器学习

建议时间：2-3 周

### 目标

理解 Triton 作为业界较成功的 AI kernel compiler 实践，如何把 Python DSL、tile-level programming model、MLIR dialect、GPU codegen 和 autotuning 结合起来。

Triton 学习不是为了替代 MLIR 主线，而是为了回答几个工程问题：

```text
1. 一个成功的 AI kernel DSL 如何设计抽象？
2. tile-level programming model 如何对应 GPU 执行？
3. Triton 如何基于 MLIR dialect 表达 kernel？
4. TTIR / TTGIR / LLVM IR / PTX 的 lowering 分层如何组织？
5. autotuning 和专家经验如何进入编译流程？
6. Triton 与 MLIR Linalg/Vector/GPU 路线有什么异同？
```

### 必学内容

```text
Triton Python DSL:
  @triton.jit
  tl.load / tl.store
  tl.arange
  tl.dot
  masks
  program_id

Triton 编译流程:
  Python AST / JIT frontend
  TTIR
  TTGIR
  LLVM IR
  PTX / CUBIN

Triton 优化:
  block-level tiling
  num_warps
  num_stages
  shared memory / register reuse
  dot lowering
  coalescing
  pipelining
  autotune configs

Triton 与 MLIR:
  Triton Dialect
  TritonGPU Dialect
  TritonNvidiaGPU Dialect
  MLIR pass pipeline
  backend-specific lowering
```

### 参考文档

如果已经下载 Triton 源码，优先看本地源码。否则先执行：

```bash
cd /Users/yuermei/Documents
git clone https://github.com/triton-lang/triton.git
```

建议参考：

```text
Triton 官方文档:
  https://triton-lang.org/main/

Triton tutorials:
  tutorials/

Triton compiler code:
  python/triton/compiler/compiler.py
  python/triton/language/
  third_party/nvidia/backend/compiler.py
  third_party/amd/backend/compiler.py

Triton MLIR dialect:
  include/triton/Dialect/Triton/
  include/triton/Dialect/TritonGPU/
  include/triton/Dialect/TritonNvidiaGPU/
  lib/Dialect/Triton/
  lib/Dialect/TritonGPU/
  lib/Dialect/TritonNvidiaGPU/
```

### 推荐阅读源码顺序

```text
1. tutorials/01-vector-add.py
2. tutorials/03-matrix-multiplication.py
3. python/triton/language/core.py
4. python/triton/compiler/compiler.py
5. include/triton/Dialect/Triton/IR/TritonOps.td
6. include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td
7. third_party/nvidia/backend/compiler.py
8. lib/Conversion/
```

### 实践任务

```text
1. 跑通 Triton vector add tutorial
2. 跑通 Triton matmul tutorial
3. 修改 BLOCK_M / BLOCK_N / BLOCK_K，观察性能变化
4. 修改 num_warps / num_stages，观察性能变化
5. dump Triton 中间 IR，观察 TTIR / TTGIR / LLVM IR
6. 对比 Triton matmul 与 MLIR linalg.matmul 的表达差异
```

如果本机没有 NVIDIA/AMD GPU，也可以先做源码和 IR 层面的学习：

```text
1. 阅读 Triton DSL 到 compiler.compile 的调用路径
2. 阅读 TritonOps.td 和 TritonGPUOps.td
3. 画出 TTIR -> TTGIR -> LLVM/TTGIR backend 的 lowering 图
4. 分析 matmul tutorial 中每个 tl.* API 对应的编译器语义
```

### 调优与分析重点

```text
1. BLOCK_M / BLOCK_N / BLOCK_K 如何影响 data reuse？
2. num_warps 如何影响 occupancy 和 parallelism？
3. num_stages 如何影响 software pipelining？
4. tl.dot 如何 lowering 到 tensor core / mma？
5. mask load/store 对边界 tile 有什么代价？
6. autotune 搜索空间如何设计？
```

如果有 NVIDIA GPU，重点使用：

```text
nsys:
  kernel launch timeline
  host-device synchronization

ncu:
  occupancy
  memory throughput
  tensor core utilization
  warp stall reason
  register/shared memory usage
```

### Quiz

1. Triton 的 programming model 为什么是 tile-level，而不是 thread-level？
2. `tl.program_id(0)` 通常对应 GPU 执行层级中的什么概念？
3. `tl.load(ptr + offsets, mask=..., other=...)` 的 mask 在 lowering 中意味着什么？
4. `tl.dot` 和 MLIR `linalg.matmul` / `vector.contract` 分别处在哪个抽象层？
5. TTIR 和 TTGIR 的主要区别是什么？
6. `num_warps` 和 `num_stages` 分别影响什么性能因素？
7. Triton autotune 为什么比单一固定配置更适合真实模型？
8. Triton 和 MLIR Linalg pipeline 各自更擅长什么场景？

### 阶段产出

```text
一份 Triton 编译流程分析文档：
  Python DSL -> TTIR -> TTGIR -> LLVM IR -> PTX/CUBIN

一份 Triton matmul 调优报告：
  BLOCK_SIZE / num_warps / num_stages 对性能或 IR 的影响

一张对比图：
  MLIR Linalg lowering pipeline vs Triton lowering pipeline
```

---

## Phase 9: Capstone Project

建议时间：4-6 周

### 项目名称

```text
Mini AI Kernel Compiler on MLIR + Triton Study
```

### 项目目标

构建一个小型 AI kernel compiler pipeline，输入高层 tensor/linalg IR，输出 LLVM dialect 或 GPU-oriented IR，并提供优化分析报告。同时加入 Triton 作为业界对照样本：比较 MLIR Linalg pipeline 与 Triton tile-level kernel pipeline 的设计差异。

### 必须支持的输入

```text
matmul
matmul + bias
matmul + bias + relu
elementwise chain
simple reduction
```

### 必须实现的能力

```text
1. canonicalize / cse
2. linalg fusion 或等价 rewrite
3. tiling
4. one-shot-bufferize
5. convert-linalg-to-loops
6. lower to LLVM dialect
7. IR dump and analysis
8. FileCheck tests
```

### 加分能力

```text
1. vectorization path
2. gpu-oriented path
3. mock expert kernel call
4. autotuning tile size
5. simple benchmark harness
6. profiling report
7. Triton matmul 对照实现
8. Triton tuning report
```

### 项目结构建议

```text
mini-ai-compiler/
  README.md
  docs/
    lowering_pipeline.md
    tuning_report.md
    design_decisions.md
    triton_pipeline_comparison.md
  test/
    matmul.mlir
    matmul_bias_relu.mlir
    fusion.mlir
    bufferization.mlir
    expert_call.mlir
  tools/
    run_pipeline.sh
    dump_ir_stages.sh
    benchmark.sh
  lib/
    Passes.cpp
    FusionPatterns.cpp
    ExpertCallLowering.cpp
```

### README 必须回答的问题

```text
1. 输入 IR 长什么样？
2. pipeline 每个阶段做什么？
3. fusion/tiling/bufferization 前后 IR 如何变化？
4. memory traffic 为什么减少？
5. tile size 为什么这样选？
6. 哪些情况走 expert kernel？
7. 哪些情况走 compiler-generated lowering？
8. 测试如何运行？
9. 性能或 IR 指标如何变化？
10. Triton 对同一类 kernel 的抽象方式有什么不同？
```

### 最终展示标准

完成后应该能展示：

```text
1. 一个 matmul_bias_relu 从 linalg 到 llvm 的完整 lowering 过程
2. 一份 IR before/after 对比
3. 一份 tuning report
4. 一组 FileCheck 测试
5. 一个专家 kernel/custom call 集成示例
6. 一份 Triton pipeline 对照分析
```

---

## 推荐周计划

```text
Week 1:
  Phase 0: MLIR 工程基础

Week 2-3:
  Phase 1: IR 分层

Week 4-5:
  Phase 2: Linalg 表达 AI 计算

Week 6-8:
  Phase 3: Fusion / Tiling / Bufferization

Week 9-10:
  Phase 4: AI 硬件基础与性能模型

Week 11-14:
  Phase 5: Vector / GPU / LLVM Lowering

Week 15-17:
  Phase 6: 性能分析与调优

Week 18:
  Phase 7: 专家 Kernel 集成

Week 19-21:
  Phase 8: Triton 编译器学习

Week 22-27:
  Phase 9: Capstone Project
```

---

## 最重要的学习原则

1. 不要只读文档，每个概念都要落实到 `.mlir` case。
2. 不要只看最终 IR，要看每个 pass 前后的 IR diff。
3. 不要只实现优化，要解释为什么这个优化可能有效。
4. 不要只关注 compiler IR，要结合 profiler 指标验证假设。
5. 不要把专家 kernel 和 compiler generation 对立起来，真实系统通常是混合路线。

最终心智模型：

```text
AI compiler 工程师的核心能力 =
  IR 分层理解
  transformation 实现能力
  lowering 工程能力
  memory/performance 分析能力
  hardware-aware tuning 能力
  runtime/expert kernel 集成能力
  真实系统源码阅读与对照分析能力
```
