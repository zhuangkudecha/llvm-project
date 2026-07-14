# RTX 4090 GPU Kernel/CodeGen 24-Week Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not mark a week complete until its exit gate passes.

**Goal:** 在 24 周、约 264 小时内，从当前单 op MLIR rewrite 基础进阶到能够修改 Triton GPU CodeGen transformation，并在 RTX 4090 上完成 fused Linear + Bias + ReLU 的 correctness、benchmark、profiling 和优化闭环。

**Architecture:** upstream MLIR 负责训练通用 transformation 工程能力，Triton 负责真实 NVIDIA GPU CodeGen 和性能项目。所有性能实验共享冻结的 workload、Triton baseline、测量脚本和结果格式；compiler 修改必须可开关、可测试、可 ablation。

**Tech Stack:** C++17、CUDA C++、Python、PyTorch、Triton、MLIR、LLVM/NVVM/PTX、FileCheck、llvm-lit、Nsight Compute、Nsight Systems、compute-sanitizer。

**Source plan:** [AI_Compiler_Engineer_Learning_Plan.md](./AI_Compiler_Engineer_Learning_Plan.md)

---

## 1. 使用规则

### 1.1 时间盒

固定每周节奏：

| 时间 | 时长 | 默认活动 |
|---|---:|---|
| 周一 | 1h | 阅读本周核心概念并写问题 |
| 周二 | 1h | 阅读对应源码/IR |
| 周三 | 1h | 完成最小代码增量 |
| 周四 | 1h | 测试、调试和负例 |
| 周五 | 1h | 总结调用链、准备周末实验 |
| 周六 | 3h | 集中实现或 GPU 实验 |
| 周日 | 3h | 验证、记录结果、阶段复盘 |

若某天中断，不把七天内容压缩到一次通宵。优先保证周六实现和周日验证，缺失内容顺延并记录实际投入。

### 1.1.1 日历映射

| Week | 开始 | 结束 | 阶段 |
|---:|---|---|---|
| 1 | 2026-07-15 | 2026-07-21 | CUDA/Triton 基础 |
| 2 | 2026-07-22 | 2026-07-28 | CUDA/Triton 基础 |
| 3 | 2026-07-29 | 2026-08-04 | CUDA/Triton 基础 |
| 4 | 2026-08-05 | 2026-08-11 | CUDA/Triton 基础 |
| 5 | 2026-08-12 | 2026-08-18 | MLIR Transformation |
| 6 | 2026-08-19 | 2026-08-25 | MLIR Transformation |
| 7 | 2026-08-26 | 2026-09-01 | MLIR Transformation |
| 8 | 2026-09-02 | 2026-09-08 | Triton Compiler |
| 9 | 2026-09-09 | 2026-09-15 | Triton Compiler |
| 10 | 2026-09-16 | 2026-09-22 | Triton Compiler |
| 11 | 2026-09-23 | 2026-09-29 | Baseline |
| 12 | 2026-09-30 | 2026-10-06 | Baseline |
| 13 | 2026-10-07 | 2026-10-13 | Baseline |
| 14 | 2026-10-14 | 2026-10-20 | Baseline |
| 15 | 2026-10-21 | 2026-10-27 | 性能优化 |
| 16 | 2026-10-28 | 2026-11-03 | 性能优化 |
| 17 | 2026-11-04 | 2026-11-10 | 性能优化 |
| 18 | 2026-11-11 | 2026-11-17 | 性能优化 |
| 19 | 2026-11-18 | 2026-11-24 | 性能优化 |
| 20 | 2026-11-25 | 2026-12-01 | 泛化与回归 |
| 21 | 2026-12-02 | 2026-12-08 | 泛化与回归 |
| 22 | 2026-12-09 | 2026-12-15 | 泛化与回归 |
| 23 | 2026-12-16 | 2026-12-22 | 工程化交付 |
| 24 | 2026-12-23 | 2026-12-29 | 工程化交付 |

2026-12-30 至 2026-12-31 只用于处理最终复现失败或文档修正，不安排新的学习主题。

### 1.2 完成定义

一项任务只有同时满足以下条件才能勾选：

```text
代码或文档已经落盘
给出的验证命令实际运行过
结果与 Expected 一致
失败或偏差已写入周报
相关路径可以从 README 找到
```

“读完”“看懂了”“大概能解释”不能作为完成证据。

### 1.3 Stop/Go

```text
correctness 未通过：禁止比较性能
GPU 环境未确认 ncu 权限：禁止冻结 baseline
baseline 未冻结：禁止宣称优化收益
compiler change 没有测试：禁止跑正式 benchmark
没有 ablation：禁止归因于某个 transformation
阶段门槛未通过：下一阶段最多预读，不正式开始
```

---

## 2. 固定路径和目录

### 2.1 环境变量

本地 upstream MLIR：

```bash
export LLVM_PROJECT_ROOT=/home/fuhao/llvm-project
export MLIR_SRC=$LLVM_PROJECT_ROOT/mlir
export MLIR_BUILD=$LLVM_PROJECT_ROOT/build
export MLIR_BIN=$MLIR_BUILD/bin
export CAPSTONE=$MLIR_SRC/LearningSteps/GPUCodeGenCapstone
```

Triton clone 不放进 llvm-project，避免把大型第三方仓库变成 MLIR 工作树的未跟踪文件：

```bash
export TRITON_ROOT=$HOME/workspace/triton
export TRITON_VENV=$TRITON_ROOT/.venv
```

租赁服务器上保持同样的 `$TRITON_ROOT` 和 `$CAPSTONE` 逻辑；`$CAPSTONE` 应通过单独 Git 仓库或持久卷同步，不能只存在于临时实例系统盘。

### 2.2 最终目录

第 1 周创建以下结构，之后只按职责增加文件：

```text
LearningSteps/GPUCodeGenCapstone/
  README.md
  progress.md
  environment/
    local.md
    rtx4090.md
  cuda/
    CMakeLists.txt
    vector_add.cu
    reduction.cu
    matmul_naive.cu
    matmul_tiled.cu
  triton_kernels/
    vector_add.py
    fused_softmax.py
    matmul.py
    fused_linear.py
  mlir/
    add_zero/
    transform_matmul_tiling.mlir
    tests/
  compiler/
    baseline_commit.txt
    patches/
    tests/
    reproducers/
  benchmark/
    shapes.py
    correctness.py
    benchmark.py
    baseline_configs.py
  scripts/
    check_environment.sh
    dump_triton_ir.sh
    run_correctness.sh
    run_benchmark.sh
    run_ncu.sh
  results/
    baseline/
    experiments/
    regression/
  profiles/
    ncu/
    nsys/
  ir/
    ttir/
    ttgir/
    llvm/
    ptx/
    sass/
  docs/
    cuda_kernel_notes.md
    triton_compiler_pipeline.md
    baseline_report.md
    optimization_design.md
    tuning_report.md
    failed_experiments.md
```

### 2.3 每周进度格式

`progress.md` 每周追加：

```markdown
## Week N

- Planned hours: 11
- Actual hours:
- Exit gate: PASS / FAIL
- Artifacts:
- Verification commands:
- Correctness problems:
- Performance observations:
- Confirmed hypothesis:
- Rejected hypothesis:
- Carry-over:
- Next single objective:
```

---

## 3. Phase 1：CUDA 与 Triton Kernel 基础

## Week 1：项目初始化与 GPU 执行模型

**Files:**

- Create: `LearningSteps/GPUCodeGenCapstone/README.md`
- Create: `LearningSteps/GPUCodeGenCapstone/progress.md`
- Create: `LearningSteps/GPUCodeGenCapstone/environment/local.md`
- Create: `LearningSteps/GPUCodeGenCapstone/cuda/vector_add.cu`
- Create: `LearningSteps/GPUCodeGenCapstone/cuda/CMakeLists.txt`
- Create: remaining directories from Section 2.2

### 工作日

- [ ] **周一：记录本地环境**

运行：

```bash
uname -a
cmake --version
ninja --version
$MLIR_BIN/mlir-opt --version
$MLIR_BIN/llvm-lit --version
```

将输出和当前 Git commit 写入 `environment/local.md`。

- [ ] **周二：画出 CUDA 执行层级**

在 `docs/cuda_kernel_notes.md` 写出：

```text
kernel -> grid -> block -> warp -> thread
SM -> warp scheduler -> CUDA core/Tensor Core
```

为每个层级写明它负责的并行粒度和可见资源。

- [ ] **周三：编写 CUDA vector add**

实现接口：

```cpp
__global__ void vectorAdd(const float *a, const float *b, float *c, int64_t n);
```

必须包含 `idx < n` 边界判断和 CPU reference。

- [ ] **周四：加入错误检查**

为 `cudaMalloc`、`cudaMemcpy`、kernel launch 和 `cudaDeviceSynchronize` 加统一错误检查宏；添加 `n=1`、`n=1025`、`n=1<<20` 三组 case。

- [ ] **周五：准备租赁环境验收清单**

在 `environment/rtx4090.md` 预先写入：

```text
GPU exact name
VRAM
driver version
CUDA version
power limit
ncu version
nsys version
performance-counter permission
persistent storage path
```

### 周末

- [ ] **周六：建立 CUDA 构建文件和项目 README**

`cuda/CMakeLists.txt` 至少定义 `vector_add` target，并启用 C++17/CUDA17。README 写清当前目标、目录和第一条构建命令。

- [ ] **周日：本地结构验证与复盘**

运行：

```bash
test -f $CAPSTONE/cuda/vector_add.cu
test -f $CAPSTONE/environment/local.md
test -f $CAPSTONE/progress.md
find $CAPSTONE -maxdepth 2 -type d | sort
```

Expected：所有固定目录和 Week 1 文件存在。将验证命令写入 `progress.md`。

### Week 1 Exit Gate

```text
能准确解释 grid/block/warp/thread
vector_add.cu 包含边界处理、CPU reference 和 CUDA error checking
租赁环境验收表已经准备好
```

---

## Week 2：首次 RTX 4090 实验与测量纪律

**Files:**

- Create: `environment/rtx4090.md`
- Create: `scripts/check_environment.sh`
- Update: `cuda/vector_add.cu`
- Create: `results/experiments/week02-vector-add.csv`

### 工作日

- [ ] **周一：租用并验收 RTX 4090 实例**

运行：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
nvcc --version
ncu --version
nsys --version
ncu --query-metrics | head
```

Expected：GPU 名称包含 RTX 4090，`ncu --query-metrics` 成功。失败则停止 GPU 计划并更换实例，不接受无 profiler 权限的服务器。

- [ ] **周二：编译运行 vector add**

```bash
cmake -S $CAPSTONE/cuda -B $CAPSTONE/cuda/build -G Ninja
cmake --build $CAPSTONE/cuda/build --target vector_add
$CAPSTONE/cuda/build/vector_add
```

Expected：三组 size 全部与 CPU reference 一致。

- [ ] **周三：加入 CUDA Event timing**

只测 kernel execution；初始化、allocation、H2D/D2H 分开记录。至少 warmup 20 次、测量 100 次。

- [ ] **周四：运行 compute-sanitizer**

```bash
compute-sanitizer --tool memcheck $CAPSTONE/cuda/build/vector_add
```

Expected：0 errors。

- [ ] **周五：记录第一个 ncu report**

```bash
ncu --set basic --target-processes all \
  --export $CAPSTONE/profiles/ncu/week02-vector-add \
  $CAPSTONE/cuda/build/vector_add
```

### 周末

- [ ] **周六：做 block-size ablation**

测量 block size `64/128/256/512`，每项至少三轮；CSV 列：

```text
timestamp,gpu,driver,cuda,n,block_size,median_us,p20_us,p80_us,bandwidth_gbps
```

- [ ] **周日：解释结果**

在 `docs/cuda_kernel_notes.md` 回答：

```text
为什么极小输入主要受 launch latency 影响？
为什么 block size 改变不一定线性改变性能？
有效带宽如何计算？
```

### Week 2 Exit Gate

```text
RTX 4090、ncu、nsys 权限确认
vector add correctness 和 memcheck 通过
有第一份可复现 CSV 和 ncu report
```

---

## Week 3：Reduction 与 CUDA Matmul

**Files:**

- Create: `cuda/reduction.cu`
- Create: `cuda/matmul_naive.cu`
- Create: `cuda/matmul_tiled.cu`
- Create: `results/experiments/week03-cuda-matmul.csv`

### 工作日

- [ ] **周一：实现 reduction baseline**

实现一个 block 内 shared-memory tree reduction；使用 FP32 输入，和 CPU double accumulation reference 比较。

- [ ] **周二：实现 naive matmul**

接口固定为 row-major：

```text
A[M,K] × B[K,N] -> C[M,N]
```

每个 thread 计算一个 `C[m,n]`，先只覆盖 `M=N=K=512`。

- [ ] **周三：实现 shared-memory tiled matmul**

先固定 `TILE=16`，包含 K 维循环、两次 `__syncthreads()` 和边界 mask。

- [ ] **周四：加入非整除 case**

至少加入：

```text
(M,N,K)=(127,251,509)
(M,N,K)=(513,769,1025)
```

- [ ] **周五：检查访存模式**

在笔记中逐式写出 A/B global address，并说明哪个维度连续、哪些 load 能 coalesce。

### 周末

- [ ] **周六：比较 naive 与 tiled**

```bash
cmake --build $CAPSTONE/cuda/build
compute-sanitizer --tool memcheck $CAPSTONE/cuda/build/matmul_tiled
$CAPSTONE/cuda/build/matmul_naive --m 1024 --n 1024 --k 1024
$CAPSTONE/cuda/build/matmul_tiled --m 1024 --n 1024 --k 1024
```

记录 latency、effective TFLOPS 和 correctness。

- [ ] **周日：采集 ncu 并复盘**

```bash
ncu --set full --kernel-name regex:matmul \
  --export $CAPSTONE/profiles/ncu/week03-matmul \
  $CAPSTONE/cuda/build/matmul_tiled --m 1024 --n 1024 --k 1024
```

重点记录 memory throughput、occupancy、register、shared memory 和 stall reason。

### Week 3 Exit Gate

```text
naive/tiled matmul 均通过整除和非整除 correctness
能解释 tiled 版本减少了哪些 global load
能指出当前版本仍未接近 cuBLAS/Triton 的至少三个原因
```

---

## Week 4：Triton Kernel 与 Fused Linear

**Files:**

- Create: `triton_kernels/vector_add.py`
- Create: `triton_kernels/fused_softmax.py`
- Create: `triton_kernels/matmul.py`
- Create: `triton_kernels/fused_linear.py`
- Create: `results/experiments/week04-triton.csv`

### 工作日

- [ ] **周一：运行并重写 Triton vector add**

先运行官方 tutorial，再关闭 tutorial，从空文件实现 `tl.program_id/tl.arange/tl.load/tl.store` 版本。

- [ ] **周二：运行 fused softmax 并解释 row mapping**

记录一个 Triton program instance 处理哪一行、mask 如何处理边界、为什么 fusion 减少 global traffic。

- [ ] **周三：运行 matmul tutorial**

固定当前环境的 Triton package/version，完成官方 correctness 和 benchmark。

- [ ] **周四：实现 fused Linear + Bias + ReLU**

固定语义：

```text
FP16 A/B/bias
FP32 accumulator
acc += bias in FP32
relu in FP32
FP16 output
```

- [ ] **周五：加入三组最小 correctness**

```text
(32,4096,4096)
(512,4096,4096)
(127,251,509)
```

reference 使用 PyTorch FP32 matmul + bias + ReLU 后 cast FP16。

### 周末

- [ ] **周六：增加 autotune**

至少搜索：

```text
BLOCK_M: 32,64,128
BLOCK_N: 32,64,128
BLOCK_K: 32,64
num_warps: 4,8
num_stages: 2,3,4
```

过滤明显超资源或不合法组合，key 固定为 `M,N,K`。

- [ ] **周日：做第一轮 Triton 结果分析**

使用 `triton.testing.do_bench` 输出 median/P20/P80；写清 CUDA thread-level 与 Triton blocked programming model 的差异。

### Phase 1 Exit Gate

```text
独立实现 CUDA tiled matmul
独立实现 Triton fused Linear
能够解释 coalescing/shared memory/occupancy/tile trade-off
RTX 4090 benchmark 和 ncu 已真实运行
```

未通过时重复 Week 3-4，不进入 compiler internals。

---

## 4. Phase 2：MLIR Transformation 桥接

## Week 5：隔离 Rewrite、Fold 与 Driver

**Files:**

- Modify: `LearningSteps/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp`
- Modify: `test/lib/Transforms/AddZeroPatternPass.cpp`
- Modify: `LearningSteps/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir`
- Create: `GPUCodeGenCapstone/mlir/add_zero/rewrite-mechanics.md`

### 工作日

- [ ] **周一：记录当前 green baseline**

```bash
$MLIR_BIN/mlir-opt \
  LearningSteps/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir \
  -pass-pipeline='builtin.module(func.func(add-zero-pattern))' \
  | $MLIR_BIN/FileCheck \
    LearningSteps/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir
```

Expected：exit 0。

- [ ] **周二：关闭 greedy folding**

在 pass 中使用：

```cpp
GreedyRewriteConfig config;
config.enableFolding(false);
applyPatternsGreedily(getOperation(), std::move(patterns), config);
```

- [ ] **周三：做 red 验证**

临时移除 `patterns.add<AddZeroPattern>`，重新构建并运行测试。

Expected：FileCheck 失败，输出仍包含 `arith.addi`。保存失败摘要后恢复 pattern 注册。

- [ ] **周四：恢复并做 green 验证**

```bash
cmake --build $MLIR_BUILD --target mlir-opt
$MLIR_BIN/mlir-opt \
  LearningSteps/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir \
  -pass-pipeline='builtin.module(func.func(add-zero-pattern))' \
  | $MLIR_BIN/FileCheck \
    LearningSteps/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir
```

- [ ] **周五：比较内建 fold**

阅读：

```text
lib/Dialect/Arith/IR/ArithOps.cpp
include/mlir/Transforms/GreedyPatternRewriteDriver.h
lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
```

记录 pattern、rewriter、driver、fold、DCE 各自职责。

### 周末

- [ ] **周六：补测试边界**

加入 multiple-use constant、non-zero、不同 integer width 和至少一个 shaped integer case；不支持的 case 要说明原因，不能静默忽略。

- [ ] **周日：同步双份源码并复盘**

```bash
diff -u LearningSteps/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp \
  test/lib/Transforms/AddZeroPatternPass.cpp
```

Expected：除明确记录的集成差异外内容一致。长期真实编译入口标记为 `test/lib/Transforms/AddZeroPatternPass.cpp`。

### Week 5 Exit Gate

```text
能够用 red/green 证明确实是自定义 pattern 完成 rewrite
能够解释 replaceOp、DCE 和 fold 的先后关系
能够解释 FuncOp pass anchor
```

---

## Week 6：Transform Dialect 驱动 Matmul Tiling

**Files:**

- Create: `mlir/transform_matmul_tiling.mlir`
- Create: `mlir/tests/transform_matmul_tiling.mlir`
- Create: `docs/mlir_tiling_notes.md`

### 工作日

- [ ] **周一：运行仓库现有 tiling 测试**

```bash
$MLIR_BIN/llvm-lit -sv \
  test/Interfaces/TilingInterface/tile-using-scfforall.mlir
$MLIR_BIN/llvm-lit -sv \
  test/Dialect/Linalg/transform-op-tile.mlir
```

- [ ] **周二：抽取最小 linalg.matmul case**

输入固定为 `tensor<128x256xf32> × tensor<256x512xf32>`，使用 `tensor.empty`、`linalg.fill` 和 `linalg.matmul`。

- [ ] **周三：加入 named sequence 和 match**

Transform IR 必须只匹配目标 `linalg.matmul`，不能依赖文件中只有一个 op。

- [ ] **周四：tile 为 scf.forall/scf.for**

第一版 tile size 固定 `64x64x32`，保存 transformation 前后 IR。

- [ ] **周五：解释 TilingInterface**

阅读：

```text
include/mlir/Interfaces/TilingInterface.h
lib/Dialect/SCF/Transforms/TileUsingInterface.cpp
```

写出 destination、iteration domain、tiled implementation 和 result replacement 的关系。

### 周末

- [ ] **周六：加入边界 shape**

增加 `127x251x509`，检查生成的 offset/size/min 逻辑；不得只测试整除 tile。

- [ ] **周日：写 FileCheck**

检查：

```text
scf.forall/scf.for 数量
step/tile size
tensor.extract_slice 边界
tiled linalg.matmul
result insertion
```

### Week 6 Exit Gate

```text
两个 upstream lit 测试通过
自定义 tiling case 的整除/非整除测试通过
能解释 tile transformation 不是简单文本替换
```

---

## Week 7：C++ 参数化 Tiling Transformation

**Files:**

- Create: `mlir/MatmulTilingPass.cpp`
- Create: `mlir/tests/matmul-tiling-pass.mlir`
- Create: `mlir/tests/matmul-tiling-invalid.mlir`
- Update: `docs/mlir_tiling_notes.md`

### 工作日

- [ ] **周一：定义 pass contract**

固定 pass 参数：

```text
tile-m
tile-n
tile-k
```

前置条件：正整数、目标为 `linalg.matmul`、静态 rank-2 tensor semantics。其他 case 产生诊断或明确跳过。

- [ ] **周二：写失败测试**

为 `tile-m=0`、缺失目标 op 和非 rank-2 case 写 expected-error/负例。

- [ ] **周三：实现最小 C++ pass**

使用 `TilingInterface`/SCF tiling API，不复制 Linalg 内部 tiling 实现。

- [ ] **周四：加入整除 FileCheck**

检查 tile size、循环层级和 tiled op。

- [ ] **周五：加入非整除 FileCheck**

检查尾块尺寸由原始 shape 与 offset 计算得出。

### 周末

- [ ] **周六：构建并运行全部测试**

将 pass 接入与 AddZero 相同的 test-only 构建路径，运行目标 build 和两个测试文件。

- [ ] **周日：写 transformation legality 说明**

必须回答：

```text
为什么 tile size 必须校验？
哪些 op 由 transformation 新建？
原 op 的 uses 如何替换？
失败时是否修改了部分 IR？
哪些 analysis 被 invalidated？
```

### Phase 2 Exit Gate

```text
能够从空文件实现参数化 matmul tiling pass
正例、负例、边界测试通过
能够解释 folding/pattern/tiling/legality 的区别
```

---

## 5. Phase 3：Triton Compiler 内部

## Week 8：固定 Triton 源码与构建

**Files:**

- Create: `compiler/baseline_commit.txt`
- Create: `environment/triton-build.md`
- Update: `README.md`

### 工作日

- [ ] **周一：clone Triton 并固定 commit**

```bash
mkdir -p $HOME/workspace
git clone https://github.com/triton-lang/triton.git $TRITON_ROOT
git -C $TRITON_ROOT rev-parse HEAD | tee $CAPSTONE/compiler/baseline_commit.txt
```

- [ ] **周二：创建独立虚拟环境**

```bash
python3 -m venv $TRITON_VENV
source $TRITON_VENV/bin/activate
python -m pip install --upgrade pip
python -m pip install -r $TRITON_ROOT/python/requirements.txt
```

- [ ] **周三：从源码构建 editable install**

```bash
source $TRITON_VENV/bin/activate
cd $TRITON_ROOT
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_WITH_CCACHE=true \
python -m pip install -e . --no-build-isolation
```

- [ ] **周四：验证 Python package 与解释器**

```bash
python -c 'import triton; print(triton.__version__); print(triton.__file__)'
TRITON_INTERPRET=1 python $CAPSTONE/triton_kernels/vector_add.py
```

- [ ] **周五：定位 triton-opt 和 pass pipeline**

```bash
find $TRITON_ROOT -type f -name triton-opt -perm -111
rg -n 'add_stages|make_ttir|make_ttgir|make_llir|make_ptx' \
  $TRITON_ROOT/third_party/nvidia $TRITON_ROOT/python/triton/compiler
```

### 周末

- [ ] **周六：运行 Triton 单测子集**

先运行无 GPU 测试入口，再在 RTX 4090 上运行两个官方 tutorial smoke：

```bash
make -C $TRITON_ROOT test-nogpu
source $TRITON_VENV/bin/activate
python $TRITON_ROOT/python/tutorials/01-vector-add.py
python $TRITON_ROOT/python/tutorials/03-matrix-multiplication.py
```

若固定 commit 的 Make target 或 tutorial 路径发生变化，使用该 commit 的 README 中“Running tests”命令替换，并把最终实际命令、测试数和耗时写入 `triton-build.md`。

- [ ] **周日：记录构建复现信息**

保存 Triton commit、其 LLVM hash、Python、compiler、CMake、Ninja 和 build flags。不得只写“使用最新版”。

### Week 8 Exit Gate

```text
editable Triton 从源码 import
triton-opt 路径已定位
baseline commit 和 LLVM revision 已记录
至少一组 compiler/kernel smoke test 通过
```

---

## Week 9：TTIR、TTGIR、LLVM IR、PTX 对照

**Files:**

- Create: `scripts/dump_triton_ir.sh`
- Create: `docs/triton_compiler_pipeline.md`
- Populate: `ir/ttir`, `ir/ttgir`, `ir/llvm`, `ir/ptx`

### 工作日

- [ ] **周一：dump vector add**

```bash
source $TRITON_VENV/bin/activate
TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=$CAPSTONE/results/experiments/week09-vector-add-dump \
python $CAPSTONE/triton_kernels/vector_add.py
```

- [ ] **周二：dump fused Linear**

用相同环境变量运行 `fused_linear.py`，固定一个小 shape 避免 dump 混入多个 autotune candidate。

- [ ] **周三：标注 TTIR**

对 `tl.load/tl.dot/bias/relu/tl.store` 各找出对应 op；写明 shape、type 和 SSA data flow。

- [ ] **周四：标注 TTGIR**

找出 module target、layout encoding、CTA/warp mapping、dot operand 和 layout conversion。

- [ ] **周五：标注 LLVM/PTX**

找出 address space、barrier、shared-memory、load/store 和 MMA 相关片段；不能只粘贴整份 IR。

### 周末

- [ ] **周六：画出一条 op lowering 链**

至少跟踪一个 `tl.dot` 和一个 bias broadcast，从 TTIR 一直跟到 PTX，并记录每一层丢失/增加的信息。

- [ ] **周日：验证 dump 可复现**

清理 Triton cache 后重新运行一次；确认脚本能重新生成同类 artifact，并记录 cache 对 dump 的影响。

### Week 9 Exit Gate

```text
四级 IR/PTX artifact 可复现
能解释 TTIR 与 TTGIR 职责差异
能跟踪至少一个 op 的完整 lowering 链
```

---

## Week 10：第一个 Triton Compiler 改动

**Files:**

- Create: `compiler/reproducers/week10.mlir`
- Create: `compiler/tests/week10.mlir`
- Create: `compiler/patches/week10.patch`
- Update: `docs/triton_compiler_pipeline.md`

### 工作日

- [ ] **周一：选择一个低风险 transformation**

使用以下命令从固定 commit 选择只影响局部 IR、已有测试充分的 pass：

```bash
rg -n 'RemoveLayout|Coalesce|Canonicalize|combine|fold' \
  $TRITON_ROOT/lib $TRITON_ROOT/third_party/nvidia/lib | head -80
```

选择标准写入文档：输入/输出 IR 明确、无需改变 ABI、可用单个 reproducer、预期不改变数值语义。第一项改动不选择 MMA lowering 或 shared-memory allocator。

- [ ] **周二：从 kernel 捕获 reproducer**

```bash
TRITON_REPRODUCER_PATH=$CAPSTONE/compiler/reproducers/week10.mlir \
TRITON_ALWAYS_COMPILE=1 \
python $CAPSTONE/triton_kernels/fused_linear.py
```

- [ ] **周三：先写 compiler test**

从 reproducer 最小化出一个只覆盖目标 transformation 的 test，运行 baseline 并保存 FileCheck output。

- [ ] **周四：实现一个行为可见的小改动**

要求：生成 IR 有稳定差异、语义不变、改动不超过一个 pass/utility 的职责范围。

- [ ] **周五：运行目标 compiler test**

使用 Triton 仓库实际 test harness；记录命令、PASS 数和耗时。

### 周末

- [ ] **周六：做 red/green 回退验证**

```text
修改存在：test PASS
临时撤销修改：新增检查 FAIL
恢复修改：test PASS
```

保存三次结果摘要。

- [ ] **周日：导出 patch 并复盘**

```bash
git -C $TRITON_ROOT diff > $CAPSTONE/compiler/patches/week10.patch
```

解释 pass anchor、匹配条件、生成 op、失败条件和测试边界。

### Phase 3 Exit Gate

```text
能够从源码构建和调试 Triton
能够生成 reproducer
完成一个真实 compiler transformation 改动
新增 test 经 red/green 验证
```

---

## 6. Phase 4：冻结 Fused Linear Baseline

## Week 11：标准 Shape 与 Correctness Harness

**Files:**

- Create: `benchmark/shapes.py`
- Create: `benchmark/correctness.py`
- Create: `scripts/run_correctness.sh`
- Create: `results/baseline/correctness.json`

### 工作日

- [ ] **周一：编码六组冻结 shape**

`shapes.py` 只包含：

```python
SHAPES = [
    (32, 4096, 4096),
    (128, 4096, 4096),
    (512, 4096, 4096),
    (512, 16384, 4096),
    (512, 4096, 16384),
    (4096, 4096, 4096),
]
```

- [ ] **周二：实现 FP32 reference**

固定 `torch.matmul(a.float(), b.float()) + bias.float()`，ReLU 后 cast FP16。

- [ ] **周三：实现十个固定 seed**

每个 shape 至少覆盖两个 seed；完整十 seed 可在分批运行中完成，结果必须记录 seed。

- [ ] **周四：实现边界输入**

加入全负 bias、零附近输入、非整除 shape `127x251x509` 和 NaN/Inf 检查。

- [ ] **周五：输出 machine-readable JSON**

至少包含：shape、seed、max_abs、max_rel、pass/fail、dtype、commit。

### 周末

- [ ] **周六：RTX 4090 完整 correctness**

```bash
bash $CAPSTONE/scripts/run_correctness.sh
```

Expected：全部 case PASS；任何失败先修 correctness，不进入 Week 12。

- [ ] **周日：运行 compute-sanitizer/边界复盘**

对可独立 launch 的最小 kernel 执行 memcheck，记录 mask 和 boundary tile 行为。

### Week 11 Exit Gate

```text
六组 shape、十组 seed 和边界 case 全部正确
结果可由 JSON 审计
无 NaN/Inf 和越界
```

---

## Week 12：Autotune 与 Benchmark Harness

**Files:**

- Create: `benchmark/baseline_configs.py`
- Create: `benchmark/benchmark.py`
- Create: `scripts/run_benchmark.sh`
- Create: `results/baseline/raw.csv`

### 工作日

- [ ] **周一：冻结 autotune search space**

写明 BLOCK_M/N/K、GROUP_M、num_warps、num_stages 候选和过滤条件。

- [ ] **周二：实现 timing protocol**

固定 warmup 100、measurement 500、三轮完整测量，输出 median/P20/P80。

- [ ] **周三：分离 compile/autotune 与 steady state**

第一次运行记录 compile/autotune 时间；正式 latency 使用已编译 cache，二者写入不同列。

- [ ] **周四：计算 effective TFLOPS**

主 GEMM 使用：

```text
2 * M * N * K / latency_seconds / 1e12
```

同时明确该数值未计 bias/ReLU FLOPs，仅用于和相同语义 baseline 比较。

- [ ] **周五：固定 CSV schema**

```text
timestamp,provider,triton_commit,compiler_patch,gpu,driver,cuda,
M,N,K,dtype,config,round,median_us,p20_us,p80_us,tflops,compile_ms
```

### 周末

- [ ] **周六：运行三轮完整 baseline**

```bash
bash $CAPSTONE/scripts/run_benchmark.sh --provider baseline --rounds 3
```

- [ ] **周日：计算噪声和异常值**

报告每个 shape 三轮 median 的最大相对偏差；如果温度、时钟或共享实例负载导致异常，重新测量并保留被废弃记录及原因。

### Week 12 Exit Gate

```text
autotune search space 已冻结
三轮 baseline 数据完整
compile time 与 steady-state latency 分离
测量噪声有量化结果
```

---

## Week 13：Baseline Profiler 与 IR/SASS Artifact

**Files:**

- Create: `scripts/run_ncu.sh`
- Populate: `profiles/ncu/baseline/`
- Populate: `profiles/nsys/baseline/`
- Populate: `ir/*/baseline/`
- Create: `results/baseline/profiler-summary.csv`

### 工作日

- [ ] **周一：为每组 shape 找到最终 autotune config**

保存 BLOCK、warps、stages 和 kernel name，后续 ncu 不重新触发全量 autotune。

- [ ] **周二：采集 nsys timeline**

确认 fused workload 只有目标 kernel launch，没有额外 bias/ReLU kernel。

- [ ] **周三：采集 ncu SpeedOfLight/Occupancy**

先使用较小 section set，记录 duration、occupancy、register、shared memory。

- [ ] **周四：对重点 shape 采集 full metrics**

重点选择小 M、大 K 和大型方阵三组，避免对六组全部使用高开销 full replay。

- [ ] **周五：导出 PTX/SASS**

保存与 profiler 中 kernel 对应的 PTX/SASS，并记录 hash，避免把不同 binary 的结果混在一起。

### 周末

- [ ] **周六：建立瓶颈矩阵**

每个 shape 填写：

```text
CTA count
waves per SM
occupancy
register/thread
shared memory/CTA
DRAM throughput
L2 hit rate
Tensor Core utilization
top stall reason
```

- [ ] **周日：写三个候选假设，不修改代码**

每个假设必须包含对应证据和预期指标变化。此周禁止提前实现优化。

### Week 13 Exit Gate

```text
baseline kernel launch 边界确认公平
重点 shape 有 ncu report 和对应 binary artifact
候选假设由指标触发，而非凭直觉选择
```

---

## Week 14：冻结 Baseline

**Files:**

- Create: `docs/baseline_report.md`
- Create: `results/baseline/MANIFEST.md`
- Update: `compiler/baseline_commit.txt`
- Update: `README.md`

### 工作日

- [ ] **周一：核对 workload contract**

逐项确认 dtype、layout、FP32 accumulation、fusion、六组 shape 和误差标准未漂移。

- [ ] **周二：核对环境 contract**

确认 GPU、driver、CUDA、Triton commit、PyTorch、power/clock 和测量命令完整。

- [ ] **周三：核对 artifact hash**

为 baseline CSV、kernel source、configs、PTX/SASS 和 profiler report 记录 hash/commit。

- [ ] **周四：写 baseline report**

必须回答总计划 Section 10.3 的七个问题。

- [ ] **周五：做冷启动复现**

在新 shell 中仅根据 README 运行 correctness 和一个 benchmark shape。

### 周末

- [ ] **周六：完整冻结验证**

重跑六组 correctness 和一轮 benchmark，和 Week 12 数据比较。

- [ ] **周日：签署 baseline freeze**

在 `MANIFEST.md` 写入日期、commit、配置、命令和冻结声明。后续若改 baseline，必须重新运行全部实验。

### Phase 4 Exit Gate

```text
六组 correctness PASS
三轮 benchmark 可复现
baseline workload/config/environment 已冻结
baseline report 和 manifest 完整
```

---

## 7. Phase 5：Profiler 驱动的 Compiler 优化

## Week 15：选择唯一主瓶颈并设计实验

**Files:**

- Create: `docs/optimization_design.md`
- Create: `results/experiments/exp01/README.md`

### 工作日

- [ ] **周一：按证据分类瓶颈**

使用以下决策：

```text
小 M 且 CTA/waves 不足 -> tile/mapping/K-parallelism 候选
layout conversion 占显著 IR/PTX -> layout propagation/elimination 候选
register/thread 高且 occupancy 低 -> tile/stages/lifetime 候选
shared bank conflict 高 -> shared layout/swizzle 候选
DRAM 高、L2 低 -> ordering/reuse 候选
```

- [ ] **周二：选一个主问题**

只允许一个主指标和一个主要 compiler transformation。把未选问题写入 future work。

- [ ] **周三：写可证伪假设**

格式：

```text
If <compiler change>, then <IR/PTX change>,
therefore <ncu metric> should change from X toward Y,
and latency should improve on <shape set>,
while <known cost> may regress <other set>.
```

- [ ] **周四：设计 feature flag**

compiler change 必须能够显式开关；baseline 和 experiment 由同一 binary/commit 下的 flag 区分，或精确记录两个 commit。

- [ ] **周五：设计 compiler test**

先定义输入 reproducer 和稳定 IR 检查，不使用 latency 作为 compiler 单测。

### 周末

- [ ] **周六：完成 design review checklist**

检查语义、适用范围、失败方式、compile-time、code-size、回退策略和 test coverage。

- [ ] **周日：只跑 baseline sanity**

确认 baseline 仍与冻结数据一致，然后批准进入实现。

### Week 15 Exit Gate

```text
唯一主瓶颈已由 profiler 证据确定
假设包含预期 IR 和硬件指标变化
compiler change 可开关
测试设计先于实现
```

---

## Week 16：Experiment 01 实现与 Compiler Test

**Files:**

- Modify: exact Triton pass selected in Week 15
- Create: `compiler/tests/exp01.mlir`
- Create: `compiler/reproducers/exp01.mlir`
- Create: `compiler/patches/exp01.patch`

### 工作日

- [ ] **周一：写并运行失败测试**

Expected：baseline compiler 下新增 FileCheck 失败，因为预期 transformation 尚未发生。

- [ ] **周二：实现最小 compiler change**

只实现足以让单个 reproducer 变化的最小逻辑，不同时重构相邻 pass。

- [ ] **周三：运行目标 test**

Expected：新增 test PASS；保存测试命令和输出摘要。

- [ ] **周四：运行相关 Triton test 子集**

至少覆盖目标 dialect/pass 目录和 fused_linear kernel smoke test。

- [ ] **周五：检查生成 IR diff**

确认差异符合 Week 15 预测，没有意外 dtype、layout 或 fusion 变化。

### 周末

- [ ] **周六：跑完整 correctness**

六组 shape 和十 seed 全部 PASS 后才允许跑性能。

- [ ] **周日：导出 patch 和 artifact**

保存 compiler diff、test、TTIR/TTGIR/LLVM/PTX before/after。

### Week 16 Exit Gate

```text
新增 compiler test 经 red/green 验证
相关 test 子集通过
完整 correctness 通过
IR diff 与设计预测一致
```

---

## Week 17：Experiment 01 Benchmark、Profiler 与 Ablation

**Files:**

- Create: `results/experiments/exp01/results.csv`
- Create: `results/experiments/exp01/ncu-summary.csv`
- Update: `results/experiments/exp01/README.md`

### 工作日

- [ ] **周一：运行 baseline sanity**

选择三组代表 shape，确认当日机器状态与冻结 baseline 偏差可接受。

- [ ] **周二：运行 experiment 三轮**

使用和 baseline 完全相同的 warmup、measurement、shape 和 autotune policy。

- [ ] **周三：计算 delta**

报告每个 shape：

```text
experiment_us / baseline_us
speedup_percent
variance
```

- [ ] **周四：采集相同 ncu sections**

只比较同一 shape、同一 kernel 语义和相同 profiler 设置。

- [ ] **周五：检查预期指标**

逐项标记 predicted/observed/mismatch；不能只看 latency。

### 周末

- [ ] **周六：做 ablation**

关闭 feature flag 或恢复 baseline pass；重新运行代表 shape。Expected：IR/指标/性能向 baseline 恢复。

- [ ] **周日：结论分类**

只允许三类结论：

```text
SUPPORTED：证据支持假设
REJECTED：证据否定假设
INCONCLUSIVE：噪声或指标不足
```

### Week 17 Exit Gate

```text
三轮结果和 ncu 对照完整
完成 ablation
结论不依赖单次最快值
```

---

## Week 18：Experiment 02 针对首轮结论迭代

**Files:**

- Create: `results/experiments/exp02/README.md`
- Create: `compiler/tests/exp02.mlir`
- Create: `compiler/patches/exp02.patch`
- Create: `results/experiments/exp02/results.csv`

### 工作日

- [ ] **周一：从 Exp01 结论定义单变量变化**

若 Exp01 rejected，改变假设而不是调参掩盖；若 supported，只改变一个参数/策略扩大适用区间。

- [ ] **周二：写 Exp02 compiler test**

- [ ] **周三：实现最小变化并运行 test**

- [ ] **周四：运行相关 test 子集和 correctness**

```bash
bash $CAPSTONE/scripts/run_correctness.sh
```

Expected：六组冻结 shape 和固定 seed 全部 PASS。

- [ ] **周五：生成四级 IR/PTX diff**

### 周末

- [ ] **周六：三轮 benchmark 和 ncu**

```bash
bash $CAPSTONE/scripts/run_benchmark.sh --provider exp02 --rounds 3
bash $CAPSTONE/scripts/run_ncu.sh --provider exp02
```

- [ ] **周日：ablation 和结论**

### Week 18 Exit Gate

```text
Exp02 与 Exp01 只存在一个主要变量差异
compiler test/correctness/benchmark/profiler/ablation 完整
```

---

## Week 19：Experiment 03 与优化候选冻结

**Files:**

- Create: `results/experiments/exp03/README.md`
- Create: `compiler/tests/exp03.mlir`
- Create: `compiler/patches/exp03.patch`
- Create: `results/experiments/summary.csv`
- Update: `docs/failed_experiments.md`

### 工作日

- [ ] **周一：定义最后一个高信息量实验**

目标是区分仍竞争的两个解释，不是盲目继续搜索配置。

- [ ] **周二：写 test 和实现**

- [ ] **周三：运行 compiler tests**

- [ ] **周四：运行 correctness**

```bash
bash $CAPSTONE/scripts/run_correctness.sh
```

- [ ] **周五：运行代表 shape benchmark**

```bash
bash $CAPSTONE/scripts/run_benchmark.sh --provider exp03 --rounds 3
```

### 周末

- [ ] **周六：完整六 shape benchmark/profiler**

```bash
bash $CAPSTONE/scripts/run_benchmark.sh --provider exp03 --rounds 3 --all-shapes
bash $CAPSTONE/scripts/run_ncu.sh --provider exp03
```

- [ ] **周日：冻结最佳候选**

比较 Exp01-03，选出一个进入泛化阶段；其余实验无论成功失败都写入记录。

### Phase 5 Exit Gate

```text
至少三次完整实验
至少允许并记录失败实验
最佳候选具有 compiler test、correctness、ablation 和 profiler 证据
至少一个硬件指标按预测方向变化
```

---

## 8. Phase 6：泛化与回归

## Week 20：邻近 Shape 与边界泛化

**Files:**

- Create: `benchmark/generalization_shapes.py`
- Create: `results/regression/week20-generalization.csv`
- Create: `docs/optimization_scope.md`

### 工作日

- [ ] **周一：定义邻近 shape**

围绕获益区间增加 M/N/K 上下邻点和非整除点，不根据已知结果挑点。

- [ ] **周二：运行全部 correctness**

- [ ] **周三：运行 baseline 三轮**

- [ ] **周四：运行 optimized 三轮**

- [ ] **周五：计算 speedup distribution**

### 周末

- [ ] **周六：分析边界在哪里反转**

定位从获益到退化的 shape 区域，并关联 CTA waves、register、occupancy 或 memory 指标。

- [ ] **周日：写适用条件**

条件必须能由 compiler 在编译期判断，例如静态 M/N/K、layout、dtype，而不是“某些情况更快”。

### Week 20 Exit Gate

```text
邻近 shape 不是事后挑选
优化适用条件可由 IR/shape 判断
退化边界有数据
```

---

## Week 21：性能回归 Harness

**Files:**

- Create: `benchmark/regression.py`
- Create: `scripts/run_regression.sh`
- Create: `results/regression/thresholds.json`

### 工作日

- [ ] **周一：定义 correctness gate**

regression 脚本首先跑 correctness，失败立即终止性能阶段。

- [ ] **周二：定义性能阈值**

硬标准：六组中至少四组达到 baseline 90%，任何一组不低于 80%。阈值考虑已测量噪声，但不能放宽硬标准。

- [ ] **周三：实现 CSV/JSON 输出**

- [ ] **周四：实现非零退出码**

correctness 或硬性能门槛失败时脚本返回非零。

- [ ] **周五：制造一次预期失败**

临时使用明显不利配置，验证 regression harness 确实失败；随后恢复候选。

### 周末

- [ ] **周六：运行正式 regression**

```bash
bash $CAPSTONE/scripts/run_regression.sh
```

- [ ] **周日：记录 compile-time/code-size**

比较 baseline 与候选的 compiler time、PTX/SASS size 和 cache artifact。

### Week 21 Exit Gate

```text
regression harness 有真实 red/green 验证
错误会返回非零
correctness、性能、compile-time 和 code-size 均被记录
```

---

## Week 22：完整泛化报告与最终 Patch

**Files:**

- Create: `compiler/patches/final.patch`
- Create: `results/regression/final.csv`
- Finalize: `docs/optimization_scope.md`
- Update: `docs/failed_experiments.md`

### 工作日

- [ ] **周一：清理 compiler diff**

移除 debug print、无关格式变化和未使用 flag；不做与优化无关的大重构。

- [ ] **周二：运行 compiler test 子集**

- [ ] **周三：运行完整 correctness**

- [ ] **周四：运行正式 regression**

- [ ] **周五：导出 final patch 和 artifact hash**

### 周末

- [ ] **周六：计算最终摘要**

必须报告 best、geometric mean、worst regression、variance、compile-time delta 和 code-size delta。

- [ ] **周日：完成适用范围文档**

明确支持条件、回退条件、已知限制和未解决问题。

### Phase 6 Exit Gate

```text
final patch 干净且有 compiler test
完整 correctness/regression 通过硬门槛
泛化、最坏退化和适用范围明确
```

---

## 9. Phase 7：工程化交付

## Week 23：冷环境复现与文档闭环

**Files:**

- Finalize: `README.md`
- Finalize: `environment/*.md`
- Create: `docs/reproduction.md`
- Create: `results/MANIFEST.md`

### 工作日

- [ ] **周一：从 README 审计所有命令**

每条命令必须包含工作目录、环境变量和预期输出。

- [ ] **周二：建立 manifest**

列出 source commit、patch、脚本、CSV、ncu/nsys、IR/PTX/SASS 和 hash。

- [ ] **周三：在干净 Python venv 重建 kernel 环境**

- [ ] **周四：在干净 Triton worktree 应用 final patch**

```bash
git -C $TRITON_ROOT worktree add $HOME/workspace/triton-repro \
  $(cat $CAPSTONE/compiler/baseline_commit.txt)
git -C $HOME/workspace/triton-repro apply \
  $CAPSTONE/compiler/patches/final.patch
```

- [ ] **周五：运行 compiler tests 和一个 correctness smoke**

### 周末

- [ ] **周六：租赁 RTX 4090 做冷环境完整复现**

仅依据 `docs/reproduction.md` 完成环境检查、correctness、benchmark 和一个 ncu report。

- [ ] **周日：修正文档缺口**

任何依赖口头记忆的步骤都补进文档；重新执行修正后的命令。

### Week 23 Exit Gate

```text
final patch 可应用到冻结 baseline commit
干净环境可完成 build/test/run
结果 artifact 可由 manifest 定位
```

---

## Week 24：最终报告与自我答辩

**Files:**

- Finalize: `docs/triton_compiler_pipeline.md`
- Finalize: `docs/baseline_report.md`
- Finalize: `docs/optimization_design.md`
- Finalize: `docs/tuning_report.md`
- Finalize: `docs/failed_experiments.md`
- Finalize: `progress.md`

### 工作日

- [ ] **周一：完成 compiler pipeline 章节**

用一个 fused Linear 跟踪 TTIR -> TTGIR -> LLVM IR -> PTX/SASS。

- [ ] **周二：完成 baseline 与实验表格**

不只展示最佳值，必须展示六组 shape、几何平均和最坏退化。

- [ ] **周三：完成 profiler 归因**

将每条性能结论链接到 IR/PTX diff 和 ncu 指标。

- [ ] **周四：整理失败实验**

至少包含 observation、hypothesis、change、result 和 rejected reason。

- [ ] **周五：准备 20 分钟答辩提纲**

结构：问题 2 分钟、baseline 3 分钟、compiler pipeline 4 分钟、优化设计 4 分钟、结果 4 分钟、限制与下一步 3 分钟。

### 周末

- [ ] **周六：进行一次无稿自我答辩**

录音或记录问题，必须能回答：

```text
为什么选择这个 IR 层修改？
为什么收益不是 autotune 或噪声？
为什么某些 shape 退化？
如何保证语义正确？
如果支持 dynamic shape，设计会怎样变化？
```

- [ ] **周日：最终验收**

运行：

```bash
bash $CAPSTONE/scripts/run_correctness.sh
bash $CAPSTONE/scripts/run_regression.sh
git -C $TRITON_ROOT worktree add --detach $HOME/workspace/triton-verify \
  $(cat $CAPSTONE/compiler/baseline_commit.txt)
git -C $HOME/workspace/triton-verify apply --check \
  $CAPSTONE/compiler/patches/final.patch
```

Expected：correctness 和 regression 返回 0，`final.patch` 能在干净 baseline worktree 中通过 `apply --check`。

### Final Exit Gate

```text
硬验收全部满足
六组 shape correctness 全部通过
至少四组达到 Triton baseline 90%
任何一组不低于 baseline 80%
compiler patch/test/reproducer 可复现
性能结论有 ablation 和 profiler 证据
成功与失败实验均已文档化
```

---

## 10. 统一实验模板

每个 `results/experiments/expNN/README.md` 使用：

```markdown
# Experiment NN: <name>

## Environment

- GPU:
- Driver/CUDA:
- Triton baseline commit:
- Experiment commit/flag:
- Power/clock/temperature:

## Observation

## Hypothesis

## Compiler Change

## Expected IR/PTX Change

## Expected Hardware Metric Change

## Correctness

## Benchmark

## Profiler

## Ablation

## Result

SUPPORTED / REJECTED / INCONCLUSIVE

## Scope and Regressions

## Next Action
```

---

## 11. 阶段检查清单

### Phase 1

- [ ] CUDA vector add/reduction/naive matmul/tiled matmul 可运行
- [ ] Triton vector add/softmax/matmul/fused Linear 可运行
- [ ] RTX 4090 profiler 权限确认
- [ ] 能解释线程映射、访存、shared memory、occupancy

### Phase 2

- [ ] AddZero red/green 证明自定义 pattern 生效
- [ ] Transform Dialect matmul tiling 整除/非整除测试通过
- [ ] C++ 参数化 tiling pass 正例/负例/边界测试通过

### Phase 3

- [ ] Triton 固定 commit 源码构建成功
- [ ] TTIR/TTGIR/LLVM/PTX artifact 可复现
- [ ] 一个真实 compiler change 经 red/green test 验证

### Phase 4

- [ ] 六组 shape 和十 seed correctness 通过
- [ ] autotune/timing/CSV schema 冻结
- [ ] baseline benchmark/profiler/manifest 完整

### Phase 5

- [ ] 三个实验均有假设、测试、correctness、benchmark、profiler、ablation
- [ ] 失败实验未被删除
- [ ] 最佳候选由完整证据选择

### Phase 6

- [ ] 邻近 shape 和边界 shape 完整
- [ ] regression harness 经预期失败验证
- [ ] final patch 和适用范围冻结

### Phase 7

- [ ] 干净环境复现通过
- [ ] 最终报告覆盖所有硬验收
- [ ] 完成无稿自我答辩

---

## 12. 调整规则

### 落后 1 周

保留本周 exit gate，将阅读内容压缩到最直接的官方文档和源码，不删除 correctness/test/profiling。

### 落后 2-3 周

删除挑战目标和第三次优化迭代，保留：

```text
CUDA/Triton 基础
MLIR tiling bridge
Triton compiler change
冻结 baseline
一次完整优化实验
泛化/回归
最终交付
```

### GPU 租赁不可用

可继续做 IR、compiler test、reproducer 和 PTX 静态分析，但 Phase 4 baseline freeze 不得标记完成，最终性能验收顺延。

### 优化未超过 Triton

不改变硬验收。如果达到 90%-100%、完成完整 profiler/ablation 并解释失败原因，项目仍然合格；禁止通过缩减 Triton baseline 能力制造胜出。

### Triton upstream 目录或 API 变化

固定 `compiler/baseline_commit.txt` 后不追逐 main。若必须升级，重新构建、重跑 compiler tests、correctness 和 baseline，并在 manifest 记录迁移原因。

---

## 13. 官方参考入口

```text
Triton repository:
  https://github.com/triton-lang/triton

Triton programming guide:
  https://triton-lang.org/main/programming-guide/

Triton tutorials:
  https://triton-lang.org/main/getting-started/tutorials/

Triton matrix multiplication tutorial:
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

NVIDIA CUDA C++ Programming Guide:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/

NVIDIA Nsight Compute:
  https://docs.nvidia.com/nsight-compute/

NVIDIA Nsight Systems:
  https://docs.nvidia.com/nsight-systems/
```
