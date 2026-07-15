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
| 8 | 2026-09-02 | 2026-09-08 | MLIR Transformation |
| 9 | 2026-09-09 | 2026-09-15 | MLIR Transformation |
| 10 | 2026-09-16 | 2026-09-22 | MLIR Transformation |
| 11 | 2026-09-23 | 2026-09-29 | Triton Compiler |
| 12 | 2026-09-30 | 2026-10-06 | Triton Compiler |
| 13 | 2026-10-07 | 2026-10-13 | Triton Compiler |
| 14 | 2026-10-14 | 2026-10-20 | Baseline |
| 15 | 2026-10-21 | 2026-10-27 | Baseline |
| 16 | 2026-10-28 | 2026-11-03 | Baseline |
| 17 | 2026-11-04 | 2026-11-10 | 性能优化 |
| 18 | 2026-11-11 | 2026-11-17 | 性能优化 |
| 19 | 2026-11-18 | 2026-11-24 | 性能优化 |
| 20 | 2026-11-25 | 2026-12-01 | 性能优化 |
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

## 3. 阅读顺序与 Phase 导航

本页是全局执行契约和总索引；7 个 Phase 文件是逐周、逐日执行手册。按以下顺序使用：

1. 首先阅读本页，固定时间盒、环境变量、目录、进度格式、实验 schema、Stop/Go 和调整规则。
2. 进入当前 Phase 文件，按 Week 顺序执行任务，并把验证命令、artifact 和偏差写入 `progress.md`。
3. 当前 Phase Exit Gate 未通过时，只允许预读下一阶段，不得正式开始、产出阶段结论或挪用后续工时。
4. Gate 通过后，在 `progress.md` 记录证据和 PASS，再进入导航表下一行。

| Phase | Week / 工时 | 详细计划 | 输入 → 核心能力 → 核心产出 | Exit Gate 摘要 |
|---|---:|---|---|---|
| Phase 1：CUDA 与 Triton Kernel 基础 | 1–4 / 44h | [计划](./Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md) · [Gate](./Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md#phase-1-exit-gate) | **输入：** 本地工具链、RTX 4090、§2 目录。<br>**能力：** CUDA/Triton kernel、执行模型与测量纪律。<br>**产出：** 环境记录、CUDA/Triton 基础 kernel 和首批测量 artifact。 | kernel 可运行；profiler 权限确认；能解释线程映射、访存、shared memory 与 occupancy。 |
| Phase 2：MLIR Transformation Bridge | 5–10 / 66h | [计划](./Phases/Phase2_MLIR_Transformation_Bridge.md) · [Gate](./Phases/Phase2_MLIR_Transformation_Bridge.md#phase-2-exit-gate) | **输入：** Phase 1 Gate、upstream MLIR、fused linear 语义。<br>**能力：** rewrite/fold、DialectConversion、结构化 tiling、IR evolution。<br>**产出：** 多 Op transformation、conversion、Transform/C++ tiling 正负例和分阶段 IR。 | 全部正负例通过；能解释 transformation 合法性和语义保持。 |
| Phase 3：Triton Compiler Internals | 11–13 / 33h | [计划](./Phases/Phase3_Triton_Compiler_Internals.md) · [Gate](./Phases/Phase3_Triton_Compiler_Internals.md#phase-3-exit-gate) | **输入：** Phase 2 证据、Triton 源码与 venv。<br>**能力：** 构建、TTIR→TTGIR→LLVM IR→PTX lowering、compiler change。<br>**产出：** 固定 commit/构建记录、lowering 对照、patch/test/reproducer。 | 固定 commit 可构建；artifacts 可复现；compiler change 有 red/green test。 |
| Phase 4：Baseline Freeze | 14–16 / 33h | [计划](./Phases/Phase4_Baseline_Freeze.md) · [Gate](./Phases/Phase4_Baseline_Freeze.md#phase-4-exit-gate) | **输入：** Phase 3 pipeline、fused workload、RTX 4090 profiler。<br>**能力：** correctness、autotune、benchmark、profiler、baseline identity。<br>**产出：** 六组 shape correctness、冻结 config/schema、profile、IR/PTX/SASS、manifest/report。 | correctness 通过；timing/schema 与 baseline artifacts/manifest 冻结。 |
| Phase 5：Performance Optimization | 17–20 / 44h | [计划](./Phases/Phase5_Performance_Optimization.md) · [Gate](./Phases/Phase5_Performance_Optimization.md#phase-5-exit-gate) | **输入：** Phase 4 immutable baseline/profile/测量契约。<br>**能力：** profiler 假设、可开关 compiler 实验、paired benchmark、ablation。<br>**产出：** 正常计划完整预注册 Exp01–03、失败记录和唯一候选。 | **Normal：** 三次完整实验。**Reduced：** 仅限结果前预声明，至少一次完整实验并标记 `PASS-REDUCED`；两者都需完整证据和有效候选。 |
| Phase 6：Generalization & Regression | 21–22 / 22h | [计划](./Phases/Phase6_Generalization_Regression.md) · [Gate](./Phases/Phase6_Generalization_Regression.md#phase-6-exit-gate) | **输入：** Phase 5 `PASS`；或结果前已启用降级、满足 Reduced Gate 的 `PASS-REDUCED`；以及唯一冻结候选。<br>**能力：** 泛化、regression gate、支持范围冻结。<br>**产出：** 泛化矩阵、regression harness、final patch/reproducer/范围。 | 泛化矩阵完整；harness 经 red/green；final patch 和适用范围冻结。 |
| Phase 7：Engineering Delivery | 23–24 / 22h | [计划](./Phases/Phase7_Engineering_Delivery.md) · [Gate](./Phases/Phase7_Engineering_Delivery.md#final-exit-gate) | **输入：** Phase 6 patch/candidate identity、protected manifest、只读 artifacts。<br>**能力：** 冷环境复现、manifest、最终报告与答辩。<br>**产出：** cold replay、运行手册、交付 manifest、报告和交付包。 | 冷环境复现和硬验收通过；文档闭环；能基于 artifact 无稿答辩。 |

计划总计：**24 周，264h**（每周 11h）。

---

## 4. 全局最终验收

Phase 7 的 Final Exit Gate 必须同时满足：

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

## 5. 统一实验模板

<a id="global-benchmark-csv-schema"></a>

### 5.1 全局 benchmark CSV schema

所有 baseline 和优化实验保留原始快照定义的 18 个基础字段及顺序：

```text
timestamp,provider,triton_commit,compiler_patch,gpu,driver,cuda,
M,N,K,dtype,config,round,median_us,p20_us,p80_us,tflops,compile_ms
```

| 字段 | 定义 |
|---|---|
| `timestamp` | 本轮测量记录的时间戳；同一结果集使用统一时区并在环境记录中声明。 |
| `provider` | 被测实现/变体标识，例如 `baseline`、`exp01`；必须能映射到唯一实验目录。 |
| `triton_commit` | 本轮使用的 Triton 40 位 commit；baseline 和实验均以 `baseline_commit.txt` 为来源。 |
| `compiler_patch` | compiler 修改标识；baseline 使用 `none`，实验记录可核验的 patch 路径或内容 hash。 |
| `gpu` | GPU 精确型号，本计划正式数据应为 RTX 4090。 |
| `driver` | NVIDIA driver 版本。 |
| `cuda` | CUDA toolkit/runtime 版本；具体取值来源在环境记录中固定。 |
| `M`、`N`、`K` | fused Linear 主 GEMM 的三个整数维度。 |
| `dtype` | 输入、权重和输出所采用的冻结 dtype 标识。 |
| `config` | 本 shape 已冻结的 Triton launch/autotune config 标识，必须可解析到完整配置。 |
| `round` | 相同 provider、shape、dtype、config 下的独立测量轮次编号。 |
| `median_us` | 本轮 steady-state latency 的 P50，单位微秒。 |
| `p20_us` | 本轮 steady-state latency 的 P20，单位微秒。 |
| `p80_us` | 本轮 steady-state latency 的 P80，单位微秒。 |
| `tflops` | 用本轮 `median_us` 按 `2*M*N*K/latency_seconds/1e12` 计算的主 GEMM effective TFLOPS，不计 bias/ReLU FLOPs。 |
| `compile_ms` | 与 steady-state latency 分离的 compile/autotune host time，单位毫秒。 |

详细 Phase 可以在这 18 列之后追加 `sample_count`、`raw_samples_path`、`timing_backend` 等证据列，但不得删除、重命名或改变基础列顺序。正式结果必须保存 raw samples，并从 raw 值复算 P20/P50/P80；compile/autotune 时间不得混入 latency。

### 5.2 单次实验 README

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

## 6. 阶段检查清单

### [Phase 1](./Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md#phase-1-exit-gate)

- [ ] CUDA vector add/reduction/naive matmul/tiled matmul 可运行
- [ ] Triton vector add/softmax/matmul/fused Linear 可运行
- [ ] RTX 4090 profiler 权限确认
- [ ] 能解释线程映射、访存、shared memory、occupancy

### [Phase 2](./Phases/Phase2_MLIR_Transformation_Bridge.md#phase-2-exit-gate)

- [ ] 能用 IR 样例解释 SSA、use-def、dominance、pass anchor 和 analysis invalidation
- [ ] AddZero red/green 证明自定义 pattern 生效
- [ ] Linear + Bias + ReLU 多 Op rewrite 的正例、multi-use、side-effect 和不匹配测试通过
- [ ] 最小 DialectConversion pass 覆盖 legality、type conversion、materialization 和失败诊断
- [ ] 能解释 DestinationStyleOpInterface、TilingInterface 和 reification 在结构化变换中的职责
- [ ] Transform Dialect matmul tiling 整除/非整除测试通过
- [ ] C++ 参数化 tiling pass 正例/负例/边界测试通过
- [ ] 保存 tiling、bufferization、vectorization 各阶段 IR，并解释每一步的合法性与语义保持条件

### [Phase 3](./Phases/Phase3_Triton_Compiler_Internals.md#phase-3-exit-gate)

- [ ] Triton 固定 commit 源码构建成功
- [ ] TTIR/TTGIR/LLVM/PTX artifact 可复现
- [ ] 一个真实 compiler change 经 red/green test 验证

### [Phase 4](./Phases/Phase4_Baseline_Freeze.md#phase-4-exit-gate)

- [ ] 六组 shape 和十 seed correctness 通过
- [ ] autotune/timing/CSV schema 冻结
- [ ] baseline benchmark/profiler/manifest 完整

### [Phase 5](./Phases/Phase5_Performance_Optimization.md#phase-5-exit-gate)

- [ ] Normal Gate：Exp01–03 三个预注册实验均有假设、测试、correctness、benchmark、profiler、ablation
- [ ] Reduced Gate：仅在结果前预声明；至少一个完整实验，状态为 `PASS-REDUCED`
- [ ] 失败实验未被删除
- [ ] 最佳候选由完整证据选择

### [Phase 6](./Phases/Phase6_Generalization_Regression.md#phase-6-exit-gate)

- [ ] 邻近 shape 和边界 shape 完整
- [ ] regression harness 经预期失败验证
- [ ] final patch 和适用范围冻结

### [Phase 7](./Phases/Phase7_Engineering_Delivery.md#final-exit-gate)

- [ ] 干净环境复现通过
- [ ] 最终报告覆盖所有硬验收
- [ ] 完成无稿自我答辩

---

## 7. 调整规则

### 落后 1 周

保留本周 exit gate，将阅读内容压缩到最直接的官方文档和源码，不删除 correctness/test/profiling。

### 落后 2-3 周

降级不是对失败结果的豁免。只有在看到任何 Phase 5 experiment correctness、benchmark 或 profiler 结果之前，才能在 `progress.md` 和 `docs/optimization_design.md` 同时记录启用日期、落后原因、剩余工时、计划完成的 experiment 集合及“至少一次完整实验”的不变证据要求。记录一旦产生不得根据结果撤销或反向补记；已看到任何实验结果后，禁止再启用降级。

启用后可删除挑战目标和 Exp02/Exp03 迭代，但必须保留：

```text
CUDA/Triton 基础
MLIR Transformation bridge：多 Op rewrite、DialectConversion、结构化 tiling
Triton compiler change
冻结 baseline
至少一次预注册的完整优化实验（compiler test、correctness、benchmark、profiler、ablation）
泛化/回归
最终交付
```

Reduced Gate 只能标记为 `PASS-REDUCED`，不能写正常 `PASS`。进入 Phase 6 仍要求完成的实验产生一个满足同等 correctness、identity、off-drift、paired benchmark、profiler 和 ablation 证据的唯一冻结候选；否则 Phase 5 为 `FAIL`。最终报告必须披露降级启用记录、未执行实验、未满足正常三次实验 Gate，以及因此缩小的结论范围。禁止在实验失败或收益不足后用降级状态逃避 Normal Gate。

### GPU 租赁不可用

可继续做 IR、compiler test、reproducer 和 PTX 静态分析，但 Phase 4 baseline freeze 不得标记完成，最终性能验收顺延。

### 优化未超过 Triton

不改变硬验收。如果达到 90%-100%、完成完整 profiler/ablation 并解释失败原因，项目仍然合格；禁止通过缩减 Triton baseline 能力制造胜出。

### Triton upstream 目录或 API 变化

固定 `compiler/baseline_commit.txt` 后不追逐 main。若必须升级，重新构建、重跑 compiler tests、correctness 和 baseline，并在 manifest 记录迁移原因。

---

## 8. 官方参考入口

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
