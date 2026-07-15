# RTX 4090 GPU CodeGen Execution Plan Phase Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RTX 4090 GPU CodeGen 24 周执行计划拆成一个总索引和七个 Phase 文档，并为每个逐日 Step 配置带阅读目的的分级参考资料。

**Architecture:** 总索引只承担全局约束、日历、目录、导航、统一模板和总验收；七个 Phase 文件各自承担连续周次的任务、资料和 Exit Gate。迁移以现有大文档为唯一内容源，先创建并验证 Phase 文件，再缩减总索引，避免任务丢失。

**Tech Stack:** Markdown、MLIR/LLVM 源码与 tests、Triton/CUDA/NVIDIA 官方文档、`rg`、`awk`、`git diff --check`。

---

## File Map

**Modify:**

- `LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`：缩减为总索引。

**Create:**

- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md`：Week 1-4。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md`：Week 5-10。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md`：Week 11-13。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase4_Baseline_Freeze.md`：Week 14-16。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase5_Performance_Optimization.md`：Week 17-20。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase6_Generalization_Regression.md`：Week 21-22。
- `LearningSteps/Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md`：Week 23-24。

**Design source:**

- `docs/superpowers/specs/2026-07-14-rtx4090-execution-plan-phase-split-design.md`

本任务不创建 Git commit；每个 Task 结束时用限定路径的 `git diff --check` 和统计快照代替提交检查点。

---

### Task 1: 冻结迁移基线

**Files:**

- Read: `LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`
- Create: `/tmp/RTX4090_GPU_CodeGen_24Week_Execution_Plan.before-split.md`

- [ ] **Step 1: 保存迁移前只读快照**

Run:

```bash
cp LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  /tmp/RTX4090_GPU_CodeGen_24Week_Execution_Plan.before-split.md
```

Expected：`cmp` 确认快照与当前执行计划相同；后续从快照计算迁移基线。

- [ ] **Step 2: 记录 Week、checkbox、命令块和 Exit Gate 数量**

Run:

```bash
PLAN=/tmp/RTX4090_GPU_CodeGen_24Week_Execution_Plan.before-split.md
rg -c '^## Week ([1-9]|1[0-9]|2[0-4])：' "$PLAN"
rg -c '^- \[ \]' "$PLAN"
rg -c '^### (Week [0-9]+|Phase [1-7]) Exit Gate' "$PLAN"
awk '/^```/{n++} END{print n}' "$PLAN"
```

Expected：Week 标题为 24；其余三个数字作为迁移后的最低覆盖基线，并可随时从快照重新计算。

- [ ] **Step 3: 检查原始结构清单**

Run:

```bash
rg -n '^## Week|^### (Week [0-9]+|Phase [1-7]) Exit Gate|^- \[ \]' \
  /tmp/RTX4090_GPU_CodeGen_24Week_Execution_Plan.before-split.md
```

Expected：输出覆盖 Week 1-24 和所有逐日 checkbox；快照不属于 Git 跟踪文件。

- [ ] **Step 4: 检查工作树边界**

Run:

```bash
git status --short -- LearningSteps docs/superpowers
```

Expected：识别用户已有修改；后续只编辑 File Map 中列出的执行计划文件和本 plan/spec。

---

### Task 2: 创建 Phase 目录和统一文档骨架

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase4_Baseline_Freeze.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase5_Performance_Optimization.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase6_Generalization_Regression.md`
- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md`

- [ ] **Step 1: 为每个文件写固定头部**

Each file must start with:

```markdown
# Phase N：[阶段名称]

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

**周期：** Week X-Y，共 Zh
**前置条件：** 上一 Phase Exit Gate 已通过
**阶段目标：** 一句话描述可验证能力

## 本阶段产出

## Week X

## Phase N Exit Gate

## 下一阶段
```

Expected：七个文件都有返回链接、周期、前置条件、产出、Week、Exit Gate 和下一阶段入口。

- [ ] **Step 2: 校验文件名和返回链接**

Run:

```bash
for f in LearningSteps/Plans/GPU_CodeGen/Phases/*.md; do
  test -f "$f"
  rg -q '\[← 返回 24 周总索引\]\(../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md\)' "$f"
done
```

Expected：七个文件全部通过。

---

### Task 3: 迁移 Phase 1 并补充 Kernel 基础资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md`
- Source section: original plan `## 3. Phase 1` through `### Phase 1 Exit Gate`

- [ ] **Step 1: 迁移 Week 1-4 原文**

Preserve：所有 Files、周一至周日 checkbox、命令、Expected、Week Exit Gate 和 Phase Exit Gate。

- [ ] **Step 2: 为 Week 1-2 Step 配置资料**

Use and annotate:

- NVIDIA CUDA C++ Programming Guide：thread hierarchy、memory hierarchy、execution model。
- CUDA C++ Best Practices Guide：timing、warm-up、coalescing、occupancy。
- Triton official vector-add tutorial：program id、block pointer、mask。
- Triton official fused-softmax tutorial：row-wise mapping 和 fusion。
- Nsight Compute CLI/User Guide：权限、metric collection 和 report export。
- Local artifacts：`cuda/vector_add.cu`、`triton_kernels/vector_add.py` 和 environment notes 的验证目的。

- [ ] **Step 3: 为 Week 3-4 Step 配置资料**

Use and annotate:

- CUDA Programming Guide 的 shared memory、synchronization 和 matrix multiply 示例。
- CUDA Best Practices Guide 的 shared-memory bank conflict 和 effective bandwidth。
- Triton matrix multiplication tutorial：blocked matmul、grouped ordering、autotune。
- Triton layer normalization/fused-attention tutorials中与 reduction/fusion 直接相关的部分；弱相关内容不加入必读。
- Nsight Compute Profiling Guide：Memory Workload Analysis、Scheduler Statistics 和 Roofline 的用途。

- [ ] **Step 4: 验证 Phase 1 覆盖**

Run:

```bash
F=LearningSteps/Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md
rg -c '^## Week [1-4]：' "$F"
rg -n '必读文档|Upstream Test|实践参考|扩展阅读' "$F"
```

Expected：四个 Week 各出现一次；每个逐日 Step 至少有一项“必读”资料及阅读目的。

---

### Task 4: 迁移 Phase 2 并补充 MLIR Transformation 资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md`
- Source section: original plan `## 4. Phase 2` through `### Phase 2 Exit Gate`

- [ ] **Step 1: 迁移 Week 5-10 原文**

Preserve：统一构建约定、IR/SSA、Rewrite、multi-op、DialectConversion、interfaces、Transform Dialect、C++ tiling 和 IR evolution 全部内容。

- [ ] **Step 2: 为 Week 5 IR/Pass Step 配置本地资料**

Use exact local anchors:

- `docs/LangRef.md`：Operation、Region、Block、Value 和 block argument。
- `include/mlir/IR/Operation.h`、`Value.h`、`Visitors.h`：owner/use/walk API。
- `include/mlir/IR/Dominance.h`：`DominanceInfo`。
- `docs/PassManagement.md`：pass anchor、nesting、analysis preservation。
- `test/IR/` 和 `test/Pass/` 中与 verifier、walk、pass pipeline 对应的最小测试。

- [ ] **Step 3: 为 Week 6-8 Rewrite/Conversion Step 配置本地资料**

Use exact local anchors:

- `docs/PatternRewriter.md`、`docs/Canonicalization.md`。
- `include/mlir/IR/PatternMatch.h`：`OpRewritePattern`、`PatternBenefit`、`notifyMatchFailure`。
- `include/mlir/Transforms/GreedyPatternRewriteDriver.h` 和实现文件：fold/worklist/strictness。
- `docs/DialectConversion.md`、`include/mlir/Transforms/DialectConversion.h`。
- `test/Transforms/test-legalize-patterns.mlir` 与 `test/lib/Transforms/TestDialectConversion.cpp`：legality、partial/full conversion 和失败测试。
- Existing `LearningSteps/Transforms/PatternRewriter_Summary.md`、`DialectConversion_Summary.md`：实践参考，不替代官方必读。

- [ ] **Step 4: 为 Week 9-10 Interfaces/Tiling Step 配置本地资料**

Use exact local anchors:

- `include/mlir/Interfaces/DestinationStyleOpInterface.td`。
- `include/mlir/Interfaces/TilingInterface.td` 和 `.h`。
- `include/mlir/Interfaces/InferTypeOpInterface.td`。
- `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp`。
- `test/Interfaces/TilingInterface/tile-using-scfforall.mlir`。
- `test/Dialect/Linalg/transform-op-tile.mlir`。
- `docs/Dialects/Transform.md` 和 `docs/Tutorials/transform/`。
- Bufferization/Vectorization 官方文档与对应 tests，阅读重点限定为 tiling 后 IR 如何继续演化。

- [ ] **Step 5: 校验所有本地引用存在**

Run：抽取 Phase 2 中反引号包裹且以 `docs/`、`include/`、`lib/`、`test/`、`LearningSteps/` 开头的路径，逐项执行 `test -e`；符号名不作为文件路径检查。

Expected：所有声明为本地文件的引用存在；不存在的候选必须改成当前 checkout 的真实路径。

---

### Task 5: 迁移 Phase 3 并补充 Triton Compiler 资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md`
- Source section: original plan `## 5. Phase 3` through `### Phase 3 Exit Gate`

- [ ] **Step 1: 迁移 Week 11-13 原文**

Preserve：固定 commit、源码构建、IR dump、pipeline 对照、compiler change 和 red/green test。

- [ ] **Step 2: 配置 Triton 官方资料和固定源码入口**

Use and annotate:

- Triton repository build-from-source instructions。
- Triton language/programming guide。
- `python/triton/compiler/compiler.py`：compile stages 和 artifact flow；路径以固定 commit 验证。
- `lib/Dialect/Triton/Transforms/`、`lib/Dialect/TritonGPU/Transforms/`：TTIR/TTGIR pass；路径以固定 commit 验证。
- Triton repository tests 中目标 pass 的 lit/pytest test。
- Triton MLIR dialect/layout documentation。
- LLVM NVPTX backend and PTX ISA documentation：LLVM IR/NVVM/PTX 边界。

- [ ] **Step 3: 为 compiler change Step 明确源码阅读链**

Reference chain must identify：pass registration → pass implementation → op/interface definition → test → generated TTGIR/PTX artifact。每个链路写明本周要回答的问题，禁止仅列目录。

- [ ] **Step 4: 标注版本边界**

Expected：所有 Triton 源码路径注明“在 `compiler/baseline_commit.txt` 固定后核验”；不把 main 分支当前路径当作永久 API。

---

### Task 6: 迁移 Phase 4 并补充 correctness/benchmark/profiler 资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase4_Baseline_Freeze.md`
- Source section: original plan `## 6. Phase 4` through `### Phase 4 Exit Gate`

- [ ] **Step 1: 迁移 Week 14-16 原文**

Preserve：六组 shape、十 seed、容差、autotune、三轮 timing、CSV schema、profiler、IR/SASS artifact 和 cold-shell freeze。

- [ ] **Step 2: 配置 correctness 资料**

Use and annotate：PyTorch numerical accuracy note、`torch.testing.assert_close` documentation、Triton testing helpers、CUDA compute-sanitizer documentation，以及 fused epilogue 的 reference expression。

- [ ] **Step 3: 配置 benchmark/profiler 资料**

Use and annotate：Triton benchmarking API/tutorial、CUDA Events timing guidance、Nsight Compute CLI/User Guide、Nsight Systems User Guide、CUDA Binary Utilities `cuobjdump`/`nvdisasm` documentation。

- [ ] **Step 4: 校验基线冻结规则**

Expected：每个正式性能 Step 均引用 warm-up、重复次数、统计量、GPU 状态或 profiler metric 的权威依据；correctness 和性能资料明确分开。

---

### Task 7: 迁移 Phase 5 并补充性能归因资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase5_Performance_Optimization.md`
- Source section: original plan `## 7. Phase 5` through `### Phase 5 Exit Gate`

- [ ] **Step 1: 迁移 Week 17-20 原文**

Preserve：唯一主瓶颈、三个 experiment、hypothesis、compiler test、correctness、benchmark、profiler、ablation、失败实验和候选冻结。

- [ ] **Step 2: 按实验类型配置资料**

Use and annotate：

- Nsight Compute Metrics Guide：memory throughput、occupancy、scheduler、instruction statistics。
- CUDA Best Practices Guide：coalescing、shared memory、register pressure 和 occupancy trade-off。
- PTX ISA：vectorized memory op、predicate、barrier 和 relevant instruction semantics。
- Triton layout/encoding、conversion 和 GPU lowering source/tests：每个实验绑定实际修改点。
- Roofline 原始论文或 NVIDIA 官方 roofline guidance：只作为归因框架，不替代 measured evidence。

- [ ] **Step 3: 为每个实验 Step 建立资料到证据的映射**

Expected：每项资料对应 hypothesis、要观察的 metric 或要修改的 compiler symbol；不存在与实验无关的通用书单。

---

### Task 8: 迁移 Phase 6 并补充泛化/回归资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase6_Generalization_Regression.md`
- Source section: original plan `## 8. Phase 6` through `### Phase 6 Exit Gate`

- [ ] **Step 1: 迁移 Week 21-22 原文**

Preserve：邻近 shape、边界 shape、几何平均、最坏退化、expected failure、regression harness 和 final patch scope。

- [ ] **Step 2: 配置统计与测试资料**

Use and annotate：Python `statistics` documentation、Triton testing/benchmark helpers、pytest expected-failure/parameterization documentation，以及 CUDA/Triton 对非整除 shape 和 mask 的官方说明。

- [ ] **Step 3: 强化回归门槛**

Expected：资料直接支撑 geometric mean、worst-case regression、red/green harness 和边界 correctness；明确不能用算术平均掩盖 shape 退化。

---

### Task 9: 迁移 Phase 7 并补充可复现交付资料

**Files:**

- Create: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md`
- Source section: original plan `## 9. Phase 7` through Week 24 completion gate

- [ ] **Step 1: 迁移 Week 23-24 原文**

Preserve：cold environment、README、manifest、artifact、最终报告、自我答辩和完成定义。

- [ ] **Step 2: 配置工程交付资料**

Use and annotate：Python packaging/venv documentation、Git bundle/patch documentation、NVIDIA environment query commands、Triton build instructions、MLIR testing guide，以及可复现实验报告所需的 manifest 字段来源。

- [ ] **Step 3: 校验独立复现路径**

Expected：每个交付 Step 指向明确的安装、构建、测试、benchmark 或 artifact reference；不依赖作者记忆和交互式修复。

---

### Task 10: 在线核验外部资料

**Files:**

- Modify: all seven Phase files

- [ ] **Step 1: 核验官方入口**

Browse primary sources only for MLIR/LLVM/Triton/CUDA/Nsight/PyTorch/Python/pytest technical references. Open each selected page and verify title, scope and current URL.

Expected：所有外链直接指向支撑当前 Step 的页面，不指向搜索结果或不稳定的聚合页。

- [ ] **Step 2: 筛选实践参考**

Prefer official tutorials. A non-official blog/video may be included only if it adds a concrete worked example absent from official material; label it “实践参考”或“扩展阅读”，不得作为唯一必读依据。

- [ ] **Step 3: 标注核验和版本信息**

At each Phase reference-policy note record：`外部链接核验日期：2026-07-14`。Triton source paths additionally depend on `compiler/baseline_commit.txt`。

---

### Task 11: 将原大文档缩减为总索引

**Files:**

- Modify: `LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`

- [ ] **Step 1: 保留全局章节**

Keep：header、使用规则、日历、完成定义、Stop/Go、固定路径和目录、`progress.md` 格式、统一实验模板、阶段检查清单、调整规则和官方参考入口。

- [ ] **Step 2: 用 Phase 导航表替换逐周正文**

Table columns：Phase、Week、工时、核心能力、Exit Gate、文档链接。七个链接必须指向 File Map 中的相对路径。

- [ ] **Step 3: 添加阅读顺序说明**

State：先读总索引全局规则，再进入当前 Phase；Phase Exit Gate 未通过时只允许预读下一 Phase，不正式开始。

- [ ] **Step 4: 检查索引尺寸和职责**

Run:

```bash
wc -l LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
rg -n '^## Week ' LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
```

Expected：总索引约 300-500 行，允许因完整全局模板略超；不再包含 Week 1-24 逐日正文。

---

### Task 12: 执行完整性和 Markdown 验证

**Files:**

- Verify: total index and all seven Phase files

- [ ] **Step 1: 验证 Week 唯一性**

Run:

```bash
rg '^## Week ([1-9]|1[0-9]|2[0-4])：' LearningSteps/Plans/GPU_CodeGen/Phases/*.md \
  | sed -E 's/.*Week ([0-9]+)：.*/\1/' | sort -n | uniq -c
```

Expected：1 到 24 每个数字的计数均为 1。

- [ ] **Step 2: 对比迁移前后 checkbox 和 Exit Gate**

Run：统计七个 Phase 文件的 `^- \[ \]` 和 Exit Gate 数量，与 `/tmp/RTX4090_GPU_CodeGen_24Week_Execution_Plan.before-split.md` 的相同统计对比。

Expected：checkbox 不少于迁移前，因为资料细化可以增加检查项；原有 Exit Gate 数量不得减少。

- [ ] **Step 3: 验证每个 Step 都有资料**

Run：逐 Phase 检查相邻 checkbox 之间存在 `参考资料` 和至少一个 `必读` 项；对多行区块使用小型 `awk` validator，不以全文总计代替逐 Step 检查。

Expected：所有逐日 Step 通过；Phase checklist checkbox 不强制使用逐 Step 资料模板。

- [ ] **Step 4: 验证本地链接和路径**

Run：检查 Markdown 相对链接目标；抽取声明为本地文件的路径并执行 `test -e`。带变量的未来 capstone artifact 只检查父目录规范，不误判为当前已存在文件。

Expected：所有现有文档/源码/test 引用可解析，七个导航链接双向有效。

- [ ] **Step 5: 验证代码围栏和格式**

Run:

```bash
for f in LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
         LearningSteps/Plans/GPU_CodeGen/Phases/*.md; do
  awk '/^```/{n++} END{if (n % 2) {print FILENAME ": odd fences"; exit 1}}' "$f"
done
git diff --check -- LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  LearningSteps/Plans/GPU_CodeGen/Phases docs/superpowers
```

Expected：无 odd fences、尾随空白或 patch 格式错误。

- [ ] **Step 6: 最终差异审阅**

Run:

```bash
git diff --stat -- LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  LearningSteps/Plans/GPU_CodeGen/Phases docs/superpowers
git status --short
```

Expected：差异只覆盖设计批准的文档；不提交、不覆盖其他用户修改。
