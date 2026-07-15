# Phase 2：MLIR Transformation Bridge

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 5–10（2026-08-12 至 2026-09-22）
- 计划工时：66h

## 前置条件

[Phase 1 Exit Gate](./Phase1_CUDA_Triton_Kernel_Fundamentals.md#phase-1-exit-gate) 已通过。

## 可验证阶段目标

- 能从 SSA、use-def 与 pass driver 角度解释 transformation 行为。
- 能实现带正负例测试的 rewrite、fold、DialectConversion 与 tiling。
- 能记录 transformation 前后 IR evolution，并定位失败的 legality 或匹配条件。

## 本阶段产出

- 隔离的 MLIR rewrite/fold 示例与测试。
- producer-consumer transformation 与 DialectConversion 实验。
- Transform Dialect、结构化 op interface 和参数化 tiling 示例。
- 每项 transformation 的输入、输出、负例与验证命令。

### 统一构建约定

本阶段 C++ pass 的可编译源码统一放入 upstream 的 test-only 工具链，学习输入、测试副本和说明文档放在 `$CAPSTONE`：

```text
test/lib/Transforms/<Pass>.cpp          # 可编译源码唯一真源
test/lib/Transforms/CMakeLists.txt      # 源文件与所需 MLIR link library
tools/mlir-opt/mlir-opt.cpp             # pass 声明与注册调用
$CAPSTONE/mlir/tests/*.mlir             # 本项目的 lit/FileCheck 输入
$CAPSTONE/docs/*.md                     # 设计、IR 对照与失败记录
```

每增加一个 pass，都必须先让 `cmake --build $MLIR_BUILD --target mlir-opt` 成功，再用
`$MLIR_BIN/mlir-opt --help` 确认参数已注册。不得维护两份会独立演化的 C++ 实现。

## Week 5：IR、SSA、Use-Def 与 Pass 基础

**Files:**

- Create: `mlir/ir_ssa_walk.mlir`
- Create: `test/lib/Transforms/IRSSAInspectionPass.cpp`
- Modify: `test/lib/Transforms/CMakeLists.txt`
- Modify: `tools/mlir-opt/mlir-opt.cpp`
- Create: `mlir/tests/ir-ssa-inspection.mlir`
- Create: `docs/mlir_ir_ssa_notes.md`

### 工作日

- [ ] **周一：手写多 Region/Block IR**

在 `ir_ssa_walk.mlir` 中至少包含：

```text
builtin.module
func.func
scf.if 的两个 region
scf.for 的 block argument
一个值具有两个 users
一个跨 block 使用且满足 dominance 的值
```

运行：

```bash
$MLIR_BIN/mlir-opt $CAPSTONE/mlir/ir_ssa_walk.mlir -verify-each
```

Expected：IR verifier 返回 0。

**参考资料：**

- 必读：`docs/LangRef.md` 的 “High-Level Structure” 与 region/block 章节 — 建立 Module、Func、Region、Block 与 block argument 的准确层级模型。

- [ ] **周二：画出 SSA use-def 图**

在 `mlir_ir_ssa_notes.md` 对每个 `Value` 写明：owner、type、defining op/block argument、users；区分 op result 与 block argument。

**参考资料：**

- 必读：`include/mlir/IR/Value.h` 中 `Value`、`OpResult`、`BlockArgument` 与 use iterator — 逐项确定值的 owner、定义位置和 users，而不是只从文本名称猜测。

- [ ] **周三：写 inspection pass**

实现 `OperationPass<func::FuncOp>`，使用 `getOperation()->walk(...)` 统计 operation、region、block、value 和 use 数量；输出 deterministic test result。

**参考资料：**

- 必读：`include/mlir/IR/Visitors.h` 中 `walk` API，以及 `include/mlir/IR/Operation.h` 中 region/block/operand/result 访问器 — 选择合适的 walk 顺序并实现可重复的统计结果。

- [ ] **周四：加入 dominance 查询**

使用 `DominanceInfo` 检查选定 definition 是否支配两个 uses；增加一个合法多 block case。

**参考资料：**

- 必读：`include/mlir/IR/Dominance.h` 中 `DominanceInfo` 查询接口 — 理解 op、block 与 value dominance 查询的适用边界。

- [ ] **周五：比较 FuncOp 与 ModuleOp anchor**

分别记录：`getOperation()` 类型、一次 pipeline 中运行次数、可见作用域、analysis 生命周期和并行机会。运行：

```bash
$MLIR_BIN/mlir-opt $CAPSTONE/mlir/ir_ssa_walk.mlir \
  -pass-pipeline='builtin.module(func.func(ir-ssa-inspection))'
```

**参考资料：**

- 必读：`docs/PassManagement.md` 的 operation pass、nesting 与 dynamic pipeline 章节 — 比较 `func.func` 与 `builtin.module` anchor 的调度次数、作用域和并行边界。

### 周末

- [ ] **周六：写正例和 verifier 负例**

正例检查 walk 统计和 dominance 结果；负例使用 `-verify-diagnostics` 构造不满足 dominance 或类型不匹配的 IR，并检查诊断。

**参考资料：**

- 必读：`docs/LangRef.md` 的 IR well-formedness、SSA 与 block 章节 — 构造 verifier 真正会拒绝的 dominance/type 负例并写准 expected diagnostic。

- [ ] **周日：解释 analysis invalidation**

在笔记中回答：

```text
只读 pass 可以 preserve 什么？
rewrite 以后为什么旧 DominanceInfo 可能失效？
PassManager 如何按 anchor 调度 nested pass？
Operation::walk 是否自动跨越 nested regions？
```

**参考资料：**

- 必读：`docs/PassManagement.md` 的 dependent dialects、analysis management 与 pass failure 章节 — 解释 analysis cache、preservation/invalidation 和 nested pass 调度。

### Week 5 Exit Gate

```text
能区分 op result 与 block argument
能从 IR 画出 use-def 和 dominance 关系
inspection pass 的正例/负例测试通过
能解释 FuncOp/ModuleOp anchor 和 analysis invalidation
```

---

## Week 6：隔离 Rewrite、Fold 与 Driver

**Files:**

- Modify: `LearningSteps/Practices/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp`
- Modify: `test/lib/Transforms/AddZeroPatternPass.cpp`
- Modify: `LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir`
- Create: `GPUCodeGenCapstone/mlir/add_zero/rewrite-mechanics.md`

### 工作日

- [ ] **周一：记录当前 green baseline**

```bash
$MLIR_BIN/mlir-opt \
  LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir \
  -pass-pipeline='builtin.module(func.func(add-zero-pattern))' \
  | $MLIR_BIN/FileCheck \
    LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir
```

Expected：exit 0。

**参考资料：**

- 必读：`LearningSteps/Practices/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp` 与 `LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir` — 先读懂当前 pattern、pass anchor 与 CHECK 约束，固定可复现 baseline。

- [ ] **周二：关闭 greedy folding**

在 pass 中使用：

```cpp
GreedyRewriteConfig config;
config.enableFolding(false);
applyPatternsGreedily(getOperation(), std::move(patterns), config);
```

**参考资料：**

- 必读：`include/mlir/Transforms/GreedyPatternRewriteDriver.h` 中 `GreedyRewriteConfig::enableFolding` 与 `applyPatternsGreedily` — 确认关闭的是 driver folding，而不是移除自定义 rewrite pattern。

- [ ] **周三：做 red 验证**

临时移除 `patterns.add<AddZeroPattern>`，重新构建并运行测试。

Expected：FileCheck 失败，输出仍包含 `arith.addi`。保存失败摘要后恢复 pattern 注册。

**参考资料：**

- 必读：`docs/PatternRewriter.md` 的 pattern registration/application 章节 — 让移除 pattern 的 red 结果能够证明改写来源，而非只观察偶然输出。

- [ ] **周四：恢复并做 green 验证**

```bash
cmake --build $MLIR_BUILD --target mlir-opt
$MLIR_BIN/mlir-opt \
  LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir \
  -pass-pipeline='builtin.module(func.func(add-zero-pattern))' \
  | $MLIR_BIN/FileCheck \
    LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir
```

**参考资料：**

- 必读：`test/lib/Transforms/AddZeroPatternPass.cpp` 与 `include/mlir/IR/PatternMatch.h` 中 `RewritePatternSet` — 恢复真实编译入口的 pattern 注册并核对成功改写契约。

- [ ] **周五：比较内建 fold**

阅读：

```text
lib/Dialect/Arith/IR/ArithOps.cpp
include/mlir/Transforms/GreedyPatternRewriteDriver.h
lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
```

记录 pattern、rewriter、driver、fold、DCE 各自职责。

**参考资料：**

- 必读：`lib/Dialect/Arith/IR/ArithOps.cpp` 与 `lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp` — 沿源码区分 op fold、pattern 应用、worklist、DCE 和迭代收敛。

### 周末

- [ ] **周六：补测试边界**

加入 multiple-use constant、non-zero、不同 integer width 和至少一个 shaped integer case；不支持的 case 要说明原因，不能静默忽略。

**参考资料：**

- 必读：`docs/Canonicalization.md` 的 globally applied rules 与 pass pipeline 章节 — 为 multiple-use、non-zero、不同位宽和 shaped case 判断哪些变化可能来自 fold/DCE。

- [ ] **周日：同步双份源码并复盘**

```bash
diff -u LearningSteps/Practices/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp \
  test/lib/Transforms/AddZeroPatternPass.cpp
```

Expected：除明确记录的集成差异外内容一致。长期真实编译入口标记为 `test/lib/Transforms/AddZeroPatternPass.cpp`。

**参考资料：**

- 必读：`test/lib/Transforms/AddZeroPatternPass.cpp` — 确认 test-only 工具链中的唯一可编译真源及其与学习副本的明确差异。

### Week 6 Exit Gate

```text
能够用 red/green 证明确实是自定义 pattern 完成 rewrite
能够解释 replaceOp、DCE 和 fold 的先后关系
能够解释 FuncOp pass anchor
```

---

## Week 7：多 Op Producer-Consumer Transformation

**Files:**

- Create: `test/lib/Transforms/FusedLinearEpiloguePass.cpp`
- Modify: `test/lib/Transforms/CMakeLists.txt`
- Modify: `tools/mlir-opt/mlir-opt.cpp`
- Create: `mlir/tests/fused-linear-epilogue.mlir`
- Create: `mlir/tests/fused-linear-epilogue-invalid.mlir`
- Create: `docs/multi_op_rewrite_notes.md`

### 工作日

- [ ] **周一：构造 Linear + Bias + ReLU IR**

使用 `linalg.matmul` 和两个 elementwise `linalg.generic` 表达：

```text
matmul(A, B) -> add broadcast(bias) -> max(value, 0)
```

同时加入一个 matmul result 具有第二个 user 的 case。

**参考资料：**

- 必读：`include/mlir/Dialect/Linalg/IR/LinalgOps.td` 中 matmul 与 `linalg.generic` 定义 — 按真实 operands、indexing maps 和 iterator semantics 搭出 Linear-Bias-ReLU 链。

- [ ] **周二：实现只读 chain matcher**

从 ReLU consumer 反向检查 defining op、operand index、iterator types、indexing maps、dtype、单 user 条件和 side effects；用 `notifyMatchFailure` 记录失败原因。

**参考资料：**

- 必读：`include/mlir/IR/PatternMatch.h` 中 `PatternRewriter::notifyMatchFailure`、`PatternBenefit` 与 rewrite pattern API — 在修改 IR 前完整验证 defining op、use、类型、映射和 side effect 条件。

- [ ] **周三：加入 PatternBenefit 和受控 pattern 集**

为 epilogue chain pattern 设置明确 benefit；构造一个会与简单 elementwise fold 竞争的 case，记录 greedy driver 选择顺序。

**参考资料：**

- 必读：`docs/PatternRewriter.md` 的 benefit、application recursion 与 bounded rewrite 章节 — 理解 benefit 只控制候选排序，并设计可观测的竞争 case。

- [ ] **周四：实现语义保持的 epilogue fusion**

从 `include/mlir/Dialect/Linalg/Transforms/Transforms.h` 选择当前 checkout 的真实入口：用
`populateElementwiseOpsFusionPatterns` 注册受 `ControlFusionFn` 控制的 greedy patterns，或对已验证的
producer-consumer operand 直接调用 `fuseElementwiseOps`。将 bias 与 ReLU 合并为单个 elementwise
consumer；不在本周尝试将 reduction matmul 强行内联到 elementwise region。

**参考资料：**

- 必读源码：`include/mlir/Dialect/Linalg/Transforms/Transforms.h` 中 `fuseElementwiseOps`、`ControlFusionFn` 与 `populateElementwiseOpsFusionPatterns` — 明确直接 transformation 与 pattern population 两条入口，以及 control callback 的职责。
- Upstream Test：`test/Dialect/Linalg/fusion-elementwise.mlir` — 复用 `-test-linalg-elementwise-fusion-patterns=fuse-generic-ops-control` 的真实测试入口，并检查 multi-result、unused operand 与 side-effect case。

- [ ] **周五：处理 multi-use 和 side-effect 负例**

multi-use、错误 broadcast map、不匹配 dtype、非 ReLU max、额外 side effect 必须拒绝或保持原 IR，不能部分改写。

**参考资料：**

- 必读：`include/mlir/Interfaces/SideEffectInterfaces.h` 与 `include/mlir/IR/Value.h` 的 use 查询 — 把 single-use、side-effect、dtype/map 等拒绝条件落实到真实 API。

### 周末

- [ ] **周六：写结构性 FileCheck**

正例检查 elementwise op 数量减少且计算语义保留；负例检查原链保持。运行目标 pass 两次，确认不会继续改写或无限增长。

**参考资料：**

- 必读：`docs/PatternRewriter.md` 的 mutation、replacement 与 erasure 限制 — 用结构性检查证明 replacement 完整且二次运行达到 fixpoint。

- [ ] **周日：解释 rewrite legality 和终止性**

回答：

```text
为什么 single-use 可能是 fusion 前置条件？
PatternBenefit 如何影响选择但不保证全局最优？
如何证明 rewrite 使 IR 朝终止方向变化？
为什么不能在 match 成功后再发现一半条件不满足？
```

**参考资料：**

- 必读：`lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp` 的 worklist 处理与迭代上限 — 从驱动器行为论证 rewrite 的单调度量、终止性和先 match 后 rewrite 的必要性。

### Week 7 Exit Gate

```text
能匹配真实 producer-consumer chain
正例、multi-use、layout/dtype 和 side-effect 负例通过
能解释 PatternBenefit、legality 和 termination
```

---

## Week 8：DialectConversion、Legality 与 TypeConverter

**Files:**

- Create: `test/lib/Transforms/MiniArithToLLVMConversionPass.cpp`
- Modify: `test/lib/Transforms/CMakeLists.txt`
- Modify: `tools/mlir-opt/mlir-opt.cpp`
- Create: `mlir/tests/mini-arith-to-llvm.mlir`
- Create: `mlir/tests/mini-arith-to-llvm-invalid.mlir`
- Create: `docs/dialect_conversion_notes.md`

### 工作日

- [ ] **周一：阅读真实 conversion pass**

重点阅读：

```text
docs/DialectConversion.md
test/lib/Conversion/FuncToLLVM/TestConvertCallOp.cpp
include/mlir/Conversion/Passes.td
```

画出 ConversionTarget、TypeConverter、ConversionPattern、materialization 与 driver 的关系。

**参考资料：**

- 必读：`docs/DialectConversion.md` 与 `test/lib/Conversion/FuncToLLVM/TestConvertCallOp.cpp` — 从真实 pass 画出 target、type converter、pattern、materialization 与 driver 的协作关系。

- [ ] **周二：定义 ConversionTarget**

构造最小 module：将选定 `arith` op 标记 illegal，将 LLVM dialect 和必要容器 op 标记 legal；分别记录 static、dynamic 和 recursive legality 的用途。

**参考资料：**

- 必读：`include/mlir/Transforms/DialectConversion.h` 中 `ConversionTarget` 和 legality API — 区分 static、dynamic、recursive legality，并精确定义允许残留的容器 op。

- [ ] **周三：接入 LLVMTypeConverter 和 patterns**

创建 `LLVMTypeConverter`，并从各 lowering 的公开 header 分别调用
`arith::populateArithToLLVMConversionPatterns`、`populateFuncToLLVMConversionPatterns` 和
`cf::populateControlFlowToLLVMConversionPatterns`；不手写已有的 LLVM lowering。打印转换前后
function signature、operand/result type。

**参考资料：**

- 必读源码：`include/mlir/Conversion/LLVMCommon/TypeConverter.h` — 理解三个 pattern population 函数共享的 `LLVMTypeConverter` 配置与生命周期。
- 必读源码：`include/mlir/Conversion/ArithToLLVM/ArithToLLVM.h` 中 `arith::populateArithToLLVMConversionPatterns`、`include/mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h` 中 `populateFuncToLLVMConversionPatterns`、`include/mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h` 中 `cf::populateControlFlowToLLVMConversionPatterns` — 使用当前 checkout 的精确公开声明注册 Arith、Func、ControlFlow lowering patterns。

- [ ] **周四：比较 partial 与 full conversion**

在同一 `ConversionTarget` 中保留一个既未标记 legal、也未标记 illegal 的 unknown op，分别调用
`applyPartialConversion` 和 `applyFullConversion`。Expected：partial conversion 成功并允许该
pre-existing unknown op 原样残留；full conversion 失败，因为它要求所有输入 op 最终都被 target
判定为 legal。不要用显式 illegal op 演示差异：显式 illegal op 在两种模式下都必须完成转换。

**参考资料：**

- 必读：`docs/DialectConversion.md` 的 “Modes of Conversion” 与 unknown operation 说明，以及 `include/mlir/Transforms/DialectConversion.h` 中 `applyPartialConversion`/`applyFullConversion` — 精确区分 partial 对 pre-existing unknown op 的容忍和 full 的全量 legality 要求。
- Upstream Test：`test/Transforms/test-legalizer.mlir` 与 `test/Transforms/test-legalizer-full.mlir` — 对照 partial 中未 legalize op 的保留/remark，以及 full 对未获 legality 的 unknown op 的失败诊断。

- [ ] **周五：观察 materialization**

构造需要 source/target materialization 的类型边界，记录 unrealized conversion cast 何时出现、为什么最终需要 reconcile 或合法 materialization。

**参考资料：**

- 必读：`docs/DialectConversion.md` 的 type conversion 与 materialization 章节 — 追踪 source/target materialization、unrealized cast 与 reconcile 的产生条件。

### 周末

- [ ] **周六：写 conversion 正例和失败测试**

覆盖合法 module、残留 illegal op、无法转换类型和动态 legality；失败测试使用 `-verify-diagnostics`。

**参考资料：**

- 必读：`test/lib/Transforms/TestDialectConversion.cpp` 与 `test/Transforms/test-legalize-type-conversion.mlir` — 复用当前 checkout 的 conversion 测试 pass、诊断和 dynamic legality 写法。

- [ ] **周日：写转换不变量**

说明：

```text
哪些 dialect/op 在 pass 后必须全部消失？
容器 op 为什么可能保持 legal？
TypeConverter 为什么不仅转换 result type？
conversion failure 后能否假设 IR 自动 rollback？
```

**参考资料：**

- 必读：`docs/DialectConversion.md` 的 rewrite rollback 与 conversion modes 章节 — 明确哪些 mutation 可回滚、哪些 op 必须消失，以及不能从 failure 推导自动恢复全部外部状态。

### Week 8 Exit Gate

```text
能独立配置 ConversionTarget 和 TypeConverter
能解释 partial/full conversion 和 materialization
正例与残留 illegal op 负例通过
```

---

## Week 9：Transform Dialect 与结构化 Op Interfaces

**Files:**

- Create: `mlir/transform_matmul_tiling.mlir`
- Create: `mlir/tests/transform_matmul_tiling.mlir`
- Create: `docs/mlir_tiling_notes.md`
- Create: `docs/structured_op_interfaces.md`

### 工作日

- [ ] **周一：运行 tiling 测试并定位 interfaces**

```bash
$MLIR_BIN/llvm-lit -sv \
  $MLIR_BUILD/tools/mlir/test/Interfaces/TilingInterface/tile-using-scfforall.mlir
$MLIR_BIN/llvm-lit -sv \
  $MLIR_BUILD/tools/mlir/test/Dialect/Linalg/transform-op-tile.mlir
```

当前 integrated monorepo 布局中，`$MLIR_BUILD` 是 LLVM build root；
`$MLIR_BUILD/tools/mlir/test/lit.site.cfg.py` 是 CMake 生成的 MLIR suite 配置。Lit 会把上述
test exec path 映射回 source-tree test，并在对应 `Output/` 目录执行。

阅读：

```text
include/mlir/Interfaces/DestinationStyleOpInterface.td
include/mlir/Interfaces/TilingInterface.td
include/mlir/Interfaces/InferTypeOpInterface.td
include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td
```

**参考资料：**

- 必读：`test/CMakeLists.txt` 中 `configure_lit_site_cfg`/`add_lit_testsuite` 与 `test/lit.cfg.py` 中 `test_source_root`/`test_exec_root` — 理解为何单测命令传 build-tree exec path，同时由生成的 suite 配置加载 source-tree 测试。
- Upstream Test：`test/Interfaces/TilingInterface/tile-using-scfforall.mlir` 与 `test/Dialect/Linalg/transform-op-tile.mlir` — 由 RUN/CHECK 反查 tiling interface 和 Transform op 的真实入口。

- [ ] **周二：抽取最小 linalg.matmul case**

输入固定为 `tensor<128x256xf32> × tensor<256x512xf32>`，使用 `tensor.empty`、`linalg.fill` 和 `linalg.matmul`。

**参考资料：**

- 必读：`include/mlir/Dialect/Linalg/IR/LinalgOps.td` 与 `docs/Dialects/Linalg/_index.md` — 核对 tensor-semantics matmul、init/destination 与静态 shape 的合法 IR 形式。

- [ ] **周三：加入 named sequence 和 match**

Transform IR 必须只匹配目标 `linalg.matmul`，不能依赖文件中只有一个 op。

**参考资料：**

- 必读：`docs/Tutorials/transform/Ch1.md` 与 `docs/Dialects/Transform.md` 的 named sequence/matching 部分 — 让 handle 通过结构和约束命中目标，而不依赖文件中 op 的数量或顺序。

- [ ] **周四：tile 为 scf.forall/scf.for**

第一版 tile size 固定 `64x64x32`，保存 transformation 前后 IR。

**参考资料：**

- 必读：`docs/Tutorials/transform/Ch2.md` 与 `test/Dialect/Linalg/transform-op-tile.mlir` — 从可运行语法选择 `scf.forall`/`scf.for` tiling 参数并保存前后 IR。

- [ ] **周五：解释 DestinationStyle、Tiling 与 Reify Interfaces**

阅读：

```text
include/mlir/Interfaces/TilingInterface.h
lib/Dialect/SCF/Transforms/TileUsingInterface.cpp
```

写出 destination/init、iteration domain、tiled implementation、result replacement 和 shape reification 的关系；说明 tiling algorithm 为什么依赖接口而不是硬编码每个 Linalg op。

**参考资料：**

- 必读：`include/mlir/Interfaces/DestinationStyleOpInterface.td`、`include/mlir/Interfaces/TilingInterface.td`、`include/mlir/Interfaces/InferTypeOpInterface.td` 与 `lib/Dialect/SCF/Transforms/TileUsingInterface.cpp` — 沿接口方法到实现串起 destination、domain、tile、replacement 与 shape reification。

### 周末

- [ ] **周六：加入边界 shape**

增加 `127x251x509`，检查生成的 offset/size/min 逻辑；不得只测试整除 tile。

**参考资料：**

- 必读：`lib/Dialect/SCF/Transforms/TileUsingInterface.cpp` 中 tile size/offset/bounds 计算 — 验证非整除 shape 的尾块由动态 min/bounds 逻辑产生。

- [ ] **周日：写 FileCheck**

检查：

```text
scf.forall/scf.for 数量
step/tile size
tensor.extract_slice 边界
tiled linalg.matmul
result insertion
```

**参考资料：**

- 必读：`test/Interfaces/TilingInterface/tile-using-scfforall.mlir` 的 CHECK 组织方式 — 为 loop、slice、tiled op 和 insertion 选择稳定的结构性断言。

### Week 9 Exit Gate

```text
两个 upstream lit 测试通过
自定义 tiling case 的整除/非整除测试通过
能解释 DestinationStyleOpInterface、TilingInterface 和 shape reification 的分工
能解释 tile transformation 不是简单文本替换
```

---

## Week 10：C++ 参数化 Tiling 与 IR Evolution

**Files:**

- Create: `test/lib/Transforms/MatmulTilingPass.cpp`
- Modify: `test/lib/Transforms/CMakeLists.txt`
- Modify: `tools/mlir-opt/mlir-opt.cpp`
- Create: `mlir/tests/matmul-tiling-pass.mlir`
- Create: `mlir/tests/matmul-tiling-invalid.mlir`
- Update: `docs/mlir_tiling_notes.md`
- Create: `mlir/tests/matmul-tiling-bufferize-vectorize.mlir`

### 工作日

- [ ] **周一：定义 pass contract**

固定 pass 参数：

```text
tile-m
tile-n
tile-k
```

前置条件：正整数、目标为 `linalg.matmul`、静态 rank-2 tensor semantics。其他 case 产生诊断或明确跳过。

**参考资料：**

- 必读：`include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h` 与 `include/mlir/Interfaces/TilingInterface.h` — 由 API 的 options/result 类型定义 pass 参数、支持范围和失败契约。

- [ ] **周二：写失败测试**

为 `tile-m=0`、缺失目标 op 和非 rank-2 case 写 expected-error/负例。

**参考资料：**

- 必读：`docs/PassManagement.md` 的 pass options 与 failure handling 章节 — 为非法 tile size、缺失目标与不支持 rank 设计可验证且一致的诊断策略。

- [ ] **周三：实现最小 C++ pass**

使用 `TilingInterface`/SCF tiling API，不复制 Linalg 内部 tiling 实现。

**参考资料：**

- 必读：`lib/Dialect/SCF/Transforms/TileUsingInterface.cpp` 与 `include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h` — 直接调用 upstream tiling API，并正确处理 failure/result replacement。

- [ ] **周四：加入整除 FileCheck**

检查 tile size、循环层级和 tiled op。

**参考资料：**

- 必读：`test/Interfaces/TilingInterface/tile-using-scfforall.mlir` — 复用对 tile size、循环层级和 tiled implementation 的稳定 FileCheck 模式。

- [ ] **周五：加入非整除 FileCheck**

检查尾块尺寸由原始 shape 与 offset 计算得出。

**参考资料：**

- 必读：`test/Dialect/Linalg/transform-op-tile.mlir` 中非整除/动态边界 cases — 核对尾块 size 与原始 shape、offset 的关系，不把常量整除结果写死。

### 周末

- [ ] **周六：构建并运行全部测试**

将 pass 接入与 AddZero 相同的 test-only 构建路径，运行目标 build 和三个测试文件。增加一条 IR evolution pipeline：

```text
linalg.matmul
  -> tiled linalg + scf
  -> one-shot-bufferize
  -> vectorization candidate / vector IR
```

记录 tiling 前后 destination、extract/insert slice、buffer allocation/copy 和 vector op 变化；此周不要求完整 lowering 到可执行 GPU binary。

**参考资料：**

- 必读：`docs/Bufferization.md`、`docs/Dialects/Vector.md`、`test/Dialect/Linalg/one-shot-bufferize.mlir` 与 `test/Dialect/Linalg/transform-op-vectorize.mlir` — 记录 tiling 后 destination/slice 如何进入 one-shot bufferization，并识别真实 vectorization candidate 与 vector IR。

- [ ] **周日：写 transformation legality 说明**

必须回答：

```text
为什么 tile size 必须校验？
哪些 op 由 transformation 新建？
原 op 的 uses 如何替换？
失败时是否修改了部分 IR？
哪些 analysis 被 invalidated？
```

**参考资料：**

- 必读：`include/mlir/IR/PatternMatch.h` 的 replacement/mutation 契约与 `docs/PassManagement.md` 的 analysis invalidation 章节 — 逐项说明新建 op、use replacement、失败原子性和 analysis 失效。

### Phase 2 Exit Gate

```text
能够从空文件实现参数化 matmul tiling pass
正例、负例、边界测试通过
能够解释 folding/pattern/conversion/tiling/legality 的区别
能够说明 tiling 如何影响后续 bufferization/vectorization
```

---

[进入 Phase 3：Triton Compiler Internals →](./Phase3_Triton_Compiler_Internals.md)
