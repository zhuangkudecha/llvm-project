# Illustrated MLIR Transformation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a runnable, Chinese, diagram-led Transformation booklet that separately explains MLIR Pass infrastructure, Pattern Rewriting, Dialect Conversion, and Transform Dialect, then reconnects them through one `linalg.matmul` pipeline.

**Architecture:** Add a self-contained booklet under `mlir/LearningSteps/IllustratedMLIR/Transformation/` without modifying the existing technical summaries. Two checked-in MLIR inputs are the source of truth: one pure payload example and one payload-plus-Transform-sequence example; seven Markdown chapters explain the four subsystems with stable semantic colors, current source anchors, and locally verified commands.

**Tech Stack:** Markdown, Mermaid 11, MLIR textual IR, `mlir-opt`, Linalg, SCF, Bufferization, Transform Dialect, MLIR C++/TableGen sources

---

## Working Directory and Guard Rails

Run every command from the LLVM monorepo root:

```bash
cd /home/fuhao/llvm-project
```

Only add files below these paths:

```text
mlir/LearningSteps/IllustratedMLIR/Transformation/
mlir/docs/superpowers/plans/2026-07-15-illustrated-mlir-transformation.md
```

Do not modify existing files below `mlir/LearningSteps/Transforms/`. The worktree already contains unrelated user changes, so every commit must use an exact `git add` path list rather than `git add .`.

## File Structure

- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/00_Reading_Map.md`: booklet entry point, subsystem map, reading routes, color legend, run commands, and status.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/01_What_Transforms_IR.md`: common taxonomy and responsibility boundaries.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/02_Pass_Infrastructure.md`: PassManager, nested pipelines, pass boundaries, analyses, and failure.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/03_Pattern_Rewriter.md`: RewritePattern, PatternRewriter, drivers, worklist, and mutation contract.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/04_Dialect_Conversion.md`: legality, conversion patterns, adaptor, TypeConverter, and materialization.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/05_Transform_Dialect.md`: dual IR, handles, interpreter, effects, failure, and invalidation.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/06_End_To_End_Matmul.md`: replay the complete matmul pipeline and attribute each state transition to its mechanism.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir`: pure tensor-semantics payload with one removable dead constant.
- Create `mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir`: the same payload plus a named Transform sequence that matches and tiles `linalg.matmul`.

## Shared Content Rules

- All reader-facing headings, Mermaid nodes, edges, legends, captions, and walkthroughs are Chinese.
- Exact identifiers such as `PassManager`, `PatternRewriter`, `ConversionTarget`, `transform.sequence`, and `linalg.matmul` stay in English and receive Chinese explanations.
- Use light Mermaid `base` theme, white background, dark text, and these stable classes:

```mermaid
classDef operation fill:#dbeafe,stroke:#2563eb,color:#1e3a8a;
classDef value fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
classDef region fill:#dcfce7,stroke:#16a34a,color:#14532d;
classDef block fill:#fef3c7,stroke:#d97706,color:#78350f;
classDef dialect fill:#f3e8ff,stroke:#9333ea,color:#581c87;
classDef context fill:#f1f5f9,stroke:#64748b,color:#334155;
```

- Blue means Operation/payload IR, red means Value/SSA/Transform handle, green means Region, yellow means Block, purple means Dialect/Type/semantics, and gray means Pass/driver/tool/source context.
- Every Mermaid diagram is followed by prose that names the exact edge, ownership relation, or state transition to inspect.
- Every mechanism chapter follows: question → before/after IR → responsibility diagram → execution timeline → boundaries/misconceptions → source anchors → runnable verification → recap diagram.
- Generated normalized IR and rendered SVG files stay under `/tmp` and are not committed.

### Task 1: Add and Validate the Shared Matmul Inputs

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir`
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir`

- [ ] **Step 1: Create the pure payload example**

Use this exact content:

```mlir
module {
  func.func @matmul(%lhs: tensor<8x16xf32>, %rhs: tensor<16x4xf32>)
      -> tensor<8x4xf32> {
    %zero = arith.constant 0.0 : f32
    %dead = arith.constant 1 : index
    %init = tensor.empty() : tensor<8x4xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<16x4xf32>)
        outs(%filled : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }
}
```

- [ ] **Step 2: Create the payload-plus-Transform-sequence example**

Use this exact content:

```mlir
module attributes {transform.with_named_sequence} {
  func.func @matmul(%lhs: tensor<8x16xf32>, %rhs: tensor<16x4xf32>)
      -> tensor<8x4xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8x4xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<16x4xf32>)
        outs(%filled : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %matmul
        tile_sizes [4, 2]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
```

- [ ] **Step 3: Verify both files parse**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  -o /tmp/illustrated-transformation-matmul-parsed.mlir
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  -o /tmp/illustrated-transformation-matmul-transform-parsed.mlir
```

Expected: both commands exit `0` without diagnostics.

- [ ] **Step 4: Verify the intended transformations before writing prose**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
  -o /tmp/illustrated-transformation-matmul-clean.mlir
! rg -n "arith.constant 1" /tmp/illustrated-transformation-matmul-clean.mlir

build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  --transform-interpreter \
  -o /tmp/illustrated-transformation-matmul-tiled.mlir
rg -n "scf.forall|linalg.matmul" \
  /tmp/illustrated-transformation-matmul-tiled.mlir

build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-loops \
  -o /tmp/illustrated-transformation-matmul-lowered.mlir
rg -n "memref.alloc|scf.for" \
  /tmp/illustrated-transformation-matmul-lowered.mlir
! rg -n "linalg.matmul|tensor<" \
  /tmp/illustrated-transformation-matmul-lowered.mlir
```

Expected: the dead constant is absent; the tiled output contains `scf.forall`; the lowered output contains `memref.alloc` and `scf.for`, with no `linalg.matmul` or tensor type.

- [ ] **Step 5: Commit the two source-of-truth examples**

```bash
git add \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir
git commit -m "docs: add illustrated MLIR transformation examples"
```

### Task 2: Create the Reading Map and Common Transformation Taxonomy

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/00_Reading_Map.md`
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/01_What_Transforms_IR.md`
- Read: `mlir/LearningSteps/IllustratedMLIR/00_Reading_Map.md`
- Read: `mlir/LearningSteps/IllustratedMLIR/01_Why_Everything_Is_An_Operation.md`
- Read: `mlir/LearningSteps/Transforms/Phase2_AI_Compiler_Beginner_Guide.md`
- Read: `mlir/LearningSteps/Transforms/Phase2_Core_Infrastructure_Summary.md`

- [ ] **Step 1: Write the reading map skeleton and navigation**

Use these headings in order:

```markdown
# 图解 MLIR Transformation：阅读地图
## 这套小册解决什么问题
## 四个子系统各自负责什么
## 推荐阅读顺序
## 按问题查阅
## 图中的颜色约定
## 如何运行共享示例
## 与现有学习总结的关系
## 当前完成状态
```

List and link chapters `01` through `06`. Link both example files. Link all six existing files under `../Transforms` through the correct relative path `../../Transforms/...`.

- [ ] **Step 2: Add the overview diagram to the reading map**

Represent these exact relationships in Chinese Mermaid labels:

```text
PassManager / pipeline -> 在 Operation 边界运行 Pass
Pass -> 调用 Pattern Driver
Pass -> 调用 Conversion Driver
Transform Dialect Interpreter -> 解释 Transform IR
Transform IR handle -> 选择 payload Operation
Pattern Driver -> 修改 payload IR
Conversion Driver -> 修改 payload IR
所有路径 -> 同一个 linalg.matmul payload IR
```

Follow the diagram with prose that asks the reader to inspect the difference between scheduling arrows and IR-mutation arrows.

- [ ] **Step 3: Write the common taxonomy chapter**

Use these headings in order:

```markdown
# MLIR 中到底是谁在变换 IR
## 从同一段 Before / After IR 开始
## 四个问题对应四个机制
## Pass：安排在哪里和何时运行
## PatternRewriter：执行一条局部改写
## Dialect Conversion：保证目标 IR 合法
## Transform Dialect：用 IR 编排变换
## 四者如何连接
## 四个最容易混淆的说法
## 当前源码入口
## 运行共享示例
## 一张图总结本章
## 继续阅读
```

Include this responsibility table exactly:

| 机制 | 它回答的问题 | 它不等于什么 |
|---|---|---|
| Pass 基础设施 | 何时、在哪个 Operation 上运行什么？ | 一条具体 RewritePattern |
| Pattern Rewriting | 一条局部规则怎样匹配并安全修改 IR？ | 完整 pipeline 调度器 |
| Dialect Conversion | 怎样判断并完成目标合法化？ | 普通 canonicalization 的别名 |
| Transform Dialect | 怎样用 Transform IR 选择并编排 payload 变换？ | 底层改写算法的替代品 |

Add one Mermaid responsibility map and one execution timeline. Explicitly state that `include/mlir/Transform/` is not the Pattern Rewriting framework or the Transform Dialect.

- [ ] **Step 4: Add source anchors and runnable checks**

Link these current files:

```text
docs/PassManagement.md
docs/PatternRewriter.md
docs/DialectConversion.md
docs/Dialects/Transform.md
include/mlir/Pass/PassManager.h
include/mlir/IR/PatternMatch.h
include/mlir/Transforms/DialectConversion.h
include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h
```

From a file inside `Transformation/`, use `../../../docs/...` for MLIR docs and `../../../include/...` for MLIR headers so every link resolves from the Markdown file itself.

Record the parser command from Task 1 and the four relevant `mlir-opt --help` entries: `canonicalize`, `one-shot-bufferize`, `convert-linalg-to-loops`, and `transform-interpreter`.

- [ ] **Step 5: Render both documents**

```bash
mkdir -p /tmp/illustrated-transformation/map/assets
mkdir -p /tmp/illustrated-transformation/taxonomy/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/00_Reading_Map.md \
  -o /tmp/illustrated-transformation/map/rendered.md \
  -a /tmp/illustrated-transformation/map/assets -t default -b white
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/01_What_Transforms_IR.md \
  -o /tmp/illustrated-transformation/taxonomy/rendered.md \
  -a /tmp/illustrated-transformation/taxonomy/assets -t default -b white
```

Expected: both commands exit `0`; SVG counts equal Mermaid fence counts.

- [ ] **Step 6: Commit the entry documents**

```bash
git add \
  mlir/LearningSteps/IllustratedMLIR/Transformation/00_Reading_Map.md \
  mlir/LearningSteps/IllustratedMLIR/Transformation/01_What_Transforms_IR.md
git commit -m "docs: introduce illustrated MLIR transformations"
```

### Task 3: Write the Pass Infrastructure Chapter

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/02_Pass_Infrastructure.md`
- Read: `mlir/LearningSteps/Transforms/PassManagement_Summary.md`
- Read: `mlir/LearningSteps/Transforms/PassManagement_OperationPass_Restrictions_Summary.md`
- Read: `mlir/docs/PassManagement.md`
- Read: `mlir/include/mlir/Pass/Pass.h`
- Read: `mlir/include/mlir/Pass/PassManager.h`

- [ ] **Step 1: Create the chapter with this heading order**

```markdown
# Pass 基础设施：谁安排变换的执行顺序
## 为什么有 Pattern 还需要 Pass
## 从一条文本 pipeline 开始
## PassManager 与 OpPassManager
## Operation 锚点和嵌套 pipeline
## 一次 Pass 执行的生命周期
## Analysis 如何缓存和失效
## Pass Failure 如何传播
## 并行执行为什么限制 Pass 的访问范围
## 回到 C++ 源码验证
## 运行共享 matmul 示例
## 三个常见误区
## 一张图总结本章
## 继续阅读
```

- [ ] **Step 2: Add the nested-pipeline diagram**

Show this exact structure:

```text
builtin.module PassManager
  -> func.func nested OpPassManager
    -> canonicalize
    -> cse
```

Explain that nesting is an Operation execution boundary, not merely visual grouping. Contrast `builtin.module(canonicalize)` with `builtin.module(func.func(canonicalize,cse))`.

- [ ] **Step 3: Add the pass/analysis timeline**

Show:

```text
Pass initialization
  -> find eligible current Operation
  -> request or reuse Analysis
  -> runOnOperation
  -> mark preserved analyses
  -> invalidate unpreserved analyses
  -> propagate success or failure
```

State the current-operation mutation rules and link the focused restriction summary for details.

- [ ] **Step 4: Add the real pipeline command and assertions**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  --pass-pipeline='builtin.module(func.func(canonicalize,cse))' \
  -o /tmp/illustrated-transformation-pass.mlir
rg -n "func.func @matmul|linalg.fill|linalg.matmul" \
  /tmp/illustrated-transformation-pass.mlir
! rg -n "arith.constant 1" /tmp/illustrated-transformation-pass.mlir
```

Explain that the command proves scheduling and observable output, while internal analysis caching is established from source/docs rather than inferred from printed IR.

- [ ] **Step 5: Render and commit the chapter**

```bash
mkdir -p /tmp/illustrated-transformation/pass/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/02_Pass_Infrastructure.md \
  -o /tmp/illustrated-transformation/pass/rendered.md \
  -a /tmp/illustrated-transformation/pass/assets -t default -b white
git add mlir/LearningSteps/IllustratedMLIR/Transformation/02_Pass_Infrastructure.md
git commit -m "docs: illustrate MLIR pass infrastructure"
```

Expected: rendering exits `0`; every Mermaid fence has an SVG.

### Task 4: Write the Pattern Rewriter Chapter

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/03_Pattern_Rewriter.md`
- Read: `mlir/LearningSteps/Transforms/PatternRewriter_Summary.md`
- Read: `mlir/docs/PatternRewriter.md`
- Read: `mlir/include/mlir/IR/PatternMatch.h`
- Read: `mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h`

- [ ] **Step 1: Create the chapter with this heading order**

```markdown
# PatternRewriter：一条局部规则如何安全修改 IR
## 从消失的 dead constant 开始
## RewritePattern 描述什么
## matchAndRewrite 的成功契约
## 为什么所有修改必须经过 PatternRewriter
## replace、erase 与原地修改
## Driver 才决定 Pattern 如何被应用
## Greedy Driver 的工作队列
## 递归应用与 bounded recursion
## 回到 C++ 源码验证
## 运行内置 canonicalization patterns
## 四个常见误区
## 一张图总结本章
## 继续阅读
```

- [ ] **Step 2: Add the match-rewrite state machine**

Represent:

```text
Driver selects root Operation
  -> Pattern match
    -> failure: root unchanged, try next pattern
    -> success: mutation must occur through PatternRewriter
      -> root replaced / erased / updated in place
      -> affected Operations return to driver worklist
```

State that success without mutation violates the pattern contract.

- [ ] **Step 3: Add the driver comparison and worklist diagram**

Compare greedy, walk, and conversion drivers in a table. The worklist diagram must show that a pattern does not schedule itself; the driver owns traversal, pattern ordering, requeueing, and convergence policy.

- [ ] **Step 4: Add the verified canonicalize command**

Use the same command and assertions as Task 3, but show the before line `%dead = arith.constant 1 : index` and its absence after canonicalization. State precisely that this proves a registered canonicalization path modified the IR, not that the sample defines a custom pattern.

Do not require `-debug-only=greedy-rewriter`; the current Optimized build may not expose a stable debug trace. Link current source and tests instead.

- [ ] **Step 5: Render and commit the chapter**

```bash
mkdir -p /tmp/illustrated-transformation/pattern/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/03_Pattern_Rewriter.md \
  -o /tmp/illustrated-transformation/pattern/rendered.md \
  -a /tmp/illustrated-transformation/pattern/assets -t default -b white
git add mlir/LearningSteps/IllustratedMLIR/Transformation/03_Pattern_Rewriter.md
git commit -m "docs: illustrate MLIR pattern rewriting"
```

### Task 5: Write the Dialect Conversion Chapter

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/04_Dialect_Conversion.md`
- Read: `mlir/LearningSteps/Transforms/DialectConversion_Summary.md`
- Read: `mlir/docs/DialectConversion.md`
- Read: `mlir/include/mlir/Transforms/DialectConversion.h`
- Read: `mlir/test/Transforms/test-legalize-type-conversion.mlir`
- Read: `mlir/test/Transforms/test-legalizer-full.mlir`

- [ ] **Step 1: Create the chapter with this heading order**

```markdown
# Dialect Conversion：怎样保证降层后的 IR 合法
## 普通改写为什么不足以表达完整降层
## 三种 Conversion 模式
## ConversionTarget 定义终点
## Legality 如何逐个判断 Operation
## ConversionPattern 与普通 RewritePattern
## Adaptor 为什么不是 op.getOperands()
## TypeConverter 如何改变类型
## Materialization 如何跨越类型边界
## Region Signature Conversion
## Rollback 与 no-rollback 的观察差异
## 回到 C++ 源码验证
## 运行 matmul lowering
## 四个常见误区
## 一张图总结本章
## 继续阅读
```

- [ ] **Step 2: Add the legality decision diagram**

Show the decision order for explicit legal/illegal, dynamic legality, unknown operations, recursive legality, and full-conversion failure. Make clear that a rewrite can be locally valid yet the conversion still fails because illegal Operations remain.

- [ ] **Step 3: Add the operand-view and type-bridge diagram**

Represent this exact distinction:

```text
Original source Operation -> op.getOperands() -> original source-typed Values
Conversion driver -> remapping -> adaptor operands -> current converted Values
TypeConverter -> source/target materialization -> bridge Values when required
ConversionPattern -> build target Operation from adaptor view
```

Include a small source-type/target-type example and state that target-op construction normally uses adaptor operands.

- [ ] **Step 4: Add and run the real lowering pipeline**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-loops \
  -o /tmp/illustrated-transformation-converted.mlir
rg -n "memref.alloc|scf.for" \
  /tmp/illustrated-transformation-converted.mlir
! rg -n "linalg.matmul|tensor<" \
  /tmp/illustrated-transformation-converted.mlir
```

Explain that One-Shot Bufferize and Linalg-to-loops are concrete registered transformations used to expose the state transition; the chapter's legality/adaptor internals are grounded in Dialect Conversion docs/source and must not be falsely attributed to every implementation detail of those passes.

- [ ] **Step 5: Render and commit the chapter**

```bash
mkdir -p /tmp/illustrated-transformation/conversion/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/04_Dialect_Conversion.md \
  -o /tmp/illustrated-transformation/conversion/rendered.md \
  -a /tmp/illustrated-transformation/conversion/assets -t default -b white
git add mlir/LearningSteps/IllustratedMLIR/Transformation/04_Dialect_Conversion.md
git commit -m "docs: illustrate MLIR dialect conversion"
```

### Task 6: Write the Transform Dialect Chapter

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/05_Transform_Dialect.md`
- Read: `mlir/docs/Dialects/Transform.md`
- Read: `mlir/docs/Tutorials/transform/Ch1.md`
- Read: `mlir/include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h`
- Read: `mlir/include/mlir/Dialect/Transform/IR/TransformOps.td`
- Read: `mlir/lib/Dialect/Transform/Transforms/InterpreterPass.cpp`

- [ ] **Step 1: Create the chapter with this heading order**

```markdown
# Transform Dialect：为什么把变换本身也写成 IR
## Payload IR 与 Transform IR 是两套 IR
## transform.named_sequence 是执行入口
## Handle 如何关联 payload Operation
## Interpreter 如何逐条执行 Transform IR
## tile_using_forall 修改了什么
## Handle 消耗与失效
## Effects 为什么是正确性契约
## Silenceable Failure 与 Definite Failure
## Transform Dialect 如何复用底层机制
## 回到 C++ 与 TableGen 源码验证
## 运行 matmul Transform sequence
## 四个常见误区
## 一张图总结本章
## 继续阅读
```

- [ ] **Step 2: Add the dual-IR handle diagram**

Show separate containers for Transform IR and payload IR. Red handle arrows point from `%root`, `%matmul`, `%tiled`, and `%loop` to associated payload Operations. Explicitly state that Transform handles are not payload SSA Values and do not own payload Operations.

- [ ] **Step 3: Add the execution and invalidation timeline**

Represent:

```text
Interpreter finds @__transform_main
  -> bind %root to payload module
  -> structured.match binds %matmul
  -> tile_using_forall consumes %matmul
  -> create tiled linalg.matmul and scf.forall
  -> bind result handles %tiled and %loop
  -> later use of consumed %matmul is invalid
```

Explain readonly vs consumed effects and why expensive checks diagnose invalid handle use.

- [ ] **Step 4: Add and run the interpreter command**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  --transform-interpreter \
  -o /tmp/illustrated-transformation-tiled.mlir
rg -n "scf.forall|linalg.matmul" \
  /tmp/illustrated-transformation-tiled.mlir
```

Expected: `scf.forall` and the tiled inner `linalg.matmul` are present. Explain that the named Transform sequence remains printable in the module; do not mistake it for payload code.

- [ ] **Step 5: Render and commit the chapter**

```bash
mkdir -p /tmp/illustrated-transformation/transform/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/05_Transform_Dialect.md \
  -o /tmp/illustrated-transformation/transform/rendered.md \
  -a /tmp/illustrated-transformation/transform/assets -t default -b white
git add mlir/LearningSteps/IllustratedMLIR/Transformation/05_Transform_Dialect.md
git commit -m "docs: illustrate the MLIR Transform dialect"
```

### Task 7: Write the End-to-End Matmul Chapter

**Files:**
- Create: `mlir/LearningSteps/IllustratedMLIR/Transformation/06_End_To_End_Matmul.md`
- Read: all files created in Tasks 1-6

- [ ] **Step 1: Create the chapter with this heading order**

```markdown
# 端到端复盘：一条 matmul 是怎样被逐步变换的
## 冻结输入和实验身份
## 阶段 0：原始 Tensor Linalg IR
## 阶段 1：Pass pipeline 清理冗余
## 阶段 2：Transform IR 选择并 Tile matmul
## 阶段 3：Bufferization 改变数据表示
## 阶段 4：Linalg Conversion 生成循环
## 每个阶段是谁调度、谁修改
## 四类失败怎样定位
## 完整可复现命令
## 最终 Operation 清单
## 一张总时间线
## 本册完成检查
```

- [ ] **Step 2: Add the experiment identity and stage table**

Record:

```text
input: examples/matmul_transform.mlir
tool: build/bin/mlir-opt
transform tile sizes: [4, 2]
bufferization: one-shot-bufferize with bufferize-function-boundaries
lowering: convert-linalg-to-loops
outputs: /tmp/illustrated-transformation-e2e-*.mlir
```

Add columns for stage, scheduler/orchestrator, mutation mechanism, expected new Operations, and expected removed Operations.

- [ ] **Step 3: Run and document each isolated stage**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  -o /tmp/illustrated-transformation-e2e-parsed.mlir
rg -n "linalg.fill|linalg.matmul|transform.named_sequence" \
  /tmp/illustrated-transformation-e2e-parsed.mlir

build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  --transform-interpreter \
  -o /tmp/illustrated-transformation-e2e-tiled.mlir
rg -n "scf.forall|linalg.matmul" \
  /tmp/illustrated-transformation-e2e-tiled.mlir
```

Expected: the parsed stage contains the original Linalg payload and named sequence; the tiled stage contains `scf.forall`. Show these focused excerpts in the chapter rather than copying full generated files.

- [ ] **Step 4: Add and run the complete pipeline**

```bash
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  --transform-interpreter \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-loops \
  -o /tmp/illustrated-transformation-e2e-lowered.mlir
rg -n "memref.alloc|scf.forall|scf.for" \
  /tmp/illustrated-transformation-e2e-lowered.mlir
! rg -n "^[[:space:]]+%.* = linalg\.matmul|tensor<" \
  /tmp/illustrated-transformation-e2e-lowered.mlir
```

Expected: `memref.alloc`, `scf.forall`, and nested `scf.for` are present; payload `linalg.matmul` and tensor types are absent. The Transform sequence can still contain the textual identifier `linalg.matmul`, so scope the absence explanation to payload IR and mention that the final file still prints Transform IR.

- [ ] **Step 5: Add the failure-routing table and total timeline**

Route failures as follows:

| Symptom | First boundary to inspect |
|---|---|
| Pass 未执行 | pipeline 嵌套和 Operation 锚点 |
| 局部 IR 没有变化 | 已注册 pattern 和 driver 工作队列 |
| 非法 Operation 仍存在 | ConversionTarget、adaptor、TypeConverter、materialization |
| Transform sequence 执行失败 | 入口、handle 绑定、effects、invalidation |

The final Mermaid timeline must name both “who schedules” and “who mutates” at every stage.

- [ ] **Step 6: Render and commit the chapter**

```bash
mkdir -p /tmp/illustrated-transformation/e2e/assets
/home/fuhao/.npm-global/bin/mmdc \
  -i mlir/LearningSteps/IllustratedMLIR/Transformation/06_End_To_End_Matmul.md \
  -o /tmp/illustrated-transformation/e2e/rendered.md \
  -a /tmp/illustrated-transformation/e2e/assets -t default -b white
git add mlir/LearningSteps/IllustratedMLIR/Transformation/06_End_To_End_Matmul.md
git commit -m "docs: connect MLIR transformation mechanisms end to end"
```

### Task 8: Validate the Complete Transformation Booklet

**Files:**
- Verify: `mlir/LearningSteps/IllustratedMLIR/Transformation/*.md`
- Verify: `mlir/LearningSteps/IllustratedMLIR/Transformation/examples/*.mlir`
- Guard: `mlir/LearningSteps/Transforms/`

- [ ] **Step 1: Re-run both source inputs and the full pipeline**

```bash
build/bin/mlir-opt --version
/home/fuhao/.npm-global/bin/mmdc --version
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir \
  -o /tmp/illustrated-transformation-final-payload.mlir
build/bin/mlir-opt \
  mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir \
  --transform-interpreter \
  --one-shot-bufferize='bufferize-function-boundaries' \
  --convert-linalg-to-loops \
  -o /tmp/illustrated-transformation-final-lowered.mlir
rg -n "memref.alloc|scf.forall|scf.for" \
  /tmp/illustrated-transformation-final-lowered.mlir
```

Expected: all commands exit `0`, and all three expected lower-level Operation forms are present.

- [ ] **Step 2: Render all seven Markdown documents into fresh directories**

```bash
render_root="/tmp/illustrated-transformation-final-$$"
for pair in \
  00_Reading_Map:00 \
  01_What_Transforms_IR:01 \
  02_Pass_Infrastructure:02 \
  03_Pattern_Rewriter:03 \
  04_Dialect_Conversion:04 \
  05_Transform_Dialect:05 \
  06_End_To_End_Matmul:06; do
  source_name="${pair%%:*}"
  slot="${pair##*:}"
  mkdir -p "${render_root}/${slot}/assets"
  /home/fuhao/.npm-global/bin/mmdc \
    -i "mlir/LearningSteps/IllustratedMLIR/Transformation/${source_name}.md" \
    -o "${render_root}/${slot}/rendered.md" \
    -a "${render_root}/${slot}/assets" -t default -b white
  test "$(find "${render_root}/${slot}/assets" -maxdepth 1 -name '*.svg' | wc -l)" \
    -eq "$(rg -c '^```mermaid$' "mlir/LearningSteps/IllustratedMLIR/Transformation/${source_name}.md")"
done
echo "rendered under ${render_root}"
```

Expected: all seven `mmdc` invocations exit `0`, every count comparison succeeds, and no renderer reports a syntax error.

- [ ] **Step 3: Check Chinese persistence, incomplete markers, and required concepts**

```bash
rg -l "[一-龥]" mlir/LearningSteps/IllustratedMLIR/Transformation/*.md
! rg -n "T[B]D|T[O]DO|F[I]XME|【待补】|<待补>" \
  mlir/LearningSteps/IllustratedMLIR/Transformation
rg -l "PassManager|PatternRewriter|ConversionTarget|transform-interpreter" \
  mlir/LearningSteps/IllustratedMLIR/Transformation/*.md
```

Expected: every Markdown file contains Chinese; no incomplete marker exists; required concepts appear in the relevant chapters.

- [ ] **Step 4: Check files, links, whitespace, and scope**

```bash
test -f mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul.mlir
test -f mlir/LearningSteps/IllustratedMLIR/Transformation/examples/matmul_transform.mlir
test "$(find mlir/LearningSteps/IllustratedMLIR/Transformation -maxdepth 1 -name '*.md' | wc -l)" -eq 7
python3 - <<'PY'
from pathlib import Path
import re

root = Path("mlir/LearningSteps/IllustratedMLIR/Transformation")
missing = []
for source in root.glob("*.md"):
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", source.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        path = (source.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            missing.append(f"{source}: {target}")
assert not missing, "missing links:\n" + "\n".join(missing)
PY
git diff --check HEAD~7..HEAD
git diff --quiet HEAD~7..HEAD -- mlir/LearningSteps/Transforms
```

Expected: all files and relative-link targets exist, no whitespace errors are reported, and existing summary files are unchanged.

- [ ] **Step 5: Verify exact commit scope and record final state**

```bash
git log -8 --oneline
git status --short -- mlir/LearningSteps/IllustratedMLIR/Transformation
git diff --name-only HEAD~7..HEAD
```

Expected: seven implementation commits cover only the nine new booklet files; the target directory has no uncommitted changes. Report unrelated pre-existing worktree changes separately rather than claiming the whole worktree is clean.
