# LearningSteps Topic Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize every current `LearningSteps` artifact into stable topic directories, add a complete reading index, and repair repository-local references without losing the existing RTX4090 work.

**Architecture:** Keep `LearningSteps/README.md` as the only root file and make it the navigation layer over subject directories. Move content without rewriting it, then apply a small, explicit path-migration set and validate file counts, local links, stale paths, whitespace, and the pre-existing dirty RTX4090 content.

**Tech Stack:** Markdown, MLIR test files, C++, Mermaid/SVG assets, Git, POSIX shell, Python 3 link validation

---

## File Structure

Create these navigation and subject boundaries:

- `LearningSteps/README.md`: recommended reading order and complete topic index.
- `LearningSteps/Plans/`: the general MLIR learning plan.
- `LearningSteps/Plans/GPU_CodeGen/`: GPU/code-generation plans and their phase documents.
- `LearningSteps/Foundations/`: introductory IR, language, toolchain, and core-header notes.
- `LearningSteps/Dialects/`: dialect-definition and Builtin/Func references.
- `LearningSteps/ODS/`: ODS, TableGen, and Toy ODS practice notes.
- `LearningSteps/Transforms/`: pattern rewriting, dialect conversion, passes, and Phase 2 overviews.
- `LearningSteps/Tutorials/`: Toy tutorial walkthroughs.
- `LearningSteps/Tools/`: FileCheck documentation.
- `LearningSteps/Practices/`: runnable `.mlir` inputs, their notes, and the rewrite-pass example.
- `LearningSteps/Assets/`: Mermaid and SVG diagrams.

The exact source-to-destination mapping is defined in
`docs/superpowers/specs/2026-07-15-learningsteps-topic-reorganization-design.md`.

### Task 1: Record the Dirty-Tree Baseline

**Files:**
- Read: `LearningSteps/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`
- Read: `LearningSteps/RTX4090_GPU_CodeGen_Phases/*.md`
- Create outside repository: `/tmp/learningsteps-topic-reorganization-before/`

- [ ] **Step 1: Confirm the known dirty files before moving anything**

Run:

```bash
git status --short
```

Expected: the RTX4090 index is modified, the seven phase files and their
2026-07-14 plan/spec are untracked, and no unexpected `LearningSteps` file is
already staged.

- [ ] **Step 2: Snapshot the pre-existing RTX4090 content**

Run:

```bash
mkdir -p /tmp/learningsteps-topic-reorganization-before
cp LearningSteps/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  /tmp/learningsteps-topic-reorganization-before/
cp -a LearningSteps/RTX4090_GPU_CodeGen_Phases \
  /tmp/learningsteps-topic-reorganization-before/
sha256sum LearningSteps/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  LearningSteps/RTX4090_GPU_CodeGen_Phases/*.md
```

Expected: one checksum for the index and seven checksums for phase documents;
the snapshot directory contains all eight files.

- [ ] **Step 3: Record baseline file counts in the execution log**

Run:

```bash
find LearningSteps -type f | wc -l
find LearningSteps -type f -name '*.md' | wc -l
find LearningSteps -type f -name '*.mlir' | wc -l
```

Expected: `39` total files, `32` Markdown files, and `4` MLIR files.

### Task 2: Move Every Artifact Into Its Topic Directory

**Files:**
- Move: all current files and directories below `LearningSteps/`
- Create: ten target directories below `LearningSteps/`

- [ ] **Step 1: Create the target directories**

Run:

```bash
mkdir -p LearningSteps/Plans/GPU_CodeGen/Phases \
  LearningSteps/Foundations \
  LearningSteps/Dialects \
  LearningSteps/ODS \
  LearningSteps/Transforms \
  LearningSteps/Tutorials \
  LearningSteps/Tools \
  LearningSteps/Practices \
  LearningSteps/Assets
```

Expected: every target directory exists and is empty.

- [ ] **Step 2: Move plans and GPU phase documents**

Run:

```bash
mv LearningSteps/MLIR_Learning_Plan.md LearningSteps/Plans/
mv LearningSteps/AI_Compiler_Engineer_Learning_Plan.md \
  LearningSteps/Plans/GPU_CodeGen/
mv LearningSteps/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  LearningSteps/Plans/GPU_CodeGen/
mv LearningSteps/RTX4090_GPU_CodeGen_Phases/*.md \
  LearningSteps/Plans/GPU_CodeGen/Phases/
rmdir LearningSteps/RTX4090_GPU_CodeGen_Phases
```

Expected: the three plan files and seven phase files appear at their new paths;
the former phase directory no longer exists.

- [ ] **Step 3: Move foundation, dialect, ODS, transform, tutorial, and tool notes**

Run:

```bash
mv LearningSteps/Step1_IR_Structure_Summary.md \
  LearningSteps/Step2_Language_Reference_Summary.md \
  LearningSteps/Step3_mlir_opt_Toolchain_Summary.md \
  LearningSteps/Step5_Core_IR_Headers_Summary.md \
  LearningSteps/Foundations/
mv LearningSteps/Builtin_Dialect_Summary.md \
  LearningSteps/DefiningDialects_Summary.md \
  LearningSteps/FuncDialect_Summary.md \
  LearningSteps/Func_Dialect_Summary.md \
  LearningSteps/Dialects/
mv LearningSteps/Operations_Summary.md \
  LearningSteps/TableGen_ODS_Notes.md \
  LearningSteps/ODS_Toy_Dialect_Practical_Summary.md \
  LearningSteps/ODS/
mv LearningSteps/PatternRewriter_Summary.md \
  LearningSteps/DialectConversion_Summary.md \
  LearningSteps/PassManagement_Summary.md \
  LearningSteps/PassManagement_OperationPass_Restrictions_Summary.md \
  LearningSteps/Phase2_Core_Infrastructure_Summary.md \
  LearningSteps/Phase2_AI_Compiler_Beginner_Guide.md \
  LearningSteps/Transforms/
mv LearningSteps/Step4_Toy_Tutorial_Summary.md \
  LearningSteps/Step4_Ch7_Code_Structure_Summary.md \
  LearningSteps/Tutorials/
mv LearningSteps/FileCheck_Summary.md LearningSteps/Tools/
```

Expected: each note exists in exactly the subject directory defined by the
design; both Func dialect summaries remain separate.

- [ ] **Step 4: Move practices and assets**

Run:

```bash
mv LearningSteps/ir_ssa_walk.mlir \
  LearningSteps/mlir_ir_ssa_notes.md \
  LearningSteps/practice_canonicalize_cse.mlir \
  LearningSteps/Practices/
mv LearningSteps/rewrite-pattern-addi-zero LearningSteps/Practices/
mv LearningSteps/Builtin_Dialect_IR_Overview.mmd \
  LearningSteps/Builtin_Dialect_IR_Overview.svg \
  LearningSteps/Assets/
```

Expected: only directories remain at the `LearningSteps` root.

- [ ] **Step 5: Check the move before changing content**

Run:

```bash
find LearningSteps -maxdepth 1 -type f -print
find LearningSteps -type f | wc -l
```

Expected: the first command prints nothing and the second prints `39`.

### Task 3: Repair Paths Affected by the Move

**Files:**
- Modify: `LearningSteps/Plans/MLIR_Learning_Plan.md`
- Modify: `LearningSteps/Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md`
- Modify: `LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`
- Modify: `LearningSteps/Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md`
- Modify: `LearningSteps/Tools/FileCheck_Summary.md`
- Modify: `LearningSteps/Transforms/PassManagement_Summary.md`
- Modify: `LearningSteps/Practices/rewrite-pattern-addi-zero/CMakeLists_notes.md`
- Modify: `docs/superpowers/plans/2026-07-14-rtx4090-execution-plan-phase-split.md`
- Modify: `docs/superpowers/plans/2026-07-14-illustrated-mlir-phase-one.md`
- Modify: `docs/superpowers/specs/2026-07-14-rtx4090-execution-plan-phase-split-design.md`

- [ ] **Step 1: Repair links between plans, transforms, and practices**

Apply these exact substitutions with `apply_patch`:

```text
Plans/MLIR_Learning_Plan.md:
  Phase2_AI_Compiler_Beginner_Guide.md
    -> ../Transforms/Phase2_AI_Compiler_Beginner_Guide.md
  Phase2_Core_Infrastructure_Summary.md
    -> ../Transforms/Phase2_Core_Infrastructure_Summary.md

Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md:
  LearningSteps/rewrite-pattern-addi-zero/
    -> LearningSteps/Practices/rewrite-pattern-addi-zero/

Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md:
  ./RTX4090_GPU_CodeGen_Phases/
    -> ./Phases/

Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md:
  LearningSteps/rewrite-pattern-addi-zero/
    -> LearningSteps/Practices/rewrite-pattern-addi-zero/

Transforms/PassManagement_Summary.md:
  mlir/LearningSteps/PassManagement_OperationPass_Restrictions_Summary.md
    -> mlir/LearningSteps/Transforms/PassManagement_OperationPass_Restrictions_Summary.md

Practices/rewrite-pattern-addi-zero/CMakeLists_notes.md:
  mlir/LearningSteps/rewrite-pattern-addi-zero/
    -> mlir/LearningSteps/Practices/rewrite-pattern-addi-zero/
  add_subdirectory(LearningSteps/rewrite-pattern-addi-zero)
    -> add_subdirectory(LearningSteps/Practices/rewrite-pattern-addi-zero)
```

Expected: adjacent links such as phase-to-index, Toy Ch6-to-Ch7, Phase 2
beginner-to-core, and SSA-notes-to-input remain unchanged because their files
moved together.

- [ ] **Step 2: Repair FileCheck command examples and the source link**

Apply these exact substitutions to
`LearningSteps/Tools/FileCheck_Summary.md`:

```text
./mlir/LearningSteps/practice_canonicalize_cse.mlir
  -> ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir
/home/fuhao/llvm-project/mlir/LearningSteps/practice_canonicalize_cse.mlir:1
  -> /home/fuhao/llvm-project/mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir:1
```

Expected: all five command/source occurrences point to `Practices/`.

- [ ] **Step 3: Update active 2026-07-14 implementation artifacts**

Apply these global path substitutions only in the three 2026-07-14 files listed
for this task:

```text
LearningSteps/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
  -> LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
LearningSteps/RTX4090_GPU_CodeGen_Phases/
  -> LearningSteps/Plans/GPU_CodeGen/Phases/
LearningSteps/AI_Compiler_Engineer_Learning_Plan.md
  -> LearningSteps/Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md
LearningSteps/rewrite-pattern-addi-zero/
  -> LearningSteps/Practices/rewrite-pattern-addi-zero/
LearningSteps/PatternRewriter_Summary.md
  -> LearningSteps/Transforms/PatternRewriter_Summary.md
LearningSteps/DialectConversion_Summary.md
  -> LearningSteps/Transforms/DialectConversion_Summary.md
mlir/LearningSteps/Step1_IR_Structure_Summary.md
  -> mlir/LearningSteps/Foundations/Step1_IR_Structure_Summary.md
mlir/LearningSteps/Operations_Summary.md
  -> mlir/LearningSteps/ODS/Operations_Summary.md
mlir/LearningSteps/Builtin_Dialect_Summary.md
  -> mlir/LearningSteps/Dialects/Builtin_Dialect_Summary.md
mlir/LearningSteps/Phase2_Core_Infrastructure_Summary.md
  -> mlir/LearningSteps/Transforms/Phase2_Core_Infrastructure_Summary.md
```

Also update the illustrated-plan instruction that links from a future
`LearningSteps/IllustratedMLIR/*.md` file:

```text
../Step1_IR_Structure_Summary.md
  -> ../Foundations/Step1_IR_Structure_Summary.md
../Operations_Summary.md
  -> ../ODS/Operations_Summary.md
../Builtin_Dialect_Summary.md
  -> ../Dialects/Builtin_Dialect_Summary.md
../Phase2_Core_Infrastructure_Summary.md
  -> ../Transforms/Phase2_Core_Infrastructure_Summary.md
```

Expected: active plan commands and referenced files use the new structure;
temporary `/tmp/RTX4090_...` snapshot names stay unchanged.

- [ ] **Step 4: Scan for stale actionable paths**

Run:

```bash
rg -n --glob '*.md' \
  'LearningSteps/(RTX4090_GPU_CodeGen|AI_Compiler_Engineer|rewrite-pattern-addi-zero|PatternRewriter_Summary|DialectConversion_Summary|Step1_IR_Structure_Summary|Operations_Summary|Builtin_Dialect_Summary|Phase2_Core_Infrastructure_Summary|practice_canonicalize_cse)' \
  LearningSteps docs/superpowers \
  -g '!**/specs/2026-07-15-learningsteps-topic-reorganization-design.md' \
  -g '!**/plans/2026-07-15-learningsteps-topic-reorganization.md'
```

Expected: no output. Bare filenames in documents that moved together are
allowed because those relative links remain valid.

### Task 4: Add the LearningSteps Entry Point

**Files:**
- Create: `LearningSteps/README.md`

- [ ] **Step 1: Create the complete topic index**

Create `LearningSteps/README.md` with this content using `apply_patch`:

```markdown
# MLIR LearningSteps

这里集中存放 MLIR 学习路线、专题笔记、实践代码和 GPU CodeGen 训练计划。目录按稳定主题组织；本文档是统一入口，具体学习进度以计划文件为准。

## 推荐阅读顺序

1. [MLIR 学习计划](Plans/MLIR_Learning_Plan.md)
2. 基础概念：[IR 结构](Foundations/Step1_IR_Structure_Summary.md) → [语言参考](Foundations/Step2_Language_Reference_Summary.md) → [mlir-opt 工具链](Foundations/Step3_mlir_opt_Toolchain_Summary.md)
3. Toy 教程：[Ch1-Ch6](Tutorials/Step4_Toy_Tutorial_Summary.md) → [Ch7 代码结构](Tutorials/Step4_Ch7_Code_Structure_Summary.md)
4. 图解 MLIR：[阅读地图](IllustratedMLIR/00_Reading_Map.md) → [为什么一切都是 Operation](IllustratedMLIR/01_Why_Everything_Is_An_Operation.md)
5. [Core IR Headers](Foundations/Step5_Core_IR_Headers_Summary.md)
6. ODS：[TableGen 与 ODS](ODS/TableGen_ODS_Notes.md) → [Operations](ODS/Operations_Summary.md) → [Toy ODS 实践](ODS/ODS_Toy_Dialect_Practical_Summary.md)
7. Dialect：[定义 Dialect](Dialects/DefiningDialects_Summary.md) → [Builtin](Dialects/Builtin_Dialect_Summary.md) → [Func 概念总结](Dialects/Func_Dialect_Summary.md) → [Func 源码导读](Dialects/FuncDialect_Summary.md)
8. 变换框架：[Phase 2 初学者导读](Transforms/Phase2_AI_Compiler_Beginner_Guide.md) → [Phase 2 基础设施总结](Transforms/Phase2_Core_Infrastructure_Summary.md) → [PatternRewriter](Transforms/PatternRewriter_Summary.md) → [DialectConversion](Transforms/DialectConversion_Summary.md) → [PassManagement](Transforms/PassManagement_Summary.md)
9. 测试与实践：[FileCheck](Tools/FileCheck_Summary.md) → [canonicalize/CSE 输入](Practices/practice_canonicalize_cse.mlir) → [rewrite pattern 示例](Practices/rewrite-pattern-addi-zero/)

## 主题索引

### 学习计划

- [MLIR Learning Plan](Plans/MLIR_Learning_Plan.md)
- [GPU Kernel/CodeGen Engineer Learning Plan](Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md)
- [RTX4090 GPU CodeGen 24 周执行计划](Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)
- [RTX4090 分阶段计划](Plans/GPU_CodeGen/Phases/)
  - [Phase 1：CUDA 与 Triton Kernel 基础](Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md)
  - [Phase 2：MLIR Transformation Bridge](Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md)
  - [Phase 3：Triton Compiler Internals](Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md)
  - [Phase 4：Baseline Freeze](Plans/GPU_CodeGen/Phases/Phase4_Baseline_Freeze.md)
  - [Phase 5：Performance Optimization](Plans/GPU_CodeGen/Phases/Phase5_Performance_Optimization.md)
  - [Phase 6：Generalization & Regression](Plans/GPU_CodeGen/Phases/Phase6_Generalization_Regression.md)
  - [Phase 7：Engineering Delivery](Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md)

### Foundations

- [Step 1：IR 结构](Foundations/Step1_IR_Structure_Summary.md)
- [Step 2：语言参考](Foundations/Step2_Language_Reference_Summary.md)
- [Step 3：mlir-opt 工具链](Foundations/Step3_mlir_opt_Toolchain_Summary.md)
- [Step 5：Core IR Headers](Foundations/Step5_Core_IR_Headers_Summary.md)

### Dialects

- [Defining Dialects](Dialects/DefiningDialects_Summary.md)
- [Builtin Dialect](Dialects/Builtin_Dialect_Summary.md)
- [Func Dialect 概念总结](Dialects/Func_Dialect_Summary.md)：适合先建立函数 IR 心智模型。
- [Func Dialect 源码导读](Dialects/FuncDialect_Summary.md)：内容更完整，包含初始化、接口、pass 和源码关系。
- [Builtin IR 结构图（Mermaid）](Assets/Builtin_Dialect_IR_Overview.mmd)
- [Builtin IR 结构图（SVG）](Assets/Builtin_Dialect_IR_Overview.svg)

### ODS

- [Operations / ODS 详细总结](ODS/Operations_Summary.md)
- [TableGen 与 ODS 笔记](ODS/TableGen_ODS_Notes.md)
- [Toy Dialect ODS 实践](ODS/ODS_Toy_Dialect_Practical_Summary.md)

### Transforms 与 Passes

- [Phase 2 初学者导读](Transforms/Phase2_AI_Compiler_Beginner_Guide.md)
- [Phase 2 Core Infrastructure](Transforms/Phase2_Core_Infrastructure_Summary.md)
- [PatternRewriter](Transforms/PatternRewriter_Summary.md)
- [DialectConversion](Transforms/DialectConversion_Summary.md)
- [PassManagement](Transforms/PassManagement_Summary.md)
- [OperationPass 并行限制](Transforms/PassManagement_OperationPass_Restrictions_Summary.md)

### Tutorials 与 Tools

- [Toy Tutorial Ch1-Ch6](Tutorials/Step4_Toy_Tutorial_Summary.md)
- [Toy Tutorial Ch7 代码结构](Tutorials/Step4_Ch7_Code_Structure_Summary.md)
- [图解 MLIR 阅读地图](IllustratedMLIR/00_Reading_Map.md)
- [图解 MLIR 第 1 章：为什么一切都是 Operation](IllustratedMLIR/01_Why_Everything_Is_An_Operation.md)
- [图解 MLIR 共用示例](IllustratedMLIR/examples/accumulate.mlir)
- [FileCheck](Tools/FileCheck_Summary.md)

### Practices

- [SSA use-def 输入](Practices/ir_ssa_walk.mlir)与[分析笔记](Practices/mlir_ir_ssa_notes.md)
- [canonicalize/CSE 与 FileCheck 输入](Practices/practice_canonicalize_cse.mlir)
- [AddZeroPattern Pass 示例](Practices/rewrite-pattern-addi-zero/)
- [AddZeroPattern 构建说明](Practices/rewrite-pattern-addi-zero/CMakeLists_notes.md)

## 维护规则

- 新文档按主题放入对应目录，不再直接堆放到 `LearningSteps/` 根目录。
- 新增或移动文档时同步更新本索引和仓库内引用。
- 专题总结继续使用一个源文档对应一个 `*_Summary.md` 的方式；不同定位的文档应在索引中说明差异，未经比较不要直接合并。
- 实践输入、实现和说明放在 `Practices/`，图表资源放在 `Assets/`。
```

Expected: the README links all 34 existing Markdown documents directly, plus
the practice inputs and asset files. The count includes the two
`IllustratedMLIR` documents committed concurrently during execution.

- [ ] **Step 2: Confirm root cleanliness and new total**

Run:

```bash
find LearningSteps -maxdepth 1 -type f -printf '%f\n'
find LearningSteps -type f | wc -l
find LearningSteps -type f -name '*.md' | wc -l
```

Expected: root output is only `README.md`; totals are `43` files and `35`
Markdown files. Relative to the original baseline, this includes `README.md`
and three concurrently committed `IllustratedMLIR` files.

### Task 5: Verify Structure, Links, and Content Preservation

**Files:**
- Verify: all files below `LearningSteps/`
- Verify: modified files below `docs/superpowers/`

- [ ] **Step 1: Validate local Markdown targets under LearningSteps**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
from urllib.parse import unquote
import re

root = Path("LearningSteps")
pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
missing = []
for source in root.rglob("*.md"):
    lines = []
    in_fence = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(re.sub(r"`[^`]*`", "", line))
    text = "\n".join(lines)
    for raw in pattern.findall(text):
        target = raw.strip().split(maxsplit=1)[0].strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not target:
            continue
        if target.startswith("/home/"):
            target = re.sub(r":\d+$", "", target)
            resolved = Path(target)
        else:
            resolved = source.parent / target
        if not resolved.exists():
            missing.append(f"{source}: {raw} -> {resolved}")
if missing:
    raise SystemExit("\n".join(missing))
print("all LearningSteps local Markdown links resolve")
PY
```

Expected: `all LearningSteps local Markdown links resolve`.

- [ ] **Step 2: Compare the RTX4090 snapshot with moved files**

Run:

```bash
diff -u \
  /tmp/learningsteps-topic-reorganization-before/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md \
  LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
diff -ru \
  /tmp/learningsteps-topic-reorganization-before/RTX4090_GPU_CodeGen_Phases \
  LearningSteps/Plans/GPU_CodeGen/Phases
```

Expected: the index diff contains only
`./RTX4090_GPU_CodeGen_Phases/` -> `./Phases/`; the phase diff contains only
the Phase 2 practice-path changes from `LearningSteps/rewrite-pattern-addi-zero/`
to `LearningSteps/Practices/rewrite-pattern-addi-zero/`. `diff` returns `1`
because these intentional differences exist.

- [ ] **Step 3: Verify moved practice paths exist**

Run:

```bash
test -f LearningSteps/Practices/practice_canonicalize_cse.mlir
test -f LearningSteps/Practices/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp
test -f LearningSteps/Practices/rewrite-pattern-addi-zero/add_zero_pattern_test.mlir
test -f LearningSteps/Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md
```

Expected: exit status `0` with no output.

- [ ] **Step 4: Run final static checks**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: `git diff --check` has no output; status shows only the intended
LearningSteps reorganization, the path-reference edits, and the pre-existing
2026-07-14 plan/spec. The stat contains no upstream MLIR source files.

- [ ] **Step 5: Leave the implementation uncommitted for review**

Do not stage or commit the implementation automatically. The worktree already
contains user-owned, previously uncommitted RTX4090 work, so the final handoff
must present the complete status and let the user decide whether those changes
should be committed together or separated.
