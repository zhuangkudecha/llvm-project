# MLIR SSA Guided Exercise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a guided, answer-free SSA use-def worksheet beside the existing completed analysis and add it to the LearningSteps index.

**Architecture:** Keep `mlir_ir_ssa_notes.md` as the immutable answer document. Create one standalone exercise file that links to the same MLIR input, preserves the value names and analysis categories, and replaces all conclusions with blank tables; then link it from `LearningSteps/README.md` and validate the no-answer contract.

**Tech Stack:** Markdown, MLIR, Git, POSIX shell, Python 3 validation

---

### Task 1: Record the Original-Note Baseline

**Files:**
- Read: `LearningSteps/Practices/mlir_ir_ssa_notes.md`
- Read: `LearningSteps/Practices/ir_ssa_walk.mlir`

- [ ] **Step 1: Confirm the source and destination state**

Run:

```bash
test -f LearningSteps/Practices/mlir_ir_ssa_notes.md
test -f LearningSteps/Practices/ir_ssa_walk.mlir
test ! -e LearningSteps/Practices/mlir_ir_ssa_exercise.md
```

Expected: exit status `0` with no output.

- [ ] **Step 2: Record the answer document checksum**

Run:

```bash
sha256sum LearningSteps/Practices/mlir_ir_ssa_notes.md
```

Expected: one checksum retained for comparison after creating the exercise.

### Task 2: Create the Guided Exercise

**Files:**
- Create: `LearningSteps/Practices/mlir_ir_ssa_exercise.md`

- [ ] **Step 1: Create the answer-free worksheet**

Create `LearningSteps/Practices/mlir_ir_ssa_exercise.md` with this exact content using `apply_patch`:

```markdown
# `ir_ssa_walk.mlir`：SSA use-def 填空练习

对应输入文件：[`ir_ssa_walk.mlir`](./ir_ssa_walk.mlir)

这份练习用于手工分析输入 IR。先独立填写所有表格，再与完整分析笔记核对；练习过程中不要修改输入文件。

## 1. 区分两种 Value

先总结 `OpResult` 和 `BlockArgument` 的判断规则。

| 类别 | 如何从 IR 中识别 | owner 是什么 | 是否有 defining operation |
|---|---|---|---|
| OpResult |  |  |  |
| BlockArgument |  |  |  |

## 2. Value 清单

逐行填写每个 Value 的类别、owner、类型、定义位置和 users。`owner` 必须写出具体 operation 或 block；`users` 应写使用它的 operation，并在必要时标明 operand 的作用。

| Value | 类别 | owner | type | 定义位置 | users |
|---|---|---|---|---|---|
| `%cond` |  |  |  |  |  |
| `%n` |  |  |  |  |  |
| `%base` |  |  |  |  |  |
| `%shared` |  |  |  |  |  |
| `%selected` |  |  |  |  |  |
| `%then_value` |  |  |  |  |  |
| `%else_value` |  |  |  |  |  |
| `%zero` |  |  |  |  |  |
| `%one` |  |  |  |  |  |
| `%result` |  |  |  |  |  |
| `%iv` |  |  |  |  |  |
| `%acc` |  |  |  |  |  |
| `%next` |  |  |  |  |  |

## 3. SSA use-def 关系

先记录真正的 operand use，再单独记录 SCF 在 region 边界上传递值的语义关系。不要把语义映射重复计入 `Value::use`。

### 3.1 Operand uses

| # | Value | user operation | operand 的作用 | use 次数依据 |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |
| 6 |  |  |  |  |
| 7 |  |  |  |  |
| 8 |  |  |  |  |
| 9 |  |  |  |  |
| 10 |  |  |  |  |
| 11 |  |  |  |  |
| 12 |  |  |  |  |
| 13 |  |  |  |  |
| 14 |  |  |  |  |
| 15 |  |  |  |  |
| 16 |  |  |  |  |
| 17 |  |  |  |  |
| 18 |  |  |  |  |

### 3.2 SCF 语义映射

| # | 外层值或 operation | region 内对应对象 | 映射发生的时机 | 是否属于额外的 `Value::use` |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

### 3.3 Uses 与 users

| Value | uses 数量 | users 数量 | 两者是否相同 | 从 IR 中找到的证据 |
|---|---|---|---|---|
| `%shared` |  |  |  |  |
| `%base` |  |  |  |  |
| `%selected` |  |  |  |  |

## 4. 手工分析记录

按轮次扫描 IR，并记录本轮新发现的信息。不要只写最终结论。

| 扫描轮次 | 本轮任务 | 新发现或修正 |
|---|---|---|
| 1 | 登记所有 block/function/region 参数 |  |
| 2 | 登记所有 operation results |  |
| 3 | 填写每个 Value 的 type 和 owner |  |
| 4 | 扫描 operands，累计 uses 和 users |  |
| 5 | 检查作用域和 dominance |  |

## 5. Scope 与 Dominance

对每个案例判断 Value 在目标位置是否可见、定义是否支配 use，并写出理由。

| Value | 定义所在区域或 block | 待检查的使用位置 | 是否可见 | 是否满足 dominance | 理由 |
|---|---|---|---|---|---|
| `%base` |  | then region |  |  |  |
| `%base` |  | else region |  |  |  |
| `%base` |  | loop body |  |  |  |
| `%then_value` |  | else region |  |  |  |
| `%then_value` |  | `scf.if` 之后 |  |  |  |
| `%selected` |  | loop 初始化位置 |  |  |  |
| `%acc` |  | loop body |  |  |  |
| `%next` |  | loop body 的 `scf.yield` |  |  |  |

## 6. 自查清单

| 检查项 | 完成情况 | 备注 |
|---|---|---|
| 已登记输入 IR 中的全部 SSA Value |  |  |
| 已区分 OpResult 与 BlockArgument |  |  |
| 已为每个 Value 写出 owner 和 type |  |  |
| 已区分 use 次数与 user 数量 |  |  |
| 已区分 operand use 与 SCF 语义映射 |  |  |
| 已检查作用域和 dominance |  |  |
```

Expected: the file contains guided prompts and blank answer cells, with no completed Mermaid graph or copied conclusions.

### Task 3: Add the Exercise to the LearningSteps Index

**Files:**
- Modify: `LearningSteps/README.md`

- [ ] **Step 1: Add the exercise beside its answer note**

Replace:

```markdown
- [SSA use-def 输入](Practices/ir_ssa_walk.mlir)与[分析笔记](Practices/mlir_ir_ssa_notes.md)
```

with:

```markdown
- SSA use-def：[输入](Practices/ir_ssa_walk.mlir) · [填空练习](Practices/mlir_ir_ssa_exercise.md) · [完整分析](Practices/mlir_ir_ssa_notes.md)
```

Expected: the exercise and answer document are clearly distinguished.

### Task 4: Verify the No-Answer Contract

**Files:**
- Verify: `LearningSteps/Practices/mlir_ir_ssa_notes.md`
- Verify: `LearningSteps/Practices/mlir_ir_ssa_exercise.md`
- Verify: `LearningSteps/README.md`

- [ ] **Step 1: Confirm the original checksum is unchanged**

Run:

```bash
sha256sum LearningSteps/Practices/mlir_ir_ssa_notes.md
```

Expected: exactly the checksum recorded in Task 1.

- [ ] **Step 2: Validate the inventory and blank cells**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
import re

path = Path("LearningSteps/Practices/mlir_ir_ssa_exercise.md")
text = path.read_text(encoding="utf-8")
expected = [
    "%cond", "%n", "%base", "%shared", "%selected", "%then_value",
    "%else_value", "%zero", "%one", "%result", "%iv", "%acc", "%next",
]
inventory = text.split("## 2. Value 清单", 1)[1].split("## 3.", 1)[0]
for value in expected:
    row = rf"^\| `{re.escape(value)}` \|  \|  \|  \|  \|  \|$"
    if not re.search(row, inventory, re.MULTILINE):
        raise SystemExit(f"missing blank inventory row: {value}")
if "```mermaid" in text:
    raise SystemExit("completed Mermaid graph leaked into exercise")
for leaked in ("arith.constant 10", "共两个 users", "支配三个", "scf.yield 本身没有 result"):
    if leaked in text:
        raise SystemExit(f"answer text leaked: {leaked}")
print("all 13 values have blank inventory rows; no known answers leaked")
PY
```

Expected: `all 13 values have blank inventory rows; no known answers leaked`.

- [ ] **Step 3: Validate links and formatting**

Run:

```bash
test -f LearningSteps/Practices/ir_ssa_walk.mlir
rg -q 'Practices/mlir_ir_ssa_exercise.md' LearningSteps/README.md
git diff --check
if rg -n '[[:blank:]]+$' \
  LearningSteps/Practices/mlir_ir_ssa_exercise.md LearningSteps/README.md; then
  exit 1
fi
```

Expected: exit status `0` with no output.

- [ ] **Step 4: Leave the exercise uncommitted for user review**

Do not stage or commit the exercise automatically. The worktree still contains the previously approved LearningSteps reorganization and RTX4090 changes; show the new file and README diff separately in the final handoff.
