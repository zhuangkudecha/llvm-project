# `ir_ssa_walk.mlir`：SSA use-def 填空练习

对应输入文件：[`ir_ssa_walk.mlir`](./ir_ssa_walk.mlir)

这份练习用于手工分析输入 IR。先独立填写所有表格，再与完整分析笔记核对；练习过程中不要修改输入文件。

## 1. 区分两种 Value

先总结 `OpResult` 和 `BlockArgument` 的判断规则。

| 类别 | 如何从 IR 中识别 | owner 是什么 | 是否有 defining operation |
|---|---|---|---|
| OpResult | 由Operation 产生 | 产生OpResult的Operation | 是 |
| BlockArgument | Operation的参数，一般在operation 右侧 | 他所在的Block | 否 |

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
