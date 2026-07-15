# MLIR Phase 2: AI 编译器初学者入门总结

## 这份文档解决什么问题

`Phase2_Core_Infrastructure_Summary.md` 更像源码复盘笔记，适合已经知道 MLIR 基本概念后回头查类和接口。作为 AI 编译器初学者，建议先读这份导读：它不从 C++ 类层次开始，而是从一次 AI 模型 IR 变换开始，解释 Phase 2 中每个组件为什么存在。

AI 编译器里的典型流程可以简化成：

```text
模型图 / Tensor 程序
  -> 高层 MLIR Dialect        例如 tensor、linalg、stablehlo、tosa
  -> 中层结构化计算 Dialect   例如 linalg、scf、affine、memref
  -> 低层目标 Dialect         例如 llvm、gpu、nvvm、spirv
  -> 目标代码 / Runtime 调用
```

Phase 2 关注的是：MLIR 框架如何承载这些 IR，如何调度 Pass，如何用 Pattern 改写 IR。

---

## 一、先抓住一条主线

把 Phase 2 的所有类放进一条变换链里：

```text
PassManager
  选择并调度一组 Pass

Pass / OperationPass
  在 module、func 或某类 operation 上执行

RewritePattern
  匹配某个局部 IR 形状

PatternRewriter / OpBuilder
  创建新 Operation，替换旧 Operation

Value / Use-Def
  自动把旧结果的使用者改到新结果上

Dialect
  决定每个 Operation、Type、Attribute 的语义
```

一个 AI 编译器里的例子：

```text
linalg.matmul
  -> tiling pass 把大矩阵乘法切成 tile
  -> vectorization pass 变成 vector op
  -> lowering pass 降到 scf/memref/llvm/gpu
```

每一步本质上都是：某个 Pass 找到一段 IR，用 Pattern 或手写逻辑改写成另一段 IR。

---

## 二、Operation：所有 IR 的基本单位

在 MLIR 里，不管是高层的 `linalg.matmul`，还是低层的 `llvm.call`，都是 Operation。

一个 Operation 通常包含：

- 名字：例如 `linalg.matmul`、`arith.addi`、`func.func`
- 输入值：operands
- 输出值：results
- 属性：attributes，例如常量、布局、符号名
- Region：嵌套的代码区域，例如函数体、循环体、if 分支

初学时可以先记住：

```text
Operation = 一条 IR 指令，也可以是一个带嵌套结构的大容器
```

这就是 MLIR 比传统 LLVM IR 更适合 AI 编译器的原因之一：高层算子、循环、函数、模块都能用统一的 Operation 模型表示。

---

## 三、Value：AI 张量数据流怎么连起来

MLIR 使用 SSA。每个计算结果是一个 Value，每个 Value 只定义一次，可以被多个 Operation 使用。

Value 主要有两类来源：

- `OpResult`：某个 Operation 的结果
- `BlockArgument`：某个 Block 的参数，常用于函数参数、循环迭代变量、控制流传值

例如：

```mlir
%0 = arith.addi %a, %b : i32
%1 = arith.muli %0, %0 : i32
```

这里 `%0` 是 `arith.addi` 的结果，也是 `arith.muli` 的输入。优化 Pass 替换 `%0` 时，MLIR 可以通过 use-def 链找到所有使用者。

对 AI 编译器来说，Value 就是张量和标量数据流的连线。

---

## 四、Region 和 Block：为什么 MLIR 能表示复杂控制流

AI 模型不总是简单的直线图。现代模型里会有控制流、循环、动态 shape、函数调用。MLIR 用 Region 和 Block 表示这些结构。

```text
Operation
  -> Region
      -> Block
          -> Operation
```

例如 `scf.for` 是一个 Operation，但它内部有 Region 表示循环体；`func.func` 是一个 Operation，但它内部有 Region 表示函数体。

这让 MLIR 可以在同一套 IR 系统里表示：

- 高层张量算子
- 函数
- if/for 控制流
- affine loop
- low-level memory operation

---

## 五、Dialect：AI 编译器为什么需要多层 IR

Dialect 是 MLIR 的扩展机制。每个 Dialect 定义一组 Operation、Type、Attribute。

可以把 Dialect 理解为“某个抽象层的语言”：

| Dialect | 适合表达什么 |
|---------|--------------|
| `tensor` | 不可变张量值 |
| `linalg` | 结构化线性代数计算 |
| `scf` | 结构化控制流 |
| `affine` | 可分析的仿射循环和访存 |
| `memref` | 显式内存视图 |
| `vector` | SIMD/vector 计算 |
| `gpu` / `nvvm` / `spirv` | GPU 目标相关 IR |
| `llvm` | 接近 LLVM IR 的低层表示 |

AI 编译器不是一次从模型图跳到机器码，而是不断在 Dialect 之间 lowering。

---

## 六、PassManager：谁来安排变换顺序

PassManager 负责组织和运行 Pass。

一个典型 pipeline 可能长这样：

```text
builtin.module(
  canonicalize,
  func.func(cse),
  convert-linalg-to-loops,
  lower-affine,
  convert-scf-to-cf,
  convert-func-to-llvm
)
```

要点：

- Module 级 Pass 处理整个模块
- Function 级 Pass 可以嵌套在 `func.func(...)` 里
- PassManager 还负责验证、分析缓存、失败传播、多线程调度等基础设施

初学时不用先记所有 API，先理解“PassManager 是编译管线的调度器”。

---

## 七、PatternRewriter：局部 IR 怎么被改写

很多 AI 编译优化都是局部改写：

```text
transpose(transpose(x)) -> x
x + 0 -> x
高层 matmul -> 循环 nest
tensor op -> memref op
```

MLIR 用 Pattern 描述“匹配什么”，用 PatternRewriter 描述“替换成什么”。

```text
RewritePattern
  判断这个 Operation 是否能改

PatternRewriter
  创建新 Operation
  替换旧 Operation
  删除死 Operation
  通知 driver 更新 worklist
```

这套机制的价值是：开发者专注写局部规则，框架负责调度、use-def 更新和收敛控制。

---

## 八、Canonicalization：不是正确性保证，而是清理 IR

Canonicalization 是 MLIR 里非常常见的清理 Pass。

它会做这类事情：

- `x + 0 -> x`
- 常量折叠
- 删除无用结果
- 简化平凡控制流
- 把 IR 变成更容易匹配的形式

但要注意：Canonicalizer 是 best-effort，不保证把所有 IR 都变成唯一标准形式。一个正确的 lowering 或 optimization Pass 不能依赖 canonicalizer 才正确；canonicalizer 只能提升后续优化效果。

---

## 九、初学者推荐阅读顺序

建议不要一开始就深挖 `TrailingObjects` 和 `StorageUniquer`。推荐顺序：

1. 读一段 `.mlir`，确认能看懂 Operation、Value、Type。
2. 看 `docs/Tutorials/UnderstandingTheIRStructure.md`，建立 IR 层级模型。
3. 看 `docs/LangRef.md` 里和 Operation/Region/Block/Value 相关的部分。
4. 跑 `mlir-opt --canonicalize --cse`，观察 IR 怎么变化。
5. 看 `docs/PassManagement.md`，理解 PassManager 和嵌套 pipeline。
6. 看 `docs/PatternRewriter.md`，理解 Pattern 如何局部改写 IR。
7. 回到 `Phase2_Core_Infrastructure_Summary.md`，把源码类和前面的概念对应起来。

---

## 十、读源码时抓哪些入口

初学时先看这些入口就够了：

| 目标 | 文件 |
|------|------|
| IR 基本单位 | `include/mlir/IR/Operation.h` |
| SSA 值 | `include/mlir/IR/Value.h` |
| 嵌套结构 | `include/mlir/IR/Region.h`, `include/mlir/IR/Block.h` |
| Dialect | `include/mlir/IR/Dialect.h` |
| IR 构建 | `include/mlir/IR/Builders.h` |
| Pass 基类 | `include/mlir/Pass/Pass.h` |
| Pass 管线 | `include/mlir/Pass/PassManager.h` |
| Pattern | `include/mlir/IR/PatternMatch.h` |
| Greedy driver | `include/mlir/Transforms/GreedyPatternRewriteDriver.h` |

---

## 十一、最小心智模型

如果只记一张图，记这个：

```text
Dialect 定义 IR 语义
Operation 承载 IR 节点
Value 连接数据流
Region/Block 表示嵌套结构和控制流
PassManager 调度变换
PatternRewriter 改写局部 IR
Canonicalizer 清理常见冗余形式
```

对 AI 编译器来说，这些机制共同支撑“高层张量程序逐步降低到可执行目标代码”的过程。
