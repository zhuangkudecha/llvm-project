# MLIR PatternRewriter.md - 学习总结

## 概述

本文总结 `mlir/docs/PatternRewriter.md` 的核心内容。这个文档讲的是 MLIR 的通用 DAG-to-DAG 重写基础设施，也就是很多 MLIR 变换背后的共同机制：

```text
RewritePattern
  定义一个局部 IR 匹配和改写规则

PatternRewriter
  负责创建、替换、删除、原地修改 Operation

PatternApplicator / Pattern Driver
  选择哪些 pattern 应用到哪些 operation 上

Greedy / Walk / DialectConversion Driver
  提供不同场景下的 pattern 应用策略
```

如果从 AI 编译器学习角度看，这篇文档回答的问题是：

> MLIR 如何把 `linalg.matmul -> scf/vector/memref/llvm` 这类局部 IR 变换写成可组合、可调试、可由 pass driver 自动调度的 rewrite pattern？

可以把它放在这条主线里理解：

```text
PassManager
  -> Pass
      -> RewritePatternSet
          -> RewritePattern::matchAndRewrite
              -> PatternRewriter 改 IR
                  -> Driver 继续调度下一轮匹配
```

---

## 一、PatternRewriter.md 在讲什么

这篇文档不是在讲某一个具体优化，而是在讲 MLIR 的 **pattern rewriting framework**。

它把 IR 变换拆成两部分：

1. **Pattern Definition**：如何定义一个 pattern，说明它匹配什么、收益是多少、如何改写 IR。
2. **Pattern Application**：如何把一组 pattern 交给 driver，由 driver 决定遍历顺序、收益模型、是否重复应用、何时停止。

对应到实际编译器工作：

```text
你写的代码:
  "看到 A 形状的 IR，就改写成 B 形状的 IR"

MLIR 框架负责:
  "什么时候看哪个 op、pattern 谁先试、改完之后是否重新入队、如何防止无限递归"
```

这也是为什么 `canonicalize`、dialect conversion、部分 lowering pass 都大量使用 pattern。

---

## 二、RewritePattern：定义一个局部改写规则

Pattern 通过继承 `RewritePattern` 定义。一个 pattern 主要包含几个信息：

```text
RewritePattern
  -> Benefit
  -> Root Operation Name
  -> matchAndRewrite()
  -> recursion/debug/initialization metadata
```

### 2.1 Benefit：pattern 的静态收益

`Benefit` 表示应用这个 pattern 的预期收益。driver 会用它来决定多个 pattern 都能匹配时谁优先。

注意：benefit 在 pattern 构造后是静态的，不能在每次 match 时动态变化。

文档给出的理由是：静态 benefit 便于 pattern fusion，也便于把 pattern 编译成更高效的状态机。如果确实需要不同收益，可以构造多个 pattern 实例，并用 match predicate 区分不同场景。

可以简单理解为：

```text
PatternBenefit = 这个规则在局部竞争中的优先级
```

### 2.2 Root Operation Name：pattern 匹配的根 op

pattern 可以声明自己只匹配某种 root operation：

```cpp
RewritePattern(MyOp::getOperationName(), benefit, context)
```

这样 driver 只会把 `MyOp` 交给这个 pattern。

如果不指定 root operation，pattern 可以匹配任意 operation，但需要显式使用 `MatchAnyOpTypeTag` 表达这个意图：

```cpp
RewritePattern(benefit, MatchAnyOpTypeTag())
```

学习时建议优先写有 root op 的 pattern，因为：

- driver 更容易过滤候选 pattern
- cost model 更容易分析
- debug 输出更清楚
- 不容易误匹配无关 op

### 2.3 matchAndRewrite：匹配和改写的核心

`matchAndRewrite` 同时承担两个职责：

```text
先判断当前 op 是否匹配
匹配成功后，通过 PatternRewriter 修改 IR
```

典型结构：

```cpp
class MyPattern : public RewritePattern {
public:
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // 1. 检查 op 是否满足条件
    // 2. 条件不满足时 return failure()
    // 3. 条件满足后再通过 rewriter 修改 IR
    // 4. 修改成功后 return success()
  }
};
```

这里最重要的约束是：

> 在确认 match 成功之前，不应该修改 IR。

也就是说，不要先改一点 IR，然后发现条件不满足再试图回滚。普通 pattern 的 `matchAndRewrite` 不是事务系统。

---

## 三、Pattern 的三个核心限制

原文列了三个很关键的 restriction，实际写 MLIR pattern 时经常踩坑。

### 3.1 所有 IR 修改都必须通过 PatternRewriter

不能绕过 rewriter 直接改 IR：

```cpp
// 不推荐：绕过 driver 的通知机制
op->erase();
```

应该写：

```cpp
rewriter.eraseOp(op);
```

原因是 pattern driver 内部维护了 worklist、监听回调、递归检测、debug tracing 等状态。你绕过 `PatternRewriter` 修改 IR，driver 可能不知道某个 op 被删了、替换了、或者新 op 应该重新入队。

所以要记住：

```text
PatternRewriter 不只是 OpBuilder。
它是 pattern 和 driver 之间的协议层。
```

### 3.2 root operation 必须被处理掉或原地更新

一次成功的 rewrite，root op 必须满足三者之一：

- 被原地更新
- 被替换
- 被删除

这能让 driver 明确知道这个 pattern 对当前 root op 做了什么。

### 3.3 success 必须等价于 IR 被修改

`matchAndRewrite` 的返回值语义是：

```text
success <=> IR 被实际修改
failure <=> 没有修改 IR
```

错误写法：

```cpp
if (matchedButNoNeedToChange)
  return success(); // 错
```

如果没有修改 IR，就应该返回 `failure()`。否则 greedy driver 可能认为发生了进展，导致多余迭代甚至无限循环。

---

## 四、递归应用：为什么需要 setHasBoundedRewriteRecursion

有些 pattern 可能会匹配自己刚生成的结果。

例如一个“剥离循环一轮迭代”的 pattern：

```text
scf.for 10 次
  -> peel 1 次
scf.for 9 次
  -> peel 1 次
scf.for 8 次
  -> ...
```

从 pattern 作者角度看，这个递归是有界的，因为循环次数会下降。但 driver 不一定知道这一点。

MLIR 默认保守假设：

```text
pattern 递归应用可能是 bug
```

如果 pattern 明确支持有界递归，需要在初始化中声明：

```cpp
void initialize() {
  setHasBoundedRewriteRecursion();
}
```

否则 driver 检测到递归应用时可能失败，用来防止无限 rewrite。

---

## 五、Debug Name 和 Label：让 pattern 可调试、可过滤

pattern 可以设置 debug name 和 debug labels：

```cpp
class MyPattern : public RewritePattern {
public:
  using RewritePattern::RewritePattern;

  void initialize() {
    setDebugName("MyPattern");
    addDebugLabels("MyRewritePass");
  }
};
```

也可以在加入 pattern set 时加 label：

```cpp
patterns.addWithLabel<MyPattern>("MyRewritePatterns", ctx);
```

这些信息主要用于：

- debug log 中显示 pattern 名称
- 通过命令行过滤启用或禁用某些 pattern
- 缩小 testcase 时只跑一部分 pattern

实际开发建议：给复杂 lowering/canonicalization pattern 都加上稳定 debug name。否则 debug greedy rewriter 时只看到一堆匿名 pattern，很难定位问题。

---

## 六、Pattern 初始化和构造

文档建议使用：

```cpp
RewritePattern::create<T>(...)
```

或者通过 `RewritePatternSet` 的 `add` 系列接口构造 pattern，因为这些工具会确保 pattern 被正确初始化。

pattern 的额外状态可以放在 `initialize()` hook 中：

```cpp
class MyPattern : public RewritePattern {
public:
  using RewritePattern::RewritePattern;

  void initialize() {
    setHasBoundedRewriteRecursion();
    setDebugName("MyPattern");
  }
};
```

这样就不用为了设置 metadata 重写一堆构造函数。

---

## 七、PatternRewriter：pattern 修改 IR 的唯一入口

`PatternRewriter` 是 pattern 和 driver 的通信接口。它继承自 `OpBuilder`，所以也能创建 operation、type、attribute。

常用 API 可以按行为分成几类：

### 7.1 删除 operation

```cpp
rewriter.eraseOp(op);
```

要求被删除 op 没有结果，或者结果已经没有 uses。

### 7.2 替换 operation

```cpp
rewriter.replaceOp(op, newValues);
rewriter.replaceOpWithNewOp<NewOp>(op, ...);
```

这会把旧 op 的结果使用者改到新 value 上，然后删除旧 op。

示意：

```text
改写前:
  %0 = old.op
  use %0

replaceOp 后:
  %1 = new.op
  use %1
```

### 7.3 原地修改 operation

如果只是改属性、operand、successor 等，可以用原地修改事务：

```cpp
rewriter.startOpModification(op);
// 修改 op 的属性、operand、successor 等
rewriter.finalizeOpModification(op);
```

也可以用包装 API：

```cpp
rewriter.modifyOpInPlace(op, [&] {
  // 修改 op
});
```

如果中途发现不能修改，可以 `cancelOpModification`。

这个事务式 API 的意义是：driver 可以准确知道一个 op 发生了 in-place update，并决定是否重新处理它。

### 7.4 解释 match 失败原因

```cpp
return rewriter.notifyMatchFailure(op, "reason");
```

这对 debug 很有用。配合 greedy rewriter debug log，可以看到某个 pattern 为什么没有匹配。

---

## 八、Pattern Application：driver 如何应用 pattern

定义 pattern 只是第一步。真正执行时，需要把 pattern 放进 `RewritePatternSet`，再交给某个 driver。

整体流程：

```text
RewritePatternSet
  -> FrozenRewritePatternSet
      -> PatternApplicator
          -> applyCostModel
              -> matchAndRewrite(op, rewriter)
```

driver 通常负责：

- 初始化 worklist 或遍历顺序
- 提供 driver-specific `PatternRewriter`
- 定义 cost model
- 调用 `PatternApplicator::matchAndRewrite`
- 根据 IR 修改结果更新 worklist
- 控制是否迭代到 fixed point

可以把职责拆开理解：

```text
Pattern:
  我知道如何把一个局部 IR 形状改成另一个形状

Driver:
  我知道以什么顺序尝试这些 pattern，以及改完后是否继续尝试
```

---

## 九、常见 Pattern Driver

MLIR 提供了几个常用 driver，适合不同场景。

### 9.1 Dialect Conversion Driver

Dialect conversion driver 用于 dialect 之间或 dialect 内部的合法化转换。

它的核心概念是：

```text
ConversionTarget
  定义哪些 op/type 是合法的

RewritePattern
  把非法 op 改写成合法 op

TypeConverter
  处理类型转换
```

典型场景：

```text
tosa / stablehlo
  -> linalg / tensor
  -> scf / memref
  -> llvm / gpu
```

如果你的目标是“把某个 dialect 系统性 lowering 到另一个 dialect”，通常应该考虑 dialect conversion，而不是手写 walk。

### 9.2 Walk Pattern Rewrite Driver

Walk driver 是快速、简单的 driver：

- 对给定 op 的 region 做 post-order 遍历
- 不访问容器 op 本身
- 每个位置选择局部 benefit 最高的 pattern
- 不重新访问被修改或新替换出来的 op
- 不支持同一个 op 的渐进式多轮 rewrite

它适合简单、单次遍历能完成的局部改写。

如果 pattern 需要不断把新 op 重新入队、反复应用到 fixed point，就应该用 greedy driver。

调试参数：

```bash
--debug-only=walk-rewriter
```

### 9.3 Greedy Pattern Rewrite Driver

Greedy driver 是最常见的 pattern driver 之一，`canonicalize` pass 就使用它。

它的特点：

- 使用 worklist 驱动
- 局部选择 benefit 最高的 pattern
- pattern 应用后会继续处理受影响的 op
- 迭代直到 fixed point 或达到最大迭代次数
- 支持配置初始遍历顺序和 strict mode

两个常用入口：

```cpp
applyPatternsGreedily(...)
applyOpPatternsGreedily(...)
```

可以理解为：

```text
Walk driver:
  走一遍，能改就改，不追着新结果反复改

Greedy driver:
  维护 worklist，改完继续看，直到没有 pattern 能继续应用
```

这就是为什么 canonicalization 适合 greedy driver：规范化通常需要多步局部简化逐渐暴露机会。

例如：

```text
(x + 0) * 1
  -> x * 1
  -> x
```

第一步 rewrite 后，第二个 rewrite 机会才出现。

---

## 十、GreedyRewriteConfig 和 strict mode

Greedy driver 可以通过 `GreedyRewriteConfig` 配置。

区域级 greedy driver 的初始 worklist 可以有两种构建方式：

```text
Top-down:
  从容器向内部 pre-order 遍历
  通常编译时间更高效

Bottom-up:
  默认模式，postorder 后反转
  对存在歧义的大 pattern set，可能更容易先匹配大结构
```

strict mode 控制哪些 op 可以被重新加入 worklist：

```text
GreedyRewriteStrictness::AnyOp
  不额外排除 op，除非超出 driver scope

GreedyRewriteStrictness::ExistingAndNewOps
  只处理初始已有 op 和新创建 op

GreedyRewriteStrictness::ExistingOps
  只处理初始已有 op
```

这里的重点不是背枚举，而是理解 worklist 范围：

> greedy driver 不是“全 IR 无限乱跑”，它有 scope、strictness、最大迭代次数等边界。

---

## 十一、调试 pattern rewrite

### 11.1 greedy rewriter debug log

可以用：

```bash
-debug-only=greedy-rewriter
```

这会打印树状 debug 输出，展示：

- 当前处理哪个 operation
- 尝试了哪些 pattern
- 哪些 pattern 失败
- 失败原因是什么
- 哪个 pattern 成功
- 插入、替换、删除了哪些 op

文档中的例子是 `cf.cond_br` 被简化成 `cf.br`：

```text
Processing operation : 'cf.cond_br' {
  * Pattern SimplifyConstCondBranchPred -> failure
  * Pattern SimplifyCondBranchIdenticalSuccessors {
    ** Insert  : 'cf.br'
    ** Replace : 'cf.cond_br'
  } -> success
}
```

这类输出对定位 pattern 为什么没触发非常有用。

### 11.2 Pattern filtering

`FrozenRewritePatternSet` 支持按 debug name 或 label 过滤 pattern：

```cpp
FrozenRewritePatternSet(rewritePatterns,
                        disabledPatterns,
                        enabledPatterns);
```

语义：

```text
disabledPatterns:
  指定禁用哪些 pattern name/label

enabledPatterns:
  只启用哪些 pattern name/label

如果一个 pattern 同时命中 disabled 和 enabled:
  disabled 优先，仍然被过滤掉
```

这对缩小 testcase 很有用。比如某个 canonicalization 结果不对，可以只启用一个 label 下的 pattern 看行为。

### 11.3 通用 rewrite pass options

MLIR 在 `mlir/Rewrite/PassUtil.td` 中提供了通用 rewrite pass 选项。自定义 pass 可以继承：

```tablegen
def MyRewritePass : Pass<"..."> {
  let summary = "...";
  let constructor = "createMyRewritePass()";
  let options = RewritePassUtils.options;
}
```

常见选项包括：

```text
disable-patterns
enable-patterns
```

这样不同 pass 的 pattern 过滤体验会保持一致。

---

## 十二、和 Pass / Canonicalization / Conversion 的关系

`PatternRewriter.md` 可以和其他 MLIR 文档串起来看：

```text
PassManagement.md
  解释 pass 如何被组织、调度、嵌套、并行执行

PatternRewriter.md
  解释 pass 内部如何用 pattern 描述局部 IR 改写

Canonicalization.md
  解释 canonicalize pass 如何使用 pattern 做规范化

DialectConversion.md
  解释如何用 conversion pattern 做合法化 lowering
```

在真实 MLIR pass 中，经常是这种结构：

```cpp
void MyPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateMyPatterns(patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
```

这说明：

```text
Pass 负责确定作用范围和 pass 生命周期
Pattern 负责描述局部变换规则
Driver 负责把 pattern 应用到 IR 上
PatternRewriter 负责安全地修改 IR
```

---

## 十三、一个最小心智模型

初学时可以先记住下面这张图：

```text
                RewritePatternSet
                        |
                        v
Operation ---> Pattern Driver ---> PatternApplicator
    ^                  |                  |
    |                  v                  v
    |          PatternRewriter <--- matchAndRewrite()
    |                  |
    +------ replace / erase / modify / create
```

含义：

1. driver 选择一个 operation。
2. applicator 根据 root op、benefit、cost model 找候选 pattern。
3. pattern 的 `matchAndRewrite` 判断是否匹配。
4. 匹配成功后，pattern 必须通过 `PatternRewriter` 修改 IR。
5. rewriter 通知 driver IR 发生了什么变化。
6. driver 根据自己的策略决定是否继续处理新 op 或被修改的 op。

---

## 十四、写 pattern 时的实践 checklist

写 MLIR rewrite pattern 时，可以按这个 checklist 检查：

- 是否尽量指定了 root operation name？
- `matchAndRewrite` 是否在确认匹配成功前没有修改 IR？
- 所有 IR 创建、替换、删除、原地修改是否都通过 `PatternRewriter`？
- 返回 `success()` 时是否真的修改了 IR？
- 返回 `failure()` 时是否没有修改 IR？
- root op 是否被替换、删除或原地更新？
- 可能递归应用的 pattern 是否声明了 `setHasBoundedRewriteRecursion()`？
- 复杂 pattern 是否设置了 debug name 和 label？
- 失败原因是否可以用 `notifyMatchFailure` 帮助调试？
- 需要反复应用到 fixed point 时，是否用了 greedy driver？
- 需要 dialect 合法化和类型转换时，是否应该用 dialect conversion driver？

---

## 总结

`PatternRewriter.md` 的核心不是某几个 API，而是一套分工：

```text
RewritePattern:
  描述局部 IR 形状和改写规则

PatternRewriter:
  作为唯一合法的 IR 修改入口，并把修改通知 driver

PatternApplicator:
  根据 root op、benefit、cost model 选择 pattern

Pattern Driver:
  控制遍历顺序、worklist、递归、fixed point 和调试输出
```

对 AI 编译器来说，PatternRewriter 是理解 MLIR lowering 的关键入口。很多看起来复杂的 lowering pipeline，本质上都是大量 pattern 在 driver 控制下，把高层 dialect 的 operation 一步步改写成更低层 dialect 的 operation。

