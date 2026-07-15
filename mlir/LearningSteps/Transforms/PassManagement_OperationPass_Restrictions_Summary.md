# MLIR OperationPass 并行限制 - 学习总结

## 概述

本文总结 `OperationPass` 在 MLIR PassManager 中的几条重要限制，重点解释：

- 什么是 current operation
- 什么是 sibling operation / sibling function
- 为什么 `func.func` pass 不能随便访问其他函数
- 为什么 pass 不能保存跨 `runOnOperation()` 的可变状态
- 为什么 pass 必须可以复制
- 如果确实需要全局视角，应该如何选择 pass 运行层级

核心结论：

> `OperationPass` 的读写范围必须和它的运行层级一致。  
> 如果 pass 跑在 `func.func` 上，就只能把当前函数及其内部 IR 当成自己的工作区；如果需要看整个 module，就应该写 `ModuleOp` pass。

---

## 一、源码位置

官方文档：

```text
mlir/docs/PassManagement.md
```

核心接口：

```text
mlir/include/mlir/Pass/Pass.h
```

`OperationPass` 的源码注释强调：

```cpp
/// Operation passes must not:
///   - modify any other operations within the parent region, as other threads
///     may be manipulating them concurrently.
///   - modify any state within the parent operation, this includes adding
///     additional operations.
```

这说明这些限制不是风格建议，而是 MLIR PassManager 并行执行模型的一部分。

---

## 二、current operation 是什么

在 `runOnOperation()` 中：

```cpp
void runOnOperation() override {
  auto op = getOperation();
}
```

`getOperation()` 返回的就是当前 pass 正在处理的 operation，也就是文档里说的 **current operation**。

例如：

```cpp
struct MyFuncPass
    : public PassWrapper<MyFuncPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
  }
};
```

这个 pass 的 current operation 就是某一个 `func.func`。

如果 IR 是：

```mlir
module {
  func.func @foo() {
    %0 = arith.constant 1 : i32
    return
  }

  func.func @bar() {
    %0 = arith.constant 2 : i32
    return
  }
}
```

当 `runOnOperation()` 正在处理 `@foo` 时：

```text
current operation = func.func @foo
```

---

## 三、什么是 sibling function

`sibling` 的意思是“同级节点”。在 MLIR IR 树中，如果两个 operation 有同一个 parent，它们就是 sibling operations。

例如：

```text
module
├── func.func @foo
├── func.func @bar
└── func.func @baz
```

`@foo`、`@bar`、`@baz` 都直接位于同一个 `module` 下，所以它们互相是 sibling functions，也可以理解为：

```text
兄弟函数 / 同级函数
```

如果当前 pass 跑在 `@foo` 上：

```text
module          = parent / ancestor operation
func.func @foo  = current operation
func.func @bar  = sibling operation
func.func @baz  = sibling operation
```

所以在 `OperationPass<func::FuncOp>` 中，`@bar` 和 `@baz` 就是当前函数的 sibling functions。

---

## 四、为什么不能检查 sibling operation

限制：

```text
Must not inspect the state of operations that are siblings of the current operation.
Must neither access operations nested under those siblings.
```

含义是：

- 当前 pass 处理 `@foo` 时，不能读取 `@bar` 的状态
- 也不能读取 `@bar` 内部嵌套的 operation
- 但可以读取 parent / ancestor operation 的状态

原因是 PassManager 可能并行执行：

```text
线程 1: MyFuncPass 处理 func.func @foo
线程 2: MyFuncPass 处理 func.func @bar
线程 3: MyFuncPass 处理 func.func @baz
```

如果线程 1 在处理 `@foo` 时去读取 `@bar`，而线程 2 正在修改 `@bar`，就会产生数据竞争，可能读到不稳定的 IR 状态。

错误示例：

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();
  ModuleOp module = current->getParentOfType<ModuleOp>();

  for (auto func : module.getOps<func::FuncOp>()) {
    // 错误：func 可能是 current 的 sibling function。
    // 当前 pass 不能检查 sibling function 的 body。
    func.walk([](Operation *op) {
      // inspect op
    });
  }
}
```

正确示例：

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();

  // 正确：只访问当前函数内部的 IR。
  current.walk([](Operation *op) {
    // inspect op nested under current function
  });

  // 正确：允许读取 ancestor operation 的状态。
  ModuleOp module = current->getParentOfType<ModuleOp>();
  Attribute attr = module->getAttr("some_attr");
}
```

注意：允许读取 parent / ancestor 的状态，不代表可以随便修改 parent。

---

## 五、为什么不能修改 current operation 之外的 IR

限制：

```text
Must not modify the state of operations other than the operations that are
nested under the current operation.
```

含义是：

- 可以修改 current operation 内部嵌套的 IR
- 不能修改 sibling operation
- 不能往 parent block 中添加、删除、移动 operation
- 不能修改 parent operation 的结构

例如当前 pass 跑在 `func.func @foo` 上：

```text
module
├── func.func @foo   <- current operation
└── func.func @bar   <- sibling operation
```

错误示例：在 `func.func` pass 里往 `module` 里新增函数。

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();
  ModuleOp module = current->getParentOfType<ModuleOp>();

  OpBuilder builder(module.getBodyRegion());

  // 错误：这是在 parent module 中添加新的 sibling operation。
  builder.create<func::FuncOp>(current.getLoc(), "new_func",
                               builder.getFunctionType({}, {}));
}
```

正确示例：只修改当前函数内部的 operation。

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();

  current.walk([&](arith::AddIOp add) {
    // 正确：arith.addi 是嵌套在当前 func 内部的 op。
    add->setAttr("visited", UnitAttr::get(&getContext()));
  });
}
```

---

## 六、为什么 current operation 只能改 attribute

文档里有一个例外：

```text
As an exception, the attributes of the current operation may be modified freely.
This is the only way that the current operation may be modified.
```

也就是说，当前 operation 自己只能自由修改 attributes。

允许：

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();

  // 正确：修改当前 operation 的 attribute。
  current->setAttr("my_pass.visited", UnitAttr::get(&getContext()));
}
```

不允许：

```cpp
void runOnOperation() override {
  func::FuncOp current = getOperation();

  // 不推荐：修改 current operation 的 operands、regions、parent block
  // 这会影响 parent 层面的结构稳定性。
}
```

为什么 attribute 是例外？

因为 attribute 通常是当前 operation 自己携带的元数据，不会改变 parent block 的 operation 链表结构，也不会改变 sibling operation 的遍历边界。

---

## 七、为什么不能保存跨 runOnOperation 的可变状态

限制：

```text
Must not maintain mutable pass state across invocations of runOnOperation.
```

错误示例：

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  int functionIndex = 0;
  SmallVector<StringRef> seenFunctions;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // 错误：依赖跨 runOnOperation 的可变状态。
    seenFunctions.push_back(func.getName());
    ++functionIndex;
  }
};
```

这个写法隐含了几个错误假设：

```text
1. 同一个 pass 实例会处理所有函数
2. 函数处理顺序是固定的
3. runOnOperation 是串行执行的
```

但 MLIR 不保证这些。并行情况下，PassManager 可以复制多个 pass 实例：

```text
clone #1 -> 处理 @foo
clone #2 -> 处理 @bar
clone #3 -> 处理 @baz
```

某个具体 pass 实例不一定能看到所有 operation，也不保证执行顺序。

正确写法：把本次处理需要的临时状态放在局部变量中。

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();

  SmallVector<Operation *> worklist;
  func.walk([&](Operation *op) {
    worklist.push_back(op);
  });

  // worklist 只服务这一次 runOnOperation。
}
```

---

## 八、为什么不能维护全局可变状态

限制：

```text
Must not maintain any global mutable state.
```

错误示例：

```cpp
static unsigned NumVisitedFunctions = 0;

void runOnOperation() override {
  ++NumVisitedFunctions;
}
```

问题：

- 多线程下会产生 data race
- 结果依赖 pass 调度顺序
- 多次运行 pipeline 可能得到不同结果
- 多个 MLIRContext 或多个编译任务之间可能互相污染

正确做法：

- pass 配置放在 pass option 中
- 临时状态放在 `runOnOperation()` 局部变量中
- 跨 IR 的信息用 analysis 或者提升 pass 运行层级

示例：

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  Option<int> threshold{*this, "threshold", llvm::cl::init(4)};

  void runOnOperation() override {
    unsigned localCount = 0;

    getOperation().walk([&](Operation *) {
      ++localCount;
    });
  }
};
```

---

## 九、为什么 pass 必须 copy-constructible

限制：

```text
Must be copy-constructible.
```

原因是 PassManager 为了并行执行，可能会复制 pass 实例：

```text
原始 pass
├── clone #1 -> func.func @foo
├── clone #2 -> func.func @bar
└── clone #3 -> func.func @baz
```

所以 pass 不能依赖不可复制的运行时资源，也不能把可变状态设计成“只有一个实例才正确”的形式。

不推荐：

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  std::unique_ptr<MyMutableState> state;

  void runOnOperation() override {
    // 如果 state 代表跨函数共享状态，复制 pass 后语义会变得混乱。
  }
};
```

推荐：

```cpp
struct MyPass : public PassWrapper<MyPass, OperationPass<func::FuncOp>> {
  Option<int> threshold{*this, "threshold", llvm::cl::init(4)};

  void runOnOperation() override {
    LocalAnalyzer analyzer(threshold);
    analyzer.run(getOperation());
  }
};
```

也就是说：

```text
pass 实例中适合保存配置；
runOnOperation 中适合保存本次运行的临时状态。
```

---

## 十、什么时候应该写 ModuleOp pass

如果一个 pass 需要查看或修改多个 sibling functions，就不应该写成 `OperationPass<func::FuncOp>`。

例如需求：

- 统计整个 module 里所有函数
- 根据 `@foo` 的信息修改 `@bar`
- 新增、删除、重命名多个函数
- 构建跨函数调用图
- 在 module 级别统一插入 helper function

这类需求应该写成 `OperationPass<ModuleOp>`：

```cpp
struct MyModulePass
    : public PassWrapper<MyModulePass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 正确：所有 func.func 都 nested under 当前 ModuleOp。
    for (auto func : module.getOps<func::FuncOp>()) {
      // 可以检查这些函数。
    }
  }
};
```

此时 current operation 是整个 `module`：

```text
current operation = module

module
├── func.func @foo   <- nested under current operation
├── func.func @bar   <- nested under current operation
└── func.func @baz   <- nested under current operation
```

因为这些函数都嵌套在当前 `ModuleOp` 下，所以 module pass 可以统一访问它们。

---

## 十一、判断规则

写 pass 时可以用一句话判断：

```text
我现在是否只访问 getOperation() 这棵子树？
```

如果答案是 yes，通常符合 `OperationPass` 的并行限制：

```text
读写当前 func 内部 op
修改当前 func 的 attribute
查询当前 func 的 analysis
在当前 func 内部做 rewrite
```

如果答案是 no，就要警惕：

```text
遍历 module 中其他 func
读取 sibling function 的 body
修改 module body
新增或删除 sibling function
保存跨 function 的 mutable vector/map
依赖所有 function 都经过同一个 pass 实例
```

如果确实需要全局视角，优先把 pass 提升到更外层：

```text
只处理函数内部局部优化       -> OperationPass<func::FuncOp>
需要看整个 module 的函数集合 -> OperationPass<ModuleOp>
需要看多个 module            -> 更外层的驱动逻辑或 pipeline 设计
```

---

## 十二、和 AI 编译器学习的关系

从 AI 编译器角度看，很多优化 pass 都可以分成两类：

```text
局部函数内优化：
  linalg / tensor / memref / scf / vector 的局部 pattern rewrite
  通常适合 func.func pass

全模块优化：
  跨函数分析、helper function 插入、符号表统一修改
  通常适合 ModuleOp pass
```

因此，设计 pass 之前应该先确定优化作用域：

```text
优化只需要一个函数内部信息吗？
  是 -> func.func pass
  否 -> 考虑 ModuleOp pass
```

这也是 MLIR PassManager 并行模型想表达的核心思想：

> pass 的运行层级，就是 pass 合法观察和修改 IR 的边界。

