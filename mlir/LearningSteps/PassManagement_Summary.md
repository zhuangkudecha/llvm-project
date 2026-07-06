# MLIR PassManagement.md - 学习总结

## 概述

本文总结 `mlir/docs/PassManagement.md` 的核心内容。这个文档讲的是 MLIR Pass 基础设施如何把很多 IR 变换、分析、调试工具组织成一个完整的编译 pipeline。

可以用一条主线理解：

```text
Operation
  -> OperationPass
  -> AnalysisManager
  -> PassManager / OpPassManager
  -> Pass registration
  -> mlir-opt textual pipeline
  -> TableGen pass declaration
  -> Instrumentation / timing / IR printing
  -> crash reproducer
```

如果从 AI 编译器学习角度看，这篇文档回答的问题是：

> MLIR 如何把一系列 lowering、canonicalize、CSE、bufferization、conversion pass 安全、可配置、可调试地串成 compiler pipeline？

---

## 一、OperationPass：Pass 的基本执行单元

MLIR 中 pass 的核心抽象是 `OperationPass`。它不是固定只能跑在 module 或 function 上，而是跑在某个 **current operation** 上。

例如：

```cpp
struct MyFuncPass
    : public PassWrapper<MyFuncPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
  }
};
```

这里的 current operation 就是某一个 `func.func`。

如果 IR 是：

```mlir
module {
  func.func @foo() {
    return
  }

  func.func @bar() {
    return
  }
}
```

PassManager 可能分别把 `@foo` 和 `@bar` 作为 current operation 调度给这个 pass。

```text
OperationPass<func::FuncOp>
  -> runOnOperation(@foo)
  -> runOnOperation(@bar)
```

### 1.1 OperationPass 的并行限制

MLIR PassManager 支持多线程执行，因此 pass 必须遵守几个限制：

- 不能检查 current operation 的 sibling operation
- 不能访问 sibling operation 内部嵌套的 operation
- 不能修改 current operation 子树之外的 IR
- 不能跨 `runOnOperation()` 保存可变状态
- 不能使用全局可变状态
- pass 必须可以复制

核心原因是：PassManager 可能并行处理同一层级的多个 operation。

```text
线程 1: pass 处理 func.func @foo
线程 2: pass 处理 func.func @bar
线程 3: pass 处理 func.func @baz
```

如果线程 1 在处理 `@foo` 时读取或修改 `@bar`，就可能和线程 2 冲突。

更详细的专题笔记见：

```text
mlir/LearningSteps/PassManagement_OperationPass_Restrictions_Summary.md
```

---

## 二、Op-Agnostic Operation Pass

`op-agnostic` pass 指的是不绑定具体 operation 类型的 pass。

例如：

```cpp
struct MyOperationPass
    : public PassWrapper<MyOperationPass, OperationPass<>> {
  void runOnOperation() override {
    Operation *op = getOperation();
  }
};
```

它不假设 `getOperation()` 一定是 `ModuleOp`、`func::FuncOp` 或其他具体 op。

这种 pass 运行在哪种 operation 上，取决于它被添加到哪个 PassManager 层级：

```cpp
PassManager pm = PassManager::on<ModuleOp>(ctx);
pm.addPass(createCanonicalizerPass());
```

这里 canonicalize 被加到 module 层。

```cpp
OpPassManager &funcPM = pm.nest<func::FuncOp>();
funcPM.addPass(createCanonicalizerPass());
```

这里 canonicalize 被加到每个 `func.func` 层。

典型 op-agnostic pass：

- `canonicalize`
- `cse`

它们的逻辑通常是“在当前 operation 子树内做通用简化”，不要求 current operation 一定是某个固定类型。

---

## 三、Filtered Operation Pass：限制 pass 能跑在哪里

有些 pass 不能跑在任意 operation 上，需要声明静态约束。这就是 static filtering。

### 3.1 按接口过滤

例如只允许跑在实现 `FunctionOpInterface` 的 operation 上：

```cpp
bool canScheduleOn(RegisteredOperationName opInfo) const override {
  return opInfo.hasInterface<FunctionOpInterface>();
}
```

这样在 pass 内部就可以安全转换：

```cpp
FunctionOpInterface op = cast<FunctionOpInterface>(getOperation());
```

### 3.2 按具体 op 类型过滤

更常见的是直接写：

```cpp
OperationPass<func::FuncOp>
```

这表示 pass 只能跑在 `func.func` 上。

### 3.3 按接口写 pass

也可以写：

```cpp
InterfacePass<FunctionOpInterface>
```

这表示 pass 可以跑在任何实现了 `FunctionOpInterface` 的 operation 上，而不是只限制在 `func.func`。

### 3.4 static filtering 对 any(...) 的影响

这一点很容易误解。

假设有 pipeline：

```text
any(cse,my-function-pass)
```

其中：

```text
cse              可以跑在很多 operation 上
my-function-pass 只能跑在 FunctionOpInterface 上
```

实际结果不是：

```text
cse 到处跑
my-function-pass 只在函数上跑
```

而是：

```text
整个 any(...) 只会调度到 FunctionOpInterface operation 上
```

原因是一个 `OpPassManager` 表示一组 pass 作为一个 pipeline 一起调度。只要其中一个 pass 有 static filtering，整个 op-agnostic pass manager 都会继承这个约束。

---

## 四、Dependent Dialects：声明 pass 依赖的 Dialect

如果 pass 会创建某个 dialect 的 op、type 或 attribute，就需要声明 dependent dialect。

例如：

```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<arith::ArithDialect, func::FuncDialect>();
}
```

原因是 MLIR 多线程 pass pipeline 执行前，需要提前加载好相关 dialect。不能在多线程运行中途临时加载未知 dialect。

典型场景：

```text
pass 中使用 OpBuilder 创建 arith.constant
  -> 需要声明 arith dialect

pass 中创建 func.call / func.return
  -> 需要声明 func dialect
```

---

## 五、Initialization：pass 执行前初始化

`initialize()` 是 pass 执行前的初始化 hook。

适合做：

```text
解析复杂 option
构造只读配置
预构造动态 pipeline
```

不适合做：

```text
getOperation()
getAnalysis()
访问当前 IR
```

因为 `initialize()` 运行时，pass 还没有绑定到某个 current operation。

---

## 六、Analysis Management：分析结果的按需计算和缓存

MLIR 中 analysis 不是 pass，而是普通类。它的特点是：

```text
按需计算
缓存结果
不能修改 IR
依赖 pass 的 preserve/invalidate 机制保持正确
```

一个 analysis 通常提供这样的构造函数：

```cpp
struct MyAnalysis {
  MyAnalysis(Operation *op);
};
```

如果 analysis 依赖其他 analysis，可以拿到 `AnalysisManager`：

```cpp
struct MyAnalysisWithDependency {
  MyAnalysisWithDependency(Operation *op, AnalysisManager &am) {
    MyAnalysis &dep = am.getAnalysis<MyAnalysis>();
  }
};
```

### 6.1 在 pass 中查询 analysis

```cpp
void runOnOperation() override {
  MyAnalysis &analysis = getAnalysis<MyAnalysis>();
}
```

如果 analysis 没有被计算过，框架会自动构造并缓存。

也可以只查询缓存，不触发计算：

```cpp
auto cached = getCachedAnalysis<MyAnalysis>();
```

### 6.2 preserve analysis

pass 默认会让已有 analysis 失效，因为 pass 可能改了 IR。

如果 pass 没有修改 IR，可以写：

```cpp
markAllAnalysesPreserved();
```

如果只保留某些 analysis：

```cpp
markAnalysesPreserved<MyAnalysis>();
```

核心理解：

```text
AnalysisManager 负责缓存；
Pass 负责告诉框架自己有没有破坏缓存。
```

---

## 七、Pass Failure：受控失败

如果 pass 发现自己的 invariant 被破坏，可以调用：

```cpp
signalPassFailure();
```

效果是：

```text
当前 pass 失败
后续 pipeline 停止执行
PassManager::run 返回 failure
```

示例：

```cpp
void runOnOperation() override {
  if (some_broken_invariant)
    return signalPassFailure();
}
```

这和 crash 不同。`signalPassFailure()` 是受控失败，用于报告 pass 无法继续安全处理。

---

## 八、PassManager / OpPassManager：组织 pipeline

`PassManager` 是顶层入口；`OpPassManager` 是某个 operation 层级上的 pass 容器。

例如：

```cpp
auto pm = PassManager::on<ModuleOp>(ctx);

pm.addPass(std::make_unique<MyModulePass>());

OpPassManager &funcPM = pm.nest<func::FuncOp>();
funcPM.addPass(createCSEPass());
funcPM.addPass(createCanonicalizerPass());
```

对应 pipeline 结构：

```text
builtin.module(
  my-module-pass,
  func.func(
    cse,
    canonicalize
  )
)
```

含义是：

```text
在 module 层执行 MyModulePass
然后进入每个 func.func
在函数层执行 CSE 和 canonicalize
```

### 8.1 op-specific pass manager

```cpp
pm.nest<func::FuncOp>()
```

表示这个 pass manager 只调度到 `func.func`。

### 8.2 op-agnostic pass manager

```cpp
pm.nestAny()
```

表示这个 pass manager 可以调度到任意合适的 operation。

例如：

```cpp
OpPassManager &anyPM = pm.nestAny();
anyPM.addPass(createCanonicalizerPass());
anyPM.addPass(createCSEPass());
```

---

## 九、Dynamic Pass Pipelines：pass 内部动态运行 pipeline

有些 pass 需要在运行时根据 IR 状态决定是否运行一段 pipeline。

例如 inliner 可能在内联过程中运行函数内简化 pipeline，以便得到更好的 cost model。

MLIR 提供：

```cpp
LogicalResult runPipeline(OpPassManager &pipeline, Operation *op);
```

示例：

```cpp
void MyModulePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (hasSomeSpecificProperty(module)) {
    OpPassManager dynamicPM("builtin.module");

    if (failed(runPipeline(dynamicPM, module)))
      return signalPassFailure();
  }
}
```

注意点：

```text
不要在 pass 内部自己构造一个独立 PassManager；
应该用 runPipeline，让 analysis、instrumentation、failure 处理都接入现有框架。
```

---

## 十、Pass Options：每个 pass 实例的配置

MLIR pass 可以定义 option：

```cpp
struct MyPass ... {
  Option<int> exampleOption{
      *this, "flag-name", llvm::cl::desc("...")};

  ListOption<int> exampleListOption{
      *this, "list-flag-name", llvm::cl::desc("...")};
};
```

这些 option 是每个 pass 实例独立解析的。

例如命令行中可能写：

```bash
mlir-opt input.mlir -my-pass='flag-name=4'
```

pipeline 本身也可以有 option，通过 `PassPipelineOptions` 定义。

---

## 十一、Pass Statistics：统计 pass 做了什么

Statistic 用来记录 pass 的效果，例如：

```text
删除了多少 operation
重写了多少 pattern
创建了多少 buffer
合并了多少公共表达式
```

示例：

```cpp
Statistic numRewrites{this, "num-rewrites", "Number of rewrites"};
```

在 pass 中递增：

```cpp
++numRewrites;
```

它适合回答这类问题：

```text
这个 pass 有没有真的触发？
这个 pass 在 pipeline 的哪个位置更有效？
重复跑 CSE 是否还有收益？
```

---

## 十二、Pass Registration：注册 pass 给 mlir-opt 使用

如果想在 `mlir-opt` 里通过名字调用 pass，需要注册：

```cpp
void registerMyPass() {
  PassRegistration<MyPass>();
}
```

注册后，`mlir-opt` 可以识别类似：

```bash
mlir-opt input.mlir -my-pass
```

如果 pass 不能默认构造，也可以用回调注册：

```cpp
PassRegistration<MyParametricPass>(
  []() -> std::unique_ptr<Pass> {
    return std::make_unique<MyParametricPass>(/* config */);
  });
```

注意：pass 仍然必须可以正确复制，因为 PassManager 可能为了并行运行复制 pass 实例。

---

## 十三、Pass Pipeline Registration：注册一组 pipeline

除了注册单个 pass，MLIR 还可以注册一整套 pipeline。

```cpp
void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<MyPass>());
  pm.addPass(std::make_unique<MyOtherPass>());
}

void registerMyPipeline() {
  PassPipelineRegistration<>(
      "my-pipeline", "Run my pipeline", pipelineBuilder);
}
```

这样工具中可以使用：

```bash
mlir-opt input.mlir -pass-pipeline='builtin.module(my-pipeline)'
```

这适合封装类似 `-O1`、`-O2`、某个 dialect lowering pipeline 这样的常用组合。

---

## 十四、Textual Pass Pipeline：命令行 pipeline 语法

MLIR 支持用字符串描述完整 pipeline。

核心语法：

```text
pipeline          ::= op-anchor `(` pipeline-element (`,` pipeline-element)* `)`
pipeline-element  ::= pipeline | (pass-name | pass-pipeline-name) options?
options           ::= `{` (key (`=` value)?)+ `}`
```

常见例子：

```bash
mlir-opt foo.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize),convert-func-to-llvm)'
```

含义：

```text
builtin.module(
  func.func(
    cse,
    canonicalize
  ),
  convert-func-to-llvm
)
```

也就是：

```text
在每个 func.func 上跑 cse 和 canonicalize
然后在 module 层跑 convert-func-to-llvm
```

带 option 的 pass：

```bash
mlir-opt foo.mlir \
  -pass-pipeline='builtin.module(convert-func-to-llvm{use-bare-ptr-memref-call-conv=1})'
```

`any(...)` 表示 op-agnostic anchor：

```bash
mlir-opt foo.mlir \
  -pass-pipeline='builtin.module(any(cse,canonicalize))'
```

---

## 十五、Declarative Pass Specification：用 TableGen 声明 pass

手写 pass 会有很多样板代码：

```text
createMyPass()
registerMyPass()
getArgument()
getDescription()
option 定义
statistic 定义
```

MLIR 支持用 TableGen 声明 pass，然后自动生成这些样板。

示例：

```tablegen
def MyPass : Pass<"my-pass", "ModuleOp"> {
  let summary = "My Pass Summary";
  let description = [{
    This pass does something useful.
  }];

  let options = [
    Option<"option", "example-option", "bool", /*default=*/"true",
           "An example option">
  ];

  let statistics = [
    Statistic<"statistic", "example-statistic", "An example statistic">
  ];
}
```

然后通过：

```text
mlir-tblgen -gen-pass-decls
```

生成注册、声明、base class、option/statistic 相关代码。

这和 ODS 定义 operation 的思路类似：

```text
把结构化元信息交给 TableGen
减少手写重复代码
保证注册和文档一致
```

---

## 十六、Pass Instrumentation：观察 pass 执行过程

Instrumentation 是 PassManager 的观察机制，可以监听：

```text
runBeforePipeline
runAfterPipeline
runBeforePass
runAfterPass
runAfterPassFailed
runBeforeAnalysis
runAfterAnalysis
```

它适合做：

```text
统计 pass 耗时
打印 pass 前后的 IR
记录 analysis 计算次数
调试 pipeline 执行顺序
```

例如统计 `DominanceInfo` 被计算了多少次：

```cpp
struct DominanceCounterInstrumentation : public PassInstrumentation {
  unsigned &count;

  DominanceCounterInstrumentation(unsigned &count) : count(count) {}

  void runAfterAnalysis(StringRef, TypeID id, Operation *) override {
    if (id == TypeID::get<DominanceInfo>())
      ++count;
  }
};
```

---

## 十七、标准 Instrumentation 工具

### 17.1 Pass Timing

查看 pass 和 analysis 的耗时：

```bash
mlir-opt foo.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-timing
```

列表模式：

```bash
-mlir-timing-display=list
```

树形模式：

```bash
-mlir-timing-display=tree
```

### 17.2 IR Printing

打印每个 pass 后的 IR：

```bash
mlir-opt foo.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  -mlir-print-ir-after-all
```

只打印某个 pass 后：

```bash
-mlir-print-ir-after=cse
```

只打印某个 pass 前：

```bash
-mlir-print-ir-before=cse
```

这对学习 lowering pipeline 非常有用，因为可以直接看到每个 pass 如何改变 IR。

---

## 十八、Crash and Failure Reproduction：生成可复现用例

PassManager 支持在 crash 或 pass failure 时生成 reproducer。

开启方式包括：

```text
PassManager::enableCrashReproducerGeneration
```

或命令行：

```bash
-mlir-pass-pipeline-crash-reproducer=repro.mlir
```

生成的 `.mlir` 文件会包含：

```text
原始 IR
pipeline 配置
threading 配置
verify_each 配置
```

之后可以用：

```bash
mlir-opt -run-reproducer repro.mlir
```

重新复现问题。

### 18.1 Local Reproducer

local reproducer 会尝试生成失败 pass 前的局部 IR 和更小 pipeline。

命令行：

```bash
-mlir-pass-pipeline-local-reproducer
```

注意：local reproducer 要求关闭多线程：

```bash
-mlir-disable-threading
```

---

## 十九、整体架构图

```text
mlir-opt
  |
  | parse textual pipeline
  v
PassManager
  |
  +-- OpPassManager: builtin.module
        |
        +-- Module-level passes
        |
        +-- OpPassManager: func.func
        |     |
        |     +-- cse
        |     +-- canonicalize
        |
        +-- OpPassManager: any
              |
              +-- op-agnostic passes

During execution:
  |
  +-- OperationPass::runOnOperation()
  +-- AnalysisManager::getAnalysis()
  +-- markAnalysesPreserved()
  +-- PassInstrumentation hooks
  +-- signalPassFailure() / crash reproducer
```

---

## 二十、学习重点

初学 MLIR Pass 基础设施时，建议按下面顺序掌握：

1. `OperationPass` 的 current operation 和并行限制
2. `OperationPass<func::FuncOp>` 与 `OperationPass<>` 的区别
3. `op-specific` / `op-agnostic` PassManager 的区别
4. static filtering 如何影响 `any(...)`
5. `getAnalysis()`、`markAllAnalysesPreserved()`、analysis invalidation
6. textual pipeline 如何映射到嵌套 `OpPassManager`
7. 如何用 `-mlir-print-ir-after-all` 和 `-mlir-timing` 调试 pipeline
8. 如何用 reproducer 复现 pass crash

一句话总结：

> `PassManagement.md` 的核心不是某个单独 pass 怎么写，而是 MLIR 如何把 pass、analysis、pipeline、注册、调试和复现机制组合成一个可扩展的编译器基础设施。

