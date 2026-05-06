# MLIR Phase 2: Core Infrastructure - 学习总结

## 概述

Phase 2 深入 MLIR 的 C++ 基础设施层。如果说 Phase 1 学会了"使用"MLIR（读写 IR、跑 Pass），那么 Phase 2 的目标是理解 MLIR "内部怎么实现"的。本阶段覆盖 6 个核心主题：

```
Step 5:  Core IR   (Operation, Value, Block, Region, Type, Attribute)
Step 6:  Pass       (Pass, OperationPass, PassManager)
Step 7:  Pass 管理  (Pass 嵌套、验证、管道化)
Step 8:  规范化     (Canonicalization)
Step 9:  Transform  (GreedyPatternRewriteDriver)
Step 10: Rewriter   (Pattern, RewritePattern, OpRewritePattern)
```

---

## 一、Core IR：MLIR 的核心数据结构 (Step 5)

### 1.1 Operation — MLIR 的基本执行单元

**意义**：Operation 是 MLIR 中一切计算的原子单位。每个指令（加法、函数调用、循环……）都是一个 Operation。理解它是理解整个 MLIR 的前提。

**源码位置**：`include/mlir/IR/Operation.h:84`

```cpp
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    detail::OpProperties, BlockOperand, Region,
                                    OpOperand> {
```

**关键设计**：

1. **内存布局**：Operation 使用 `TrailingObjects` 将 operands、results、regions 等打包在一次堆分配中，而非分别 new：
   ```
   [Result2, Result1, Result0, Operation, Operands..., Regions...]
                                 ^-- Operation* 指向这里
   ```
   这极大地减少了内存碎片和分配开销。

2. **核心创建方法**（`Operation.h:90-108`）：
   ```cpp
   // 从 OperationState 创建
   static Operation *create(const OperationState &state);
   // 从各个组件创建
   static Operation *create(Location location, OperationName name,
                            TypeRange resultTypes, ValueRange operands,
                            NamedAttrList &&attributes,
                            OpaqueProperties properties,
                            BlockRange successors = {},
                            RegionRange regions = {});
   ```

3. **访问结果和操作数**（`Operation.h:270-274`）：
   ```cpp
   // 替换所有使用
   template <typename ValuesT>
   void replaceAllUsesWith(ValuesT &&values) {
     getResults().replaceAllUsesWith(std::forward<ValuesT>(values));
   }
   ```

4. **层级关系导航**：
   ```cpp
   Block *getBlock() { return block; }                    // 父 Block
   Region *getParentRegion() { return block->getParent(); } // 父 Region
   Operation *getParentOp() { return block->getParentOp(); } // 父 Op
   ```

**意义总结**：Operation 的设计体现了 MLIR 的核心哲学——用单一、通用的数据结构表示所有计算，通过 Dialect 赋予语义。

---

### 1.2 Value — SSA 值体系

**意义**：MLIR 严格遵循 SSA（Static Single Assignment）模型。每个值只被定义一次，但有零或多个使用。Value 是连接 Operation 之间数据流的桥梁。

**源码位置**：`include/mlir/IR/Value.h`

Value 有两种子类，形成完整的 SSA 值来源：

```cpp
// Value 的内部实现（Value.h:350-400）
class Value {
public:
  Type getType() const { return impl->getType(); }
  Operation *getDefiningOp() const;  // 如果是 OpResult，返回定义它的 Op
  void replaceAllUsesWith(Value newValue); // SSA 重写的核心操作
  void dropAllUses();
private:
  detail::ValueImpl *impl;
};
```

**两种 Value 子类**：

```cpp
// OpResult: Operation 的输出值（Value.h:457）
class OpResult : public Value {
  Operation *getOwner() const;       // 拥有此结果的 Operation
  unsigned getResultNumber() const;   // 这是第几个结果
};

// BlockArgument: Block 的输入参数（Value.h:380 附近）
class BlockArgument : public Value {
  Block *getOwner() const;            // 拥有此参数的 Block
  unsigned getArgNumber() const;       // 这是第几个参数
};
```

**Use-Def 链**：每个 Value 维护一个 use-list（使用链），支持高效的 `replaceAllUsesWith`：

```cpp
// Operation.h:267-268
void replaceUsesOfWith(Value from, Value to);  // 替换本 Op 中对 from 的使用
```

**意义**：Value 体系使得 MLIR 的所有数据流都是显式的、可追踪的，这是编译优化的基础。

---

### 1.3 Region 和 Block — 层级嵌套结构

**意义**：MLIR 的 Operation 可以包含 Region（区域），Region 包含 Block（基本块），Block 包含 Operation。这形成了 Operation → Region → Block → Operation 的递归树结构，使得 MLIR 能够表示任意嵌套的控制流和作用域。

**Region**（`include/mlir/IR/Region.h:26`）：

```cpp
class Region {
public:
  using BlockListType = llvm::iplist<Block>;
  BlockListType &getBlocks() { return blocks; }
  bool empty() { return blocks.empty(); }
  bool hasOneBlock() { return !empty() && std::next(begin()) == end(); }

  // 委托给第一个 Block 的参数管理
  BlockArgument addArgument(Type type, Location loc) {
    return front().addArgument(type, loc);
  }

  // 获取父 Operation
  Operation *getParentOp();
private:
  BlockListType blocks;
};
```

**Block**（`include/mlir/IR/Block.h:32`）：

```cpp
class alignas(8) Block : public IRObjectWithUseList<BlockOperand>,
                         public llvm::ilist_node_with_parent<Block, Region> {
public:
  Region *getParent() const;              // 父 Region
  Operation *getParentOp();               // 父 Operation

  BlockArgListType getArguments();        // Block 参数（SSA phi 的替代）
  // Operation 列表
  using OpListType = llvm::ilist<Operation>;
  OpListType &getOperations() { return operations; }
private:
  OpListType operations;
  MutableArrayRef<BlockArgument> arguments;
};
```

**嵌套关系示意**：

```
ModuleOp (Operation)
  └── Region
       └── Block
            ├── BlockArgument(0): memref<10xf32>
            ├── arith.constant ... (Operation)
            ├── scf.for (Operation)
            │    └── Region
            │         └── Block
            │              ├── BlockArgument(0): index
            │              ├── ... (Operations)
            │              └── scf.yield
            └── func.return
```

**意义**：Region/Block 嵌套是 MLIR 区别于 LLVM IR 的关键特性。LLVM IR 是扁平的，而 MLIR 允许在 Operation 内部嵌套任意复杂的结构，这使得不同抽象级别可以在同一个 IR 中共存。

---

### 1.4 Type 和 Attribute — 类型和属性系统

**意义**：Type 描述 Value 的数据类型，Attribute 描述 Operation 的编译期常量属性。它们共享基于 StorageUniquer 的唯一化（uniquing）机制，确保相同内容只存一份。

**Type**（`include/mlir/IR/Types.h`）：

```cpp
class Type {
public:
  TypeID getTypeID() const;           // 类型 ID
  Dialect &getDialect() const;        // 所属方言
  MLIRContext *getContext() const;    // 所属上下文
  bool isInteger() const;
  bool isFloat() const;
  // ...
private:
  const AbstractType *abstractTy;     // 类型的抽象描述
  Impl *impl;                         // 类型的具体数据
};
```

**Attribute**（`include/mlir/IR/Attributes.h`）：

```cpp
class Attribute {
public:
  TypeID getTypeID() const;
  Dialect &getDialect() const;
  MLIRContext *getContext() const;
  // 常用子类:
  // StringAttr, IntegerAttr, FloatAttr, BoolAttr,
  // ArrayAttr, DictionaryAttr, DenseElementsAttr ...
};
```

**StorageUniquer 机制**：Type 和 Attribute 都通过 `StorageUniquer` 实现唯一化——相同的 Type/Attribute 在全局只创建一个实例，后续通过哈希表查找复用：

```cpp
// Dialect.h:337 - 类型注册
template <typename T>
void addType() {
  addType(T::getTypeID(), AbstractType::get<T>(*this));
  detail::TypeUniquer::registerType<T>(context);
}
```

**意义**：唯一化大幅减少内存开销，同时使得 Type 和 Attribute 的比较退化为指针比较（O(1)），是编译器性能的关键。

---

### 1.5 Dialect — 方言系统

**意义**：Dialect 是 MLIR 可扩展性的核心。每个方言定义一组 Operation、Type 和 Attribute，形成独立的语义空间。不同方言可以共存并相互转换。

**源码位置**：`include/mlir/IR/Dialect.h:54`

```cpp
class Dialect {
public:
  StringRef getNamespace() const { return name; }
  MLIRContext *getContext() const { return context; }

  // 注册 Operations
  template <typename... Args>
  void addOperations() {
    (void)std::initializer_list<int>{
        0, (RegisteredOperationName::insert<Args>(*this), 0)...};
  }

  // 注册 Types
  template <typename... Args>
  void addTypes() {
    (void)std::initializer_list<int>{0, (addType<Args>(), 0)...};
  }

  // 解析/打印钩子
  virtual Type parseType(DialectAsmParser &parser) const;
  virtual void printType(Type, DialectAsmPrinter &) const;
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type) const;
  virtual void printAttribute(Attribute, DialectAsmPrinter &) const;

  // 常量折叠钩子
  virtual Operation *materializeConstant(OpBuilder &builder, Attribute value,
                                         Type type, Location loc) {
    return nullptr;
  }

  // 方言接口
  template <typename InterfaceT>
  InterfaceT *getRegisteredInterface() { ... }
private:
  StringRef name;
  TypeID dialectID;
  MLIRContext *context;
  bool unknownOpsAllowed = false;
  DenseMap<TypeID, std::unique_ptr<DialectInterface>> registeredInterfaces;
};
```

**意义**：Dialect 使 MLIR 成为一个"元编译器"——你可以定义自己的 IR 层级，通过 DialectConversion 进行层间转换，而不需要修改框架本身。

---

### 1.6 MLIRContext — 全局上下文

**意义**：MLIRContext 是整个 MLIR 系统的"引擎"。它管理所有 Dialect、Type、Attribute 的注册和唯一化，是每个 MLIR 程序的起点。

**源码位置**：`include/mlir/IR/MLIRContext.h`

```cpp
class MLIRContext {
public:
  explicit MLIRContext(Threading multithreading = Threading::ENABLED);

  // Dialect 管理
  template <typename T>
  T *getOrLoadDialect() {
    return static_cast<T *>(
        getOrLoadDialect(T::getDialectNamespace(), TypeID::get<T>(),
          [this]() { return std::make_unique<T>(this); }));
  }

  // 唯一化系统
  StorageUniquer &getTypeUniquer();
  StorageUniquer &getAttributeUniquer();

  // 多线程支持
  bool isMultithreadingEnabled();
  void disableMultithreading(bool disable = true);
  void enableMultithreading(bool enable = true);

  // Operation 注册表
  ArrayRef<RegisteredOperationName> getRegisteredOperations();
  bool isOperationRegistered(StringRef name);
};
```

**Context 初始化过程**（`lib/IR/MLIRContext.cpp:289`）：

```cpp
MLIRContext::MLIRContext(const DialectRegistry &registry, Threading setting)
    : impl(new MLIRContextImpl(...)) {
  registry.appendTo(impl->dialectsRegistry);
  getOrLoadDialect<BuiltinDialect>();  // 始终预加载 Builtin 方言
  // 初始化常用类型
  impl->f32Ty = TypeUniquer::get<Float32Type>(this);
  impl->i32Ty = TypeUniquer::get<IntegerType>(this, 32, IntegerType::Signless);
}
```

**意义**：MLIRContext 统一管理所有全局状态，使得多个编译管线可以安全地共享类型系统和方言注册。

---

### 1.7 OpBuilder — IR 构建器

**意义**：OpBuilder 是构建 MLIR IR 的主要工具。它封装了创建 Operation、管理插入位置、构建常用类型的便捷方法。

**源码位置**：`include/mlir/IR/Builders.h`

```cpp
class OpBuilder : public Builder {
public:
  // 模板化创建 Operation（最常用）
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args &&...args) {
    OperationState state(location,
                         getCheckRegisteredInfo<OpTy>(location.getContext()));
    OpTy::build(*this, state, std::forward<Args>(args)...);
    auto *op = create(state);
    return cast<OpTy>(op);
  }

  // 插入点管理
  void setInsertionPoint(Block *block, Block::iterator insertPoint);
  void setInsertionPoint(Operation *op);
  void setInsertionPointAfter(Operation *op);
  void setInsertionPointToStart(Block *block);
  void setInsertionPointToEnd(Block *block);

  // RAII 插入点保护
  class InsertionGuard {
    InsertionGuard(OpBuilder &builder)
        : builder(&builder), ip(builder.saveInsertionPoint()) {}
    ~InsertionGuard() { builder->restoreInsertionPoint(ip); }
  };
};
```

**实际使用示例**（`lib/Dialect/Func/Utils/Utils.cpp:283`）：

```cpp
func::FuncOp func::createFnDecl(OpBuilder &b, SymbolOpInterface symTable,
                                StringRef name, FunctionType funcT, ...) {
  OpBuilder::InsertionGuard g(b);                    // RAII 保存/恢复插入点
  b.setInsertionPointToStart(&symTable->getRegion(0).front());
  func::FuncOp funcOp =
      func::FuncOp::create(b, symTable->getLoc(), name, funcT);
  if (setPrivate)
    funcOp.setPrivate();
  return funcOp;
}
```

**意义**：OpBuilder 的 `create<OpTy>` 模板方法 + 插入点管理是 MLIR IR 构建的标准模式，InsertionGuard 的 RAII 设计防止了插入点错乱的 bug。

---

## 二、Pass 基础设施 (Step 6-7)

### 2.1 Pass 基类

**意义**：Pass 是 MLIR 中所有变换/分析的基类。每个 Pass 在特定的 Operation 上执行，可以查询分析结果、修改 IR。

**源码位置**：`include/mlir/Pass/Pass.h:52`

```cpp
class Pass {
public:
  virtual StringRef getName() const = 0;
  virtual void getDependentDialects(DialectRegistry &registry) const {}
  virtual StringRef getArgument() const { return ""; }
  virtual StringRef getDescription() const { return ""; }

  // Pass 选项（支持命令行参数传递）
  template <typename DataType, typename OptionParser = ...>
  struct Option : public detail::PassOptions::Option<DataType, OptionParser> {};

  // Pass 统计
  class Statistic : public llvm::Statistic { ... };

protected:
  // 核心方法：子类必须实现
  virtual void runOnOperation() = 0;

  // 获取当前 Operation
  Operation *getOperation() {
    return getPassState().irAndPassFailed.getPointer();
  }

  // 获取分析结果
  template <typename AnalysisT>
  AnalysisT &getAnalysis() { ... }

  // 标记分析为已保留
  void markAllAnalysesPreserved() { ... }
};
```

**OperationPass 模板**（`include/mlir/Pass/Pass.h` 后续部分）：

```cpp
template <typename OpT = void>
class OperationPass : public Pass {
protected:
  OpT getOperation() { return cast<OpT>(Pass::getOperation()); }
  // canScheduleOn 确保只运行在正确的 Op 上
};
```

**意义**：Pass 基类定义了 Pass 的生命周期接口。`runOnOperation` 是唯一必须实现的方法，它接收一个 Operation 并对其（及其内部的 Region）进行变换。

---

### 2.2 PassManager — Pass 管道

**意义**：PassManager 管理 Pass 的组织、调度和验证。它支持嵌套——一个 Module 级别的 Pass 管道内可以嵌套 Function 级别的 Pass 管道。

**源码位置**：`include/mlir/Pass/PassManager.h:46`

```cpp
class OpPassManager {
public:
  // 嵌套子 PassManager
  OpPassManager &nest(OperationName nestedName);
  template <typename OpT>
  OpPassManager &nest() { return nest(OpT::getOperationName()); }

  // 添加 Pass
  void addPass(std::unique_ptr<Pass> pass);

  // 嵌套添加
  template <typename OpT>
  void addNestedPass(std::unique_ptr<Pass> pass) {
    nest<OpT>().addPass(std::move(pass));
  }
};
```

**使用示例**（命令行）：

```bash
# Module 级别 canonicalize，嵌套的 func 级别运行 cse
mlir-opt --pass-pipeline="builtin.module(canonicalize,func.func(cse))" input.mlir
```

**对应 C++ 代码**：

```cpp
PassManager pm(&context);
pm.addPass(createCanonicalizerPass());
pm.addNestedPass<func::FuncOp>(createCSEPass());
if (failed(pm.run(module)))
  ... // 处理错误
```

**AnalysisManager**（`include/mlir/Pass/AnalysisManager.h`）：

```cpp
class AnalysisManager {
public:
  // 获取分析结果（按需计算，缓存）
  template <typename AnalysisT>
  AnalysisT &getAnalysis(Operation *op);

  // 通知分析仍然有效
  void preserve(TypeID id);
  void markAllPreserved();
};
```

**意义**：PassManager 的嵌套模型与 MLIR 的 Operation-Region-Block 层级结构完美对应。AnalysisManager 避免重复计算，PreservedAnalyses 机制确保 Pass 之间正确地失效和重用分析。

---

## 三、规范化 (Step 8)

### 3.1 什么是 Canonicalization

**意义**：规范化是将 IR 转换为"标准形式"的过程，使得后续 Pass 可以假设 IR 遵循某些约定（如常量总是在右侧、没有恒等操作等）。这是 MLIR 中最基本、最常用的优化。

**注册规范化模式**：

```cpp
// 在 Dialect 中注册方言级别的规范化
// Dialect.h
virtual void getCanonicalizationPatterns(RewritePatternSet &results) const {}

// 在 Operation 中注册 Op 级别的规范化（ODS 生成）
// 每个 Op 可以定义 getCanonicalizationPatterns
```

**内置规范化 Pass**（`include/mlir/Transforms/Passes.h:56`）：

```cpp
// 创建规范化 Pass
std::unique_ptr<Pass> createCanonicalizerPass();
std::unique_ptr<Pass> createCanonicalizerPass(
    const GreedyRewriteConfig &config,
    ArrayRef<std::string> disabledPatterns = {},
    ArrayRef<std::string> enabledPatterns = {});
```

**规范化的典型规则**：
- 消除恒等操作：`x + 0 → x`、`x * 1 → x`
- 常量折叠：`arith.constant 1 + arith.constant 2 → arith.constant 3`
- 死代码消除
- 简化控制流

**意义**：规范化是 MLIR 的"卫生保洁"机制。Pass 的作者不需要手动处理所有边界情况，只要 IR 不满足规范化假设，后续的 Canonicalize Pass 会自动清理。

---

## 四、Pattern Rewriting 框架 (Step 9-10)

### 4.1 Pattern 继承体系

**意义**：Pattern Rewriting 是 MLIR 中实现 IR 变换的核心机制。它将"匹配"和"重写"分离为声明式的 Pattern，由驱动器（Driver）自动调度。

**继承层次**：

```
Pattern                     (基类：模式元数据)
  └── RewritePattern        (添加 matchAndRewrite 接口)
        └── OpRewritePattern<SourceOp>  (针对特定 Op 的便捷模板)
              └── OpInterfaceRewritePattern<Interface>  (针对接口的模板)
```

**Pattern 基类**（`include/mlir/IR/PatternMatch.h:73`）：

```cpp
class Pattern {
public:
  PatternBenefit getBenefit() const { return benefit; }    // 模式收益（调度优先级）
  std::optional<OperationName> getRootKind() const;        // 匹配的目标 Op
  ArrayRef<OperationName> getGeneratedOps() const;          // 可能生成的 Op
  bool hasBoundedRewriteRecursion() const;                   // 是否有界递归
  MLIRContext *getContext() const;
  StringRef getDebugName() const;
};
```

**RewritePattern**（`include/mlir/IR/PatternMatch.h:238`）：

```cpp
class RewritePattern : public Pattern {
public:
  // 核心接口：匹配并重写
  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const = 0;
};
```

**OpRewritePattern 模板**（`include/mlir/IR/PatternMatch.h:293`）：

```cpp
template <typename SourceOp>
struct OpOrInterfaceRewritePatternBase : public RewritePattern {
  // 自动转型，提供类型安全的接口
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op), rewriter);
  }
  // 子类实现这个
  virtual LogicalResult matchAndRewrite(SourceOp op,
                                        PatternRewriter &rewriter) const = 0;
};

template <typename SourceOp>
class OpRewritePattern : public detail::OpOrInterfaceRewritePatternBase<SourceOp> {
  // 构造时自动注册 root operation
};
```

**意义**：OpRewritePattern 是编写变换规则的标准方式。通过模板参数绑定目标 Op 类型，框架自动完成类型转换和分发。

---

### 4.2 PatternRewriter — IR 修改器

**意义**：PatternRewriter 封装了所有修改 IR 的操作，确保修改过程的一致性和正确性（如自动维护 use-def 链）。

**关键方法**：

```cpp
class PatternRewriter : public OpBuilder {
public:
  // 替换 Operation 的所有结果
  void replaceOp(Operation *op, ValueRange newValues);
  void replaceOpWithNewOp<OpTy>(Operation *op, Args &&...args);

  // 删除 Operation
  void eraseOp(Operation *op);

  // 修改 Block
  void inlineBlockBefore(Block *source, Block *dest, Block::iterator destBefore);
  void splitBlock(Block *block, Block::iterator splitBefore);

  // 修改 Region
  void inlineRegionBefore(Region &region, Region &dest, Region::iterator destBefore);

  // 通知 Pattern 失败/成功
  void notifyMatchFailure(Location loc, StringRef reason);
};
```

**意义**：PatternRewriter 的所有修改操作都会通知驱动器，驱动器据此更新内部状态（如 worklist）。这是 GreedyPatternRewriteDriver 能正确工作的关键。

---

### 4.3 真实示例：Arith 方言的 Op 展开模式

**源码位置**：`lib/Dialect/Arith/Transforms/ExpandOps.cpp:61`

```cpp
/// 将 CeilDivUIOp (n, m) 展开为: n == 0 ? 0 : ((n-1) / m) + 1
struct CeilDivUIOpConverter : public OpRewritePattern<arith::CeilDivUIOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(arith::CeilDivUIOp op,
                                PatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value a = op.getLhs();
    Value b = op.getRhs();
    Value zero = createConst(loc, a.getType(), 0, rewriter);
    Value compare =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, a, zero);
    Value one = createConst(loc, a.getType(), 1, rewriter);
    Value minusOne = arith::SubIOp::create(rewriter, loc, a, one);
    Value quotient = arith::DivUIOp::create(rewriter, loc, minusOne, b);
    Value plusOne = arith::AddIOp::create(rewriter, loc, quotient, one);
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, compare, zero, plusOne);
    return success();
  }
};
```

**要点分析**：
1. 继承 `OpRewritePattern<arith::CeilDivUIOp>` —— 自动匹配 CeilDivUIOp
2. 使用 `rewriter.create<>()` 构建新 Op —— 保证正确的插入点和通知
3. 最终 `replaceOpWithNewOp` —— 原子的替换操作，自动维护 use-def 链

---

### 4.4 RewritePatternSet — 模式集合

**意义**：RewritePatternSet 收集一组 RewritePattern，交给驱动器统一调度。

```cpp
class RewritePatternSet {
public:
  // 添加模式
  template <typename... Args>
  RewritePatternSet &add(Args &&...args);

  // 添加带标签的模式
  template <typename... Args>
  RewritePatternSet &addWithLabel(StringRef label, Args &&...args);

  // 获取所有模式
  FrozenRewritePatternSet freeze();
};
```

---

### 4.5 GreedyPatternRewriteDriver — 贪心重写驱动器

**意义**：GreedyPatternRewriteDriver 是最常用的模式驱动器。它反复应用所有 Pattern 直到不动点（fixed point）或达到迭代上限。

**工作原理**：

```
1. 将所有 Pattern 按 benefit 排序
2. 遍历 IR 中的每个 Operation
3. 对每个 Op 尝试所有匹配的 Pattern
4. 如果有 Pattern 成功：修改 IR，将受影响的 Op 加入 worklist
5. 重复直到 worklist 为空或达到最大迭代次数
```

**使用方式**：

```cpp
// 在 Pass 中使用 GreedyPatternRewriteDriver
void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  // 添加自定义模式
  patterns.add<MyPattern1, MyPattern2>(patterns.getContext());
  // 也可以添加方言的规范化模式
  if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                          std::move(patterns)))) {
    signalPassFailure();
  }
}
```

**意义**：GreedyPatternRewriteDriver 实现了"编写模式、框架负责调度"的编程模型。开发者只需关注局部变换逻辑，驱动器负责全局收敛和正确性。

---

### 4.6 PatternBenefit — 模式优先级

**源码位置**：`include/mlir/IR/PatternMatch.h:34`

```cpp
class PatternBenefit {
  enum { ImpossibleToMatchSentinel = 65535 };
public:
  PatternBenefit(unsigned benefit);  // 0（低优先级）到 65534（高优先级）
  static PatternBenefit impossibleToMatch();  // 永不匹配
  unsigned short getBenefit() const;
};
```

**意义**：Benefit 值越大，Pattern 越优先被尝试。这允许在高开销的模式之前先尝试低开销的模式，或者在多个模式可以匹配同一个 Op 时控制优先级。

---

## 五、各组件关系总览

```
┌──────────────────────────────────────────────────┐
│                  MLIRContext                      │
│  ┌───────────┐ ┌──────────┐ ┌─────────────────┐  │
│  │ Dialects  │ │ Types    │ │ Attributes      │  │
│  │ Registry  │ │ Uniquer  │ │ Uniquer         │  │
│  └─────┬─────┘ └────┬─────┘ └───────┬─────────┘  │
│        │             │               │            │
│  ┌─────▼─────────────▼───────────────▼─────────┐  │
│  │              IR 层级                         │  │
│  │  Operation ──→ Region ──→ Block ──→ Operation│  │
│  │       │                  │                   │  │
│  │       ├── OpResult       ├── BlockArgument   │  │
│  │       │   (Value)        │    (Value)        │  │
│  │       └── Attributes     └── Operations      │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │           PassManager                        │  │
│  │  ┌─────┐  ┌─────┐  ┌─────┐                 │  │
│  │  │Pass1│→│Pass2│→│Pass3│  (管道化)          │  │
│  │  └──┬──┘  └──┬──┘  └──┬──┘                 │  │
│  │     │         │        │                     │  │
│  │  ┌──▼─────────▼────────▼──┐                 │  │
│  │  │   AnalysisManager      │  (缓存分析)     │  │
│  │  └────────────────────────┘                 │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │       Pattern Rewriting                      │  │
│  │  RewritePatternSet                           │  │
│  │    ├── OpRewritePattern<OpA>  (benefit=1)    │  │
│  │    ├── OpRewritePattern<OpB>  (benefit=2)    │  │
│  │    └── InterfacePattern<>     (benefit=1)    │  │
│  │          │                                    │  │
│  │          ▼                                    │  │
│  │  GreedyPatternRewriteDriver                  │  │
│  │    → 反复应用直到不动点                      │  │
│  └──────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## 六、关键设计模式总结

| 设计模式 | 在 MLIR 中的体现 | 意义 |
|---------|-----------------|------|
| **TrailingObjects** | Operation 打包 operands/results/regions | 减少内存分配次数 |
| **SSA Value** | Value = OpResult \| BlockArgument | 显式数据流，支持高效优化 |
| **唯一化 (Uniquing)** | Type、Attribute 通过 StorageUniquer 全局唯一 | 内存节省 + O(1) 比较 |
| **CRTP/模板** | OpRewritePattern\<OpT\> 自动类型转换 | 类型安全 + 消除手写 cast |
| **RAII** | OpBuilder::InsertionGuard 保存/恢复插入点 | 防止资源泄漏 |
| **观察者模式** | PatternRewriter 通知驱动器修改事件 | 驱动器自动维护 worklist |
| **策略模式** | Pass 的 runOnOperation + AnalysisManager | 灵活的变换/分析组合 |
| **管道模式** | PassManager 嵌套 OpPassManager | 与 IR 层级对应的变换管道 |
| **不动点迭代** | GreedyPatternRewriteDriver | 自动的收敛式重写 |

---

## 七、从源码中学到的工程实践

### 7.1 写一个自定义 Pass 的标准模板

```cpp
// 1. 定义 Pass 类
struct MyPass : public impl::MyPassBase<MyPass> {
  void runOnOperation() override {
    // 获取操作
    auto op = getOperation();

    // 设置模式集
    RewritePatternSet patterns(&getContext());
    patterns.add<MyPattern1>(patterns.getContext());
    // 收集目标 Op 的规范化模式
    MyDialect::getCanonicalizationPatterns(patterns, &getContext());

    // 应用贪心重写
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};

// 2. 注册 Pass
std::unique_ptr<Pass> createMyPass() { return std::make_unique<MyPass>(); }
```

### 7.2 写一个自定义 Pattern 的标准模板

```cpp
struct MyPattern : public OpRewritePattern<my_dialect::SomeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(my_dialect::SomeOp op,
                                PatternRewriter &rewriter) const override {
    // 1. 检查匹配条件
    if (!someCondition(op))
      return rewriter.notifyMatchFailure(op, "condition not met");

    // 2. 构建新 IR
    auto newOp = rewriter.create<my_dialect::AnotherOp>(op.getLoc(), ...);

    // 3. 替换旧 Op
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }
};
```

---

## 八、下一步：Phase 3 预告

Phase 3 将进入 **Dialects Deep Dive**，重点学习：
- 如何使用 TableGen/ODS 创建自定义方言
- 深入理解 Arith、SCF、Affine、Linalg 等核心方言
- DialectConversion 框架的完整使用

Phase 2 建立的基础——Operation/Value/Pass/Pattern Rewriting——将贯穿整个 Phase 3 的学习。
