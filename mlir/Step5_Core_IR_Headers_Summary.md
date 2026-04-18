# MLIR Step 5: Core IR Headers (`include/mlir/IR/`) - 学习总结

## 概述

Phase 2 Step 5 的目标是理解 MLIR 的 C++ 核心基础设施。`include/mlir/IR/` 下的头文件定义了整个 MLIR IR 系统的基础类，是所有 Dialect、Pass、Transform 的基石。

本文档深入分析五个核心头文件及其相互关系：

| 文件 | 核心类 | 角色 |
|------|--------|------|
| `MLIRContext.h` | `MLIRContext` | 全局上下文，管理一切 |
| `Dialect.h` | `Dialect` | Dialect 注册与管理 |
| `Types.h` | `Type` | 类型系统 |
| `Operation.h` | `Operation` | IR 的核心单元 |
| `Builders.h` | `OpBuilder` | IR 构建工具 |

另外补充两个密切相关的重要类：`Value.h`（Value/OpResult/BlockArgument）和 `Attributes.h`（Attribute）。

---

## 一、MLIRContext — 一切的核心

> 源文件：`include/mlir/IR/MLIRContext.h`，实现：`lib/IR/MLIRContext.cpp`

### 1.1 角色定位

`MLIRContext` 是 MLIR 的**顶层管理对象**，所有 IR 对象（Operation、Type、Attribute、Dialect）都存活在某个 Context 中。它类似于 LLVM 的 `LLVMContext`。

```
MLIRContext
├── Loaded Dialects        （已加载的 Dialect 集合）
├── DialectRegistry        （Dialect 注册表）
├── TypeUniquer            （类型唯一化器）
├── AttributeUniquer       （属性唯一化器）
├── AffineUniquer          （仿射表达式唯一化器）
├── ThreadPool             （多线程支持）
├── DiagnosticEngine       （诊断引擎）
└── RegisteredOperations   （已注册的 Operation 列表）
```

### 1.2 核心 API

```cpp
class MLIRContext {
  // ---- Dialect 管理 ----
  Dialect *getLoadedDialect(StringRef name);        // 获取已加载的 Dialect
  template <typename T> T *getOrLoadDialect();      // 获取或加载 Dialect
  void loadAllAvailableDialects();                   // 加载所有可用 Dialect
  void appendDialectRegistry(const DialectRegistry &);  // 追加注册表
  std::vector<Dialect *> getLoadedDialects();       // 获取所有已加载 Dialect

  // ---- Operation 查询 ----
  ArrayRef<RegisteredOperationName> getRegisteredOperations();
  bool isOperationRegistered(StringRef name);

  // ---- 线程控制 ----
  bool isMultithreadingEnabled();
  void disableMultithreading(bool disable = true);
  void setThreadPool(llvm::ThreadPoolInterface &pool);

  // ---- 诊断控制 ----
  bool shouldPrintOpOnDiagnostic();
  void printOpOnDiagnostic(bool enable);

  // ---- 唯一化器（内部使用）----
  StorageUniquer &getTypeUniquer();
  StorageUniquer &getAttributeUniquer();
  StorageUniquer &getAffineUniquer();
};
```

### 1.3 Dialect 加载流程

```cpp
// MLIRContext::getOrLoadDialect 的内部实现逻辑：
Dialect *MLIRContext::getOrLoadDialect(StringRef namespace, TypeID id,
                                       function_ref<unique_ptr<Dialect>()> ctor) {
  // 1. 检查是否已加载
  auto it = loadedDialects.find(namespace);
  if (it != loadedDialects.end())
    return it->second.get();

  // 2. 调用构造函数创建 Dialect
  auto dialect = ctor();  // 例如 new ArithDialect(context)

  // 3. 应用注册表中的 extensions
  dialectsRegistry.applyExtensions(dialect.get());

  // 4. 存入 loadedDialects 映射
  loadedDialects[namespace] = std::move(dialect);
  return dialect.get();
}
```

### 1.4 生命周期要点

- **MLIRContext 比所有 IR 对象活得长**：Type、Attribute 等"不朽对象"由 Context 拥有
- **一个 Context 通常对应一次编译**：不同 Context 之间的 IR 对象不能混用
- **线程安全**：Context 内部通过 mutex 保护共享状态，支持多线程 Pass 执行

---

## 二、Dialect — 扩展的基本单元

> 源文件：`include/mlir/IR/Dialect.h`，实现：`lib/IR/Dialect.cpp`

### 2.1 角色定位

Dialect 将一组相关的 Operation、Type、Attribute 组织在一起，通过**命名空间**隔离。它是 MLIR 可扩展性的基础。

```
Dialect "arith"
├── Operations: addi, addf, muli, mulf, constant, ...
├── Types: (无自定义类型，使用 builtin)
├── Attributes: FastMathFlagsAttr, RoundingModeAttr, ...
└── Interfaces: ArithEmulateUnsupportedInterface, ...
```

### 2.2 核心 API

```cpp
class Dialect {
  // ---- 构造 ----
  Dialect(StringRef name, MLIRContext *context, TypeID id);

  // ---- 注册 Operation ----
  template <typename... Args>
  void addOperations();         // 注册一个或多个 Operation

  // ---- 注册 Type ----
  template <typename... Args>
  void addTypes();              // 注册自定义类型

  // ---- 注册 Attribute ----
  template <typename... Args>
  void addAttributes();         // 注册自定义属性

  // ---- 接口管理 ----
  template <typename InterfaceT>
  InterfaceT *getRegisteredInterface();

  template <typename InterfaceT, typename... Args>
  InterfaceT &addInterface(Args &&...args);   // 添加接口实现

  template <typename InterfaceT, typename ConcreteT>
  void declarePromisedInterface();            // 声明将来会实现的接口

  // ---- 解析/打印钩子 ----
  virtual Type parseType(DialectAsmParser &parser);
  virtual void printType(Type, DialectAsmPrinter &);
  virtual Attribute parseAttribute(DialectAsmParser &parser, Type type);
  virtual void printAttribute(Attribute, DialectAsmPrinter &);

  // ---- 验证钩子 ----
  virtual LogicalResult verifyRegionArgAttribute(
      Operation *, unsigned regionIndex, unsigned argIndex, NamedAttribute);
  virtual LogicalResult verifyOperationAttribute(Operation *, NamedAttribute);

  // ---- 基本信息 ----
  StringRef getNamespace() const;
  MLIRContext *getContext() const;
};
```

### 2.3 Dialect 的 initialize() 模式

自定义 Dialect 通常覆盖 `initialize()` 方法注册所有组件：

```cpp
class ArithDialect : public Dialect {
public:
  explicit ArithDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context, TypeID::get<ArithDialect>()) {
    initialize();
  }

  void initialize() override {
    // 注册所有 Operation（由 TableGen 生成列表）
    addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/IR/ArithOps.cpp.inc"
    >();
    // 注册自定义 Attribute
    addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Arith/IR/ArithOpsAttributes.cpp.inc"
    >();
    // 注册接口
    addInterfaces<ArithEmulateUnsupportedInterface>();
  }
};
```

### 2.4 DialectRegistry

`DialectRegistry` 是一个**延迟加载**机制——只声明 Dialect 可用，不立即实例化：

```cpp
DialectRegistry registry;
registry.insert<arith::ArithDialect>();    // 注册但不加载
registry.insert<func::FuncDialect>();

MLIRContext context(registry);             // Context 使用此注册表
context.getOrLoadDialect<arith::ArithDialect>();  // 此时才真正构造
```

---

## 三、Type — 类型系统

> 源文件：`include/mlir/IR/Types.h`

### 3.1 角色定位

`Type` 是 MLIR 中的**值类型语义**对象——它是一个轻量级的值类型（类似 `StringRef`），底层通过指针指向 Context 中唯一化的存储。

```
Type (轻量值类型, sizeof = 1 pointer)
  └── TypeImpl* (存储在 MLIRContext 中, 全局唯一)
        ├── AbstractType     (类型描述信息)
        └── 具体存储数据     (如 IntegerType 的位宽)
```

### 3.2 核心 API

```cpp
class Type {
public:
  // ---- 上下文 ----
  MLIRContext *getContext() const;

  // ---- 类型查询 ----
  bool isIndex() const;
  bool isF32() const;
  bool isF64() const;
  bool isInteger() const;
  bool isInteger(unsigned width) const;
  bool isSignedInteger() const;
  bool isUnsignedInteger() const;
  unsigned getIntOrFloatBitWidth() const;

  // ---- 类型 ID ----
  TypeID getTypeID() const;
  Dialect &getDialect() const;

  // ---- 子元素遍历与替换 ----
  void walkImmediateSubElements(
      function_ref<void(Attribute)>, function_ref<void(Type)>);
  Type replaceImmediateSubElements(ArrayRef<Attribute>, ArrayRef<Type>);

  // ---- 打印 ----
  void print(raw_ostream &os) const;
  void dump() const;

  // ---- 类型转换 (dyn_cast) ----
  template <typename U> U dyn_cast() const;
  template <typename U> U cast() const;
  template <typename U> bool isa() const;
  template <typename... Us> bool isa_any() const;
};
```

### 3.3 TypeBase — CRTP 模式定义自定义类型

```cpp
// CRTP: Curiously Recurring Template Pattern
// ConcreteType = 你定义的类型, BaseT = 基类, StorageType = 存储类
template <typename ConcreteType, typename BaseT = Type,
          typename StorageType = DefaultTypeStorage, typename... Traits>
class TypeBase : public BaseT {
  // 提供 classof(), get(), getChecked() 等方法
};
```

使用示例（`BuiltinTypes.h`）：

```cpp
class IntegerType : public TypeBase<IntegerType, Type, IntegerTypeStorage> {
public:
  using Base::Base;

  static IntegerType get(MLIRContext *context, unsigned width);
  unsigned getWidth() const;
  bool isSigned() const;
  bool isUnsigned() const;
};
```

### 3.4 Type 唯一化（Uniquing）

类型在 Context 中是**全局唯一**的：

```cpp
// 同一个 Context 中，两次 get 返回同一个底层对象
auto t1 = IntegerType::get(ctx, 32);
auto t2 = IntegerType::get(ctx, 32);
assert(t1 == t2);          // 指针相等比较，O(1)
```

通过 `StorageUniquer` 实现：
1. 计算类型的 hash
2. 在 Context 的类型表中查找
3. 如果已存在则返回现有实例
4. 如果不存在则创建并存入

**好处**：指针相等比较代替结构相等比较，高效。

### 3.5 常见 Type 层次结构

```
Type
├── IntegerType          (i1, i32, i64, ...)
├── FloatType            (f16, f32, f64, bf16, ...)
├── IndexType            (index, 用于张量维度)
├── FunctionType         ((inputs) -> (outputs))
├── RankedTensorType     (tensor<MxNxT>)
├── UnrankedTensorType   (tensor<*xT>)
├── MemRefType           (memref<MxNxT, affine_map>)
├── VectorType           (vector<MxNxT>)
├── TupleType            (tuple<T1, T2, ...>)
├── NoneType             (none)
└── ... (Dialect 自定义类型)
```

---

## 四、Operation — IR 的核心单元

> 源文件：`include/mlir/IR/Operation.h`，实现：`lib/IR/Operation.cpp`

### 4.1 角色定位

`Operation` 是 MLIR IR 的**原子单元**，表示一个计算操作。在 `.mlir` 文本中就是一行 Op：

```mlir
%result = arith.addi %lhs, %rhs : i32
```

对应一个 `Operation` 对象，包含名称、操作数、结果、属性、区域等。

### 4.2 内存布局

Operation 使用 **Trailing Objects** 模式，一次 `malloc` 分配所有空间：

```
内存布局 (低地址 → 高地址):

[prefix 空间] [Results (逆序)] [Operation 本体] [Properties] [BlockOperands] [Regions] [Operands]
                                      ↑
                              Operation* 指针指向这里
```

关键设计点：
- **Results 存在 Operation 之前**（逆序），因为结果数量在编译期可能不知道
- **Operands 尾分配**，可以动态增长（从内联迁移到堆分配）
- 前 5 个 Result 是 `InlineOpResult`（index 只需 3 bit，打包在 Type 指针中）
- 第 6 个及之后的 Result 是 `OutOfLineOpResult`（需要额外 unsigned 存 index）

### 4.3 Operation 创建流程

```cpp
// 用户代码
OpBuilder builder(context);
auto op = builder.create<arith::AddIOp>(loc, lhs, rhs);

// 内部调用链:
// 1. OpBuilder::create<AddIOp>(...) → 构造 OperationState
// 2. AddIOp::build(...) → 填充 operands, types, attributes
// 3. OpBuilder::create(OperationState&) → Operation::create(state)
// 4. Operation::create(...) → malloc + placement new
```

核心分配代码（`lib/IR/Operation.cpp`）：

```cpp
Operation *Operation::create(Location loc, OperationName name,
                             TypeRange resultTypes, ValueRange operands,
                             DictionaryAttr attrs, ...) {
  // 1. 计算总分配大小
  size_t byteSize = totalSizeToAlloc<OperandStorage, OpProperties,
                                     BlockOperand, Region, OpOperand>(
      needsOperandStorage ? 1 : 0, propSize, numSuccessors,
      numRegions, numOperands);

  // 2. 加上 prefix（results 的空间）
  size_t prefixByteSize = alignTo(
      prefixAllocSize(numTrailingResults, numInlineResults),
      alignof(Operation));

  // 3. 一次 malloc
  char *mem = (char *)malloc(byteSize + prefixByteSize);
  void *rawMem = mem + prefixByteSize;

  // 4. Placement new 构造 Operation
  Operation *op = new (rawMem) Operation(loc, name, numResults, ...);

  // 5. 初始化 results, operands, regions...
  return op;
}
```

### 4.4 核心 API

```cpp
class Operation {
  // ---- 标识 ----
  OperationName getName();                // 操作名（如 "arith.addi"）
  StringAttr getNameAttr();               // 操作名的 StringAttr
  bool isRegistered();                    // 是否在 Dialect 中注册

  // ---- 操作数 (Operands/Inputs) ----
  Value getOperand(unsigned idx);
  void setOperand(unsigned idx, Value value);
  OperandRange getOperands();
  unsigned getNumOperands();
  void setOperands(ValueRange operands);
  OpOperand &getOpOperand(unsigned idx);  // 获取 use-def 链节点

  // ---- 结果 (Results/Outputs) ----
  OpResult getResult(unsigned idx);
  ResultRange getResults();
  unsigned getNumResults();

  // ---- 属性 (Attributes) ----
  DictionaryAttr getAttrDictionary();
  Attribute getAttr(StringRef name);
  template <typename T> T getAttrOfType(StringRef name);
  void setAttr(StringRef name, Attribute value);
  void removeAttr(StringRef name);

  // ---- 区域 (Regions) ----
  Region &getRegion(unsigned idx);
  unsigned getNumRegions();
  bool hasSingleBlock();                  // 便捷：是否有且仅有一个 Block

  // ---- 后继 (Successors) ----
  Block *getSuccessor(unsigned idx);
  void setSuccessor(unsigned idx, Block *block);
  unsigned getNumSuccessors();

  // ---- 父 Block ----
  Block *getBlock();                      // 此 Operation 所在的 Block

  // ---- 克隆与折叠 ----
  Operation *clone(IRMapping &mapper = nullptr);  // 深拷贝
  LogicalResult fold(ArrayRef<Attribute> &operands);  // 常量折叠

  // ---- 遍历 ----
  template <typename OpT> OpT getParentOp();  // 向上找父 Op
  void walk(llvm::function_ref<void(Operation *)> callback);  // 遍历子 Op

  // ---- 打印 ----
  void print(raw_ostream &os, const OpPrintingFlags &flags = {});
  void dump();
};
```

### 4.5 OperationState — 构建操作的中间表示

`OperationState` 是创建 Operation 时的参数打包对象：

```cpp
struct OperationState {
  Location location;                              // 源码位置
  OperationName name;                             // 操作名称
  SmallVector<Value, 4> operands;                 // 操作数
  SmallVector<Type, 4> types;                     // 结果类型
  NamedAttrList attributes;                       // 属性列表
  SmallVector<Block *, 1> successors;             // 后继 Block
  SmallVector<std::unique_ptr<Region>, 1> regions; // 区域
  OpaqueProperties properties = nullptr;          // 属性存储
};
```

### 4.6 Op<> 模板 — 类型安全的 Operation 包装

```cpp
// Operation 是通用的，Op<> 模板提供类型安全的接口
template <typename ConcreteOp, typename... Traits>
class Op : public OpState, public Traits... {
  // 提供类型安全的 getOperand(), getResult() 等
  // 由 TableGen 生成的 Op 类继承此模板
};

// 示例：TableGen 生成的 AddIOp
class AddIOp : public Op<AddIOp, Arith_Op::elementwiseTrait,
                         SameOperandsAndResultType, ...> {
  Value getLhs() { return getOperand(0); }    // 类型安全的 getter
  Value getRhs() { return getOperand(1); }
  Value getResult() { return Op::getResult(0); }
};
```

---

## 五、Value — SSA 值

> 源文件：`include/mlir/IR/Value.h`

### 5.1 角色定位

`Value` 是 MLIR SSA 图中的**边**——它连接 Operation 的输出和另一个 Operation 的输入。

```
Value 有两种来源:
  1. OpResult    — Operation 的输出结果
  2. BlockArgument — Block 的参数（函数参数、Region 入口参数）

%1 = arith.constant 42 : i32    // %1 是 OpResult (ConstantOp 的结果)
    ^^^^^^^^^^^^^^^^
    func.func @foo(%arg0: i32)   // %arg0 是 BlockArgument
```

### 5.2 Use-Def 链

MLIR 维护双向的 use-def 链：

```
Value (定义端)
  └── 第一个 use → OpOperand → 下一个 use → OpOperand → ...
                      |                         |
                    Operation                  Operation
                    (使用这个 Value 作为 operand)

OpOperand (使用端)
  └── 指向 Value (def)
  └── 指向所属 Operation
  └── 指向下一个/上一个 use (双向链表)
```

### 5.3 核心 API

```cpp
class Value {
  // ---- 类型 ----
  Type getType() const;
  void setType(Type newType);

  // ---- 定义来源 ----
  bool isa<OpResult>();              // 是否是 Operation 的结果
  bool isa<BlockArgument>();         // 是否是 Block 参数
  Operation *getDefiningOp();        // 获取定义此 Value 的 Operation（BlockArgument 返回 nullptr）

  // ---- 使用者遍历 ----
  use_iterator use_begin();          // 遍历所有使用此 Value 的 OpOperand
  use_iterator use_end();
  user_iterator user_begin();        // 遍历所有使用此 Value 的 Operation
  user_iterator user_end();
  bool use_empty();                  // 是否没有任何使用者
  bool hasOneUse();                  // 是否恰好有一个使用者
  void replaceAllUsesWith(Value newValue);  // 替换所有使用（核心变换操作！）

  // ---- 所在 Block/Region ----
  Block *getParentBlock();
  Region *getParentRegion();
};
```

### 5.4 replaceAllUsesWith — 最重要的变换原语

```cpp
// 这是 MLIR 图变换的核心操作：
// 将所有使用 oldValue 的地方替换为 newValue
oldValue.replaceAllUsesWith(newValue);

// 示例：
//   %1 = arith.constant 42 : i32
//   %2 = arith.addi %1, %0 : i32
// 执行 %1.replaceAllUsesWith(%3) 后：
//   %1 = arith.constant 42 : i32   (仍然存在，但没有人使用)
//   %2 = arith.addi %3, %0 : i32   (%1 被替换为 %3)
```

---

## 六、Attribute — 编译期常量值

> 源文件：`include/mlir/IR/Attributes.h`

### 6.1 角色定位

`Attribute` 表示附加在 Operation 上的**编译期已知常量数据**。和 `Type` 类似，它是轻量值类型，底层由 Context 唯一化。

```
Operation
├── Operands (运行时 Value)
├── Results  (运行时 Value)
└── Attributes (编译期常量)
      ├── inherent attrs  (Op 定义的固有属性，如 ConstantOp 的 value)
      └── discardable attrs (可被 Pass 移除的附加信息)
```

### 6.2 常见 Attribute 类型

```
Attribute
├── IntegerAttr         (整数值 + 类型，如 42 : i32)
├── FloatAttr           (浮点值 + 类型)
├── StringAttr          (字符串，如 "constant")
├── BoolAttr            (布尔值)
├── ArrayAttr           (Attribute 数组)
├── DictionaryAttr      (StringAttr → Attribute 映射)
├── DenseElementsAttr   (密集张量常量，如 dense<[[1,2],[3,4]]>)
├── SymbolRefAttr       (符号引用，如 @main)
├── TypeAttr            (包装 Type 为 Attribute)
├── UnitAttr            (无值，表示"存在")
└── ... (Dialect 自定义 Attribute)
```

---

## 七、OpBuilder — IR 构建工具

> 源文件：`include/mlir/IR/Builders.h`，实现：`lib/IR/Builders.cpp`

### 7.1 角色定位

`OpBuilder` 是构建 MLIR IR 的**主要工具**，管理插入点并提供类型安全的 Operation 创建 API。

### 7.2 Builder 类层次

```
Builder                        // 基础类：创建 Type、Attribute
  └── OpBuilder                // 增加插入点管理、Operation 创建
        └── ImplicitLocOpBuilder  // 增加隐式 Location
```

### 7.3 OpBuilder 核心 API

```cpp
class OpBuilder : public Builder {
  // ==== 插入点管理 ====

  // 设置插入点
  void setInsertionPoint(Operation *op);            // 插入到 op 之前
  void setInsertionPointAfter(Operation *op);       // 插入到 op 之后
  void setInsertionPointAfterValue(Value val);      // 插入到定义 val 的 Op 之后
  void setInsertionPointToEnd(Block *block);        // 插入到 Block 末尾
  void setInsertionPointToStart(Block *block);      // 插入到 Block 开头

  // 保存/恢复插入点
  InsertPoint saveInsertionPoint();
  void restoreInsertionPoint(InsertPoint ip);

  // RAII 守卫（推荐使用）
  OpBuilder::InsertionGuard guard(builder);  // 构造时保存，析构时自动恢复

  // ==== Operation 创建 ====

  // 通用创建
  Operation *create(const OperationState &state);

  // 类型安全的模板创建（最常用）
  template <typename OpTy, typename... Args>
  OpTy create(Location loc, Args &&...args);

  // 创建或折叠（如果结果可折叠，直接返回常量，不创建新 Op）
  template <typename OpTy, typename... Args>
  void createOrFold(SmallVectorImpl<Value> &results, Location loc, Args &&...args);

  // ==== Block 创建 ====
  Block *createBlock(Region *parent, Region::iterator insertPt,
                     TypeRange argTypes = {}, ArrayRef<Location> locs = {});
  Block *createBlock(Block *insertBefore, TypeRange argTypes = {},
                     ArrayRef<Location> locs = {});

  // ==== Listener 模式 ====
  // 可以注册 Listener 监听 Operation 插入事件
  void setListener(Listener *newListener);
};
```

### 7.4 Builder 基础类 API（Type/Attribute 创建）

```cpp
class Builder {
public:
  // ---- 整数类型 ----
  IntegerType getI1Type(), getI8Type(), getI32Type(), getI64Type();
  IntegerType getIntegerType(unsigned width);

  // ---- 浮点类型 ----
  FloatType getF16Type(), getF32Type(), getF64Type(), getBF16Type();

  // ---- 复合类型 ----
  FunctionType getFunctionType(TypeRange inputs, TypeRange results);
  RankedTensorType getTensorType(ArrayRef<int64_t> shape, Type elementType);
  MemRefType getMemRefType(ArrayRef<int64_t> shape, Type elementType);
  VectorType getVectorType(ArrayRef<int64_t> shape, Type elementType);

  // ---- 常用 Type ----
  IndexType getIndexType();
  NoneType getNoneType();

  // ---- Attribute 创建 ----
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  FloatAttr getFloatAttr(Type type, double value);
  StringAttr getStringAttr(const Twine &bytes);
  BoolAttr getBoolAttr(bool value);
  ArrayAttr getArrayAttr(ArrayRef<Attribute> value);
  DictionaryAttr getDictionaryAttr(ArrayRef<NamedAttribute> value);
  DenseElementsAttr getDenseElementsAttr(ShapedType type, ArrayRef<Attribute> values);
  UnitAttr getUnitAttr();
};
```

### 7.5 使用示例

```cpp
MLIRContext context;
OpBuilder builder(&context);

// 设置插入点
auto *module = builder.create<ModuleOp>(UnknownLoc::get(&context));
builder.setInsertionPointToStart(module.getBody());

// 创建 Operation
auto loc = builder.getUnknownLoc();
auto i32 = builder.getI32Type();

// 方式 1：通过具体 Op 类创建
auto cst = builder.create<arith::ConstantOp>(loc, i32, builder.getIntegerAttr(i32, 42));

// 方式 2：通过通用 OperationState 创建
OperationState state(loc, "arith.constant");
state.types.push_back(i32);
state.addAttribute("value", builder.getIntegerAttr(i32, 42));
auto *op = builder.create(state);
```

---

## 八、各组件之间的关系图

```
                    ┌──────────────────┐
                    │   MLIRContext    │  全局管理器
                    │  ┌────────────┐  │
                    │  │ TypeUniquer│  │  唯一化 Type
                    │  │ AttrUniquer│  │  唯一化 Attribute
                    │  │ DialectReg │  │  管理已加载 Dialect
                    │  └────────────┘  │
                    └───────┬──────────┘
                            │ owns
              ┌─────────────┼──────────────┐
              │             │              │
        ┌─────▼─────┐ ┌────▼────┐  ┌──────▼──────┐
        │  Dialect   │ │  Type   │  │  Attribute  │
        │ "arith"    │ │ (i32)   │  │ (42 : i32)  │
        │ "func"     │ │ (f64)   │  │ (dense<..>) │
        │ "toy"      │ │ (tensor)│  │ (string)    │
        └─────┬──────┘ └────┬────┘  └──────┬──────┘
              │              │              │
        registers        defines         attached to
              │              │              │
        ┌─────▼──────────────▼──────────────▼──────┐
        │              Operation                    │
        │  ┌──────────────────────────────────┐    │
        │  │ name: "arith.addi"               │    │
        │  │ operands: [%lhs, %rhs] ──────────┼──→ Value (OpResult / BlockArgument)
        │  │ results: [%res]                  │    │    │
        │  │ attributes: {"overflow": "nsw"}  │    │    │ use-def chain
        │  │ regions: []                      │    │    │
        │  │ location: loc("test.mlir":5:10)  │    │    ▼
        │  └──────────────────────────────────┘    │  OpOperand (使用端)
        └──────────────────┬───────────────────────┘
                           │ contains
                    ┌──────▼──────┐
                    │   Region    │  0..N 个
                    │  ┌───────┐  │
                    │  │ Block │  │  1..N 个
                    │  │  ┌──┐ │  │
                    │  │  │Op│ │  │  0..N 个 Operation
                    │  │  └──┘ │  │
                    │  └───────┘  │
                    └─────────────┘
```

---

## 九、Operation 的创建完整流程

```
用户代码
  │
  ▼
builder.create<arith::AddIOp>(loc, lhs, rhs)
  │
  ├─ 1. 构造 OperationState
  │     state.location = loc
  │     state.name = "arith.addi"
  │     state.operands = {lhs, rhs}
  │     state.types = {i32}
  │
  ├─ 2. 调用 AddIOp::build()
  │     （由 TableGen 生成或手写的 build 方法）
  │     填充 state 的 operands, types, attributes
  │
  ├─ 3. 调用 OpBuilder::create(state)
  │     → Operation::create(state)
  │
  ├─ 4. Operation::create 内部
  │     ├─ 计算 malloc 大小（results + Operation + trailing objects）
  │     ├─ malloc 分配内存
  │     ├─ placement new 构造 Operation
  │     ├─ 初始化 OpResults（存在 Operation 之前）
  │     ├─ 初始化 OpOperands（尾分配）
  │     ├─ 初始化 Regions（如果有）
  │     └─ 建立 use-def 链（operand → value → use list）
  │
  ├─ 5. OpBuilder::insert(op)
  │     将 Operation 插入到当前插入点所在的 Block
  │
  └─ 6. 返回 AddIOp(op) — 类型安全的包装
```

---

## 十、关键设计模式总结

| 设计模式 | 应用位置 | 说明 |
|----------|---------|------|
| **Placement New + Trailing Objects** | Operation | 一次 malloc 分配所有内存，高效紧凑 |
| **CRTP (奇异递归模板)** | TypeBase, Op<> | 编译期多态，零开销抽象 |
| **Flyweight (享元)** | Type, Attribute | Context 中全局唯一，指针比较代替结构比较 |
| **Builder** | OpBuilder | 分步构建复杂对象，管理插入点 |
| **RAII** | InsertionGuard, OpBuilder | 自动恢复插入点 |
| **Registry** | DialectRegistry | 延迟加载，按需实例化 |
| **Observer/Listener** | OpBuilder::Listener | 监听 Operation 插入事件 |
| **Use-Def Chain (双向链表)** | Value ↔ OpOperand | O(1) 遍历使用者和定义 |

---

## 十一、核心要点回顾

1. **MLIRContext 是一切的容器**：所有 IR 对象都活在 Context 中，Context 负责唯一化和生命周期管理
2. **Operation 是 IR 的核心**：使用 Trailing Objects 的高效内存布局，包含 operands、results、attributes、regions
3. **Value 是 SSA 的边**：OpResult 和 BlockArgument 两种来源，通过 use-def 链追踪所有使用者
4. **Type 和 Attribute 是轻量值类型**：底层由 Context 唯一化存储，支持高效的指针比较
5. **OpBuilder 是构建 IR 的主要工具**：管理插入点，提供类型安全的 create 模板方法
6. **Dialect 是扩展的入口**：通过命名空间组织 Operation、Type、Attribute，支持延迟加载
7. **replaceAllUsesWith 是变换的核心原语**：几乎所有 IR 变换最终都通过替换 Value 的使用者来实现

---

## 参考资源

- `mlir/include/mlir/IR/Operation.h` — Operation 类定义
- `mlir/include/mlir/IR/Value.h` — Value, OpResult, BlockArgument
- `mlir/include/mlir/IR/Builders.h` — OpBuilder
- `mlir/include/mlir/IR/Types.h` — Type 系统
- `mlir/include/mlir/IR/Attributes.h` — Attribute 系统
- `mlir/include/mlir/IR/Dialect.h` — Dialect 类
- `mlir/include/mlir/IR/MLIRContext.h` — MLIRContext
- `mlir/include/mlir/IR/OperationSupport.h` — OperationState, OpResult 等
- `mlir/lib/IR/Operation.cpp` — Operation 实现（内存分配细节）
- `mlir/lib/IR/MLIRContext.cpp` — Context 实现（唯一化细节）
