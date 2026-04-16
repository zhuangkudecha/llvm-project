# MLIR Step 5: TableGen 与 ODS 机制 - 学习总结

## 概述

MLIR 使用 **TableGen** 作为代码生成工具，通过 `.td` 文件声明式地定义 Dialect、Operation、Type、Attribute 等组件，再由 `mlir-tblgen` 自动生成 C++ 样板代码。这套系统称为 **ODS (Operation Definition Specification)**。

本文档涵盖三个核心问题：
1. `.mlir` 文件与 `.td` 文件的区别
2. `.td` 生成代码与手写 C++ 的配合机制
3. TableGen (.td) 文件的语法

---

## 一、.mlir 文件与 .td 文件的区别

### 1.1 本质差异

| 维度 | `.td` 文件 | `.mlir` 文件 |
|------|-----------|-------------|
| 本质 | 代码生成模板（定义语言规则） | 用该语言写的程序（中间表示） |
| 类比 | 语法规则定义 | 用该语言写的代码 |
| 谁读它 | `mlir-tblgen`（代码生成器） | `mlir-opt`（编译器） |
| 产出 | C++ `.inc` 文件 | 变换后的 IR / LLVM IR / 机器码 |
| 何时写 | 开发 Dialect/Operation 时 | 测试/调试/写示例 IR 时 |

### 1.2 具体内容对比

**.td 文件** — 定义 Operation 的结构：
```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

**.mlir 文件** — 使用定义好的 Operation 编写程序：
```mlir
module {
  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
    %0 = toy.add %arg0, %arg0 : tensor<f64>
    return %0 : tensor<f64>
  }
}
```

**一句话总结**：`.td` 定义"语言长什么样"，`.mlir` 是"用这个语言写的程序"。

---

## 二、.td 生成代码与手写 C++ 的配合机制

### 2.1 整体流程

```
.td 文件  ──mlir-tblgen──▶  .h.inc（声明）  +  .cpp.inc（定义）
                                        ↓
手写 C++ 通过 #include 把生成的代码"嵌入"到自己的类中
```

### 2.2 CMake 声明生成规则

以 Toy 教程为例（`mlir/examples/toy/Ch5/include/toy/CMakeLists.txt`）：

```cmake
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)       # 生成 Op 类声明
mlir_tablegen(Ops.cpp.inc -gen-op-defs)       # 生成 Op 类定义
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
```

### 2.3 .td 中控制生成行为的标志

`.td` 文件中的标志位控制生成代码只放**声明**还是同时生成**默认实现**：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let hasCustomAssemblyFormat = 1;  // 只生成声明，需要手写 parse/print
  let hasVerifier = 1;              // 只生成声明，需要手写 verify
  let hasFolder = 1;                // 只生成声明，需要手写 fold
  let hasCanonicalizer = 1;         // 只生成声明，需要手写 canonicalize
}
```

设为 `1` 意味着"我自己来实现"，生成代码只放函数声明。

### 2.4 手写 C++ 如何 include 生成的代码

**头文件**（`Dialect.h`）— 引入生成的声明：

```cpp
// 引入生成的 Dialect 类声明
#include "toy/Dialect.h.inc"

// 引入生成的所有 Op 类声明
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

**源文件**（`Dialect.cpp`）— 引入生成的定义 + 手写实现：

```cpp
// ---- 1) Dialect 初始化：注册所有 Op ----
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"   // 展开为：AddOp, ConstantOp, MulOp, ...
  >();
}

// ---- 2) 手写自定义方法 ----
void ConstantOp::build(OpBuilder &builder, OperationState &state, double value) {
  // 手写 builder 逻辑（因为 .td 中声明了自定义 builder）
}

mlir::ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  // 手写解析逻辑（因为 hasCustomAssemblyFormat = 1）
}

void ConstantOp::print(OpAsmPrinter &printer) {
  // 手写打印逻辑（因为 hasCustomAssemblyFormat = 1）
}

llvm::LogicalResult ConstantOp::verify() {
  // 手写验证逻辑（因为 hasVerifier = 1）
}

// ---- 3) 文件末尾引入生成的 Op 方法定义 ----
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

### 2.5 职责划分

| 部分 | TableGen 自动生成 | 手写 C++ |
|------|------------------|----------|
| 类骨架（继承、模板参数） | ✅ | ❌ |
| operand/result 的 getter（如 `getLhs()`） | ✅ | ❌ |
| 属性访问器（如 `getValue()`） | ✅ | ❌ |
| 默认 builder | ✅ | ❌ |
| 自定义 builder 实现 | 声明 | **✅** |
| parse/print（自定义汇编格式） | 声明 | **✅** |
| verify（自定义验证） | 声明 | **✅** |
| canonicalize / fold | 声明 | **✅** |
| Dialect 注册操作列表 | ✅ | 调用 `addOperations<...>` |
| DRR 重写模式 | ✅（`.td` → `.inc`） | 可选手写 OpRewritePattern |

### 2.6 典型的 .cpp 文件结构

```cpp
// ===== 头文件包含 =====
#include "toy/Dialect.h"
#include "mlir/IR/Builders.h"
// ...

// ===== 引入生成的 Dialect 定义 =====
#include "toy/Dialect.cpp.inc"

// ===== Dialect 初始化 =====
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
}

// ===== 手写 Op 方法实现 =====
void ConstantOp::build(...) { /* 自定义逻辑 */ }
ParseResult ConstantOp::parse(...) { /* 自定义逻辑 */ }
void ConstantOp::print(...) { /* 自定义逻辑 */ }
LogicalResult ConstantOp::verify() { /* 自定义逻辑 */ }

// ===== 文件末尾引入生成的 Op 定义 =====
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

**关键约定**：`#define GET_OP_CLASSES` + `#include "... .cpp.inc"` 必须放在 **.cpp 文件末尾**，因为它会展开所有生成的内联方法定义。

---

## 三、TableGen (.td) 文件语法详解

### 3.1 基本元素

#### `class` — 模板（抽象蓝图）

```tablegen
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

- 类似 C++ 的类模板，**可带参数和默认值**
- 不会被直接生成代码，只是用来被 `def` 继承

#### `def` — 具体定义（实例化记录）

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant operation";
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}
```

- 创建一个**具名记录**，`mlir-tblgen` 会为它生成 C++ 代码
- `: ParentClass<args>` 表示继承
- `let field = value;` 设置字段值

#### `defvar` — 变量

```tablegen
defvar maxBits = 128;
```

### 3.2 数据类型

| 类型 | 示例 | 说明 |
|------|------|------|
| `string` | `"hello"` | 字符串 |
| `int` / `bit` | `42`, `1` | 整数/位 |
| `list<T>` | `[Pure, Commutative]` | 列表，用 `#` 拼接 |
| `dag` | `(ins F64Tensor:$x)` | DAG（有向无环图），最重要的类型 |
| `code` | `[{ C++ code }]` | C++ 代码块 |
| `class` | `Op` | 类引用 |

### 3.3 DAG 语法 — 最核心的语法

DAG 用 `()` 包裹，是 TableGen 最独特的语法：

```tablegen
// 格式: (operator arg1, arg2:type:$name, ...)
(ins F64ElementsAttr:$value)                    // 单个输入参数
(outs F64Tensor)                                 // 单个输出结果
(ins F64Tensor:$lhs, F64Tensor:$rhs)            // 多个命名参数
```

- `ins` / `outs` / `region` / `successor` 是 DAG 操作符
- `Type:$name` 声明一个带类型的命名参数
- `:$name` 是可选的名字绑定

### 3.4 代码块 `[{ ... }]`

嵌入 C++ 代码，用 `[{` 和 `}]` 包裹：

```tablegen
let builders = [
  OpBuilder<(ins "DenseElementsAttr":$value), [{
    build($_builder, $_state, value.getType(), value);  // C++ 代码
  }]>
];
```

特殊替换变量：
- `$_builder` → `OpBuilder &`
- `$_state` → `OperationState &`

### 3.5 MLIR .td 文件的典型结构

```tablegen
// ============================================
// 1) 引入基础定义
// ============================================
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CallInterfaces.td"

// ============================================
// 2) 定义 Dialect
// ============================================
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

// ============================================
// 3) 定义类型约束（可选）
// ============================================
def F64Tensor : TensorOf<[F64]>;

// ============================================
// 4) 定义基础 Op 类（模板）
// ============================================
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// ============================================
// 5) 定义具体 Operations
// ============================================
def AddOp : Toy_Op<"add", [Pure]> {
  let summary = "element-wise addition";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
  }];
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}

def ConstantOp : Toy_Op<"constant", [Pure, ConstantLike]> {
  let summary = "constant";
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "double":$value)>
  ];
}
```

### 3.6 Op 类的常用字段速查

```tablegen
// 输入输出
let arguments = (ins Type1:$a, Type2:$b);     // 操作数 + 属性
let results = (outs Type1);                    // 结果
let regions = (region AnyRegion:$body);        // 区域
let successors = (successor AnySuccessor:$s);  // 后继

// 特征与接口
let traits = [Pure, Commutative];              // 特征列表

// 汇编格式
let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";  // 声明式
let hasCustomAssemblyFormat = 1;               // 手写 parse/print

// 自定义行为标志
let hasVerifier = 1;         // 手写 verify()
let hasCanonicalizer = 1;    // 手写 getCanonicalizationPatterns()
let hasFolder = 1;           // 手写 fold()

// 自定义 builder
let builders = [
  OpBuilder<(ins "Type":$type, "Value":$val)>,
  OpBuilder<(ins "double":$val), [{
    build($_builder, $_state, ...);
  }]>
];

// 注入代码到生成的 C++ 类中
let extraClassDeclaration = [{
  static bool classof(const Operation *op);
  void customMethod();
}];
```

### 3.7 高级语法

#### 列表拼接

```tablegen
let traits = parentTraits # [ExtraTrait];
```

#### 条件表达式

```tablegen
defvar result = !if(!gt(size, 0), "big", "small");
```

#### foreach 遍历

```tablegen
foreach i = 0...3 in {
  def Op#i : Toy_Op<!cast<string>(i)>;
}
```

#### multiclass（批量生成多个相关 def）

```tablegen
multiclass BinaryOp<string mnemonic, list<Trait> traits = []> {
  def NAME : Toy_Op<mnemonic, traits # [Pure]>;
}

defm AddOp : BinaryOp<"add">;
defm MulOp : BinaryOp<"mul", [Commutative]>;
```

### 3.8 语法风格速查表

| 概念 | 语法 | 说明 |
|------|------|------|
| 模板 | `class Name<params> : Parent;` | 抽象，不直接生成代码 |
| 实例 | `def Name : Parent<args> { let ...; }` | 具体记录，会生成代码 |
| 参数列表 | `(ins Type:$name, ...)` | DAG 语法 |
| C++ 代码块 | `[{ C++ }]` | 嵌入到生成代码中 |
| 列表 | `[a, b]`，拼接用 `#` | 类型化列表 |
| 字段赋值 | `let field = value;` | 在 def/class 中使用 |
| 文件包含 | `include "path/file.td"` | 引入其他定义 |
| 变量 | `defvar name = value;` | 顶层变量 |
| 条件 | `!if(cond, then, else)` | 编译期条件 |
| 类型转换 | `!cast<string>(val)` | 编译期类型转换 |

---

## 四、完整实例：从 .td 到可运行的 Op

以下展示一个完整的工作流程（以 Toy AddOp 为例）：

### Step 1: 在 .td 中定义

```tablegen
def AddOp : Toy_Op<"add",
    [Pure, Commutative,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($res)";
}
```

### Step 2: CMake 配置生成规则

```cmake
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(ToyOpsIncGen)
```

### Step 3: mlir-tblgen 自动生成的代码（.h.inc 片段）

```cpp
// 自动生成 AddOp 类声明
class AddOp : public Op<AddOp, Pure, Commutative, ...> {
public:
  // 自动生成 getter
  ::mlir::Value getLhs();     // 对应 $lhs
  ::mlir::Value getRhs();     // 对应 $rhs
  ::mlir::Value getRes();     // 对应 $res

  // 自动生成 builder（因为有 assemblyFormat）
  static void build(OpBuilder &, OperationState &,
                    Type resType, Value lhs, Value rhs);

  // 自动生成 parse/print（因为有 assemblyFormat）
  static ParseResult parse(OpAsmParser &, OperationState &);
  void print(OpAsmPrinter &);
};
```

### Step 4: 无需手写额外代码

由于使用了声明式 `assemblyFormat`，parse/print 由 `mlir-tblgen` 自动生成实现，无需手写 C++ 代码。只需要在 `Dialect.cpp` 中注册：

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
  >();
}

#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

### Step 5: 在 .mlir 中使用

```mlir
%result = toy.add %a, %b : tensor<2x3xf64>
```

---

## 五、关键要点总结

1. **TableGen 是声明式 DSL**：你在 .td 文件中"声明" Operation 的结构，工具自动生成 C++ 样板代码
2. **生成代码 vs 手写代码的边界**由标志位控制：`hasVerifier`、`hasCustomAssemblyFormat`、`hasFolder` 等设为 `1` 表示你需要手写实现
3. **DAG 语法是核心**：`(ins Type:$name, ...)` 和 `(outs Type:$name, ...)` 定义输入输出
4. **assemblyFormat 大幅减少手写代码**：使用声明式汇编格式可以避免手写 parse/print
5. **典型的 include 模式**：头文件 include `.h.inc`，源文件 include `.cpp.inc`，用 `#define GET_OP_CLASSES` 控制
6. **`def` 会生成代码，`class` 不会**：`class` 是模板，`def` 才是具体实例化

---

## 参考资源

- `llvm/docs/TableGen/index.rst` — TableGen 语言参考
- `mlir/include/mlir/IR/OpBase.td` — MLIR ODS 基础定义
- `mlir/examples/toy/Ch5/include/toy/Ops.td` — 完整的 Toy Dialect 定义示例
- `mlir/include/mlir/Dialect/Arith/IR/ArithOps.td` — 生产级 Dialect 定义示例
