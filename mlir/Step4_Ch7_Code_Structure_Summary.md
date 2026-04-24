# MLIR Toy Tutorial Ch7 — 代码结构详解

## 概述

Toy Ch7 是 MLIR Toy 教程的最终章，实现了一个**完整的编译器管道**：从 Toy 源代码解析、AST 生成、MLIR IR 生成、优化、形状推断、逐级 Lowering 到 LLVM IR，最终可以 JIT 执行。

```
toy 源码 → Lexer → Parser → AST → MLIRGen → Toy MLIR
                                              ↓
                                    Inliner + Canonicalizer
                                    + ShapeInference + CSE
                                              ↓
                                    LowerToAffine (部分 lowering)
                                              ↓
                                    LowerToLLVM (完全 lowering)
                                              ↓
                                    LLVM IR → JIT 执行 / 输出
```

---

## 一、目录结构

```
Ch7/
├── CMakeLists.txt                 # 顶层构建配置
├── toyc.cpp                       # 编译器入口 (main)
├── include/toy/                   # 头文件 & .td 定义
│   ├── CMakeLists.txt             # TableGen 生成规则
│   ├── Lexer.h                    # 词法分析器
│   ├── Parser.h                   # 语法分析器 (递归下降)
│   ├── AST.h                      # AST 节点定义
│   ├── MLIRGen.h                  # AST → MLIR 入口
│   ├── Dialect.h                  # Toy Dialect + StructType 定义
│   ├── Ops.td                     # TableGen Operation 定义
│   ├── Passes.h                   # Pass 入口声明
│   ├── ShapeInferenceInterface.h  # 形状推断接口头文件
│   └── ShapeInferenceInterface.td # 形状推断接口 .td 定义
├── parser/
│   └── AST.cpp                    # AST 打印 (dump)
└── mlir/
    ├── MLIRGen.cpp                # AST → MLIR IR 生成
    ├── Dialect.cpp                # Dialect + Op 实现 (parse/print/verify/inferShapes)
    ├── ToyCombine.td              # DRR 优化模式定义
    ├── ToyCombine.cpp             # 手写优化模式 + include 生成的 DRR 模式
    ├── ShapeInferencePass.cpp     # 形状推断 Pass
    ├── LowerToAffineLoops.cpp     # Toy → Affine/Arith/MemRef 部分降级
    └── LowerToLLVM.cpp            # 完全降级到 LLVM Dialect
```

---

## 二、编译器入口 — toyc.cpp

### 2.1 程序流程

```cpp
int main(int argc, char **argv) {
  // 1. 解析命令行参数
  cl::ParseCommandLineOptions(argc, argv, "toy compiler\n");

  // 2. 如果只要求 dump AST，直接解析并输出
  if (emitAction == Action::DumpAST)
    return dumpAST();

  // 3. 创建 MLIRContext，注册 Toy Dialect
  MLIRContext context(registry);
  context.getOrLoadDialect<ToyDialect>();

  // 4. 加载并处理 MLIR
  OwningOpRef<ModuleOp> module;
  loadAndProcessMLIR(context, module);

  // 5. 根据动作输出结果
  if (isOutputingMLIR) module->dump();
  if (emitAction == DumpLLVMIR) dumpLLVMIR(*module);
  if (emitAction == RunJIT) runJit(*module);
}
```

### 2.2 支持的动作 (Action)

| 命令行参数 | 动作 | 输出 |
|-----------|------|------|
| `-emit=ast` | `DumpAST` | 打印 AST |
| `-emit=mlir` | `DumpMLIR` | 打印 Toy MLIR |
| `-emit=mlir-affine` | `DumpMLIRAffine` | 打印 Affine lowering 后的 MLIR |
| `-emit=mlir-llvm` | `DumpMLIRLLVM` | 打印 LLVM lowering 后的 MLIR |
| `-emit=llvm` | `DumpLLVMIR` | 打印 LLVM IR |
| `-emit=jit` | `RunJIT` | JIT 编译执行 |
| `-opt` | 启用优化 | 配合上述动作使用 |

### 2.3 Pass 管道

`loadAndProcessMLIR()` 中的 Pass 管道：

```
                        ModuleOp
                          │
                 ┌────────▼────────┐
                 │  Inliner Pass   │  内联所有函数到 main
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  func.func(     │  嵌套到 FuncOp 内部
                 │   Canonicalize  │  标准化（触发 DRR 优化）
                 │   ShapeInfer    │  形状推断
                 │   Canonicalize  │  再次标准化
                 │   CSE           │  公共子表达式消除
                 │  )              │
                 └────────┬────────┘
                          │
              ┌───────────▼────────────┐
              │ LowerToAffine Pass     │  Toy → Affine + Arith + MemRef
              │  func.func(            │
              │   Canonicalize + CSE   │
              │   + LoopFusion (opt)   │
              │  )                     │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │ LowerToLLVM Pass       │  全部降级到 LLVM Dialect
              └───────────┬────────────┘
                          │
                     LLVM IR / JIT
```

对应代码（`toyc.cpp:149-198`）：

```cpp
mlir::PassManager pm(module.get()->getName());

// 阶段 1: 优化（内联 + 标准化 + 形状推断）
if (enableOpt || isLoweringToAffine) {
  pm.addPass(createInlinerPass());
  auto &optPM = pm.nest<toy::FuncOp>();
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(toy::createShapeInferencePass());
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createCSEPass());
}

// 阶段 2: 降级到 Affine
if (isLoweringToAffine) {
  pm.addPass(toy::createLowerToAffinePass());
  auto &optPM = pm.nest<func::FuncOp>();
  optPM.addPass(createCanonicalizerPass());
  optPM.addPass(createCSEPass());
}

// 阶段 3: 降级到 LLVM
if (isLoweringToLLVM) {
  pm.addPass(toy::createLowerToLLVMPass());
}

pm.run(*module);
```

---

## 三、前端：Lexer + Parser + AST

### 3.1 Lexer.h — 词法分析

将 Toy 源码字符流分割为 Token：

```cpp
// Token 类型
enum Token : int {
  tok_plus, tok_minus, tok_star, tok_slash,     // + - * /
  tok_lt, tok_le, tok_gt, tok_ge,               // < <= > >=
  tok_eq, tok_ne,                                // == !=
  tok_assign,                                    // =
  tok_semicolon, tok_paren_open, tok_paren_close,
  tok_bracket_open, tok_bracket_close,
  tok_brace_open, tok_brace_close,
  tok_comma, tok_colon,                          // ; ( ) [ ] { } , :
  tok_return, tok_var, tok_def,                  // 关键字
  tok_identifier, tok_number,                    // 标识符 / 数字
  tok_eof,
};
```

### 3.2 Parser.h — 语法分析

递归下降解析器，将 Token 流构建为 AST：

```
ModuleAST = FunctionAST*
FunctionAST = PrototypeAST BlockAST
BlockAST = ExprAST*
ExprAST = LiteralExprAST | VariableExprAST | BinaryExprAST |
          CallExprAST | PrintExprAST | ...
```

### 3.3 AST.h — AST 节点层次

```
ExprAST (基类)
├── VariableExprAST          # 变量引用: a
├── LiteralExprAST           # 字面量: [[1,2],[3,4]]
├── StructLiteralExprAST     # 结构体字面量 (Ch7 新增)
├── BinaryExprAST            # 二元运算: a + b
├── CallExprAST              # 函数调用: foo(a, b)
├── PrintExprAST             # 打印: print(a)
├── ReturnExprAST            # 返回: return a
├── VarDeclExprAST           # 变量声明: var a = ...
└── IndexExprAST             # 数组索引: a[0]

ModuleAST                     # 整个模块 (多个函数 + 结构体定义)
├── FunctionAST[]             # 函数列表
└── StructAST[]               # 结构体定义 (Ch7 新增)
    └── PrototypeAST          # 函数签名
```

### 3.4 parser/AST.cpp — AST dump

`AST.cpp` 只实现 AST 的 `dump()` 方法，用于 `-emit=ast` 调试。

---

## 四、MLIR 生成 — MLIRGen.cpp

### 4.1 入口函数

```cpp
OwningOpRef<ModuleOp> mlirGen(MLIRContext &context, ModuleAST &moduleAST);
```

将整个 Toy AST Module 转换为 MLIR ModuleOp。

### 4.2 核心转换逻辑

```
mlirGen(ModuleAST)                → ModuleOp
  └─ mlirGen(FunctionAST)         → FuncOp
       └─ mlirGen(ExprAST)        → 各种 Toy Operation
            ├─ LiteralExprAST     → toy.constant
            ├─ VariableExprAST    → 符号表查找 Value
            ├─ BinaryExprAST      → toy.add / toy.mul
            ├─ CallExprAST        → toy.generic_call
            ├─ PrintExprAST       → toy.print
            ├─ ReturnExprAST      → toy.return
            ├─ VarDeclExprAST     → 无对应 Op (直接映射到 Value)
            └─ StructLiteralExprAST → toy.struct_constant (Ch7)
```

### 4.3 符号表

`MLIRGen.cpp` 维护一个符号表，将变量名映射到 MLIR Value：

```cpp
llvm::StringMap<Value> symbolTable;

// var a = expr → 将 expr 的结果 Value 以名字 "a" 存入符号表
// 后续引用 a → 从符号表查找 Value
```

---

## 五、Dialect 定义 — Dialect.h + Ops.td

### 5.1 Dialect.h — 头文件结构

```cpp
// Dialect.h

// 1. 前向声明
namespace mlir::toy::detail { struct StructTypeStorage; }

// 2. Include TableGen 生成的 Dialect 声明
#include "toy/Dialect.h.inc"

// 3. Include TableGen 生成的 Operation 声明
#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

// 4. 手写的自定义类型 StructType（不在 .td 中定义，手写实现）
class StructType : public Type::TypeBase<StructType, Type, StructTypeStorage> {
  static StructType get(ArrayRef<Type> elementTypes);
  ArrayRef<Type> getElementTypes();
};
```

### 5.2 Ops.td — Operation 定义

```tablegen
// Dialect 定义
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::mlir::toy";
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

// 类型约束
def F64Tensor : TensorOf<[F64]>;

// 基础 Op 类
class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

// 具体 Operations:
def ConstantOp      // 常量: dense<[[1,2],[3,4]]> → tensor
def StructConstantOp // 结构体常量 (Ch7)
def StructAccessOp   // 结构体字段访问 (Ch7)
def AddOp           // 加法
def MulOp           // 乘法
def TransposeOp     // 转置
def ReshapeOp       // 形状变换
def FuncOp          // 函数定义
def ReturnOp        // 返回
def GenericCallOp   // 函数调用
def PrintOp         // 打印
def CastOp          // 类型转换 (内联时类型不匹配)
```

### 5.3 CMakeLists.txt — 生成规则

```cmake
set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(ToyCh7OpsIncGen)

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ToyCh7ShapeInferenceInterfaceIncGen)
```

---

## 六、Dialect 实现 — Dialect.cpp

这是**最大**的文件（665 行），实现了所有手写逻辑。

### 6.1 文件结构

```cpp
// Dialect.cpp

// ① Include 生成的 Dialect 定义
#include "toy/Dialect.cpp.inc"

// ② ToyInlinerInterface — 内联接口实现
struct ToyInlinerInterface : public DialectInlinerInterface {
  isLegalToInline(...)     → return true;  // 所有 Op 都可以内联
  handleTerminator(...)    → 替换 return operands
  materializeCallConversion(...) → 创建 CastOp
};

// ③ 通用 parse/print 工具函数
parseBinaryOp(...)   // AddOp/MulOp 共用的解析
printBinaryOp(...)   // AddOp/MulOp 共用的打印

// ④ 各 Operation 的手写实现
// ConstantOp
ConstantOp::build(...)   → 从 double 创建 DenseElementsAttr
ConstantOp::parse(...)   → 解析属性和类型
ConstantOp::print(...)   → 打印属性
ConstantOp::verify()     → 验证张量形状匹配
ConstantOp::inferShapes() → 从属性类型推断结果类型

// AddOp / MulOp
AddOp::build(...)        → 结果设为 UnrankedTensorType
AddOp::parse/print(...)  → 委托给 parseBinaryOp/printBinaryOp
AddOp::inferShapes()     → 结果类型 = 左操作数类型

// TransposeOp
TransposeOp::build(...)      → 结果设为 UnrankedTensorType
TransposeOp::inferShapes()   → 反转维度
TransposeOp::verify()        → 检查输入/输出形状互为转置

// ReturnOp
ReturnOp::verify() → 检查返回值数量和类型匹配函数签名

// FuncOp
FuncOp::build(...)  → 构建函数 + 入口 Block
FuncOp::parse/print → 委托给 FunctionOpInterface 工具

// GenericCallOp — 实现 CallInterface
GenericCallOp::getCallableForCallee()  → 获取被调用函数引用
GenericCallOp::getArgOperands()        → 获取参数

// StructAccessOp (Ch7)
StructAccessOp::build(...)  → 从结构体类型推导结果类型
StructAccessOp::verify()    → 验证索引合法

// ⑤ StructType — 手写自定义类型实现
StructTypeStorage     → TypeStorage 子类（唯一化存储）
StructType::get()     → 通过 TypeBase::get() 获取唯一实例
ToyDialect::parseType() → 解析 struct<type, type, ...>
ToyDialect::printType() → 打印 struct<type, type, ...>

// ⑥ Include 生成的 Op 方法定义（必须在文件末尾）
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"

// ⑦ Dialect 初始化
void ToyDialect::initialize() {
  addOperations<...>();                    // 注册所有 Operation
  addInterfaces<ToyInlinerInterface>();    // 注册内联接口
  addTypes<StructType>();                  // 注册自定义类型
}

// ⑧ 常量物化（内联器需要）
ToyDialect::materializeConstant(...) → 创建 ConstantOp 或 StructConstantOp
```

### 6.2 关键设计模式

**每种 Op 的典型实现模式**：

```
.td 中声明                    Dialect.cpp 中手写
─────────                    ──────────────
arguments/results 声明    →   build() 方法填充 OperationState
hasCustomAssemblyFormat=1 →   parse() + print()
hasVerifier=1             →   verify()
DeclareOpInterfaceMethods  →   inferShapes()
hasFolder=1                →   fold()（在 ToyCombine.cpp 中）
```

---

## 七、优化 — ToyCombine.td + ToyCombine.cpp

### 7.1 DRR 声明（ToyCombine.td）

```tablegen
// 模式 1: Reshape(Reshape(x)) → Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;

// 模式 2: Reshape(Constant(x)) → Constant(reshaped_value)
def ReshapeConstant : NativeCodeCall<"$0.reshape(...)$1.getType())">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;

// 模式 3: Reshape(x) → x（当类型相同时）
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```

### 7.2 手写 Pattern（ToyCombine.cpp）

```cpp
// Include DRR 生成的 Pattern
#include "ToyCombine.inc"

// 手写 fold
ConstantOp::fold(...) → return getValue();
StructConstantOp::fold(...) → return getValue();
StructAccessOp::fold(...) → 返回结构体中对应索引的元素;

// 手写 C++ Pattern: transpose(transpose(x)) → x
struct SimplifyRedundantTranspose : OpRewritePattern<TransposeOp> {
  matchAndRewrite(op, rewriter) {
    auto innerTranspose = op.getOperand().getDefiningOp<TransposeOp>();
    if (!innerTranspose) return failure();
    rewriter.replaceOp(op, {innerTranspose.getOperand()});
    return success();
  }
};

// 注册到 Canonicalization 框架
void TransposeOp::getCanonicalizationPatterns(...) {
  results.add<SimplifyRedundantTranspose>(context);
}
void ReshapeOp::getCanonicalizationPatterns(...) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

---

## 八、形状推断 — ShapeInferencePass.cpp + ShapeInferenceInterface.td

### 8.1 接口定义（ShapeInferenceInterface.td）

```tablegen
def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let methods = [
    InterfaceMethod<"Infer and set the output shape.",
                    "void", "inferShapes">
  ];
}
```

在 `Ops.td` 中，需要形状推断的 Op 声明实现此接口：

```tablegen
def AddOp : Toy_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> { ... }
```

### 8.2 各 Op 的 inferShapes() 实现

```cpp
// ConstantOp: 结果类型 = 值的属性类型
void ConstantOp::inferShapes() {
  getResult().setType(cast<TensorType>(getValue().getType()));
}

// AddOp/MulOp: 结果类型 = 左操作数类型
void AddOp::inferShapes() { getResult().setType(getLhs().getType()); }

// TransposeOp: 结果类型 = 反转维度
void TransposeOp::inferShapes() {
  auto arrayTy = cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(reverse(arrayTy.getShape()));
  getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
}

// CastOp: 结果类型 = 输入类型
void CastOp::inferShapes() { getResult().setType(getInput().getType()); }
```

### 8.3 Pass 实现（ShapeInferencePass.cpp）

工作列表算法：

```
1. 收集所有返回动态形状的 Op 到 worklist
2. 迭代：
   a. 找到一个"ready"的 Op（所有 operand 都有确定形状）
   b. 调用 shapeOp.inferShapes()
   c. 从 worklist 移除
3. 如果 worklist 为空 → 成功；否则 → 失败
```

```cpp
void runOnOperation() override {
  SmallPtrSet<Operation*, 16> opWorklist;
  f.walk([&](Operation *op) {
    if (returnsDynamicShape(op)) opWorklist.insert(op);
  });

  while (!opWorklist.empty()) {
    auto next = find_if(opWorklist, allOperandsInferred);
    if (next == opWorklist.end()) break;
    Operation *op = *next;
    opWorklist.erase(op);
    cast<ShapeInference>(op).inferShapes();
  }

  if (!opWorklist.empty()) signalPassFailure();
}
```

---

## 九、降级 Pass 1 — LowerToAffineLoops.cpp

### 9.1 目标

将 Toy Dialect 的计算密集型 Op 降级到 **Affine + Arith + MemRef** dialect，利用仿射循环实现张量计算。

```
toy.constant dense<[[1,2],[3,4]]>
  → memref.alloc + affine.store (逐元素存储)

toy.add %a, %b : tensor<2x3xf64>
  → memref.alloc (分配结果)
  → affine.for i = 0 to 2 {
      affine.for j = 0 to 3 {
        %lhs = affine.load %a[i, j]
        %rhs = affine.load %b[i, j]
        %sum = arith.addf %lhs, %rhs
        affine.store %sum, %result[i, j]
      }
    }

toy.transpose %x
  → memref.alloc (分配转置结果)
  → affine.for i = 0 to M {
      affine.for j = 0 to N {
        %val = affine.load %x[j, i]    // 反转索引
        affine.store %val, %result[i, j]
      }
    }
```

### 9.2 Conversion Pattern 结构

```cpp
// 通用模板：二元运算降级
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : OpConversionPattern<BinaryOp> {
  matchAndRewrite(op, adaptor, rewriter) {
    lowerOpToLoops(op, rewriter, [&](builder, loopIvs) {
      auto lhs = AffineLoadOp::create(adaptor.getLhs(), loopIvs);
      auto rhs = AffineLoadOp::create(adaptor.getRhs(), loopIvs);
      return LoweredBinaryOp::create(lhs, rhs);  // arith.addf / arith.mulf
    });
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;

// 具体模式：
ConstantOpLowering   → alloc + 逐元素 affine.store
AddOpLowering        → alloc + affine.for + load + arith.addf + store
MulOpLowering        → alloc + affine.for + load + arith.mulf + store
TransposeOpLowering  → alloc + affine.for + load(reverse indices) + store
FuncOpLowering       → toy.func → func.func (只保留 main)
ReturnOpLowering     → toy.return → func.return
PrintOpLowering      → 保留 toy.print 但更新 operand 类型
```

### 9.3 DialectConversion 框架用法

```cpp
void runOnOperation() {
  // 1. 定义合法目标
  ConversionTarget target(getContext());
  target.addLegalDialect<AffineDialect, ArithDialect, FuncDialect, MemRefDialect>();
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<PrintOp>(...);  // print 特殊处理

  // 2. 注册转换 Pattern
  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, ...>(&getContext());

  // 3. 执行部分转换
  applyPartialConversion(getOperation(), target, std::move(patterns));
}
```

---

## 十、降级 Pass 2 — LowerToLLVM.cpp

### 10.1 目标

将剩余的所有 Op（包括 Affine、Arith、SCF）完全降级到 **LLVM Dialect**。

```
                    Affine Ops ──→ Standard Ops ──┐
                                                  │
                    Arith Ops ────────────────────┼──→ LLVM Dialect
                                                  │
                    Func Ops ─────────────────────┘
                                                  │
                    toy.print ──→ SCF loop + printf ──┘
```

### 10.2 核心工作

```cpp
void runOnOperation() {
  LLVMConversionTarget target(getContext());
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  // 利用已有的转换 Pattern（传递降级：A→B→C）
  populateAffineToStandardConversionPatterns(patterns);      // Affine → SCF
  populateSCFToControlFlowConversionPatterns(patterns);       // SCF → CF
  populateArithToLLVMConversionPatterns(typeConverter, patterns); // Arith → LLVM
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);// MemRef → LLVM
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);  // Func → LLVM

  // 唯一手写的 Pattern: toy.print → printf 调用
  patterns.add<PrintOpLowering>(&getContext());

  // 完全转换（所有 Op 必须被转换）
  applyFullConversion(module, target, std::move(patterns));
}
```

### 10.3 PrintOp 降级

`toy.print` 被降级为嵌套 SCF 循环 + `printf` 调用：

```
toy.print %tensor : memref<2x3xf64>
  ↓
scf.for %i = 0 to 2 {
  scf.for %j = 0 to 3 {
    %val = memref.load %tensor[%i, %j]
    llvm.call printf("%f", %val)
  }
  llvm.call printf("\n")
}
```

---

## 十一、构建系统 — CMakeLists.txt

### 11.1 TableGen 生成

```cmake
# DRR 模式生成
set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCh7CombineIncGen)

# Ops.td 生成（在 include/toy/CMakeLists.txt 中）
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

# ShapeInferenceInterface 生成
mlir_tablegen(ShapeInferenceOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ShapeInferenceOpInterfaces.cpp.inc -gen-op-interface-defs)
```

### 11.2 可执行文件构建

```cmake
add_toy_chapter(toyc-ch7
  toyc.cpp                # 入口
  parser/AST.cpp          # AST dump
  mlir/MLIRGen.cpp        # AST → MLIR
  mlir/Dialect.cpp        # Dialect 实现
  mlir/LowerToAffineLoops.cpp  # 降级 Pass 1
  mlir/LowerToLLVM.cpp         # 降级 Pass 2
  mlir/ShapeInferencePass.cpp  # 形状推断 Pass
  mlir/ToyCombine.cpp          # 优化 Pattern

  DEPENDS
  ToyCh7ShapeInferenceInterfaceIncGen
  ToyCh7OpsIncGen
  ToyCh7CombineIncGen
)
```

---

## 十二、数据流全景图

```
                          输入文件 (.toy 或 .mlir)
                                  │
                    ┌─────────────┴──────────────┐
                    │                            │
               .toy 文件                     .mlir 文件
                    │                            │
           ┌────────▼────────┐          ┌────────▼────────┐
           │  Lexer → Parser │          │  parseSourceFile│
           │       → AST     │          │    → ModuleOp    │
           └────────┬────────┘          └────────┬────────┘
                    │                            │
           ┌────────▼────────┐                   │
           │   MLIRGen.cpp   │                   │
           │  AST → ModuleOp │                   │
           └────────┬────────┘                   │
                    │                            │
                    └──────────┬─────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Inliner Pass       │  toy.generic_call → 内联函数体
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Canonicalize Pass  │  触发 DRR + 手写 Pattern
                    │  + ShapeInference   │  推断张量形状
                    │  + CSE              │  消除冗余
                    └──────────┬──────────┘
                               │
              ┌────────────────▼────────────────┐
              │  LowerToAffine Pass             │
              │  toy.add  → affine.for + arith.addf
              │  toy.mul  → affine.for + arith.mulf
              │  toy.transpose → affine.for (reverse)
              │  toy.constant → memref.alloc + affine.store
              │  tensor → memref                │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  LowerToLLVM Pass               │
              │  affine → SCF → CF → LLVM       │
              │  arith → LLVM                   │
              │  memref → LLVM struct/pointer   │
              │  toy.print → printf loops       │
              │  func → LLVM func               │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │  LLVM IR                        │
              │  → 输出 / JIT 执行 / 优化       │
              └─────────────────────────────────┘
```

---

## 十三、每个文件的职责总结

| 文件 | 行数 | 职责 | 关键机制 |
|------|------|------|---------|
| `toyc.cpp` | 333 | 编译器入口、Pass 管道编排 | PassManager、命令行解析 |
| `Lexer.h` | 236 | 词法分析 | Token 枚举、字符流扫描 |
| `Parser.h` | 683 | 语法分析 | 递归下降解析 |
| `AST.h` | 313 | AST 节点定义 | 类继承层次 |
| `AST.cpp` | 274 | AST 打印 | Visitor 模式 |
| `MLIRGen.h/cpp` | 35/691 | AST → MLIR 转换 | OpBuilder、符号表 |
| `Dialect.h` | 82 | Dialect 头文件 | include .inc、StructType |
| `Ops.td` | 459 | Operation 声明式定义 | TableGen ODS |
| `Dialect.cpp` | 665 | Dialect + Op 全部实现 | parse/print/verify/inferShapes |
| `ShapeInferenceInterface.td` | 30 | 形状推断接口定义 | OpInterface |
| `ShapeInferencePass.cpp` | 123 | 形状推断 Pass | 工作列表算法 |
| `ToyCombine.td` | 63 | DRR 优化模式 | Declarative Rewrite Rules |
| `ToyCombine.cpp` | 89 | 手写优化 + fold + 注册 Pattern | OpRewritePattern、fold |
| `LowerToAffineLoops.cpp` | 368 | Toy → Affine 降级 | DialectConversion、OpConversionPattern |
| `LowerToLLVM.cpp` | 239 | 完全降级到 LLVM | 传递降级、applyFullConversion |
| `Passes.h` | 35 | Pass 入口声明 | 工厂函数 |
| `CMakeLists.txt` (×2) | 13+59 | 构建配置 | mlir_tablegen、add_toy_chapter |

---

## 十四、关键学习要点

1. **完整的编译器管道**：Ch7 展示了从源码到可执行代码的完整链路
2. **Dialect 分层降级**：Toy → Affine + Arith + MemRef → LLVM，每层只处理自己关心的转换
3. **声明式 + 命令式混合**：`.td` 定义 Op 结构，手写 C++ 实现自定义行为
4. **DialectConversion 框架**：通过 `ConversionTarget` + `OpConversionPattern` + `TypeConverter` 实现系统化降级
5. **传递降级**：LowerToLLVM 利用已有的转换 Pattern（Affine→SCF→CF→LLVM），自己只需处理 `toy.print`
6. **Interface 扩展**：ShapeInferenceOpInterface 让形状推断与具体 Op 解耦
7. **DRR vs 手写 Pattern**：简单模式用 DRR（ToyCombine.td），复杂逻辑手写 C++（SimplifyRedundantTranspose）
