# MLIR ODS 与 Toy 方言实践总结

## 概述

ODS（Operation Definition Specification）是 MLIR 基于 TableGen 的声明式 op 定义机制。它的核心价值是：用 `.td` 文件集中描述 dialect 和 operation 的结构信息，再由 `mlir-tblgen` 自动生成 C++ 样板代码。

在 Toy 教程中，ODS 主要用于定义 Toy dialect 及其 op，例如：

```text
toy.constant
toy.add
toy.mul
toy.transpose
toy.reshape
toy.return
toy.func
toy.print
```

整体流程是：

```text
Ops.td
  -> mlir-tblgen
  -> Ops.h.inc / Ops.cpp.inc / Dialect.h.inc / Dialect.cpp.inc
  -> Dialect.h / Dialect.cpp include 生成代码
  -> 编译进 Toy compiler
```

一句话总结：

**ODS 负责声明 op 的结构和通用语义，C++ 负责实现复杂行为。**

---

## 一、为什么 ODS 能节省开发时间

如果不用 ODS，开发一个 op 通常要手写：

```cpp
class AddOp : public Op<AddOp, ...> {
public:
  static StringRef getOperationName() { return "toy.add"; }

  Value getLhs() { return getOperand(0); }
  Value getRhs() { return getOperand(1); }

  static void build(OpBuilder &builder, OperationState &state,
                    Value lhs, Value rhs);

  static ParseResult parse(OpAsmParser &parser, OperationState &state);
  void print(OpAsmPrinter &printer);
  LogicalResult verify();
};
```

Toy 中有多个 op，如果每个都这样手写，重复代码会非常多。

ODS 可以把这些结构压缩成：

```tablegen
def AddOp : Toy_Op<"add"> {
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}
```

TableGen 会自动生成：

```text
AddOp C++ wrapper 类
getLhs()
getRhs()
builder 声明或默认实现
parser/printer 声明或默认实现
verifyInvariants()
op 名称和注册信息
```

因此开发者只需要关注真正有语义的部分。

---

## 二、Toy 中 `.td` 和 C++ 如何配合

以 Toy Ch2 为例，`.td` 文件在：

```text
examples/toy/Ch2/include/toy/Ops.td
```

C++ 实现在：

```text
examples/toy/Ch2/include/toy/Dialect.h
examples/toy/Ch2/mlir/Dialect.cpp
```

头文件中引入生成的声明：

```cpp
#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"
```

源文件中注册所有 op：

```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "toy/Ops.cpp.inc"
      >();
}
```

源文件末尾引入生成的 op 方法定义：

```cpp
#define GET_OP_CLASSES
#include "toy/Ops.cpp.inc"
```

可以理解成：

```text
.td:
  声明 Toy dialect 有哪些 op，每个 op 有哪些输入、输出、属性和 trait

生成的 .inc:
  提供 C++ 类、getter、注册列表、默认 verifier 等样板

手写 C++:
  补充 build、parse/print、verify、fold、canonicalize、lowering 等复杂逻辑
```

---

## 三、Trait 的作用

Trait 是 ODS 中给 op 贴的语义标签。它可以表达结构约束、优化性质和通用行为。

Toy 中常见例子：

```tablegen
def ConstantOp : Toy_Op<"constant", [Pure]> {
  ...
}
```

`Pure` 表示这个 op 没有副作用。未使用的 pure op 可以被 DCE 删除，canonicalization 也可以更放心地处理它。

另一个例子：

```tablegen
def ReturnOp : Toy_Op<"return", [
  Pure,
  HasParent<"FuncOp">,
  Terminator
]> {
  ...
}
```

含义是：

```text
Pure:
  toy.return 没有普通内存副作用

HasParent<"FuncOp">:
  toy.return 必须出现在 toy.func 里面

Terminator:
  toy.return 是 block terminator，必须位于 block 末尾
```

常见 trait 及含义：

| Trait | 含义 |
|------|------|
| `Pure` | 无副作用，适合 DCE、fold、canonicalize |
| `ConstantLike` | 常量风格 op |
| `Terminator` | block 终结操作 |
| `HasParent<"...">` | 限制父 op |
| `IsolatedFromAbove` | region 内不能随意捕获外部 SSA 值 |
| `SameOperandsAndResultType` | operands 和 results 类型一致 |
| `SameOperandsAndResultShape` | operands 和 results shape 一致 |
| `Commutative` | 满足交换律 |
| `Elementwise` | 逐元素操作 |
| `AttrSizedOperandSegments` | 多个 variadic/optional operand 组时记录分段大小 |

Trait 适合表达通用事实，但复杂语义仍然应该放到 C++ verifier 或 interface 方法中。

---

## 四、Builder 的语义理解

Toy 中 `TransposeOp` 是理解 builder 的好例子。

`.td` 声明：

```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  let hasVerifier = 1;
}
```

这里的 builder 声明表示：

```text
请生成一个能够通过 Value input 创建 toy.transpose 的 C++ builder 入口。
```

C++ 中实现：

```cpp
void TransposeOp::build(mlir::OpBuilder &builder,
                        mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}
```

从语义上看，builder 不是在执行 transpose，而是在创建 IR 节点。

它回答的是：

```text
如果用户写 builder.create<toy::TransposeOp>(loc, input)，
这个 toy.transpose operation 应该长什么样？
```

这段 builder 填了两件事：

```cpp
state.addOperands(value);
```

表示：

```text
toy.transpose 的输入 operand 是 value
```

```cpp
state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
```

表示：

```text
toy.transpose 的结果类型暂时是 tensor<*xf64>
```

最终大概生成：

```mlir
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<*xf64>
```

Toy 这里没有直接生成 `tensor<3x2xf64>`，是因为早期阶段故意先使用 unranked tensor，后续再通过 shape inference pass 推导精确形状。

---

## 五、OperationState 是什么

`OperationState` 不是“op 的属性”。它是 **创建 Operation 之前的临时描述对象**。

它包含：

```text
op name
location
operands
result types
attributes
regions
successors
properties
```

可以把它理解成“op 创建施工单”：

```text
我要创建一个 toy.transpose
  名字：toy.transpose
  输入：%0
  输出类型：tensor<*xf64>
  属性：无
  region：无
```

`builder.create<toy::TransposeOp>(loc, input)` 的大致流程是：

```text
创建 OperationState
  -> 调用 TransposeOp::build(builder, state, input)
  -> build 填 operands/result types/attributes 等字段
  -> MLIR 根据 state 真正创建 Operation
```

因此 builder 的职责是：

```text
把高级 C++ 创建接口翻译成 OperationState。
```

---

## 六、Verifier 的配合方式

还是以 `TransposeOp` 为例。

`.td` 中：

```tablegen
let hasVerifier = 1;
```

表示 TableGen 会为 op 生成 verifier 调用框架，但具体语义由 C++ 实现。

C++ 中：

```cpp
llvm::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
  auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
  if (!inputType || !resultType)
    return mlir::success();

  auto inputShape = inputType.getShape();
  if (!std::equal(inputShape.begin(), inputShape.end(),
                  resultType.getShape().rbegin())) {
    return emitError()
           << "expected result shape to be a transpose of the input";
  }
  return mlir::success();
}
```

`.td` 自动检查基础结构：

```text
有一个 input
input 是 F64Tensor
有一个 result
result 是 F64Tensor
assembly format 合法
```

C++ verifier 检查复杂语义：

```text
如果 input/result 都是 ranked tensor，
那么 result shape 必须等于 input shape 的反转。
```

例如合法：

```mlir
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
```

非法：

```mlir
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
```

---

## 七、声明式 Rewrite Pattern

Toy Ch3 还展示了 ODS/DRR 如何节省 rewrite pattern 的开发时间。

文件：

```text
examples/toy/Ch3/mlir/ToyCombine.td
```

声明：

```tablegen
def ReshapeReshapeOptPattern :
  Pat<(ReshapeOp(ReshapeOp $arg)),
      (ReshapeOp $arg)>;
```

含义：

```text
reshape(reshape(x)) -> reshape(x)
```

如果不用 DRR，需要手写 `OpRewritePattern<ReshapeOp>`，自己检查 operand、匹配 defining op、创建新 op、替换旧 op。

使用 DRR 后，C++ 只需要 include 生成文件并注册：

```cpp
namespace {
#include "ToyCombine.inc"
}

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern,
              RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
```

这说明 ODS 不仅能定义 op，还能声明一些简单 rewrite 规则。

---

## 八、职责边界总结

| 内容 | 适合放 `.td` | 适合放 C++ |
|------|-------------|------------|
| op 名字 | 是 | 否 |
| operand/result/attribute | 是 | 否 |
| trait | 是 | 否 |
| interface 声明 | 是 | 部分方法实现 |
| 简单 assembly format | 是 | 否 |
| 复杂 parser/printer | 声明开关 | 手写实现 |
| 简单 builder | 是 | 可选 |
| 复杂 builder | 声明入口 | 手写实现 |
| 基础结构 verify | 自动生成 | 否 |
| 复杂语义 verify | 声明开关 | 手写实现 |
| fold/canonicalize | 声明开关或 DRR | 手写复杂 pattern |
| lowering | 否 | 是 |
| shape inference | interface 声明 | 方法实现 |

实践原则：

```text
能声明的结构事实放 .td
需要算法和语义判断的逻辑放 C++
```

---

## 九、核心心智模型

可以把一个 Toy op 的实现拆成几层：

```text
.td:
  这个 op 长什么样

builder:
  C++ 中如何创建这个 op

verifier:
  这个 op 是否语义合法

canonicalizer/folder:
  这个 op 如何简化

lowering:
  这个 op 如何变成更低层 dialect
```

以 `toy.transpose` 为例：

```text
Ops.td:
  toy.transpose 有一个 input 和一个 result
  输入输出都是 f64 tensor
  文本格式是 "(input : type) to result_type"
  需要自定义 builder 和 verifier

Dialect.cpp builder:
  创建时把 input 放进 operand list
  结果类型先设为 tensor<*xf64>

Dialect.cpp verifier:
  检查 ranked tensor 的输出 shape 是否为输入 shape 的反转
```

最终总结：

**ODS 让 Toy 方言开发者不用反复手写 op wrapper、getter、注册、基础 verifier 和 rewrite 样板；C++ 则保留对复杂语义、构造逻辑、优化和 lowering 的精确控制。**
