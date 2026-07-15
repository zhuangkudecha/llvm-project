# MLIR Func Dialect - Introduction

本文基于：

- `docs/Dialects/Func.md`
- `include/mlir/Dialect/Func/IR/FuncOps.td`
- `lib/Dialect/Func/IR/FuncOps.cpp`
- `include/mlir/Dialect/Func/Transforms/Passes.td 

整理。`Func.md` 本身很短，它主要通过生成文档引入\ `FuncOps`；真正有学习价值的内容在 `FuncOps.td` 和 `FuncOps.cpp` 中。

一句话理解： 

> `func` dialect 负责表达 MLIR 中的函数定义、函数调用、函数返回，以及函数值引用等高阶函数相关抽象。

它不是用来表达具体算术计算的 dialect，而是 MLIR 中组织程序边界和调用关系的基础 dialect。

---

## 一、Func Dialect 的定位

`func` dialect 的 TableGen 定义：

```tablegen
def Func_Dialect : Dialect {
  let name = "func";
  let cppNamespace = "::mlir::func";
  let hasConstantMaterializer = 1;
}
```

含义：

```text
IR 中的 dialect 名字：
  func

C++ 命名空间：
  mlir::func

支持常量物化：
  可以把 FlatSymbolRefAttr + FunctionType 物化成 func.constant
```

`func` dialect 中最核心的 operations：

| Operation | 作用 |
|---|---|
| `func.func` | 定义函数 |
| `func.call` | 直接函数调用 |
| `func.call_indirect` | 间接函数调用 |
| `func.constant` | 把函数符号引用变成 SSA 函数值 |
| `func.return` | 函数返回 |

一个典型例子：

```mlir
module {
  func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    func.return %0 : i32
  }

  func.func @main() -> i32 {
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %r = func.call @add(%c1, %c2) : (i32, i32) -> i32
    func.return %r : i32
  }
}
```

这个例子里：

```text
builtin.module
  顶层容器

func.func @add
  函数定义，也是一个 symbol

%a / %b
  函数 entry block 的 block arguments

func.call @add
  通过 symbol reference 调用函数

func.return
  返回函数结果
```

---

## 二、Func Dialect 为什么值得学

`func` dialect 虽然 operation 不多，但连接了 MLIR 的许多基础概念：

```text
module
symbol
region
block argument
function type
call graph
pass pipeline anchor
dialect conversion
LLVM lowering
```

尤其对学习 pass 和 lowering 很重要。

例如常见 pipeline：

```shell
mlir-opt input.mlir \
  -pass-pipeline='builtin.module(func.func(canonicalize,cse))'
```

这里：

```text
builtin.module
  顶层 pass manager anchor

func.func(...)
  嵌套到每个函数上运行函数级 pass
```

也就是说，`func.func` 经常是函数级 pass 的执行单位。

---

## 三、FuncDialect 初始化

C++ 实现中：

```c++
void FuncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Func/IR/FuncOps.cpp.inc"
      >();
  declarePromisedInterface<ConvertToEmitCPatternInterface, FuncDialect>();
  declarePromisedInterface<DialectInlinerInterface, FuncDialect>();
  declarePromisedInterface<ConvertToLLVMPatternInterface, FuncDialect>();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, CallOp,
                            FuncOp, ReturnOp>();
}
```

这里做了几件事：

```text
1. 注册 func dialect 的所有 operations
2. 声明它承诺支持 EmitC conversion interface
3. 声明它承诺支持 inliner interface
4. 声明它承诺支持 LLVM conversion interface
5. 声明 CallOp / FuncOp / ReturnOp 支持 bufferization interface
```

这说明 `func` dialect 在很多转换链路中都处于核心位置。

---

## 四、func.func：函数定义

ODS 定义：

```tablegen
def FuncOp : Func_Op<"func", [
  AffineScope, AutomaticAllocationScope,
  FunctionOpInterface, IsolatedFromAbove, OpAsmOpInterface
]> {
  ...
}
```

### 4.1 语义

`func.func` 表示一个函数。

它可以是：

```text
函数定义
  有 body region

函数声明 / external function
  没有 body
```

例子：

```mlir
func.func private @abort()
```

这是一个外部函数声明，没有函数体。

```mlir
func.func @count(%x: i64) -> (i64, i64) {
  func.return %x, %x : i64, i64
}
```

这是一个函数定义，有 body。

### 4.2 函数参数本质上是 block arguments

MLIR 文本里写：

```mlir
func.func @foo(%x: i32, %y: i32) -> i32 {
  %0 = arith.addi %x, %y : i32
  func.return %0 : i32
}
```

看起来 `%x`、`%y` 是函数参数。

但在 MLIR 内部，它们是函数 body region 的 entry block arguments。

可以理解成：

```text
func.func @foo
  region
    block ^entry(%x: i32, %y: i32):
      ...
```

这点很重要，因为 MLIR 的 region/block/SSA 模型中，函数参数不是特殊 AST 节点，而是 block argument。

### 4.3 IsolatedFromAbove

`func.func` 有 trait：

```tablegen
IsolatedFromAbove
```

含义：

```text
函数内部 operation 不能隐式捕获函数外部定义的 SSA value
```

如果函数内部需要外部信息，必须通过：

- 函数参数
- attribute
- symbol reference

示意：

```mlir
module {
  %x = "some.op"() : () -> i32

  func.func @bad() {
    // 不能隐式使用外部 %x
  }
}
```

这个限制让函数成为相对独立的变换单元，也便于多线程 pass、symbol table 和函数级优化。

### 4.4 FuncOp 的 arguments

ODS 中：

```tablegen
let arguments = (ins
  SymbolNameAttr:$sym_name,
  TypeAttrOf<FunctionType>:$function_type,
  OptionalAttr<StrAttr>:$sym_visibility,
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs,
  UnitAttr:$no_inline
);
```

各字段含义：

| 字段 | 含义 |
|---|---|
| `sym_name` | 函数符号名，例如 `@foo` |
| `function_type` | 函数类型，例如 `(i32, i32) -> i32` |
| `sym_visibility` | symbol 可见性，例如 `private` |
| `arg_attrs` | 函数参数属性 |
| `res_attrs` | 函数结果属性 |
| `no_inline` | 不希望被 inline 的标记 |

例子：

```mlir
func.func private @example_fn_arg(%x: i32 {swift.self = unit})
```

这里 `%x` 有参数属性。

```mlir
func.func private @example_fn_result()
  -> (f64 {dialectName.attrName = 0 : i64})
```

这里函数结果有 result attribute。

```mlir
func.func private @example_fn_attr()
  attributes {dialectName.attrName = false}
```

这里函数本身有 attribute。

注意：函数、函数参数、函数结果的 attribute dictionary 中，只允许 dialect attribute names。

### 4.5 FuncOp 的 region

ODS 中：

```tablegen
let regions = (region AnyRegion:$body);
```

也就是说 `func.func` 拥有一个 region，名字是 `body`。

如果函数是 external declaration，则没有实际 body。

C++ 接口中：

```c++
Region *getCallableRegion() {
  return isExternal() ? nullptr : &getBody();
}
```

说明：

```text
external function 没有 callable region
普通函数的 callable region 是 body
```

### 4.6 FuncOp 的常见接口和 trait

```tablegen
AffineScope
```

表示函数可以作为 affine scope。

```tablegen
AutomaticAllocationScope
```

表示自动分配对象的作用域，例如某些 stack allocation 的生命周期边界。

```tablegen
FunctionOpInterface
```

说明它实现函数接口，可以被通用函数工具处理。

```tablegen
IsolatedFromAbove
```

说明函数体不能隐式捕获外层 SSA value。

```tablegen
OpAsmOpInterface
```

提供 assembly printing/parsing 相关能力。

---

## 五、func.return：函数返回

ODS 定义：

```tablegen
def ReturnOp : Func_Op<"return", [
  Pure, HasParent<"FuncOp">,
  MemRefsNormalizable, ReturnLike, Terminator
]> {
  ...
}
```

### 5.1 语义

`func.return` 表示从函数返回。

它：

```text
可以带 0 个或多个 operands
不产生 results
必须位于 func.func 内部
必须是 terminator
```

例子：

```mlir
func.func @foo() -> (i32, f8) {
  ...
  func.return %0, %1 : i32, f8
}
```

无返回值函数：

```mlir
func.func @bar() {
  func.return
}
```

### 5.2 ReturnOp 的 verifier

C++ 中：

```c++
LogicalResult ReturnOp::verify() {
  auto function = cast<FuncOp>((*this)->getParentOp());

  const auto &results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size())
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();

  for (unsigned i = 0, e = results.size(); i != e; ++i)
    if (getOperand(i).getType() != results[i])
      return emitError() << "type of return operand " << i << " ("
                         << getOperand(i).getType()
                         << ") doesn't match function result type ("
                         << results[i] << ")"
                         << " in function @" << function.getName();

  return success();
}
```

它检查：

```text
1. return operand 数量必须等于函数结果数量
2. 每个 return operand 类型必须等于对应函数结果类型
```

例如非法：

```mlir
func.func @bad() -> i32 {
  func.return
}
```

因为函数声明返回 `i32`，但 `func.return` 没有返回值。

也非法：

```mlir
func.func @bad(%x: f32) -> i32 {
  func.return %x : f32
}
```

因为返回值类型 `f32` 不等于函数结果类型 `i32`。

---

## 六、func.call：直接函数调用

ODS 定义：

```tablegen
def CallOp : Func_Op<"call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  ...
}
```

### 6.1 语义

`func.call` 表示直接调用同一 symbol scope 中的函数。

例子：

```mlir
%2 = func.call @my_add(%0, %1) : (f32, f32) -> f32
```

其中：

```text
@my_add
  callee，被编码为名为 "callee" 的 symbol reference attribute

%0, %1
  传给 callee 的参数

(f32, f32) -> f32
  call site 的函数类型
```

### 6.2 CallOp 的 arguments/results

ODS 中：

```tablegen
let arguments = (ins
  FlatSymbolRefAttr:$callee,
  Variadic<AnyType>:$operands,
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs, 
  UnitAttr:$no_inline
);

let results = (outs Variadic<AnyType>);
```

含义：

| 字段 | 含义 |
|---|---|
| `callee` | 被调用函数的符号引用 |
| `operands` | 调用参数 |
| `arg_attrs` | call 参数属性 |
| `res_attrs` | call 结果属性 |
| `no_inline` | 不希望 inline 的标记 |
| `results` | call 的返回值 |

### 6.3 CallOp 的 assembly format

```tablegen
let assemblyFormat = [{
  $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
}];
```

对应文本：

```mlir
%r = func.call @add(%a, %b) : (i32, i32) -> i32
```

这里：

```text
$callee
  @add

$operands
  %a, %b

functional-type($operands, results)
  (i32, i32) -> i32
```

### 6.4 CallOp 的 verifier

C++ 中：

```c++
LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid function";

  auto fnType = fn.getFunctionType();
  ...
}
```

它检查：

```text
1. 必须有 callee symbol reference attribute
2. callee 必须能在最近的 symbol table 中找到对应 func.func
3. call operand 数量必须等于 callee 函数输入数量
4. call operand 类型必须等于 callee 函数输入类型
5. call result 数量必须等于 callee 函数结果数量
6. call result 类型必须等于 callee 函数结果类型
```

例如：

```mlir
func.func private @add(i32, i32) -> i32

func.func @caller(%a: i32, %b: i32) -> i32 {
  %r = func.call @add(%a, %b) : (i32, i32) -> i32
  func.return %r : i32
}
```

合法。

但下面非法：

```mlir
func.func private @add(i32, i32) -> i32

func.func @caller(%a: i32) -> i32 {
  %r = func.call @add(%a) : (i32) -> i32
  func.return %r : i32
}
```

因为 `@add` 需要两个参数，但 call site 只传了一个。

### 6.5 CallOp 和 Symbol

`func.call` 的 callee 不是 SSA value，而是：

```text
FlatSymbolRefAttr
```

也就是说它通过 symbol 名字引用函数。

这与 MLIR 多线程设计有关：函数不是直接作为 SSA operand 被引用，而是通过 symbol table 解析。

---

## 七、func.constant：函数符号变成 SSA 函数值

ODS 定义：

```tablegen
def ConstantOp : Func_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  ...
}
```

### 7.1 语义

`func.constant` 从一个 `func.func` 的 symbol reference 产生一个 SSA value。

例子：

```mlir
%f = func.constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>
```

它产生一个函数类型的 SSA value：

```text
(tensor<16xf32>, f32) -> tensor<16xf32>
```

generic form：

```mlir
%f = "func.constant"() { value = @myfn }
  : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
```

### 7.2 为什么需要 func.constant

MLIR 不允许 SSA operand 直接引用函数。

原因之一是编译器多线程设计：避免函数对象被直接当作 SSA value 捕获，能简化并行编译和 symbol 管理。

所以如果想得到“函数值”，需要：

```text
函数 symbol
  -> func.constant
  -> SSA value of function type
```

然后这个函数值可以传给：

```text
func.call_indirect
```

### 7.3 ConstantOp 的 verifier

C++ 中：

```c++
LogicalResult ConstantOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  StringRef fnName = getValue();
  Type type = getType();

  auto fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(
      this->getOperation(), StringAttr::get(getContext(), fnName));
  if (!fn)
    return emitOpError() << "reference to undefined function '" << fnName
                         << "'";

  if (fn.getFunctionType() != type)
    return emitOpError("reference to function with mismatched type");

  return success();
}
```

它检查：

```text
1. value 引用的函数必须存在
2. func.constant 的结果类型必须等于被引用函数的 FunctionType
```

### 7.4 ConstantOp 的 fold

```c++
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValueAttr();
}
```

也就是说 `func.constant` 可以 fold 成它的 symbol attribute。

这也是 `func.call_indirect` canonicalize 的基础。

---

## 八、func.call_indirect：间接函数调用

ODS 定义：

```tablegen
def CallIndirectOp : Func_Op<"call_indirect", [
  CallOpInterface,
  TypesMatchWith<"callee input types match argument types",
                 "callee", "callee_operands",
                 "::llvm::cast<FunctionType>($_self).getInputs()">,
  TypesMatchWith<"callee result types match result types",
                 "callee", "results",
                 "::llvm::cast<FunctionType>($_self).getResults()">
]> {
  ...
}
```

### 8.1 语义

`func.call_indirect` 表示调用一个函数类型的 SSA value。

这个 SSA value 可以来自：

```text
func.constant
函数参数
其他返回函数值的 operation
```

例子：

```mlir
%func = func.constant @my_func
  : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>

%result = func.call_indirect %func(%0, %1)
  : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```

这里：

```text
%func
  是函数类型的 SSA value

%0, %1
  是传给这个函数值的参数
```

### 8.2 CallIndirectOp 的 arguments/results

ODS 中：

```tablegen
let arguments = (ins
  FunctionType:$callee,
  Variadic<AnyType>:$callee_operands,
  OptionalAttr<DictArrayAttr>:$arg_attrs,
  OptionalAttr<DictArrayAttr>:$res_attrs
);

let results = (outs Variadic<AnyType>:$results);
```

含义：

| 字段 | 含义 |
|---|---|
| `callee` | 函数类型的 SSA value |
| `callee_operands` | 传给 callee 的参数 |
| `arg_attrs` | 参数属性 |
| `res_attrs` | 结果属性 |
| `results` | 调用结果 |

### 8.3 类型约束

`CallIndirectOp` 有两个 `TypesMatchWith`：

```text
callee 的输入类型必须匹配 callee_operands 的类型
callee 的结果类型必须匹配 call_indirect 的 results 类型
```

例如：

```mlir
%f = func.constant @foo : (i32, f32) -> i64
%r = func.call_indirect %f(%a, %b) : (i32, f32) -> i64
```

合法。

如果 `%a` 不是 `i32`，或 `%b` 不是 `f32`，就不合法。

### 8.4 Canonicalization：间接调用变直接调用

C++ 中：

```c++
LogicalResult CallIndirectOp::canonicalize(CallIndirectOp indirectCall,
                                           PatternRewriter &rewriter) {
  SymbolRefAttr calledFn;
  if (!matchPattern(indirectCall.getCallee(), m_Constant(&calledFn)))
    return failure();

  rewriter.replaceOpWithNewOp<CallOp>(indirectCall, calledFn,
                                      indirectCall.getResultTypes(),
                                      indirectCall.getArgOperands());
  return success();
}
```

意思是：

如果：

```mlir
%f = func.constant @foo : (i32) -> i32
%r = func.call_indirect %f(%x) : (i32) -> i32
```

可以 canonicalize 成：

```mlir
%r = func.call @foo(%x) : (i32) -> i32
```

因为 callee 是一个已知的常量函数符号，间接调用没有必要保留。

这就是一个典型的 canonicalization pattern：

```text
更间接、更复杂的形式
  -> 更直接、更规范的形式
```

---

## 九、func.constant 与 func.call_indirect 的关系

这两个 op 经常配合：

```mlir
func.func private @callee(i32) -> i32

func.func @caller(%x: i32) -> i32 {
  %f = func.constant @callee : (i32) -> i32
  %r = func.call_indirect %f(%x) : (i32) -> i32
  func.return %r : i32
}
```

语义：

```text
func.constant
  把 @callee 这个函数符号变成函数值 %f

func.call_indirect
  调用函数值 %f
```

canonicalize 后：

```mlir
func.func private @callee(i32) -> i32

func.func @caller(%x: i32) -> i32 {
  %r = func.call @callee(%x) : (i32) -> i32
  func.return %r : i32
}
```

因为 `%f` 是已知常量函数。

---

## 十、Symbol、FunctionType 和 CallInterface

### 10.1 Symbol

`func.func @foo` 定义一个 symbol：

```text
@foo
```

`func.call @foo(...)` 使用 symbol：

```text
callee = @foo
```

这个 `callee` 在 ODS 里是：

```tablegen
FlatSymbolRefAttr:$callee
```

所以直接调用不是通过 SSA value 指向函数，而是通过 symbol table 查找。

### 10.2 FunctionType

函数类型形如：

```text
(input types) -> (result types)
```

例子：

```mlir
(i32, f32) -> i64
```

在 C++ 中是：

```c++
FunctionType
```

`func.func` 的函数签名保存在：

```tablegen
TypeAttrOf<FunctionType>:$function_type
```

`func.call_indirect` 的 callee operand 类型也是：

```tablegen
FunctionType:$callee
```

### 10.3 CallOpInterface

`func.call` 和 `func.call_indirect` 都实现：

```tablegen
CallOpInterface
```

这让通用 call graph、inlining、call rewriting 工具可以用统一接口处理它们。

关键区别：

```text
func.call
  callee 是 SymbolRefAttr

func.call_indirect
  callee 是 SSA Value，类型是 FunctionType
```

---

## 十一、Func Dialect 和 Pass Pipeline

`func.func` 常作为函数级 pass anchor。

例如：

```shell
mlir-opt input.mlir \
  -pass-pipeline='builtin.module(func.func(canonicalize,cse))'
```

含义：

```text
在 builtin.module 下找到每个 func.func
对每个 func.func 运行 canonicalize 和 cse
```

如果 IR 是：

```mlir
module {
  func.func @foo() {
    ...
  }

  func.func @bar() {
    ...
  }
}
```

执行方式可理解为：

```text
对 @foo 跑 canonicalize, cse
对 @bar 跑 canonicalize, cse
```

这种设计带来：

```text
更好的缓存局部性
更自然的函数级优化边界
更容易并行处理多个函数
```

---

## 十二、Func Dialect 的 transform/pass

`include/mlir/Dialect/Func/Transforms/Passes.td` 中定义了：

```tablegen
def DuplicateFunctionEliminationPass : Pass<"duplicate-function-elimination",
    "ModuleOp"> {
  let summary = "Deduplicate functions";
  let description = [{
    Deduplicate functions that are equivalent in all aspects but their symbol
    name. The pass chooses one representative per equivalence class, erases
    the remainder, and updates function calls accordingly.
  }];
}
```

这个 pass 的作用：

```text
去重等价函数
选择每组等价函数中的一个代表
删除其他重复函数
更新函数调用，让它们指向保留下来的代表函数
```

它是 module 级 pass：

```text
Pass<"duplicate-function-elimination", "ModuleOp">
```

因为它需要观察整个 module 中多个函数之间的等价关系，并更新跨函数调用。

---

## 十三、常见 IR 例子

### 13.1 外部函数声明

```mlir
func.func private @abort()
func.func private @print_i32(i32)
```

没有 body，表示外部函数。

### 13.2 多返回值函数

```mlir
func.func @dup(%x: i64) -> (i64, i64) {
  func.return %x, %x : i64, i64
}
```

返回两个结果。

### 13.3 直接调用

```mlir
func.func private @add(i32, i32) -> i32

func.func @caller(%a: i32, %b: i32) -> i32 {
  %r = func.call @add(%a, %b) : (i32, i32) -> i32
  func.return %r : i32
}
```

### 13.4 间接调用

```mlir
func.func private @add(i32, i32) -> i32

func.func @caller(%a: i32, %b: i32) -> i32 {
  %f = func.constant @add : (i32, i32) -> i32
  %r = func.call_indirect %f(%a, %b) : (i32, i32) -> i32
  func.return %r : i32
}
```

### 13.5 参数属性

```mlir
func.func private @example(%x: i32 {mydialect.readonly})
```

### 13.6 函数属性

```mlir
func.func private @example()
  attributes {mydialect.attr = true}
```

---

## 十四、和 LLVM Lowering 的关系

`func` dialect 经常是 lowering 到 LLVM dialect 的入口。

典型 pass：

```shell
convert-func-to-llvm
```

它会处理：

```text
func.func
  -> llvm.func

func.call
  -> llvm.call

func.return
  -> llvm.return
```

函数签名、memref 参数传递约定、返回值 ABI 等也会在这个阶段处理。

所以学 lowering 时，`func` 是必须理解的桥梁 dialect。

---

## 十五、和 DialectConversion 的关系

很多 dialect conversion 会涉及函数签名修改。

例如类型转换：

```text
tensor/memref/custom type
  -> llvm-compatible type
```

这会影响：

```text
func.func 的参数和返回类型
func.call 的参数和结果类型
func.return 的返回值类型
```

因此 conversion pattern 不只要改函数体里的计算 op，还经常要改：

- 函数签名
- call site
- return op
- block arguments

这也是为什么 `FunctionOpInterface`、`CallOpInterface` 很重要：它们让转换框架能以统一方式处理函数类 operation 和调用类 operation。

---

## 十六、学习时最该抓住的点

### 16.1 三个最核心 op

优先掌握：

```text
func.func
  函数定义 / 声明

func.call
  直接调用函数 symbol

func.return
  返回函数结果
```

然后再理解：

```text
func.constant
  函数 symbol -> SSA 函数值

func.call_indirect
  调用 SSA 函数值
```

### 16.2 直接调用和间接调用的区别

```text
func.call
  callee 是 FlatSymbolRefAttr
  例子：func.call @foo(...)

func.call_indirect
  callee 是 SSA Value
  例子：func.call_indirect %f(...)
```

### 16.3 函数参数是 block arguments

```mlir
func.func @foo(%x: i32) {
  ...
}
```

内部理解为：

```text
函数 body region 的 entry block 有一个 block argument %x
```

### 16.4 verifier 关注类型匹配

`func` dialect 的核心合法性检查包括：

```text
func.call
  callee 必须存在
  参数数量/类型必须匹配 callee 函数类型
  结果数量/类型必须匹配 callee 函数类型

func.return
  返回值数量/类型必须匹配所在函数签名

func.constant
  引用的函数必须存在
  结果 FunctionType 必须匹配被引用函数类型
```

### 16.5 func.func 是函数级 pass 的 anchor

例如：

```shell
builtin.module(func.func(canonicalize,cse))
```

表示：

```text
在每个 func.func 上运行 canonicalize 和 cse
```

---

## 十七、推荐学习顺序

建议按这个顺序学 `func` dialect：

```text
1. 看懂 func.func / func.return 的函数定义结构
2. 看懂 func.call 的 symbol 调用
3. 理解函数参数是 block argument
4. 理解 FunctionType
5. 理解 func.constant + func.call_indirect
6. 理解 func.call_indirect canonicalize 成 func.call
7. 理解 func.func 作为 pass pipeline anchor
8. 理解 convert-func-to-llvm 的作用
```

---

## 十八、自测问题

1. `func.func @foo(%x: i32) -> i32` 中 `%x` 在 MLIR 内部是什么？
2. `func.call @foo(%x)` 的 callee 是 SSA value 还是 symbol reference？
3. `func.return` 的 verifier 会检查什么？
4. `func.constant @foo` 的作用是什么？
5. `func.call_indirect` 在什么情况下可以 canonicalize 成 `func.call`？
6. 为什么 `func.func` 有 `IsolatedFromAbove` trait？
7. 为什么 `duplicate-function-elimination` 是 module 级 pass，而不是 func 级 pass？

参考答案：

```text
1. 是函数 body entry block 的 block argument。
2. 是 symbol reference，具体是 FlatSymbolRefAttr。
3. 检查返回值数量和类型是否匹配所在函数的 FunctionType results。
4. 把函数 symbol 引用变成函数类型的 SSA value。
5. 当 indirect call 的 callee 来自已知的 func.constant 时。
6. 为了禁止函数体隐式捕获外部 SSA value，使函数成为独立变换单元。
7. 因为它需要比较整个 module 中的多个函数，并更新跨函数调用。
```

---

## 十九、一句话总结

`func` dialect 是 MLIR 中描述函数边界和调用关系的基础 dialect：

```text
func.func
  定义 symbol 和函数体

func.call
  通过 symbol 直接调用

func.constant + func.call_indirect
  把函数作为 SSA 值使用并间接调用

func.return
  结束函数并返回结果
```

它连接了 symbol table、region/block、FunctionType、call graph、pass pipeline 和 LLVM lowering，是学习 MLIR 中后端转换和函数级优化前必须掌握的一块。
