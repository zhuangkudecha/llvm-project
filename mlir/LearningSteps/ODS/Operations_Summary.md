# MLIR Operations.md (ODS) - 详细学习总结

## 概述

本文基于 `mlir/docs/DefiningDialects/Operations.md` 重新整理。原文是 MLIR 的 **Operation Definition Specification (ODS)** 规范文档，讲的是如何用 TableGen 声明式定义 MLIR operation，并让 `mlir-tblgen` 自动生成对应的 C++ `Op` 类、getter、builder、verifier、parser、printer、文档和 enum 辅助代码。

一句话理解：

> ODS 是 MLIR 的 operation 描述语言：你在 `.td` 文件中声明一个 op 的名字、操作数、属性、property、region、结果、successor、trait、约束、builder、assembly format、fold/canonicalize hook 等事实，MLIR 在编译期把这些事实展开成类型安全的 C++ 接口和验证/解析/打印代码。

它在 dialect 开发中的位置：

```text
Dialect 定义
  -> Operations.td
      -> ODS record 描述每个 op
          -> mlir-tblgen
              -> Op C++ class
              -> Adaptor
              -> Builder
              -> Verifier
              -> Parser / Printer
              -> Canonicalizer / Folder hook 声明
              -> Documentation
```

从 AI 编译器角度看，这篇文档回答的问题是：

> 如果要定义一个高层 tensor op、linalg-like op、shape op、buffer op 或 target dialect op，如何把它的结构、类型约束、文本语法、验证逻辑和 C++ 使用接口一次性声明清楚？

---

## 一、为什么需要 ODS

MLIR 支持可插拔 dialect。每个 dialect 都可以定义自己的 operation。如果所有 op 只靠字符串和通用 `Operation *` 操作，会出现几个问题：

- pass 中到处写字符串比较，例如 `"mydialect.matmul"`
- 访问 operand 时写 `getOperand(3)`，不直观且容易错
- 构造 op 时参数冗长，缺少默认值和类型检查
- textual IR 打印冗长
- verifier 分散在各处，或者根本没有验证
- op 文档、parser/printer、builder、canonicalizer 声明互相脱节

ODS 的目标是建立一个 **single source of truth**：

```text
一个 op 的结构事实集中写在一个 TableGen record 中
  -> 自动生成 C++ 操作接口
  -> 自动生成验证逻辑
  -> 自动生成文档
  -> 驱动 parser/printer/builders/pattern 使用
```

核心收益：

1. **单一事实来源**：读一个 op 的 `.td` record 就能看到它的参数、结果、约束、traits、文档、语法。
2. **减少样板代码**：自动生成 operand/attribute/result getter、builder、verify 方法。
3. **方便自动生成其他工具**：例如 dialect 文档、序列化、pattern 辅助、enum/string 转换。

---

## 二、TableGen 基础语法

ODS 使用 LLVM TableGen。`.td` 文件中最常见的概念：

| 概念 | 类比 | 说明 |
|------|------|------|
| `class` | C++ 类模板/基类 | 可以带模板参数，可以被继承 |
| `def` | C++ 对象/实例 | 实例化某个 class 后形成具体 record，不能再被模板化或继承 |
| `dag` | 有 operator 的参数树 | 语法为 `(operator arg0, arg1, ...)`，ODS 用它描述参数、结果、region 等 |

`dag` 可以给 operator 和参数命名：

```tablegen
(MyOp:$op_name MyArg:$arg_name)
```

ODS 中非常常见：

```tablegen
let arguments = (ins
  I32:$lhs,
  I32:$rhs,
  I64Attr:$axis
);

let results = (outs
  I32:$result
);
```

这里 `ins` / `outs` 是 `OpDefinitionsGen` 识别的特殊 marker。

---

## 三、Operation Definition 的核心构件

ODS 中定义 operation 的基础构件主要来自 `include/mlir/IR/OpBase.td`：

| 构件 | 作用 |
|------|------|
| `Op` | 定义 operation 的主 class |
| `Dialect` | 定义 operation 所属 dialect |
| `Trait` | 表示 op 的语义/结构特性，如无副作用、terminator、类型相等等 |
| `ins` | 引出 operands / attributes / properties |
| `outs` | 引出 results |
| `TypeConstraint` | operand/result 类型约束 |
| `AttrConstraint` | attribute 约束 |
| `Property` | 非 Attribute 存储的内联 property |

一个典型 op 定义：

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoMemoryEffect]> {
  let summary = "Performs average pooling on the input";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins
    TF_FpTensor:$value,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$strides,
    TF_AnyStrAttrOf<["SAME", "VALID"]>:$padding,
    DefaultValuedAttr<TF_ConvertDataFormatAttr, "NHWC">:$data_format
  );

  let results = (outs
    TF_FpTensor:$output
  );
}
```

这段 record 同时表达了：

- op 名字是 `tf.AvgPool`
- 它没有内存副作用
- 有一个 tensor operand
- 有若干带约束的 attributes
- 有一个 tensor result
- 可以生成文档、getter、builder、verifier

---

## 四、Operation Name 和文档

### 4.1 Operation Name

MLIR operation 的完整名字由：

```text
dialect name + "." + op name
```

组成，例如：

```text
tf.Add
arith.addi
linalg.matmul
func.func
```

在 ODS 中，op 名通常作为 `Op` 模板参数给出：

```tablegen
def My_AddOp : MyDialect_Op<"add", [...]> { ... }
```

这个名字用于：

- textual IR parse/print
- pattern matching
- pass 中识别 operation
- dialect 文档

注意：TableGen `def` 名和 MLIR operation 名不是一回事。`def TF_AddOp` 生成的 C++ 类可能叫 `AddOp`，但 IR 里的 operation 名可能是 `tf.Add`。

### 4.2 Operation Documentation

ODS 支持：

```tablegen
let summary = "短说明";

let description = [{
更长的 Markdown 文档。
}];
```

这些会用于生成 dialect 文档。

原文建议：

- 文档放在 operation 定义开头
- `summary` 简短一行
- `summary` 大写开头，不带句号
- 详细解释放在 `description`

---

## 五、Operation Arguments：operands / attributes / properties

ODS 中 `arguments` 可以包含三类东西：

```text
operands
  运行时 SSA value，来自其他 op 或 block argument

attributes
  编译期常量，存储在 MLIRContext 中，是 Attribute

properties
  编译期常量，但内联存储在 operation 上，不放在 MLIRContext 中
```

写法：

```tablegen
let arguments = (ins
  <type-constraint>:$operandName,
  <attr-constraint>:$attrName,
  <property>:$propName
);
```

例如：

```tablegen
let arguments = (ins
  AnyTensor:$input,
  I64Attr:$axis,
  DefaultValuedAttr<BoolAttr, "false">:$keep_dims,
  I32Prop:$mode
);
```

### 5.1 operands

operand 是运行时值：

```mlir
%0 = arith.addi %a, %b : i32
```

这里 `%a` 和 `%b` 是 `arith.addi` 的 operands。

ODS 会为具名 operand 生成 getter：

```cpp
Value getLhs();
Value getRhs();
```

或者在新式命名风格中生成对应访问器。

### 5.2 attributes

attribute 是编译期常量：

```mlir
%0 = "my.reshape"(%arg0) {shape = [2, 3]} : ...
```

ODS 中 attribute 有两类：

```text
Natural attributes
  影响 op 语义，例如 padding、axis、layout

Derived attributes
  可从 op 其他信息派生出来，主要用于接口方便或和外部框架交互
```

所有 derived attributes 必须可以 materialize 成 `Attribute`，也就是理论上能存成 MLIR attribute。

ODS 会生成两类访问：

```text
友好 getter:
  返回 returnType，例如 int64_t、StringRef、ArrayRef

raw attr getter:
  通常是 <name>Attr，返回底层 Attribute
```

raw attr getter 对 transformation pass 很有用，因为它保留了 attribute 存储形态。

### 5.3 properties

properties 类似 attributes，都是编译期常量，但区别是：

```text
attribute:
  存在 MLIRContext 中，作为 Attribute 对象

property:
  内联存在 operation 的 property storage 中
```

property 适合 operation 固有的小型结构化数据，尤其是希望避免 attribute uniquing 或希望更直接 C++ 存储的情况。

---

## 六、Variadic / Optional / Segment 信息

### 6.1 Variadic operands

变长 operand：

```tablegen
let arguments = (ins
  Variadic<AnyType>:$inputs
);
```

表示动态 IR 中可以有 0 个或多个 operands 映射到 `$inputs`。

如果一个 op 只有一个 variadic operand，框架容易判断哪些动态 operands 属于它。

### 6.2 多个可变长度 operand 的消歧

如果一个 op 有多个 variable-length operands：

```tablegen
let arguments = (ins
  Variadic<AnyType>:$lhs,
  Variadic<AnyType>:$rhs
);
```

动态 operand 列表本身无法告诉框架：

```text
前几个属于 lhs？
后几个属于 rhs？
```

此时需要额外 trait：

```text
SameVariadicOperandSize
  所有 variable-length operand 长度相同

AttrSizedOperandSegments
  使用 segment size attribute 记录每段长度
```

这在 `linalg`、`tensor`、`scf` 等复杂 op 中很常见。

### 6.3 VariadicOfVariadic operands

更复杂的情况是：

```text
一个 operand group 是 variadic
group 内部每个 sub-range 也是 variadic
```

ODS 写法：

```tablegen
VariadicOfVariadic<AnyType, "segment_sizes">:$groups
```

第二个参数是一个 `DenseI32ArrayAttr` 名字，用来记录每个 sub-range 的大小。

### 6.4 Optional operands

可选 operand：

```tablegen
Optional<I32>:$maybe_value
```

如果一个 op 中有多个 optional/variadic operands，同样需要 `SameVariadicOperandSize` 或 `AttrSizedOperandSegments` 消歧。

### 6.5 Optional attributes

可选 attribute：

```tablegen
OptionalAttr<I64Attr>:$axis
```

### 6.6 Default-valued attributes

带默认值 attribute：

```tablegen
DefaultValuedAttr<BoolAttr, "false">:$keep_dims
DefaultValuedAttr<F32Attr, "0.5f">:$alpha
DefaultValuedAttr<I64ArrayAttr, "{1, 2, 3}">:$sizes
```

注意：

- 第二个参数是 C++ 默认值字符串
- 打印 IR 时，如果 attribute 等于默认值，生成的 printer 不会打印它
- enum attribute 可用 `DefaultValuedEnumAttr<EnumAttr, EnumCase>`

### 6.7 Confining attributes

`ConfinedAttr` 用于给 attribute 加额外约束：

```tablegen
ConfinedAttr<I32Attr, [IntMinValue<10>]>:$size
```

常见 primitive constraints：

| 约束 | 含义 |
|------|------|
| `IntMinValue<N>` | 整数属性 >= N |
| `IntMaxValue<N>` | 整数属性 <= N |
| `IntNEQValue<N>` | 整数属性 != N |
| `IntPositive` | 正整数 |
| `IntNonNegative` | 非负整数 |
| `IntPowerOf2` | 大于 0 的 2 的幂 |
| `ArrayMinCount<N>` | array 至少 N 个元素 |
| `ArrayMaxCount<N>` | array 至多 N 个元素 |
| `ArrayCount<N>` | array 恰好 N 个元素 |
| `DenseArrayCount<N>` | dense array 恰好 N 个元素 |
| `DenseArrayStrictlyPositive<T>` | dense array 元素全为正 |
| `DenseArrayStrictlyNonNegative<T>` | dense array 元素全非负 |
| `DenseArraySorted<T>` | dense array 非递减 |
| `DenseArrayStrictlySorted<T>` | dense array 严格递增 |
| `IntArrayNthElemEq<I, N>` | 第 I 个元素等于 N |
| `IntArrayNthElemMinValue<I, N>` | 第 I 个元素 >= N |
| `IntArrayNthElemMaxValue<I, N>` | 第 I 个元素 <= N |
| `IntArrayNthElemInRange<I, M, N>` | 第 I 个元素在 [M, N] |
| `IsNullAttr` | optional attribute 必须为空 |

### 6.8 Optional / Default-valued properties

默认值 property：

```tablegen
DefaultValuedProp<I32Prop, "0">:$mode
```

可选 property：

```tablegen
OptionalProp<I32Prop>:$maybe_mode
```

`OptionalProp` 会把底层 property 包成 `std::optional`，默认值是 `std::nullopt`。

如果 property 的 storage type 和 interface type 不同，例如内部存 `SmallVector`，接口暴露 `ArrayRef`，`DefaultValuedProp` 可能需要第三个参数指定 storage-type 等价默认值。

### 6.9 Combining constraints

`AllAttrOf` 表示多个 attribute 约束都必须满足：

```tablegen
AllAttrOf<[Constraint0, Constraint1]>:$attr
```

这在 declarative rewrite pattern 中也可以用来限制匹配。

---

## 七、Regions / Results / Successors

### 7.1 Regions

operation 可以拥有 regions：

```tablegen
let regions = (region
  AnyRegion:$body
);
```

例如：

```text
func.func
  有函数体 region

scf.for
  有循环体 region

scf.if
  有 then/else region
```

变长 region：

```tablegen
VariadicRegion<AnyRegion>:$regions
```

限制：variadic region 当前只能出现在 region 列表最后。

### 7.2 Results

operation results 用 `outs` 声明：

```tablegen
let results = (outs
  AnyTensor:$output
);
```

变长 result：

```tablegen
Variadic<AnyType>:$results
```

多个 variadic results 时，需要类似 `SameVariadicResultSize` 的 trait 消歧。

### 7.3 Successors

terminator op 可以声明 successors：

```tablegen
let successors = (successor
  AnySuccessor:$dest
);
```

变长 successor：

```tablegen
VariadicSuccessor<AnySuccessor>:$dests
```

限制：variadic successor 当前只能放在 successor 列表最后。

---

## 八、Traits / Interfaces / Multi-entity Constraints

Traits 是 operation 的结构或语义性质，写在 `Op` 模板参数中：

```tablegen
def My_AddOp : MyDialect_Op<"add", [
  Pure,
  SameOperandsAndResultType,
  Commutative
]> { ... }
```

Trait 可以表达：

- 是否有副作用
- 是否是 terminator
- 是否 commutative
- operands/results 类型是否相同
- 是否实现某个 interface
- 多个 operands/results/attributes 之间的关系

常见区别：

```text
single-entity constraint:
  写在 operand/attribute/result 声明处
  例如 I32:$x、I64Attr:$axis

multi-entity constraint:
  写在 op trait 列表中
  例如 AllTypesMatch、SameOperandsAndResultType

native trait:
  映射到 C++ mlir::OpTrait
```

---

## 九、Builder 方法

ODS 会根据 arguments 和 results 自动生成多个 `build` 方法。

假设：

```tablegen
def MyOp : ... {
  let arguments = (ins
    I32:$i32_operand,
    F32:$f32_operand,
    I32Attr:$i32_attr,
    F32Attr:$f32_attr,
    I32Prop:$i32_prop
  );

  let results = (outs
    I32:$i32_result,
    F32:$f32_result
  );
}
```

可能生成的 builder 包括：

### 9.1 聚合形式 builder

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  ValueRange operands,
                  Properties properties,
                  ArrayRef<NamedAttribute> discardableAttributes = {});
```

用途：统一处理不同 op，适合 declarative pattern rewrite 或通用构造路径。

### 9.2 legacy attribute dictionary builder

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  ValueRange operands,
                  ArrayRef<NamedAttribute> attributes);
```

这里 inherent properties 和 discardable attributes 混在 attribute dictionary 中，是兼容旧形式。

### 9.3 分离参数 builder

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result,
                  Value i32_operand, Value f32_operand,
                  IntegerAttr i32_attr, FloatAttr f32_attr,
                  int32_t i32_prop);
```

用途：手写 C++ 代码中更类型安全、更清楚。

### 9.4 raw-value attribute builder

如果 attribute 的 `returnType` 和 `storageType` 不同，并且知道如何从 raw value 构造 attribute，则可能生成：

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  Type i32_result, Type f32_result,
                  Value i32_operand, Value f32_operand,
                  APInt i32_attr, StringRef f32_attr,
                  int32_t i32_prop);
```

也就是说调用方可以传 raw C++ 值，而不是手动构造 `IntegerAttr` / `StringAttr`。

### 9.5 result type aggregate builder

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  TypeRange resultTypes,
                  Value i32_operand, Value f32_operand,
                  IntegerAttr i32_attr, FloatAttr f32_attr,
                  int32_t i32_prop);
```

### 9.6 可推断 result type 的 builder

如果 result type 可推断，可以不传 result type：

```cpp
static void build(OpBuilder &odsBuilder, OperationState &odsState,
                  ValueRange operands,
                  Properties properties,
                  ArrayRef<NamedAttribute> discardableAttributes);
```

何时可以省略 result type？

- op 实现 `InferTypeOpInterface`
- 所有 result types 是 buildable types
- result type 与某个 operand type 相同，例如通过 `AllTypesMatch` 表达

### 9.7 默认值 attribute 的 builder 参数位置

如果 attribute 有默认值，ODS 可以给 builder 参数加 C++ 默认值。但 C++ 默认参数要求默认值参数位于参数列表末尾附近。

因此实践建议：

```text
把 default-valued attributes 尽量放在 arguments 列表后面
```

这样生成的 builder 签名更好用。

---

## 十、自定义 Builder

自动 builder 不够用时，可以写：

```tablegen
def MyOp : Op<"my_op", []> {
  let arguments = (ins F32Attr:$attr);

  let builders = [
    OpBuilder<(ins "float":$val)>
  ];
}
```

生成 C++ 声明：

```cpp
static void build(::mlir::OpBuilder &builder,
                  ::mlir::OperationState &state,
                  float val);
```

### 10.1 在 ODS 中内联 builder 实现

```tablegen
let builders = [
  OpBuilder<(ins "float":$val), [{
    $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
  }]>
];
```

特殊变量：

```text
$_builder
  对应 OpBuilder

$_state
  对应 OperationState
```

普通参数可以直接引用，例如 `val`。

实践建议：

```text
短 builder 可以内联写在 ODS
长 builder 应放到 C++ 文件中实现
```

### 10.2 CArg 默认参数

```tablegen
OpBuilder<(ins CArg<"float", "0.5f">:$val), [{
  $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
}]>
```

生成：

```cpp
// header
static void build(OpBuilder &builder, OperationState &state,
                  float val = 0.5f);

// source
MyOp::build(OpBuilder &builder, OperationState &state, float val) {
  state.addAttribute("attr", builder.getF32FloatAttr(val));
}
```

注意：C++ 默认值只出现在声明，不出现在定义。

---

## 十一、自定义 Parser / Printer

如果不使用 declarative assembly format，也可以提供自定义 parser/printer：

```tablegen
let hasCustomAssemblyFormat = 1;
```

然后在 C++ 中实现对应 parse/print 方法。

不过原文重点讲的是更推荐的 **Declarative Assembly Format**，因为它能从 ODS 字符串自动生成 parser/printer。

---

## 十二、自定义 Verifier

ODS 会根据 operands、attributes、results、traits 等约束自动生成基础 verifier。

如果还需要额外验证：

```tablegen
let hasVerifier = 1;
let hasRegionVerifier = 1;
```

会生成：

```cpp
LogicalResult verify();
LogicalResult verifyRegions();
```

选择规则：

```text
hasVerifier
  验证当前 op 自身，不依赖 region 内部 operation 已经合法

hasRegionVerifier
  需要访问 nested operations 时使用
  因为它会在 region 内部 op 验证后执行
```

### 12.1 Verification ordering

验证顺序第一阶段：

```text
1. StructuralOpTrait verifier
2. ODS 生成的 verifyInvariants
3. 不需要 region 的 traits/interfaces verifier
4. op 自定义 hasVerifier
```

如果 op 有 regions，还有第二阶段：

```text
1. 需要访问 region 的 traits/interfaces verifier
2. op 自定义 hasRegionVerifier
```

重要点：

```text
越靠后的 verifier 可以依赖前面 verifier 已经验证过的不变量
```

### 12.2 verifier 诊断输出

自定义 verifier 发诊断时，建议使用 `Error` severity。

原因：低级别诊断如 `Note` / `Remark` / `Warning` 可能使用 custom printer 打印 op，而 custom printer 有时要求 op 或 parent op 已经验证通过。Verifier 正在验证“可能非法”的 IR，因此最好让错误诊断使用 generic form。

---

## 十三、Declarative Assembly Format

ODS 可以用 declarative string 描述 op 的自定义文本格式：

```tablegen
def CallOp : Std_Op<"call", ...> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` functional-type($args, results)
  }];
}
```

这个 format 由三类元素组成：

```text
directives
  内置指令，例如 attr-dict、type($x)、functional-type(...)

literals
  关键字或标点，例如 `(`、`)`、`,`、`:`、`->`

variables
  operation 上注册的实体，例如 $callee、$args、$body
```

### 13.1 常用 directives

| Directive | 作用 |
|-----------|------|
| `attr-dict` | attribute dictionary，必须出现 |
| `attr-dict-with-keyword` | 带 `attributes` 关键字的 attribute dictionary |
| `prop-dict` | properties 转成 dictionary |
| `custom<UserDirective>(...)` | 调用用户 C++ parse/print 子逻辑 |
| `functional-type(inputs, outputs)` | 打印函数类型 |
| `oilist(...)` | 可选、顺序无关的 clause 列表 |
| `operands` | 所有 operands |
| `regions` | 所有 regions |
| `results` | 所有 results |
| `successors` | 所有 successors |
| `type(input)` | 某个 operand/result 的类型 |
| `qualified(type_or_attribute)` | 强制打印带 dialect/mnemonic 的完整形式 |
| `ref(input)` | 引用已经解析/绑定的变量传给 custom directive |

### 13.2 attr-dict 和 prop-dict

`attr-dict` 表示 operation 的 attributes：

```tablegen
let assemblyFormat = "$input attr-dict `:` type($input)";
```

规则：

- format 中没有单独打印的 inherent attributes 会出现在 `attr-dict`
- discardable attributes 总是在 `attr-dict`
- 如果存在 `prop-dict`，`attr-dict` 不包含 inherent attributes

`prop-dict` 表示 properties 转换成 dictionary：

```tablegen
let assemblyFormat = "$input prop-dict attr-dict `:` type($input)";
```

如果某些 non-attribute properties 没有在 format 中出现，`prop-dict` 必须出现。

### 13.3 oilist

`oilist` 是 optional order-independent list：

```tablegen
oilist(`keyword` elements | `otherKeyword` elements)
```

特点：

- 每个 clause 有关键字
- 每个 clause 出现 0 或 1 次
- clause 顺序任意
- clause 内只能使用 literals、types、variables
- 所有变量必须 optional 或 variadic

适合这种语法：

```text
my.op foo(...) bar(...) baz(...)
```

其中 `foo/bar/baz` 可以任意顺序出现。

### 13.4 literals

literal 用反引号包围：

```tablegen
`(` $args `)` `:` type($args)
```

合法标点包括：

```text
: , = < > ( ) { } [ ] -> ? + *
```

空白 literal：

```text
`\n`
` `
```

`\n` 会换行并缩进到 operation 开始位置。

空 literal `` 可以删除某些隐式空格。

### 13.5 variables

变量是 operation 上声明的实体：

```text
$operand
$attr
$region
$result
$successor
```

attribute variable 打印时通常带它的 value type，除非该 type 是 buildable，可以省略。

---

## 十四、Custom Directives

当 declarative assembly format 表达不了某段特殊语法时，可以用 custom directive：

```tablegen
custom<MyDirective>($operand, type($operand), attr-dict)
```

它会生成对 C++ 函数的调用：

```text
parseMyDirective(...)
printMyDirective(...)
```

### 14.1 parse 方法参数映射

`parse<UserDirective>` 的第一个参数是：

```cpp
OpAsmParser &
```

后面是根据 format 参数生成的输出参数。

常见映射：

| declarative 参数 | parse 参数 |
|------------------|------------|
| single attribute | `Attribute &` 或具体 storage type 引用 |
| optional attribute | `Attribute &`，不存在时保持空 |
| single operand | `OpAsmParser::UnresolvedOperand &` |
| optional operand | `std::optional<OpAsmParser::UnresolvedOperand> &` |
| variadic operand | `SmallVectorImpl<OpAsmParser::UnresolvedOperand> &` |
| variadic-of-variadic operand | `SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &` |
| single region | `Region &` |
| variadic region | `SmallVectorImpl<std::unique_ptr<Region>> &` |
| single successor | `Block *&` |
| variadic successor | `SmallVectorImpl<Block *> &` |
| single type | `Type &` |
| variadic type | `SmallVectorImpl<Type> &` |
| `attr-dict` | `NamedAttrList &` |
| `prop-dict` | `OperationState &` |

### 14.2 print 方法参数映射

`print<UserDirective>` 的前两个参数是：

```cpp
OpAsmPrinter &
FooOp op
```

后面是 format 中的参数。

常见映射：

| declarative 参数 | print 参数 |
|------------------|-------------|
| single attribute | `Attribute` 或具体 storage type |
| single operand | `Value` |
| variadic operand | `OperandRange` |
| variadic-of-variadic operand | `OperandRangeRange` |
| single region | `Region &` |
| variadic region | `MutableArrayRef<Region>` |
| single successor | `Block *` |
| variadic successor | `SuccessorRange` |
| single type | `Type` |
| variadic type | `TypeRange` |
| `attr-dict` | `DictionaryAttr` |
| `prop-dict` | `FooOp::Properties` |

### 14.3 C++ string 参数

custom directive 可以带一段 C++ 字符串作为参数。字符串会原样粘贴到 parse/print 调用中，并支持：

```text
$_builder
$_ctxt
```

这可以用来参数化 custom directive。

---

## 十五、Optional Groups

有些 operation 的文本语法中某段信息只在条件满足时出现，例如：

```text
func.return
func.return %0 : i32
```

ODS 用 optional group 表达：

```text
(then-elements) ? 
(then-elements) : (else-elements) ?
```

准确语法：

```text
optional-group: `(` then-elements `)` (`:` `(` else-elements `)`) ? `?`
```

### 15.1 anchor

optional group 必须有一个 anchor，使用 `^` 标记：

```tablegen
let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
```

含义：

```text
如果 $operands 存在，就打印 operands 和 type
如果 $operands 为空，就整组都不打印
```

anchor 的作用是决定 optional group 是否出现。

### 15.2 optional group 的要求

原文要求：

- then-elements 第一个元素必须可选解析，可以是 attribute、literal、operand、property、region
- then 或 else 中必须正好有一个 argument variable 或 type directive 被标为 anchor
- anchor 用 `^` 标记
- group 内只能使用 literals、variables、custom directives、type directives
- attribute 变量都可以使用，但只有 optional/default-valued attribute 可以作为 anchor
- operand/result 只能用 optional 或 variadic
- region 都可以用；固定数量 region 若 group 不出现，则 region 为空

### 15.3 UnitAttr / UnitProp

`UnitAttr` 只有一个可能值，它的意义来自“是否存在”。

如果 `UnitAttr` 作为 optional group 的 anchor，且不是 group 第一个元素，那么它本身不会被打印，而是通过 group 是否出现推断。

例子：

```tablegen
let arguments = (ins UnitAttr:$is_read_only);
let assemblyFormat = "attr-dict (`is_read_only` $is_read_only^)?";
```

打印：

```mlir
foo.op is_read_only
foo.op
```

同样规则适用于 `UnitProp`。

### 15.4 Optional else group

optional group 可以带 else：

```tablegen
let assemblyFormat = "attr-dict (`foo_is_present` $foo^):(`foo_is_absent`)?";
```

打印：

```mlir
foo.op foo_is_present
foo.op foo_is_absent
```

else group 中不能有 anchor。

---

## 十六、Assembly Format 的硬性要求

Declarative assembly format 必须满足：

1. operation name 和 results 前缀不在 format 中写，因为它们固定。
2. 所有 operands 必须出现，或者通过 `operands` directive 统一出现。
3. 所有 regions 必须出现，或者通过 `regions` directive 统一出现。
4. 所有 successors 必须出现，或者通过 `successors` directive 统一出现。
5. 所有 operand/result types 必须通过 `type(...)` 等 directive 出现，除非可推断。
6. 如果 non-attribute properties 没全部出现在 format 中，必须有 `prop-dict`。
7. `attr-dict` 必须出现。
8. 不允许重复/重叠信息，例如多个 `attr-dict`，重复打印同一个 operand type 等。

`attr-dict` 和单独 attribute variable 不算重叠。已经单独打印的 attributes 会从 `attr-dict` 中省略。

### 16.1 Type Inference

某些情况下，assembly format 可以省略 type。

可省略的依据：

#### Buildable Types

如果某个 type constraint 只有一个可构造表示，例如：

```text
I32
Index
```

它可以通过 `builderCall` 或继承 `BuildableType` 标记为 buildable。

#### Trait Equality Constraints

如果 traits 已经表达类型相等关系，printer/parser 可利用这些信息推断缺失类型。

支持的 traits 包括：

```text
AllTypesMatch
TypesMatchWith
SameTypeOperands
SameOperandsAndResultType
```

#### InferTypeOpInterface

如果 op 实现 `InferTypeOpInterface`，result types 可由 operands 推断，因此 assembly format 中可以省略 result types。

---

## 十七、Canonicalizer / Folder Hooks

ODS 有几个布尔字段用于声明 op 实现了 canonicalization 或 folding。

### 17.1 hasCanonicalizer

```tablegen
let hasCanonicalizer = 1;
```

表示这个 op 定义了：

```cpp
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context);
```

用于注册一组 rewrite patterns。

### 17.2 hasCanonicalizeMethod

```tablegen
let hasCanonicalizeMethod = 1;
```

表示 op 实现一个简单的 `canonicalize` 方法，适合单个 match-and-rewrite 风格 pattern。

如果 `hasCanonicalizer = 0`，ODS 会生成 `getCanonicalizationPatterns()`，并让它调用这个 `canonicalize` method。

### 17.3 hasFolder

```tablegen
let hasFolder = 1;
```

表示 op 实现：

```cpp
fold(...)
```

fold 比 rewrite pattern 更局部，通常不能创建新 op，主要用于常量折叠、返回已有 value/attribute、原地更新等。

---

## 十八、Extra Declarations / Definitions

### 18.1 extraClassDeclaration

如果需要在生成的 C++ op class 中塞入额外声明：

```tablegen
let extraClassDeclaration = [{
  void someHelper();
}];
```

它会被原样复制进 generated op class。

原文提醒：这是给长尾高级场景用的。如果某种需求很通用，最好改进 ODS 基础设施，而不是大量塞 raw C++。

### 18.2 extraClassDefinition

对于 TableGen 中的 base op class，如果多个 op 都继承它，可能希望提供通用函数定义。

```tablegen
let extraClassDefinition = [{
  void $cppClass::someHelper() { ... }
}];
```

`$cppClass` 会被替换成具体 C++ op 类名。

这些定义会放到 generated source file 的 op C++ namespace 中。

---

## 十九、Generated C++ Code

`OpDefinitionsGen` 会处理 `.td` 文件并生成两类 C++：

```text
-gen-op-decls
  生成声明，通常包含到 .h.inc

-gen-op-defs
  生成定义，通常包含到 .cpp.inc
```

### 19.1 GET_OP_CLASSES / GET_OP_LIST

definition file 中通常通过 macro 控制生成内容：

```cpp
#define GET_OP_CLASSES
#include "MyDialectOps.cpp.inc"
```

还可以通过：

```cpp
#define GET_OP_LIST
```

得到逗号分隔的 op 列表，用于 registration。

### 19.2 C++ class name 和 namespace

C++ op class 名由 TableGen `def` 名去掉 dialect prefix 得到。

例子：

```tablegen
def TF_AddOp : TF_Op<"Add", [...]> { ... }
```

生成类名：

```cpp
AddOp
```

原因：不同 dialect 都可能有自己的 `AddOp`，C++ namespace 用来区分。

namespace 来自 dialect 的：

```tablegen
let cppNamespace = "A::B";
```

如果没指定，使用 dialect name 作为 namespace。

因此：

```text
IR op name:
  tf.Add

C++ qualified name:
  mlir::TF::AddOp 或指定 namespace 下的 AddOp
```

两者不一定字符串完全一致。

### 19.3 Operand Adaptors

每个 op 会生成一个 operand adaptor。

目的：

> 当你只有一组 `Value`，但想用具名 getter 访问它们时，不用写 magic index。

例如 binary op 的 adaptor 可能提供：

```cpp
Value lhs();
Value rhs();
```

而不是：

```cpp
values[0]
values[1]
```

adaptor 类：

```text
AddOpAdaptor
AddOp::Adaptor
```

典型用法：

```cpp
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());
}

void process(AddOp op, ArrayRef<Value> newOperands) {
  zip(op);
  zip(AddOp::Adaptor(newOperands));
}
```

这也解释了为什么 DialectConversion 中 `OpConversionPattern` 常用 `Adaptor`：它能在“替换后的 operands 数组”上提供和原 op 类似的具名访问器。

### 19.4 Sharded Operation Definitions

大 dialect 如果有很多 op，单个 generated `.cpp.inc` 会导致编译单元过大。

`mlir-tblgen` 支持 sharding：

```text
-op-shard-count
```

生成多个 shard：

```cpp
#define GET_OP_DEFS_0
#include "MyDialectOps.cpp.inc"

#define GET_OP_DEFS_1
#include "MyDialectOps.cpp.inc"
```

实践要求：

- 不要把共享 utility 写成单个源文件里的 `static` 函数
- 把 shared parser/printer/helper 声明到公共 header
- out-of-line verifier/member functions 建议放到单独 source file
- dialect 初始化中使用生成的 registration hook：

```cpp
void MyDialect::initialize() {
  registerMyDialectOperations(this);
}
```

CMake 可用：

```cmake
set(LLVM_TARGET_DEFINITIONS MyDialectOps.td)
add_sharded_ops(MyDialectOps 8)
```

---

## 二十、Constraints 约束系统

Constraint 是 ODS 的核心之一。operation verification 和 declarative rewrite matching 都依赖约束。

ODS 中约束分三类：

```text
single-entity constraint
  只约束一个 operand / attribute / result

multi-entity constraint
  约束多个 operands / attributes / results 之间的关系

trait
  operation 本身的内在性质
```

### 20.1 Single-entity constraint

写在实体声明处：

```tablegen
I32:$x
F32Attr:$alpha
TensorOf<[F32]>:$input
```

常见层次：

```text
TypeConstraint
  -> Type
      -> I32, F32, Index, TensorOf<...>

AttrConstraint
  -> Attr
      -> I32Attr, F32Attr, StrAttr, I64ArrayAttr
```

### 20.2 Multi-entity constraint

写在 op trait 列表中：

```tablegen
def MyOp : MyDialect_Op<"my_op", [
  AllTypesMatch<["input", "output"]>
]> { ... }
```

用于表达：

- operand 和 result 类型相同
- 多个 result shape 相同
- 某个 result 类型由 operand 类型变换得到

这类约束建模为 `PredOpTrait`。

### 20.3 Trait

Traits 是 op 的内在性质：

```tablegen
Pure
NoMemoryEffect
Commutative
IsTerminator
SameOperandsAndResultType
```

ODS 中 trait 建模为 `NativeTrait`，会映射到 C++ `mlir::OpTrait`。

---

## 二十一、自定义 Constraint

自定义约束需要：

```text
predicate
  判断是否满足

description
  用于错误信息和文档
```

核心 predicate 类型：

```text
CPred
  叶子谓词，里面是 C++ 代码字符串

Compound predicates
  And / Or / Neg / SubstLeaves / Concat 等组合器
```

### 21.1 CPred

`CPred` 是 TableGen 和 C++ 的边界：

```tablegen
CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">
```

里面的字符串是 C++ 表达式，返回 bool。

特殊占位符：

| 占位符 | 含义 |
|--------|------|
| `$_builder` | 替换成 `mlir::Builder` |
| `$_op` | 当前 operation |
| `$_self` | 当前约束附着的实体 |

对于 attribute 约束：

```tablegen
BoolAttr
  CPred<"isa<BoolAttr>($_self)">
```

`$_self` 是 attribute 本身。

对于 type constraint：

```tablegen
F32:$operand
```

`$_self` 会展开成 operand/result 的 type。

### 21.2 用组合器构造复杂约束

例如 attribute 是 32-bit 或 64-bit integer：

```tablegen
And<[
  CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
  Or<[
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType().isInteger(32)">,
    CPred<"::llvm::cast<::mlir::IntegerAttr>($_self).getType().isInteger(64)">
  ]>
]>
```

实际项目中，如果已有 `I32Attr` / `I64Attr`，应该复用：

```tablegen
Or<[I32Attr.predicate, I64Attr.predicate]>
```

### 21.3 调 C++ 函数

复杂逻辑也可以写成 C++ 函数：

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

然后在 TableGen 中调用：

```tablegen
def HasSomeProperty : AttrConstraint<
  CPred<"HasSomeProperty($_self)">,
  "has some property">;
```

取舍：

```text
TableGen predicate 组合:
  更声明式，更利于未来自动生成和分析

C++ 函数:
  表达复杂逻辑更方便，但对 TableGen 来说是不透明黑盒
```

---

## 二十二、Attribute Definition

attribute 是 operation 的编译期常量。

ODS attributes 是对 C++ `mlir::Attribute` 类的包装。它们描述：

```text
storage type
  底层存储的 Attribute 类型

return type
  生成 getter 返回给用户的 C++ 类型

conversion method
  从 storage 转成 return type 的方法
```

例子：

```text
StrAttr
  映射到 StringAttr

F32Attr / F64Attr
  映射到 FloatAttr，但额外要求 bitwidth
```

### 22.1 Attribute decorators

常见装饰器：

| Decorator | 作用 |
|-----------|------|
| `DefaultValuedAttr` | attribute 带默认值 |
| `OptionalAttr` | attribute 可选 |
| `ConfinedAttr` | 增加额外约束 |
| `AllAttrOf` | 多个约束都满足 |

这些装饰器既影响 verifier，也影响 builder/printer/parser 行为。

---

## 二十三、Enum Definition

MLIR ODS 可以生成 C++ enum 及其辅助函数。

两类 enum：

```text
IntEnum
  从若干整数值中选一个

BitEnum
  bit flags，可组合
```

对应 case：

```text
EnumCase
BitEnumCase
```

TableGen backend：

```text
-gen-enum-decls
-gen-enum-defs
```

会生成：

- C++ `enum class`
- `DenseMapInfo`
- 从整数到 enum 的 `symbolize*`
- 从字符串到 enum 的转换函数
- 从 enum 到字符串的转换函数
- bit enum 的 `|`、`&`、`^`、`~` 等运算符
- bit enum 的 contains/clear 辅助函数

### 23.1 IntEnum 例子

```tablegen
def Case15: I32EnumCase<"Case15", 15>;
def Case20: I32EnumCase<"Case20", 20>;

def MyIntEnum: I32Enum<"MyIntEnum", "An example int enum",
                       [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```

生成类似：

```cpp
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

std::optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
std::optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
```

### 23.2 BitEnum 例子

```tablegen
def None: I32BitEnumCaseNone<"None">;
def Bit0: I32BitEnumCaseBit<"Bit0", 0, "tagged">;
def Bit1: I32BitEnumCaseBit<"Bit1", 1>;

def MyBitEnum: I32BitEnum<"MyBitEnum", "An example bit enum",
                          [None, Bit0, Bit1]> {
  let separator = "|";
}
```

生成值类似：

```cpp
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit0 = 1,
  Bit1 = 2,
};
```

并生成 bit 操作：

```cpp
operator|
operator&
operator^
operator~
bitEnumContainsAll
bitEnumContainsAny
bitEnumClear
```

字符串形式可类似：

```text
tagged|Bit1
```

### 23.3 Wrapping enums in attributes

把 enum 包成 Attribute 的常见方式：

```tablegen
EnumAttr<...>
```

它持有一个 enum value，并可自定义 assembly format，例如加 mnemonic 或尖括号。

旧形式包括：

```text
*IntEnumAttr
*BitEnumAttr
*EnumAttrCase
```

这些通常把值存为某个 bitwidth 的 `SignlessIntegerAttr`，并验证值在 enum 合法范围内。

### 23.4 Enum properties

enum 也可以包成 property：

```text
EnumProp
NamedEnumProp
```

效果：

- enum C++ 值成为 operation property struct 的成员
- verifier 检查 enum 值是否合法
- `NamedEnumProp` 生成更不歧义的文本语法，例如带 mnemonic 和 `<>`
- `*EnumPropWithAttrForm` 支持从 EnumAttr 透明升级，也可以保留 generic form 中的 attributes

---

## 二十四、Debugging Tips：查看生成代码

TableGen 有时不直观，最有效的调试方法是看 `mlir-tblgen` 生成了什么。

先构建：

```bash
cmake --build . --target mlir-tblgen
```

查看 op C++ class 声明：

```bash
mlir-tblgen --gen-op-decls \
  -I /path/to/mlir/include \
  /path/to/input/td/file
```

查看 op C++ class 定义：

```bash
mlir-tblgen --gen-op-defs \
  -I /path/to/mlir/include \
  /path/to/input/td/file
```

查看 dialect 文档：

```bash
mlir-tblgen --gen-dialect-doc \
  -I /path/to/mlir/include \
  /path/to/input/td/file
```

查看 op interface 声明/定义/文档：

```bash
mlir-tblgen --gen-op-interface-decls \
  -I /path/to/mlir/include \
  /path/to/input/td/file

mlir-tblgen --gen-op-interface-defs \
  -I /path/to/mlir/include \
  /path/to/input/td/file

mlir-tblgen --gen-op-interface-doc \
  -I /path/to/mlir/include \
  /path/to/input/td/file
```

实践建议：

```text
TableGen 报错看不懂时:
  先缩小 .td
  再用 mlir-tblgen --gen-op-decls / --gen-op-defs 看展开结果
```

---

## 二十五、Deprecation

### 25.1 TableGen deprecation

TableGen class/def 可以标记 deprecated：

```tablegen
def OpTraitA : NativeOpTrait<"OpTraitA">,
               Deprecated<"use `bar` instead">;
```

`mlir-tblgen` 可以根据 `-on-deprecated` 发 warning 或 error。

### 25.2 C++ deprecation

生成的 C++ entity 可以用：

```tablegen
def MyOp : Op<MyDialect, "my.op">,
           CppDeprecated<"use 'your.op' instead">;
```

区别：

```text
Deprecated
  TableGen 层面，由 mlir-tblgen 报告

CppDeprecated
  C++ 编译层面，使用生成实体时由 C++ 编译器报警
```

对于常见匿名定义，提供了便捷类，例如：

```tablegen
DeprecatedOpBuilder<"use 'build' with foo instead", (ins "int":$bar)>
```

---

## 二十六、设计原则和原文附录观点

原文最后强调：op 描述应该尽量声明式。

原因是：

```text
越声明式
  -> 工具越容易分析
  -> verifier/parser/printer/builder 越容易自动生成
  -> pattern 和 lowering 越容易复用结构信息
```

一些设计观点：

- MLIR 允许 unknown ops，所以 op 不一定必须注册才能存在。
- registered ops 的优势是可以生成类型安全的 C++ 类、accessor、verifier。
- traits、operand/result constraints、shape inference 信息应尽量写进 ODS。
- 文档应该和 op 定义放在一起，自动生成 dialect LangRef。
- generic assembly form 总是可用，但 custom/declarative assembly format 能提供更简洁 IR。
- pattern matching 与 op 描述分离，因为 MLIR 中 dialect 很多，转换关系形成 graph，不是单一后端固定规则。

---

## 二十七、一个定义 Op 的实践模板

可以用下面模板理解一个真实 op 定义应该包含什么：

```tablegen
def My_MatmulOp : MyDialect_Op<"matmul", [
  Pure,
  AllTypesMatch<["lhs", "rhs", "result"]>
]> {
  let summary = "Matrix multiplication";

  let description = [{
Computes matrix multiplication on two rank-2 tensors.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$lhs,
    TensorOf<[F32]>:$rhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );

  let results = (outs
    TensorOf<[F32]>:$result
  );

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}
```

这个模板覆盖了：

- operation name
- traits
- documentation
- operands
- default-valued attributes
- results
- custom assembly
- verifier / canonicalizer / folder hooks

---

## 二十八、学习 checklist

读完这篇文档后，应该能回答：

- ODS 为什么比手写 `mlir::Op` C++ 模板更适合 dialect 开发？
- `def` 名、C++ class 名、IR operation name 三者有什么区别？
- operands、attributes、properties 的区别是什么？
- `Optional`、`Variadic`、`VariadicOfVariadic` 分别解决什么问题？
- 多个 variadic/optional operands 为什么需要 segment size trait？
- `ConfinedAttr` 和 `AllAttrOf` 如何组合约束？
- `regions`、`results`、`successors` 如何声明？
- ODS 自动生成哪些 builder？什么时候可以省略 result type？
- `OpBuilder` 中 `$_builder` / `$_state` 是什么？
- `hasVerifier` 和 `hasRegionVerifier` 的执行顺序有什么区别？
- declarative assembly format 为什么必须出现 `attr-dict`？
- `prop-dict` 和 `attr-dict` 如何分工？
- optional group 的 anchor 是什么？
- 哪些情况下 assembly format 可以省略 type？
- `hasCanonicalizer`、`hasCanonicalizeMethod`、`hasFolder` 各自生成什么 hook？
- generated C++ class 和 operand adaptor 如何使用？
- constraints 中 `CPred`、`$_self`、`$_op`、`$_builder` 分别代表什么？
- ODS attribute 的 storage type 和 return type 为什么可能不同？
- IntEnum / BitEnum 会生成哪些 C++ 辅助函数？
- 遇到 TableGen 生成问题时如何用 `mlir-tblgen --gen-op-decls/defs` 调试？

---

## 总结

`Operations.md` 的核心不是单独教某个 TableGen 语法，而是定义一套完整的 operation 建模系统：

```text
ODS record
  描述 operation 的全部结构事实

OpDefinitionsGen
  把结构事实生成 C++ class、builder、verifier、parser/printer

Constraints / Traits / Interfaces
  把语义和不变量声明化

Declarative Assembly Format
  用声明式字符串生成简洁 textual IR

Attribute / Enum / Property
  让编译期常量和配置项有类型安全的 C++ 表示
```

对 AI 编译器学习来说，ODS 是定义新 IR 层级的入口。无论是高层 tensor 算子、中层 linalg/scf 变换，还是低层 target dialect，最终都需要用 ODS 把 operation 的结构、约束和使用接口描述清楚。

