# MLIR Operations.md (ODS) - 学习总结

## 概述

本文总结 `mlir/docs/DefiningDialects/Operations.md` 的核心内容。这个文档是 MLIR **Operation Definition Specification (ODS)** —— 操作定义规范，讲的是如何用 TableGen 以表格驱动（table-driven）的方式声明式地定义 MLIR 操作。

一句话理解：

> ODS 是 MLIR 的"操作描述语言"：你在一个 `.td` 文件里用 TableGen 语法声明一个操作的所有事实（参数、结果、约束、trait、汇编格式等），`mlir-tblgen` 在编译时把这些声明自动展开成等价的 C++ `Op` 类，包含 getter、builder、verifier、parser、printer 等全部样板代码。

可以把它放在 MLIR dialect 开发的主线里理解：

```text
Dialect (方言)
  -> Op (.td 中用 ODS 定义)
      -> mlir-tblgen 生成 C++ Op 类
          -> getter / builder / verifier / parser / printer
          -> 被 Pass / Pattern / Conversion 使用
```

从 AI 编译器学习角度看，这篇文档回答的问题是：

> 当你要给一个新的 dialect（比如 tosa、stablehlo、linalg）添加一个 operation 时，MLIR 提供了什么机制让你用最少的代码、最声明式的方式把一个 op 的全部语义描述清楚，并自动获得类型安全的 C++ 接口？

---

## 一、动机：为什么需要 ODS

MLIR 允许可插拔的 dialect，dialect 里包含一系列 operation。如果没有统一的描述机制，会面临"stringly typed IR"问题：

- 优化/分析 pass 里到处做字符串比较来判断 op 类型
- 访问操作数用 `getOperand(3)` 这种容易出错且无自文档性的方式
- 构造函数冗长且没有默认参数
- 没有集中的验证逻辑，要么到处重复，要么干脆没有

ODS 的解决方案是：把每个 op 的所有事实集中写在一个 TableGen record 里，作为**单一事实来源（single source of truth）**，然后自动生成 getter、builder、verifier 等代码。

### 核心收益

1. **单一事实来源**：一个 op 的所有信息都在一个 record 里，不需要在代码片段间跳转
2. **消除样板代码**：自动生成 operand/attribute/result getter、builder、verifier
3. **驱动自动生成**：这些 op 信息记录不仅用于 op 定义本身，还可以驱动计算图序列化等其他组件的自动生成

---

## 二、TableGen 语法基础

TableGen 是 ODS 使用的语言，文件后缀为 `.td`。核心概念：

| 概念 | 类比 | 说明 |
|------|------|------|
| `class` | C++ 类 | 可模板化、可继承 |
| `def` | C++ 对象 | 通过特化 class 声明，不能再被模板化或继承 |
| `dag` | 有向无环图 | 专有类型，语法 `(operator arg0, arg1, ...)`，可带名 `(Op:$name Arg:$argname)` |

`dag` 是 ODS 中描述操作参数和结果的核心数据结构。

---

## 三、操作定义的核心结构

MLIR 通过 `OpBase.td` 中定义的一系列构造来支持操作定义，核心的几个：

```text
Op          主构造，定义操作本身
Dialect     操作所属的方言，包含方言级信息
Trait       操作的特殊属性和约束（如是否有副作用）
ins/outs    标记，分别引出参数和结果
TypeConstraint  对操作数/结果的类型约束
AttrConstraint  对属性的约束
Property    非属性支持的、内联存储的操作属性
```

一个操作通过特化 `Op` 类来定义，例如：

```tablegen
def TF_AvgPoolOp : TF_Op<"AvgPool", [NoMemoryEffect]> {
  let summary = "Performs average pooling on the input.";
  let description = [{ ... }];

  let arguments = (ins
    TF_FpTensor:$value,
    ConfinedAttr<I64ArrayAttr, [ArrayMinCount<4>]>:$ksize,
    ...
  );

  let results = (outs
    TF_FpTensor:$output
  );
}
```

### 3.1 操作名称

操作全名 = `dialect名.op名`，如 `tf.Add`。dialect 名由 `Dialect` 提供，op 名是 `Op` 类的第二个模板参数。这是解析/打印/模式匹配的唯一标识。

### 3.2 操作文档

包含一行 `summary` 和更长的 `description`（Markdown 语法），用于自动生成 dialect 文档。

> 最佳实践：文档放在操作定义的开头；summary 简短一行，大写开头，无句号结尾。

---

## 四、操作参数：operands / attributes / properties

操作有三种参数，都在 `arguments`（`ins` 引导的 dag）中声明：

```text
operands    运行时值，由其他 op 产生
attributes  编译期常量，存储在 MLIR context 中
properties  编译期常量，内联存储在 operation 上
```

### 4.1 两类 attributes

1. **自然属性（Natural attributes）**：影响操作行为，如卷积的 padding
2. **派生属性（Derived attributes）**：不用于定义 op，而是从 op 信息派生（如输出 shape 的类型），主要用于方便接口生成或与其他框架交互。所有派生属性必须可物化为 Attribute

### 4.2 Variadic / Optional operands

- `Variadic<...>`：声明变长操作数
- `Optional<...>`：声明可选操作数
- `VariadicOfVariadic<..., "segAttr">`：声明子范围也是变长的操作数

当一个 op 有多个变长/可选操作数时，需要 `SameVariadicOperandSize` 或 `AttrSizedOperandSegments` trait 来消歧。

### 4.3 属性的装饰器

| 装饰器 | 作用 |
|--------|------|
| `OptionalAttr<...>` | 可选属性 |
| `DefaultValuedAttr<..., "...">` | 带默认值的属性 |
| `ConfinedAttr<..., [...]` | 组合更多约束 |
| `AllAttrOf<[...]>` | 多约束同时满足 |
| `DefaultValuedEnumAttr<...>` | enum 属性带默认值 |

`ConfinedAttr` 支持丰富的原始约束，如 `IntMinValue<N>`、`IntMaxValue<N>`、`ArrayMinCount<N>`、`IntArrayNthElemEq<I,N>` 等。

### 4.4 Properties

Properties 类似 attributes，但不存储在 MLIR context 中，而是内联存储在 operation 上。

- `DefaultValuedProp<..., "...">`：带默认值的 property
- `OptionalProp<...>`：可选 property，包装在 `std::optional` 中

---

## 五、操作 regions / results / successors

### 5.1 Regions

```tablegen
let regions = (region
  <region-constraint>:$<region-name>,
  ...
);
```

`VariadicRegion<...>` 用于变长 region，只能放在 region 列表最后。

### 5.2 Results

```tablegen
let results = (outs
  <type-constraint>:$<result-name>,
  ...
);
```

`Variadic<...>` 同样可用于变长结果，`SameVariadicResultSize` 处理多个变长结果。

### 5.3 Successors

用于终结操作（terminator），指定后继基本块：

```tablegen
let successors = (successor
  <successor-constraint>:$<successor-name>,
  ...
);
```

`VariadicSuccessor<...>` 只能放在列表最后。

---

## 六、Builder 方法

ODS 根据参数和结果类型自动生成多个 builder。给定一个有 operands/attributes/properties/results 的 op，会生成：

1. **聚合参数 builder**：`TypeRange resultTypes, ValueRange operands, Properties, ArrayRef<NamedAttribute>`
2. **legacy 聚合 builder**：合并 attribute 字典
3. **分离参数 builder**：每个 result-type/operand/attribute 单独参数
4. **裸值 builder**：attribute 参数用原始值（如 `APInt` 而非 `IntegerAttr`）
5. **类型推断 builder**：如果 op 实现了 `InferTypeOpInterface` 或返回类型可推断，则不需要传 result type

### 6.1 自定义 builder

自动生成的不够用时，可以在 `builders` 字段中定义：

```tablegen
let builders = [
  OpBuilder<(ins "float":$val), [{
    $_state.addAttribute("attr", $_builder.getF32FloatAttr(val));
  }]>
];
```

- `$_builder` 和 `$_state` 是特殊变量
- `CArg<"float", "0.5f">` 用于给参数加默认值
- 短 builder 内联在 ODS 中，长 builder 放到 C++ 文件里

---

## 七、自定义 verifier

约束指定的验证会自动生成。需要额外验证时：

```tablegen
let hasVerifier = 1;
let hasRegionVerifier = 1;
```

生成 `LogicalResult verify()` 和 `LogicalResult verifyRegions()` 声明。

### 验证执行顺序

第一阶段（不访问 region 内部）：
1. StructuralOpTrait
2. `verifyInvariants`（ODS 生成的类型/属性验证）
3. 其他标记为 `verifyTrait` 的 Trait/Interface
4. 自定义 `hasVerifier`

第二阶段（可访问 region 内部操作，在 region 内操作验证后执行）：
1. 标记为 `verifyRegionTrait` 的 Trait/Interface
2. 自定义 `hasRegionVerifier`

> 注意：自定义 verifier 应避免使用自定义 printer 打印操作（因为可能尚未验证），应使用 `Error` 级别诊断。

---

## 八、声明式汇编格式（Declarative Assembly Format）

这是 ODS 最强大的特性之一：用一个声明式字符串描述操作的文本格式，自动生成 parser 和 printer。

```tablegen
let assemblyFormat = [{
  $callee `(` $args `)` attr-dict `:` functional-type($args, results)
}];
```

### 8.1 格式组成

格式由三部分组成：

#### Directives（指令）

| 指令 | 作用 |
|------|------|
| `attr-dict` | 属性字典 |
| `attr-dict-with-keyword` | 带 `attributes` 关键字的属性字典 |
| `prop-dict` | properties 转为字典 |
| `custom<UserDirective>(Params)` | 用户自定义 C++ 指令 |
| `functional-type(inputs, outputs)` | 函数类型格式 |
| `oilist(...)` | 可选的顺序无关子句列表 |
| `operands` | 所有操作数 |
| `regions` | 所有 region |
| `results` | 所有结果 |
| `successors` | 所有后继 |
| `type(input)` | 输入的类型 |
| `qualified(type_or_attr)` | 带方言前缀的类型/属性 |
| `ref(input)` | 引用已解析的变量/directive |

#### Literals（字面量）

用反引号包围的关键字或标点，如 `` `(` ``、`` `:` ``、`` `->` ``。
支持 `\n` 换行和空字面量 `` `` 去除隐式空格。

#### Variables（变量）

注册在操作上的实体，如 `$callee`、`$args`。

### 8.2 自定义指令（Custom Directives）

```text
custom<UserDirective>(Params)
```

会生成对 `parseUserDirective` 和 `printUserDirective` 的调用。Params 可以是变量、type 指令、`attr-dict`、C++ 代码字符串。

参数到 C++ 方法的映射有详细规则，例如：
- 单个操作数 → `OpAsmParser::UnresolvedOperand &`（parse）/ `Value`（print）
- 变长操作数 → `SmallVectorImpl<...> &`（parse）/ `OperandRange`（print）
- `attr-dict` → `NamedAttrList &`（parse）/ `DictionaryAttr`（print）

### 8.3 可选组（Optional Groups）

```text
(then-elements):(else-elements)?
```

- 第一个元素必须可可选解析（attribute/literal/operand/property/region）
- 必须有一个元素标记为 anchor（尾部加 `^`）
- Unit attribute 做 anchor 时不会打印，由可选组的存在性自动推断
- 支持 else 组（anchor 不存在时打印）

```tablegen
// func.return：有操作数时才打印操作数和类型
let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
```

### 8.4 格式要求

1. 输出和操作名不显示（固定的）
2. 所有 operands 必须出现（单独或 `operands`）
3. 所有 regions 必须出现
4. 所有 successors 必须出现
5. 所有 operand/result 类型必须通过 `type` 指令出现
6. 除非所有非属性 properties 都在格式中，否则必须有 `prop-dict`
7. `attr-dict` 必须始终存在
8. 不能有重叠信息

### 8.5 类型推断

类型可以通过以下方式省略：
- **Buildable Types**：只有一个表示的类型（如 `I32`、`Index`），设置了 `builderCall`
- **Trait 相等约束**：`AllTypesMatch`、`TypesMatchWith`、`SameTypeOperands`、`SameOperandsAndResultType`
- **InferTypeOpInterface**：实现了该接口的 op 可省略 result 类型

---

## 九、Canonicalization 和 Folding

| 字段 | 作用 |
|------|------|
| `hasCanonicalizer` | 定义了 canonicalization patterns，需实现 `getCanonicalizationPatterns()` |
| `hasCanonicalizeMethod` | 实现了简单的 `canonicalize` 方法（matchAndRewrite 风格） |
| `hasFolder` | 定义了 fold 规则，需实现 `fold()` |

---

## 十、额外声明和定义

### 10.1 extraClassDeclaration

自动生成无法覆盖的长尾情况，代码原样复制到生成的 C++ op 类中。面向高级用户。

### 10.2 extraClassDefinition

用于定义基类 op 的通用工具函数，代码加到生成的源文件中 op 的 C++ namespace 内。`$cppClass` 会被替换为 op 的 C++ 类名。

---

## 十一、Constraints 体系

Constraint 是 ODS 的核心概念：操作验证和图匹配都基于满足约束。

### 11.1 三种范围

```text
单实体约束    只涉及一个 operand/attribute/result
多实体约束    涉及多个 operand/attribute/result（如 result shape = operand shape）
Trait         操作本身的内在属性（如是否有副作用）
```

### 11.2 Predicate 组合

约束的核心是 **Predicate**（`Pred` 类）：

1. `CPred`：原子叶子谓词，里面是返回 bool 的 C++ 代码
2. **复合谓词**：用 `And`、`Or`、`Neg`、`SubstLeaves`、`Concat` 组合

特殊占位符：
- `$_builder`：`mlir::Builder` 实例
- `$_op`：当前操作
- `$_self`：谓词所附着的实体（attribute 则是该 attribute，type 约束则是 `operand(...).getType()`）

```tablegen
// 属性是 32 位或 64 位整数
And<[
  CPred<"isa<IntegerAttr>($_self)">,
  Or<[
    CPred<"cast<IntegerAttr>($_self).getType().isInteger(32)">,
    CPred<"cast<IntegerAttr>($_self).getType().isInteger(64)">
  ]>
]>
```

> 最佳实践：优先用 `CPred` + 谓词组合器（暴露更多信息给 ODS），而非把逻辑藏在一个 C++ 函数后面。

---

## 十二、Attribute 定义

ODS attribute 是 C++ attribute 类的包装器，有三个层面：

```text
storage type    底层 mlir::Attribute（存储）
return type     getter 的 C++ 返回类型
转换方法        storage 和 return 之间的转换
```

常用装饰器：`DefaultValuedAttr`、`OptionalAttr`、`ConfinedAttr`、`AllAttrOf`。

---

## 十三、Enum 定义

ODS 可以从 `IntEnum`（值列表）和 `BitEnum`（位标志组合）生成 C++ enum。

### 13.1 IntEnum

```tablegen
def Case15: I32EnumCase<"Case15", 15>;
def Case20: I32EnumCase<"Case20", 20>;
def MyIntEnum: I32Enum<"MyIntEnum", "An example int enum",
                       [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
}
```

`mlir-tblgen -gen-enum-decls/defs` 生成：
- C++ `enum class`
- `symbolizeXxx(uint32_t)`：值 → enum
- `stringifyXxx(Enum)`：enum → 字符串
- `symbolizeXxx(StringRef)`：字符串 → enum
- `llvm::DenseMapInfo` 特化

### 13.2 BitEnum

支持位运算：`|`、`&`、`^`、`~`、`|=`、`&=`、`^=`。
还有 `bitEnumContainsAll`、`bitEnumContainsAny`、`bitEnumClear`。
字符串用 `|` 分隔（可配置 separator）。

### 13.3 Enum 包装为 Attribute / Property

- `EnumAttr`：最常用，取 `EnumInfo` 参数
- 旧形式：`*IntEnumAttr` / `*BitEnumAttr`（存为 `SignlessIntegerAttr`）
- `EnumProp`：包装为 property，内联存储
- `NamedEnumProp`：带 mnemonic 和 `<>` 语法
- `*EnumPropWithAttrForm`：可透明升级自 `EnumAttr`

---

## 十四、生成的 C++ 代码

`OpDefinitionsGen` 处理 `.td` 文件，生成两个文件：

```text
-gen-op-decls   声明文件（.h.inc）
-gen-op-defs    定义文件（.cpp.inc）
```

定义文件通过 `#define GET_OP_CLASSES` 启用，还包含 `GET_OP_LIST`（所有 op 的逗号分隔列表）。

### 14.1 类名和命名空间

- C++ 类名 = TableGen `def` 名去掉方言前缀（第一个 `_` 为分隔符），如 `TF_AddOp` → `AddOp`
- 命名空间来自 dialect 的 `cppNamespace`，如 `A::B`

### 14.2 Operand Adaptor

为每个 op 自动生成 operand adaptor 类，解决用 `Value` 列表访问操作数时的"魔数"问题。提供与 op 类同名的方法（如 `.lhs()`、`.rhs()`），可与 op 一起用在模板函数中。

```c++
template <typename BinaryOpTy>
std::pair<Value, Value> zip(BinaryOpTy &&op) {
  return std::make_pair(op.lhs(), op.rhs());
}
// 既可用于 AddOp，也可用于 Adaptor<AddOp>
```

### 14.3 分片（Sharding）

大型 dialect 可通过 `-op-shard-count` 将 op 定义分片，用 `GET_OP_DEFS_${N}` 分别编译，解决编译时间问题。CMake 用 `add_sharded_ops()`，Bazel 用 `gentbl_sharded_ops()`。

---

## 十五、调试技巧

用 `mlir-tblgen` 查看生成的内容：

```sh
# 查看 op C++ 类声明
mlir-tblgen --gen-op-decls -I /path/to/mlir/include input.td

# 查看 op C++ 类定义
mlir-tblgen --gen-op-defs -I /path/to/mlir/include input.td

# 查看 op 文档
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include input.td

# 查看 op interface
mlir-tblgen --gen-op-interface-decls input.td
mlir-tblgen --gen-op-interface-defs input.td
```

TableGen 语法晦涩时，直接看生成的 C++ 是理解问题最有效的方式。

---

## 十六、Deprecation 机制

### TableGen 层面

```tablegen
def OpTraitA : NativeOpTrait<"OpTraitA">, Deprecated<"use `bar` instead">;
```

`mlir-tblgen` 会发出 warning（默认）或 error（`-on-deprecated` 控制）。

### C++ 层面

```tablegen
def MyOp : Op<MyDialect, "my.op">, CppDeprecated<"use 'your.op' instead">;
```

不产生 tblgen 警告，而是由 C++ 编译器在使用该实体时发出警告。`DeprecatedOpBuilder` 是常用辅助类。

---

## 十七、设计理念（Appendix）

从附录的需求分析可以看出 ODS 的设计哲学：

1. **声明式优先**：op 描述应尽可能声明式，让广泛的工具能使用和查询
2. **注册与 C++ 分离**：op registry 独立于 C++ 代码，但未注册的 op 也被允许（能力受限但正确）
3. **TableGen 作为建模语言**：LLVM 后端已证明其适合 trait-based 建模，易扩展
4. **定义与未定义 op 共存**：已定义 op 有固定语义，dialect 完全由 owner 控制
5. **trait 和类型约束与 op 一起建模**：使文本输出更简洁
6. **匹配模式与 op 描述分离**：不同于 LLVM 的"基础 op 集"，MLIR 有多 dialect 形成变换图
7. **可提供参考实现**：可用标准 op 或其他参考实现来表达

---

## 十八、一个最小心智模型

把 ODS 理解为一个编译器的"前端"：

```text
                        ODS Pipeline
                     ─────────────────────

  你写的 .td 文件
  ┌──────────────────────────────────┐
  │ def MyOp : MyDialect<"my_op"> {  │
  │   let arguments = (ins I32:$x);  │
  │   let results = (outs I32:$y);   │
  │   let assemblyFormat = ...;      │
  │ }                                │
  └──────────────┬───────────────────┘
                 │ mlir-tblgen --gen-op-decls/defs
                 ▼
  ┌──────────────────────────────────┐
  │ 生成的 C++ 类 MyOp               │
  │  - getX() / getY()               │  ← getter
  │  - build(...)                    │  ← builder
  │  - verify()                      │  ← verifier
  │  - parse / print                 │  ← assembly
  │  - Adaptor                       │  ← operand adaptor
  └──────────────┬───────────────────┘
                 │ 在 C++ 中 #include .inc 文件
                 ▼
  ┌──────────────────────────────────┐
  │ Pass / Pattern / Conversion      │
  │  使用类型安全的 MyOp 接口        │
  └──────────────────────────────────┘
```

核心心智模型三句话：

1. **你在 `.td` 里声明 op 的全部事实**，TableGen 是声明语言，`dag` 是描述参数/结果的核心结构
2. **`mlir-tblgen` 把声明展开成 C++**，自动生成 getter/builder/verifier/parser/printer，你不用手写样板
3. **约束（Constraint）是验证和匹配的基础**，`CPred` 是原子，`And/Or/Neg` 是组合器，`$_self/$_op/$_builder` 是环境钩子

从 AI 编译器实践角度看，理解 ODS 是理解任何 MLIR dialect（tosa/stablehlo/linalg/affine）代码的第一步：这些 dialect 的每个 op 都是通过 ODS 定义的，你在 `.td` 文件里看到的就是那个 op 的"完整规格说明书"。
