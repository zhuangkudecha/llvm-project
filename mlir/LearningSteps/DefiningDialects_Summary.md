# MLIR Defining Dialects - 中文学习总结

本文基于 `mlir/docs/DefiningDialects/_index.md` 整理，主题是：

> 如何在 MLIR 中定义一个 Dialect，以及如何让 Dialect 支持 attributes、operations、types、parser/printer、canonicalization、bytecode 和运行时扩展。

Dialect 是 MLIR 的核心扩展机制。一个 dialect 可以定义自己的：

- operations
- attributes
- types
- interfaces
- parser / printer
- canonicalization patterns
- bytecode serialization
- dynamic operations / dynamic types / dynamic attributes

从开发角度看，定义 dialect 通常分两类：

```text
静态 dialect
  编译期通过 C++ / TableGen 定义

可扩展 / 动态 dialect
  运行时注册新的 op/type/attr
```

---

## 一、LangRef 回顾：Dialect 是什么

MLIR 的 dialect 是参与并扩展 MLIR 生态的机制。

它允许开发者定义新的：

```text
Attribute
Operation
Type
```

不同 dialect 可以表达不同层级的抽象。例如：

- `arith` 表达算术运算
- `func` 表达函数
- `scf` 表达结构化控制流
- `linalg` 表达线性代数计算
- `pdl` 表达 pattern rewrite 逻辑

一句话理解：

> Dialect 是 MLIR 中“定义一套 IR 语言子集”的单位。

---

## 二、定义 Dialect 的基本方式

最底层来说，定义 dialect 就是特化 C++ 的 `mlir::Dialect` 类。

但 MLIR 推荐用 TableGen 声明式定义 dialect，因为 TableGen 可以自动生成大量 C++ 样板代码，并降低维护成本。

一个最小 dialect 定义示例：

```tablegen
include "mlir/IR/DialectBase.td"

def MyDialect : Dialect {
  let summary = "A short one line description of my dialect.";
  let description = [{
    My dialect is a very important dialect. This section contains a much more
    detailed description that documents all of the important pieces of information
    to know about the document.
  }];

  let name = "my_dialect";
  let cppNamespace = "::my_dialect";
}
```

其中：

```tablegen
def MyDialect : Dialect
```

定义一个 dialect record。

```tablegen
let name = "my_dialect";
```

定义 IR 中的 dialect 名字。这个 dialect 里的 operation 名字会形如：

```text
my_dialect.foo
my_dialect.bar
```

```tablegen
let cppNamespace = "::my_dialect";
```

定义生成的 C++ 类所在命名空间。

实践建议：

```text
Dialect 本身最好放在单独的 .td 文件里
Operations / Attributes / Types 分别放在其他 .td 文件里
```

这样可以建立清晰的 layering，也能避免某些构件被重复生成。

---

## 三、Initialization：Dialect 初始化

每个 dialect 都需要实现初始化 hook：

```c++
void MyDialect::initialize() {
  // Dialect initialization logic should be defined in here.
}
```

这里通常注册：

- operations
- attributes
- types
- interfaces
- dynamic components
- 其他 dialect 初始化逻辑

例如一个 dialect 定义了 `MyAddOp` 和 `MyType`，就需要在 `initialize()` 中把它们加入 dialect。

概念上可以理解为：

```text
TableGen 描述 dialect 有什么
initialize 真正把这些东西注册进 MLIRContext 可用的 dialect 对象中
```

---

## 四、Documentation：文档字段

Dialect 支持：

```tablegen
let summary = "短说明";
let description = [{
  长说明，支持 Markdown 风格文本。
}];
```

区别：

```text
summary
  一行简短描述

description
  更完整的说明文档
```

这些内容可以用于生成 dialect 的 Markdown 文档。MLIR upstream dialect 文档也是通过类似机制生成的。

---

## 五、Class Name：生成的 C++ 类名

生成的 C++ dialect 类名来自 TableGen 中的 def 名字。

规则：

```text
去掉 def 名字里的下划线 _
```

例如：

```tablegen
def Foo_Dialect : Dialect
```

生成：

```c++
class FooDialect;
```

如果是：

```tablegen
def MyDialect : Dialect
```

生成的类就是：

```c++
class MyDialect;
```

---

## 六、C++ Namespace：C++ 命名空间

`cppNamespace` 控制 dialect 及其子组件生成到哪个 C++ 命名空间中。

例如：

```tablegen
let cppNamespace = "::my_dialect";
```

生成的类会位于：

```c++
namespace my_dialect {
  class MyDialect;
}
```

如果写：

```tablegen
let cppNamespace = "";
```

表示不放入任何命名空间。

如果写：

```tablegen
let cppNamespace = "A::B";
```

则生成：

```c++
namespace A {
namespace B {
  // generated classes
}
}
```

原文建议尽量使用完整 namespace，这样不同项目、不同 namespace 中的 dialect 更容易互相使用。

---

## 七、Accessor 生成规则

MLIR 为 dialect 组件生成 getter/setter 时，会把 `snake_style` 转成 camel case。

例如：

```tablegen
def MyOp : MyDialect<"op"> {
  let arguments = (ins StrAttr:$value, StrAttr:$other_value);
}
```

会生成类似：

```c++
StringAttr MyOp::getValue();
void MyOp::setValue(StringAttr newValue);

StringAttr MyOp::getOtherValue();
void MyOp::setOtherValue(StringAttr newValue);
```

也就是：

```text
value
  -> getValue / setValue

other_value
  -> getOtherValue / setOtherValue
```

---

## 八、Dependent Dialects：依赖的 Dialect

一个 dialect 可能依赖其他 dialect。

例如：

- canonicalization 中生成 `arith` op
- lowering 中生成 `func` op
- attribute/type 里复用其他 dialect 的组件

这种依赖应该显式声明：

```tablegen
def MyDialect : Dialect {
  let dependentDialects = [
    "arith::ArithDialect",
    "func::FuncDialect"
  ];
}
```

含义：

```text
加载 MyDialect 时，也应加载 arith 和 func dialect
```

这样可以确保依赖 dialect 已经注册到 `MLIRContext`，避免创建 op/type/attr 时找不到对应 dialect。

---

## 九、Extra Declarations：额外声明

TableGen 会尽量自动生成 dialect 的 C++ 逻辑。

但如果有特殊需求，可以用：

```tablegen
extraClassDeclaration
```

其中的代码会被原样复制到生成的 C++ Dialect 类中。

注意：

```text
extraClassDeclaration 适合少数特殊场景
如果某种需求很普遍，更推荐改进 MLIR 基础设施
```

---

## 十、hasConstantMaterializer：常量物化

字段：

```tablegen
let hasConstantMaterializer = 1;
```

用于声明 dialect 可以从一个 `Attribute` 和一个 `Type` 物化出常量 operation。

这通常用于 folding。

例如某个 op 被 fold 后，结果变成一个 attribute。为了把这个 attribute 放回 IR，MLIR 需要创建一个 constant-like operation。

启用后需要实现：

```c++
Operation *MyDialect::materializeConstant(OpBuilder &builder,
                                          Attribute value,
                                          Type type,
                                          Location loc) {
  ...
}
```

参数含义：

```text
builder
  用于创建 constant-like operation

value
  fold 得到的 attribute 值

type
  期望的结果类型

loc
  source location
```

成功时返回新创建的 operation，失败时返回 `nullptr`。

这个 hook 不应该改变 builder 的 insertion position。

---

## 十一、hasNonDefaultDestructor：自定义析构

如果 dialect 类需要自定义析构函数，可以设置：

```tablegen
let hasNonDefaultDestructor = 1;
```

此时生成代码只会声明析构函数，具体定义由开发者在 C++ 文件中实现。

适用场景：

```text
dialect 持有额外资源
析构时需要特殊清理逻辑
```

---

## 十二、Discardable Attribute Verification

MLIR 中有一种 attribute 叫 **discardable attribute**。

它的语义由 attribute 名字前缀对应的 dialect 定义。

例如：

```text
gpu.contained_module
```

这个 attribute 的语义由 `gpu` dialect 定义。

如果一个 dialect 想验证带有自己前缀的 discardable attributes，可以启用以下 hook。

### 12.1 hasOperationAttrVerify

字段：

```tablegen
let hasOperationAttrVerify = 1;
```

生成 hook：

```c++
LogicalResult MyDialect::verifyOperationAttribute(Operation *op,
                                                  NamedAttribute attribute);
```

作用：

```text
验证某个 operation 的 attribute dictionary 中使用的、本 dialect 前缀的 discardable attribute 是否合法
```

例如看到：

```text
my_dialect.some_attr
```

就可以交给 `MyDialect` 验证。

### 12.2 hasRegionArgAttrVerify

字段：

```tablegen
let hasRegionArgAttrVerify = 1;
```

生成 hook：

```c++
LogicalResult MyDialect::verifyRegionArgAttribute(Operation *op,
                                                  unsigned regionIndex,
                                                  unsigned argIndex,
                                                  NamedAttribute attribute);
```

作用：

```text
验证 region entry block argument 对应的 attribute dictionary 中的、本 dialect 前缀的 discardable attribute
```

注意：block argument 本身不一定有 attribute dictionary。通常是某些 operation 代为保存，例如实现 `FunctionOpInterface` 的 operation 会保存函数参数属性。

### 12.3 hasRegionResultAttrVerify

字段：

```tablegen
let hasRegionResultAttrVerify = 1;
```

生成 hook：

```c++
LogicalResult MyDialect::verifyRegionResultAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute attribute);
```

作用：

```text
验证 region result 对应 attribute dictionary 中的、本 dialect 前缀的 discardable attribute
```

这类属性也通常由某些 operation 代为保存，例如函数结果属性。

---

## 十三、Operation Interface Fallback

有些 dialect 是开放生态，不会预先注册所有可能的 operation。

这种情况下，即使某个 operation 没注册，或者没有显式实现某个 `OpInterface`，查询 interface 时也可以回退到 dialect 本身。

字段：

```tablegen
let hasOperationInterfaceFallback = 1;
```

生成 hook：

```c++
void *MyDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                             StringAttr opName);
```

作用：

```text
给定 interface 的 TypeID 和 operation 名字，
由 dialect 返回这个 operation 对应的 interface model
```

这适用于动态 operation 或开放式 dialect。

---

## 十四、默认 Attribute / Type Parser 和 Printer

如果 dialect 注册了 Attribute 或 Type，通常要实现：

```c++
Dialect::parseAttribute
Dialect::printAttribute
Dialect::parseType
Dialect::printType
```

如果 dialect 中所有 attributes/types 都有 mnemonic，MLIR 可以自动生成这些 parser/printer。

字段：

```tablegen
useDefaultAttributePrinterParser
useDefaultTypePrinterParser
```

默认值为 `1`，也就是启用。

如果你要手写 parser/printer，需要显式设为 `0`。

---

## 十五、Dialect-wide Canonicalization Patterns

一般 canonicalization pattern 是 op 级别的。

但有些 canonicalization pattern 适合放到 dialect 级别。例如：

```text
pattern 操作的是某个 interface
pattern 操作的是某个 trait
多个 op 都能复用同一个 canonicalization
```

这时可以启用：

```tablegen
let hasCanonicalizer = 1;
```

生成方法：

```c++
void MyDialect::getCanonicalizationPatterns(RewritePatternSet &results) const;
```

作用：

```text
向 results 中加入 dialect 级别的 canonicalization patterns
```

---

## 十六、为 Dialect Attributes 和 Types 定义 Bytecode 格式

默认情况下，dialect attributes 和 types 的 bytecode 序列化会使用普通 textual format。

如果想要更紧凑的 bytecode 表示，可以给 dialect 添加：

```c++
BytecodeDialectInterface
```

ODS 的 `-gen-bytecode` 可以生成一部分 reader/writer。

示例中，一个 C++ `MemRefType` 可以在 bytecode 中有两个 variant：

```text
MemRefType
  shape: svarint[]
  elementType: Type
  layout: Attribute

MemRefTypeWithMemSpace
  memorySpace: Attribute
  shape: svarint[]
  elementType: Type
  layout: Attribute
```

关键概念：

```text
cType
  表示这些 bytecode variant 对应同一个 C++ type

DialectType
  描述一个 type 的 bytecode 编码形式

printerPredicate
  控制某个 variant 何时被写出

cBuilder
  当 bytecode 字段顺序和 C++ builder 参数顺序不一致时，自定义构造表达式

ReservedOrDead
  保留或废弃的 enum slot，不生成读写或 dispatch 代码

Array
  表示序列化列表，通常先写数量，再写元素
```

生成代码会提供类似以下函数：

```c++
Attribute readAttribute(DialectBytecodeReader &reader) const override;
LogicalResult writeAttribute(Attribute attr,
                             DialectBytecodeWriter &writer) const override;

Type readType(DialectBytecodeReader &reader) const override;
LogicalResult writeType(Type type,
                        DialectBytecodeWriter &writer) const override;
```

这部分属于高级特性，主要在需要自定义 dialect bytecode 体积或性能时使用。

---

## 十七、Defining an Extensible Dialect：定义可扩展 Dialect

可扩展 dialect 指的是：

```text
可以在运行时添加新的 operations、types、attributes 的 dialect
```

这允许用户通过元编程、脚本语言或其他语言定义 dialect 内容，而不需要重新编译 C++。

### 17.1 C++ 中定义 extensible dialect

继承：

```c++
mlir::ExtensibleDialect
```

而不是：

```c++
mlir::Dialect
```

示例：

```c++
class MyDialect : public mlir::ExtensibleDialect {
  ...
};
```

### 17.2 TableGen 中定义 extensible dialect

设置：

```tablegen
let isExtensible = 1;
```

示例：

```tablegen
def Test_Dialect : Dialect {
  let isExtensible = 1;
}
```

之后可以通过：

```c++
llvm::dyn_cast<ExtensibleDialect>(dialect)
```

把普通 `Dialect *` 转成 `ExtensibleDialect *`。

---

## 十八、Defining a Dynamic Dialect：定义动态 Dialect

Dynamic dialect 是运行时定义的 extensible dialect。

它只包含动态：

- operations
- types
- attributes

可以用 `DialectRegistry::insertDynamic` 注册：

```c++
auto populateDialect = [](MLIRContext *ctx, DynamicDialect *dialect) {
  // Register dynamic operations, types, and attributes here.
};

registry.insertDynamic("dialectName", populateDialect);
```

注册到 `MLIRContext` 后，可以加载：

```c++
Dialect *dialect = ctx->getOrLoadDialect("dialectName");
```

---

## 十九、运行时定义 Operation

运行时定义的 operation 用：

```c++
DynamicOpDefinition
```

表示。

创建动态 operation definition 需要：

```text
operation name
  不带 dialect 前缀

dialect
  注册这个 operation 的 dialect

verifier
  验证 operation invariant
```

还可以可选提供：

- parser
- printer
- fold hook
- canonicalization patterns

示例结构：

```c++
StringRef name = "my_operation_name";
Dialect *dialect = ctx->getOrLoadDialect<MyDialect>();

AbstractOperation::VerifyInvariantsFn verifyFn = [](Operation *op) {
  // Verify operation invariants.
};

AbstractOperation::ParseAssemblyFn parseFn =
    [](OpAsmParser &parser, OperationState &state) {
      // Parse operation after operation name has been parsed.
    };

auto printFn = [](Operation *op, OpAsmPrinter &printer) {
  printer << op->getName();
  // Print operation after operation name has been printed.
};

auto foldHookFn = [](Operation *op,
                     ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &result) {
  // Fold implementation.
};

auto getCanonicalizationPatterns =
    [](RewritePatternSet &results, MLIRContext *context) {
      // Add patterns.
    };
```

注册：

```c++
extensibleDialect->registerDynamicOperation(std::move(opDef));
```

注意：

```text
传给 DynamicOpDefinition 的 dialect 应该就是注册这个 operation 的 dialect
```

---

## 二十、使用运行时定义的 Operation

动态 operation 没有专门的 C++ op 类，所以通常通过名字匹配：

```c++
if (op->getName().getStringRef() == "my_dialect.my_dynamic_op") {
  ...
}
```

创建时可以构造 `OperationState`：

```c++
OperationState state(location, "my_dialect.my_dynamic_op",
                     operands, resultTypes, attributes);

rewriter.createOperation(state);
```

这比静态 op 的：

```c++
rewriter.create<MyOp>(...)
```

更底层，也更通用。

---

## 二十一、运行时定义 Type

运行时定义的 type 用：

```c++
DynamicTypeDefinition
```

表示。

注意：

```text
动态 type 的参数只能是一组 Attribute
```

定义一个动态 type 需要：

- type 名字，不带 dialect 前缀
- 注册它的 dialect
- 参数 verifier

可选：

- parser
- printer

示例：

```c++
StringRef name = "my_type_name";
Dialect *dialect = ctx->getOrLoadDialect<MyDialect>();

auto verifier = [](function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> args) {
  ...
};

auto parser = [](DialectAsmParser &parser,
                 llvm::SmallVectorImpl<Attribute> &parsedParams) {
  ...
};

auto printer = [](DialectAsmPrinter &printer,
                  ArrayRef<Attribute> params) {
  ...
};
```

如果省略 parser/printer，默认格式为：

```text
!dialect.typename<arg1, arg2, ..., argN>
```

注册：

```c++
dialect->registerDynamicType(std::move(typeDef));
```

---

## 二十二、解析运行时定义的 Type

TableGen 生成的 `parseType` 可以处理动态 type。

但如果你手写覆盖了 `parseType`，需要自己添加动态 type 支持：

```c++
Type MyDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeTag;
  if (failed(parser.parseKeyword(&typeTag)))
    return Type();

  Type dynType;
  auto parseResult = parseOptionalDynamicType(typeTag, parser, dynType);
  if (parseResult.has_value()) {
    if (succeeded(parseResult.getValue()))
      return dynType;
    return Type();
  }

  ...
}
```

核心逻辑：

```text
先解析 type 名字
尝试按 dynamic type 解析
如果成功，就返回动态 type
否则继续走静态 type 解析逻辑
```

---

## 二十三、使用运行时定义的 Type

动态 type 是：

```c++
DynamicType
```

的实例。

创建：

```c++
auto typeDef = extensibleDialect->lookupTypeDefinition("my_dynamic_type");
ArrayRef<Attribute> params = ...;
auto type = DynamicType::get(typeDef, params);
```

如果一个 `Type` 已知是动态 type，可以 cast：

```c++
auto dynType = cast<DynamicType>(type);
auto typeDef = dynType.getTypeDef();
auto args = dynType.getParams();
```

---

## 二十四、运行时定义 Attribute

动态 attribute 与动态 type 很像。

运行时定义 attribute 用：

```c++
DynamicAttrDefinition
```

表示。

注意：

```text
动态 attribute 的参数也只能是一组 Attribute
```

定义需要：

- attribute 名字，不带 dialect 前缀
- 注册它的 dialect
- 参数 verifier

可选：

- parser
- printer

如果省略 parser/printer，默认格式为：

```text
!dialect.attrname<arg1, arg2, ..., argN>
```

注册：

```c++
dialect->registerDynamicAttr(std::move(attrDef));
```

---

## 二十五、解析运行时定义的 Attribute

如果你覆盖了 `parseAttribute`，需要自己加动态 attribute 解析逻辑：

```c++
Attribute MyDialect::parseAttribute(DialectAsmParser &parser,
                                    Type type) const {
  StringRef attrTag;
  if (failed(parser.parseKeyword(&attrTag)))
    return Attribute();

  Attribute dynAttr;
  auto parseResult = parseOptionalDynamicAttr(attrTag, parser, dynAttr);
  if (parseResult.has_value()) {
    if (succeeded(*parseResult))
      return dynAttr;
    return Attribute();
  }

  ...
}
```

核心逻辑与 dynamic type 类似：

```text
先解析 attribute 名字
尝试按 dynamic attribute 解析
如果成功，就返回动态 attribute
否则继续走静态 attribute 解析逻辑
```

---

## 二十六、使用运行时定义的 Attribute

动态 attribute 是：

```c++
DynamicAttr
```

的实例。

创建：

```c++
auto attrDef = extensibleDialect->lookupAttrDefinition("my_dynamic_attr");
ArrayRef<Attribute> params = ...;
auto attr = DynamicAttr::get(attrDef, params);
```

如果一个 `Attribute` 已知是动态 attribute，可以 cast：

```c++
auto dynAttr = cast<DynamicAttr>(attr);
auto attrDef = dynAttr.getAttrDef();
auto args = dynAttr.getParams();
```

---

## 二十七、Extensible Dialect 的实现细节

### 27.1 ExtensibleDialect 的角色

Extensible dialect 负责持有运行时定义的：

- operations
- types
- attributes

并提供查询和访问方法。

为了能把普通 `Dialect` cast 回 `ExtensibleDialect`，MLIR 给 extensible dialect 实现了：

```c++
IsExtensibleDialect
```

cast 时本质上是检查 dialect 是否实现这个 interface。

### 27.2 Dynamic Operation 的表示与注册

MLIR 中 operation 的抽象定义由：

```c++
AbstractOperation
```

表示。

动态 operation 和 C++ 静态 operation 一样，最终也是通过：

```c++
AbstractOperation::insert
```

注册到 dialect。

区别是：

```text
静态 operation 有 C++ 类，可以用类的 TypeID
动态 operation 没有 C++ 类，所以需要运行时创建 TypeID
```

这个运行时 `TypeID` 由：

```c++
TypeIDAllocator
```

生成。

### 27.3 Dynamic Type 的表示与注册

Type 和 operation 不同。Type 需要：

```text
storage class
  存储 type 参数

wrapper/accessor class
  提供访问 storage 的接口
```

动态 type 使用：

```c++
DynamicTypeStorage
DynamicType
```

`DynamicTypeStorage` 保存：

- 参数列表，也就是 `ArrayRef<Attribute>`
- 指向 type definition 的指针

动态 type 通过：

```c++
Dialect::addType
```

注册。

每个动态 type 都需要一个运行时生成的 `TypeID`。

由于不同 dynamic type 拥有不同 `TypeID`，不能简单用某一个固定 `TypeID` 判断它是否是 `DynamicType`。因此 MLIR 给 dynamic type 加了：

```c++
IsDynamicTypeTrait
```

将 `Type` cast 成 `DynamicType` 时，本质上会查询这个 trait。

---

## 二十八、学习重点总结

这篇文档的主线可以分成两部分。

第一部分：静态 dialect 定义。

```text
Dialect.td
  -> name
  -> cppNamespace
  -> summary / description
  -> dependentDialects
  -> hasConstantMaterializer
  -> parser/printer
  -> canonicalizer
  -> bytecode
```

第二部分：运行时可扩展 dialect。

```text
ExtensibleDialect
DynamicDialect
DynamicOpDefinition
DynamicTypeDefinition
DynamicAttrDefinition
```

初学时优先掌握：

```text
1. Dialect 是 MLIR 的扩展单位
2. Dialect 通常用 TableGen 定义
3. name 决定 IR 中的 dialect 前缀
4. cppNamespace 决定 C++ 代码所在命名空间
5. initialize 负责注册 op/type/attr/interface
6. dependentDialects 用于声明依赖
7. hasConstantMaterializer 支持 folding 后常量物化
8. parser/printer 可以默认生成，也可以自定义
```

后半部分的 extensible/dynamic dialect 属于高级功能，通常在需要运行时定义 IR 组件时才会用到。

一句话总结：

> 定义 Dialect，就是在 MLIR 中注册一套新的 IR 语言空间；TableGen 描述它的静态结构，`initialize()` 把它注册到上下文中，而 extensible dialect 进一步允许在运行时扩展这套语言空间。
