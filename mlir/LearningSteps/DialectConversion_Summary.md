# MLIR DialectConversion.md - 学习总结

## 概述

本文总结 `mlir/docs/DialectConversion.md` 的核心内容。这个文档讲的是 MLIR 中用于 **跨 dialect 或 dialect 内部合法化转换** 的框架。

一句话理解：

> Dialect Conversion 是 MLIR 用来把“不合法/高层”的 operation 和 type，系统性改写成“目标合法/低层”的 operation 和 type 的基础设施。

它常用于 lowering：

```text
toy / stablehlo / tosa / linalg
  -> affine / scf / arith / memref
  -> llvm / gpu / spirv / nvvm
```

可以把它放在 MLIR lowering 主线里：

```text
PassManager
  -> Conversion Pass
      -> ConversionTarget      定义什么 IR 是合法的
      -> RewritePatternSet     定义非法 op 如何改写
      -> TypeConverter         定义 type 如何转换
      -> applyPartialConversion / applyFullConversion
```

---

## 一、DialectConversion.md 在讲什么

普通 `PatternRewriter` 关注的是：

```text
看到某个局部 IR 形状
  -> 把它改写成另一个局部 IR 形状
```

`DialectConversion` 关注的是更强的目标：

```text
给定一个合法性目标 ConversionTarget
  -> 框架检查哪些 op/type 不合法
  -> 尝试用 conversion patterns 把它们改成合法 IR
  -> 必要时处理 type conversion、block argument、region signature
  -> 最后根据模式判断转换是否成功
```

所以它不只是“重写几个 op”，而是一个带合法性检查的系统化 lowering 框架。

核心组件有三个：

```text
ConversionTarget
  定义最终允许留下哪些 operation/dialect

Rewrite Patterns
  定义非法 operation 如何改写

TypeConverter
  可选，定义 type 如何从 source type 转成 target type
```

---

## 二、三种 Conversion 模式

Dialect conversion 有三种模式，决定“转换到什么程度才算成功”。

### 2.1 Partial Conversion

入口：

```cpp
applyPartialConversion(op, target, patterns);
```

含义：

```text
尽可能把 IR 转成目标合法形式
但允许没有显式标记为 illegal 的已有 op 留下来
```

适合场景：

- 分阶段 lowering
- 输入 IR 中还有未知 dialect
- 只想转换一部分 operation
- 某些 op 暂时允许保留到后续 pass

例如：

```text
toy.compute      illegal，必须转换
func.func        unknown 或 legal，可以保留
arith.addf       legal，可以保留
```

### 2.2 Full Conversion

入口：

```cpp
applyFullConversion(op, target, patterns);
```

含义：

```text
所有输入 operation 最终都必须是 target legal
只要还有任何非法或未知不可接受的 op，就失败
```

适合场景：

- 最终 lowering 到 LLVM / SPIR-V / GPU 等目标 dialect
- 希望转换后 IR 中只剩下明确支持的 operation
- 编译 pipeline 的强边界检查

例如 Toy Ch6 降到 LLVM dialect 时，就更适合 full conversion。

### 2.3 Analysis Conversion

入口：

```cpp
applyAnalysisConversion(op, target, patterns);
```

含义：

```text
只分析哪些 operation 如果真的转换会成功
不实际修改 IR
```

适合场景：

- 预检查 legalizability
- 调试 conversion coverage
- 分析一个 pass 是否有足够 pattern 支持某类 IR

### 2.4 遍历顺序

三种模式中，框架都会以前序遍历 operation：

```text
先检查当前 op
再检查其 region 内部嵌套 op
```

这对 conversion target 的 recursive legality 很重要。

---

## 三、ConversionTarget：定义什么是合法 IR

`ConversionTarget` 是 dialect conversion 的核心。它回答一个问题：

> 转换结束后，哪些 operation / dialect 可以留在 IR 里？

合法性动作主要有三类：

```text
Legal
  所有实例都合法

Dynamic
  某些实例合法，取决于属性、operand、type 或上下文

Illegal
  所有实例都非法，必须被转换掉
```

### 3.1 Legal

表示某个 op 或 dialect 的所有实例都合法：

```cpp
target.addLegalDialect<LLVM::LLVMDialect>();
target.addLegalOp<arith::ConstantOp>();
```

含义：

```text
LLVM dialect 的 op 可以留下
arith.constant 可以留下
```

### 3.2 Dynamic Legality

表示某个 op 或 dialect 只有部分实例合法：

```cpp
target.addDynamicallyLegalOp<func::ReturnOp>(
    [](func::ReturnOp op) {
      return /* 检查 return operand type 是否已转换 */;
    });
```

典型场景：

```text
func.func 本身可以留下
但它的 function type 必须已经从 tensor/memref 转成目标类型
```

或者：

```text
arith.addi 只在 operand 是 i32 时合法
其他整数宽度还需要继续转换
```

### 3.3 Illegal

表示某个 op 或 dialect 必须被转换：

```cpp
target.addIllegalDialect<gpu::GPUDialect>();
target.addIllegalOp<cf::BranchOp, cf::CondBranchOp>();
```

如果转换结束后这些 op 还存在，conversion 失败。

### 3.4 Unknown Operation

既没有显式标记 legal，也没有显式标记 illegal 的 operation，属于 unknown。

它们和 illegal 不一样：

```text
Partial Conversion:
  unknown op 可以保留，除非被明确标记 illegal

Full Conversion:
  unknown op 通常不能作为最终结果保留
```

也可以用：

```cpp
target.markUnknownOpDynamicallyLegal(...);
```

给 unknown op 定义动态合法性。

---

## 四、Recursive Legality：整个子树视为合法

有时某个 operation 一旦合法，它内部 region 里的 operation 也应该整体视为合法，不需要再递归转换。

这就是 recursive legality。

```cpp
target.addLegalOp<MyOp>();
target.markOpRecursivelyLegal<MyOp>();
```

含义：

```text
如果 MyOp 合法
  那么 MyOp 内部嵌套的 op 即使本来 illegal
  也被视为合法，不再强制转换
```

也可以动态决定：

```cpp
target.markOpRecursivelyLegal<MyOp>([](MyOp op) {
  return /* 只有某些 MyOp 的 region 可以跳过 */;
});
```

适合场景：

- 一个容器 op 已经代表了目标 dialect 的合法边界
- 某些 region 是 opaque 的，不希望 conversion 进入内部
- 某些 operation 内部的 IR 不属于当前 lowering 阶段处理范围

---

## 五、Rewrite Pattern：把非法 op 改写成合法 op

定义好 `ConversionTarget` 后，还要提供 legalization patterns：

```text
非法 op
  -> pattern rewrite
      -> 合法 op
```

但有一个重要点：pattern 不一定必须一步生成最终合法 op。

例如 target 只支持：

```text
foo.add
```

你提供两个 pattern：

```text
bar.add -> baz.add
baz.add -> foo.add
```

框架会推导出：

```text
bar.add 最终可以 legalize 到 foo.add
```

也就是说，dialect conversion 可以构建 conversion graph，不要求你为每一对 source/target 都写直接转换。

---

## 六、ConversionPattern：conversion 专用 pattern

普通 `RewritePattern` 的核心接口类似：

```cpp
matchAndRewrite(Operation *op, PatternRewriter &rewriter)
```

conversion 框架提供了专用的 `ConversionPattern`，接口多了 remapped operands：

```cpp
LogicalResult matchAndRewrite(Operation *op,
                              ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter) const;
```

或者在 `OpConversionPattern` 中使用 adaptor：

```cpp
LogicalResult matchAndRewrite(MyOp op,
                              MyOp::Adaptor adaptor,
                              ConversionPatternRewriter &rewriter) const;
```

这里最重要的是：

```text
op.getOperands()
  从正在匹配的旧 op 本身读取 operands
  看到的是原始 IR 视角，类型可能还是 source type

adaptor / operands
  conversion driver 传给当前 pattern 的 operands 视图
  看到的是 conversion 视角，已经考虑了 replacement、TypeConverter 和 materialization
```

写 conversion pattern 时，通常应该使用 adaptor，而不是直接读原始 operands。

可以把两者的职责分开：

```text
op
  用来读取原始 operation 的属性、位置、符号名、原始类型等信息

adaptor
  用来读取“创建目标 dialect op 时应该使用的 operands”
```

例如：

```cpp
LogicalResult matchAndRewrite(ToyPrintOp op,
                              ToyPrintOpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Location loc = op.getLoc();          // 位置来自原始 op，读 op 没问题
  Value input = adaptor.getInput();    // operand 用 adaptor，更符合 conversion 后类型

  rewriter.replaceOpWithNewOp<LoweredPrintOp>(op, input);
  return success();
}
```

不推荐这样写：

```cpp
Value input = op.getInput();
```

除非你明确就是要检查原始 IR 的 source operand 或 source type。否则这个 value 可能还停留在转换前的类型，拿去创建目标 dialect op 时容易类型不匹配。

---

## 七、Remapped Operands / Adaptor 为什么重要

假设原始 IR：

```mlir
%0 = "test.foo"() : () -> i1
"test.bar"(%0) : (i1) -> ()
```

pattern A 把 `test.foo` 改成结果类型不同的 `test.qux`：

```mlir
%1 = "test.qux"() : () -> i2
```

但 `test.bar` 还期待 `i1`。框架不能直接把 `test.bar` 的 operand 从 `%0 : i1` 换成 `%1 : i2`，否则会破坏类型语义。

概念上会出现：

```mlir
%1 = "test.qux"() : () -> i2
%r = builtin.unrealized_conversion_cast %1 : i2 to i1
"test.bar"(%r) : (i1) -> ()
```

之后 pattern B 转换 `test.bar` 时，adaptor 里拿到的可能是最新 replacement value，或者经过 materialization/cast 的值。

更贴近 lowering 的例子：

```mlir
%0 = "toy.transpose"(%arg0)
     : (tensor<2x3xf64>) -> tensor<3x2xf64>
"toy.print"(%0) : (tensor<3x2xf64>) -> ()
```

假设 lowering 目标是把 `tensor` 降成 `memref`。如果 `toy.transpose` 已经先被转换：

```mlir
%1 = "lowered.transpose"(%arg0_memref)
     : (memref<2x3xf64>) -> memref<3x2xf64>
```

接下来转换 `toy.print` 时：

```text
op.getInput()
  可能仍然看到原始输入:
  %0 : tensor<3x2xf64>

adaptor.getInput()
  conversion driver 根据替换关系和 TypeConverter 准备好的输入:
  %1 : memref<3x2xf64>
```

如果你要创建目标 dialect 的 print op，它通常期待的是 `memref`，所以应该使用：

```cpp
Value input = adaptor.getInput();
```

而不是：

```cpp
Value input = op.getInput();
```

所以要记住：

```text
matched op:
  代表原始 IR 形状

adaptor:
  代表 conversion driver 为当前 pattern 准备好的合法化 operand 视图
```

原文还强调：框架只保证 adaptor/operands 里的 **类型契约**，不保证一定是某个具体 SSA value。它可能是直接 replacement value，也可能是临时 `unrealized_conversion_cast` 的结果。

一句话总结：

```text
op 是“正在被替换掉的旧 IR”
adaptor 是“conversion driver 根据当前转换进度给你的新 operands”
```

因此 conversion pattern 的经验规则是：

```text
读属性、location、symbol name:
  从 op 读

读要传给新 op 的 operands:
  从 adaptor 读

确实要判断 source type/source operand:
  才从 op.getOperands() 读
```

---

## 八、Immediate vs Delayed IR Modification

Dialect conversion driver 有两种修改模式：

```text
rollback mode
  默认模式，允许回溯，部分修改延迟到 conversion 结束

no-rollback mode
  不允许回溯，IR 修改立即生效
```

由：

```cpp
ConversionConfig::allowPatternRollback
```

控制。

### 8.1 Rollback mode

rollback mode 的目标是：

```text
当前 legalization 路径走不通时
  可以撤销已经应用的 pattern
  再尝试其他路径
```

因此某些 IR 修改不会立即发生：

- `eraseOp` 只是标记删除，op 仍可能对 pattern/遍历可见
- `replaceOp` 不一定马上替换所有 uses
- `replaceAllUsesWith` 的实际替换可能延迟到最后
- block erasure 和 signature conversion 也可能部分延迟

好处：

- 回滚更容易
- 被删除的 op/block 指针可以在 rollback 后恢复
- pattern 仍能在一定程度上访问原始 IR

代价：

- 编译时间更高
- 内部 bookkeeping 更复杂
- debug 更难，因为 dump 出来的 IR 可能是 old/new IR 混合状态
- 部分连接关系藏在 C++ 内部数据结构里，而不是直接体现在 IR dump 中

### 8.2 No-rollback mode

no-rollback mode 中：

```text
create / insert / replace / erase / RAUW / signature conversion
  基本立即作用于 IR
```

优点：

- 更快
- debug 更直观
- IR dump 更接近真实状态

原文建议：能用 no-rollback mode 时，尽量用 no-rollback mode。

---

## 九、Type Safety：conversion pattern 的类型安全

Dialect conversion 的一个核心职责是：

> 在 type conversion 过程中，不让 operation 的 operand/use 类型被隐式破坏。





```cpp
TypeConverter converter;
converter.addConversion([](FloatType t) {
  return IntegerType::get(t.getContext(), t.getWidth());
});
```

如果某个 op 的 operand 原本是 `f32`，pattern 使用这个 converter 后，adaptor 中对应 value 会是合法化后的 integer 类型。

如果当前没有现成 replacement value，driver 可能会插入 target materialization：

```text
source value
  -> target materialization
      -> pattern 期望的 target type
```

如果 pattern 没有 TypeConverter，driver 只传递最近的 remapped values，不主动套用类型转换规则。

---

## 十、TypeConverter：定义 type 如何转换

`TypeConverter` 用来描述：

```text
source type
  -> target type 或 target types
```

例如：

```text
tensor<4xf32> -> memref<4xf32>
index         -> i64
tuple<T, U>   -> T, U
opaque type   -> 删除
```

它主要有两类功能：

```text
Conversion
  纯类型层面的转换规则，不创建 IR

Materialization
  需要在 IR 中插入 cast/adapter op 时，负责创建转换 op
```

### 10.1 addConversion

常见 1:1 转换：

```cpp
converter.addConversion([](FloatType type) -> std::optional<Type> {
  return IntegerType::get(type.getContext(), type.getWidth());
});
```

1:N 转换：

```cpp
converter.addConversion([](TupleType type,
                           SmallVectorImpl<Type> &results)
                           -> std::optional<LogicalResult> {
  for (Type element : type.getTypes())
    results.push_back(element);
  return success();
});
```

如果转换到自己，表示这个 type 本身合法：

```cpp
converter.addConversion([](Type type) { return type; });
```

### 10.2 Context-aware conversion

conversion function 可以按两类输入区分：

```text
Type -> Type
  context-unaware，不看具体 IR，容易缓存

Value -> Type
  context-aware，可以根据具体 value / IR 决定转换方式
```

context-aware 更灵活，但会影响缓存。原文建议如果需要添加 context-aware conversion，尽量早加入 conversion 列表，因为转换函数按反向顺序应用。

---

## 十一、Materialization：在 source/target 类型之间搭桥

`TypeConverter` 的 conversion 规则本身不能创建 IR。如果转换过程中确实需要一个“桥接值”，就需要 materialization。

### 11.1 Source Materialization

source materialization 用于：

```text
某个 value 已经被替换成 target type
但还有用户需要原始 source type
```

示意：

```text
new value : target type
  -> source materialization
      -> old source type
```

典型场景：

- block argument 被转换成新类型，但仍有原类型用户保留
- block argument 被删除，但仍有用户需要替代值
- operation result 被转换成新类型，但仍有原类型用户保留

### 11.2 Target Materialization

target materialization 用于：

```text
pattern 期望 target type
但当前可用 value 还是 source type 或其他类型
```

示意：

```text
old value : source type
  -> target materialization
      -> pattern 期望的 target type
```

一句话区分：

```text
未转换 op 要继续使用已转换 value:
  需要 source materialization

正在转换的 op 要使用未转换 value:
  需要 target materialization
```

如果需要 materialization 但无法构造，整个 conversion 会失败。

### 11.3 unrealized_conversion_cast

如果：

```cpp
ConversionConfig::buildMaterializations = false
```

driver 会用：

```mlir
builtin.unrealized_conversion_cast
```

代替 type converter 的 materialization callback。

这个 op 常作为 conversion 过程中的临时桥接。正常 pipeline 后续通常会把它折叠掉；如果最后还残留，往往说明某些 type conversion 没有完全闭合。

---

## 十二、Region Signature Conversion

operation result 和 operand 的类型转换相对直接，但 block argument 比较特殊。

原因是：

```text
Block 可以在 conversion 过程中移动到不同 region
Block argument 的语义经常和父 operation 绑定
```

例如：

```text
func.func 的 entry block arguments
scf.for 的 loop-carried arguments
region-based op 的 block arguments
```

这些 argument 的类型转换不能完全靠普通 op pattern 自动推断，通常需要 conversion pattern 显式调用：

```cpp
rewriter.convertRegionTypes(region, typeConverter);
```

或者只转换某个 block 的 signature：

```cpp
rewriter.applySignatureConversion(block, signatureConversion);
```

### 12.1 SignatureConversion

`TypeConverter::SignatureConversion` 描述旧 block 参数如何映射到新 block 参数：

```cpp
TypeConverter::SignatureConversion conversion(oldNumArgs);
```

可以：

```text
把一个旧参数映射成一个或多个新参数
追加新的 block 参数
把旧参数 remap 到已有 replacement value
删除某个旧参数
```

常用接口：

```cpp
conversion.addInputs(origInputNo, newTypes);
conversion.addInputs(newTypes);
conversion.remapInput(origInputNo, newInputNo, newInputCount);
conversion.remapInput(origInputNo, replacementValue);
```

实际含义：

```text
旧 block:
  ^bb0(%arg0: tensor<4xf32>, %arg1: index)

转换后:
  ^bb0(%arg0_new: memref<4xf32>, %arg1_new: i64)
```

或者 1:N：

```text
旧参数:
  %tuple : tuple<i32, f32>

新参数:
  %a : i32
  %b : f32
```

---

## 十三、调试 dialect conversion

可以使用：

```bash
-debug-only=dialect-conversion
```

它会输出树状日志，展示：

- 正在 legalize 哪个 operation
- 是否先尝试 fold
- 应用了哪个 pattern
- pattern 插入、替换、删除了哪些 op
- 新生成的 op 是否 legal
- 为什么 legalization 失败

文档里的例子是：

```text
func.return
  -> 尝试 fold，失败
  -> 应用 pattern
      -> 插入 spirv.Return
      -> 替换 func.return
  -> 检查 spirv.Return
      -> target 标记为 legal
```

注意：这个 debug 选项依赖 LLVM debug-only 支持。如果当前 `mlir-opt` 是 Release 且没启用 assertions/dump，可能不会识别该参数。

---

## 十四、最小使用模板

一个典型 conversion pass 的结构如下：

```cpp
void MyLoweringPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  Operation *op = getOperation();

  ConversionTarget target(*ctx);
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addIllegalDialect<toy::ToyDialect>();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  // typeConverter.addConversion(...);
  // typeConverter.addTargetMaterialization(...);
  // typeConverter.addSourceMaterialization(...);

  RewritePatternSet patterns(ctx);
  patterns.add<MyOpLoweringPattern>(typeConverter, ctx);

  if (failed(applyPartialConversion(op, target, std::move(patterns))))
    signalPassFailure();
}
```

如果是最终目标 lowering，更常见：

```cpp
if (failed(applyFullConversion(module, target, std::move(patterns))))
  signalPassFailure();
```

---

## 十五、和 canonicalize / 普通 PatternRewriter 的区别

### 15.1 canonicalize

`canonicalize` 的目标是清理和规范化：

```text
x + 0 -> x
常量折叠
cast 折叠
trivial control flow 简化
```

它是 best-effort，不保证 pipeline correctness。

### 15.2 普通 PatternRewriter

普通 pattern rewriting 的目标是局部 IR 改写：

```text
match 某个 op
  -> replace/erase/modify
```

它不自带“最终 IR 必须满足某个合法性目标”的框架。

### 15.3 DialectConversion

Dialect conversion 的目标是合法化：

```text
source dialect/type
  -> target dialect/type
```

它额外提供：

- `ConversionTarget`
- legal / illegal / dynamic legality
- partial / full / analysis conversion
- `ConversionPatternRewriter`
- remapped operands / adaptor
- `TypeConverter`
- source/target materialization
- region signature conversion
- rollback / no-rollback conversion control

可以这样对比：

```text
canonicalize:
  让 IR 更干净

PatternRewriter:
  让局部 rewrite 更可控

DialectConversion:
  让跨 dialect/type lowering 有合法性边界和类型安全
```

---

## 十六、学习 checklist

读完这篇文档后，应该能回答：

- `applyPartialConversion` 和 `applyFullConversion` 有什么区别？
- `ConversionTarget` 为什么是 dialect conversion 的核心？
- legal / dynamic legal / illegal / unknown op 有什么区别？
- recursive legality 为什么能让嵌套 region 免于继续转换？
- conversion pattern 中为什么要用 adaptor，而不是直接读 `op.getOperands()`？
- rollback mode 为什么会让 `eraseOp` / `replaceOp` 延迟生效？
- `TypeConverter::addConversion` 和 materialization 有什么区别？
- source materialization 和 target materialization 分别解决什么问题？
- `unrealized_conversion_cast` 为什么会出现在 conversion 过程中？
- 为什么 block argument 需要 region signature conversion？
- final lowering 到 LLVM/SPIR-V 时，为什么通常应该用 full conversion？

---

## 总结

`DialectConversion.md` 的核心是：

```text
ConversionTarget
  定义最终合法 IR

ConversionPattern
  把非法 op 改写成更合法的 op

TypeConverter
  定义类型如何转换，并在必要时 materialize 桥接值

Conversion Driver
  根据 partial/full/analysis 模式驱动整个合法化过程
```

对 AI 编译器来说，Dialect Conversion 是理解 lowering pipeline 的关键基础设施。它把“从高层算子 dialect 降到低层执行 dialect”这件事变成了一个可检查、可组合、类型安全的过程。
