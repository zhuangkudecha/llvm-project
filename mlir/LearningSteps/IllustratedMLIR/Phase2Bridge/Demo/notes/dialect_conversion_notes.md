# Week 8 复盘：DialectConversion（Mini Arith-to-LLVM）

> 填写说明：先自己回答，再对照实验证据（代码 / 输入 / 输出 / 诊断）核对。答不出的不要空着，写下「卡在哪、打算怎么查」。
> `【Review 补充】` 是复习后的校正/参考答案，不要背，理解后用自己的话复述。

## 实验概要

- 日期：
- Pass 名：`phase2-convert-to-llvm`
- Anchor：`builtin.module`（为什么不是 `func.func`？）
- 模式：full（默认）/ partial（`{partial=true}`）
- 涉及 dialect：`arith` / `func` / `cf` → `LLVM`

## 复现命令

```bash
# 构建
cd Demo && ninja -C build

# 正例 full
mlir-opt --load-pass-plugin=.../libPhase2Passes.so \
  --pass-pipeline='builtin.module(phase2-convert-to-llvm)' \
  inputs/04-convert-to-llvm.mlir

# partial/full 对比
mlir-opt ... --pass-pipeline='builtin.module(phase2-convert-to-llvm{partial=true})' \
  inputs/04-convert-partial-full.mlir

# 负例
mlir-opt ... --pass-pipeline='builtin.module(phase2-convert-to-llvm)' \
  inputs/06-convert-invalid.mlir -verify-diagnostics -split-input-file
```

---

## Exit Gate 8 问

### Q1. 能独立配置 static、dynamic、recursive legality 吗？

提示：`ConversionTarget::addLegalDialect/addIllegalDialect/addDynamicallyLegalOp/markOpRecursivelyLegal`；每种对应什么场景，recursive 和「Region 自动合法」的区别。

- 我的理解：
可以，通过ConversionTarget配置，含义不太清晰

- 【Review 补充】
三种 legality：
- **static**：`addLegalDialect` / `addIllegalDialect` / `addLegalOp` / `addIllegalOp`——无条件判定，不看 op 状态。
- **dynamic**：`addDynamicallyLegalOp<OpTy>(callback)`——按 op 当前状态运行时判断（例：签名是否已转换）。
- **recursive**：`markOpRecursivelyLegal<OpTy>(callback)`——满足条件的容器 op，其整个 nested region 不再逐个检查，整体视为 legal。
- 关键区分：recursive **不是**「Region 自动合法」，而是显式告诉 driver「这棵子树我全认」。不设 recursive 时，即使容器 op legal，driver 仍会逐个检查内部 op。

本次实验只用了 static（`addLegalDialect<LLVM>` / `addIllegalDialect<arith/func/cf>` / `addLegalOp<ModuleOp>`）。可以自己加一个 `addDynamicallyLegalOp<func::FuncOp>` 练手。

### Q2. 能说明 legal、illegal、unknown 以及 operation/dialect action 的优先级吗？

提示：operation-specific action > dialect action > unknown-op callback > partial/full 兜底。用 `tensor.empty` 说明 unknown。

- 我的理解：
legal = illegal > unknown ; operation 先于 dialect

- 【Review 补充】
优先级**不是**「legal 与 illegal 谁高」，而是「配置的具体程度」：
1. **operation-specific**（`addLegalOp<X>` / `addIllegalOp<X>` / `addDynamicallyLegalOp<X>`）——最高，覆盖 dialect action
2. **dialect action**（`addLegalDialect<X>` / `addIllegalDialect<X>`）
3. **unknown-op 动态回调**（`markUnknownOpDynamicallyLegal(callback)`）
4. 都没有 → 真 **unknown** → 由 partial/full 模式兜底

legal 和 illegal 是同一层、只是方向不同；op-specific 的 illegal 可以覆盖 dialect 的 legal。例：`addLegalDialect<arith>` + `addIllegalOp<arith.addi>` → addi 仍必须转换。
`tensor.empty` 是 unknown：tensor 方言既没标 legal 也没标 illegal，也没配 unknown-op callback。

- 实验证据：本次 target 只配置了 LLVM legal、arith/func/cf illegal、ModuleOp legal；`tensor.empty` 属于第 4 层。

### Q3. 能配置 LLVMTypeConverter 与 Arith/Func/CF patterns 吗？

提示：三个 `populate*ConversionPatterns` 共享同一个 `LLVMTypeConverter`；TypeConverter 的生命周期/配置。

- 我的理解：
没看懂问题，配置LLVMTypeConverter是什么意思

- 【Review 补充】
「配置 LLVMTypeConverter」是指：
1. **构造实例**：`LLVMTypeConverter typeConverter(&ctx);`
2. **默认类型映射**：`index→i64`、定长整型/浮点保持不变、函数类型、定长 vector、memref（可选）等。
3. **可自定义**：`addConversion(...)` 加类型映射；`addSourceMaterialization` / `addTargetMaterialization` / `addArgumentMaterialization` 加桥接回调（对应 Q5 三类）。
4. **传给 patterns**：三个 `populate*ConversionPatterns(typeConverter, patterns)` 共享**同一个** typeConverter——pattern 内部用它决定 target op 的类型。
5. **生命周期**：在 `runOnOperation` 里创建，活到 conversion 结束；patterns 持有它的引用，所以它必须比 patterns 活得久。

本实验没自定义任何 conversion/materialization，全用默认行为——这就是「最低配置」。

### Q4. 为什么 result、operand、BlockArgument、function signature 要一起转换？

提示：SSA 图一致性；函数签名变了入口 BlockArgument 必须同步；branch successor operands。

- 我的理解：

- 【参考答案】
因为 SSA 图必须保持类型一致，conversion 不是「只改产生结果的 op」：
- result 变了 → 使用它的 operand 要跟着变
- operand 变了 → producer 的 result 要对齐
- func.func 签名变了（如 i32→i64）→ 入口 **BlockArgument 类型必须同步**（region signature conversion）
- 分支（cf.br / scf.if 等）传入 successor 的 operands 必须与新 BlockArgument 对齐
- return 的 operand 要和新签名对齐

- 实验证据：`populateFuncToLLVMConversionPatterns` 把 `func.func` 转 `llvm.func` 时同时转换了函数类型和入口 block argument；`llvm.return` 的 operand 随之匹配。若只转 result 不转 operand，LLVM verifier 会立刻报类型不匹配。

### Q5. 能区分 source / target / argument materialization 吗？

提示：方向不同——旧 producer 连新 consumer / 新 producer 连旧 consumer / 新 BlockArgument 附近建旧视图。`builtin.unrealized_conversion_cast` 何时出现。

- 我的理解：

- 【参考答案】
三类桥接方向不同：
- **Target materialization**：source Value → target Value。旧 producer（还没转）连已转换的 consumer 时出现。
- **Source materialization**：target Value → source Value。新 producer（已转）连尚未转换的 consumer 时出现。
- **Argument materialization**：在新 BlockArgument 附近建立旧类型视图。region signature 已转换但内部旧 user 还在用旧类型时出现。

没有真实转换可用时，driver 临时用 `builtin.unrealized_conversion_cast` 保持 SSA 类型合法，最终 pipeline 要 reconcile 掉。

- 实验证据：本实验**没见到 cast**——arith/func/cf 的所有类型（i32 等）都能 1:1 转换，没有类型边界。思考：如果加入 LLVMTypeConverter 不认识的类型（如 tensor），就会触发 materialization。

### Q6. 能用同一个 unknown op 正确展示 partial/full 差异吗？

提示：为什么不能用显式 illegal op 演示？你的 `04-convert-partial-full.mlir` 为什么 func.func 签名必须可转换？

- 我的理解：
full 中没有unknown op, partial 对于没有声明为legal的op，也没有声明为illegal 的op会使用unknown op替代，pass的锚点为module.op 因此func.func也必须可转换。

- 【Review 补充】（重点纠正两处误区）
1. **「full 中没有 unknown op」不准确**。full 也有 unknown op，区别在**处置**：partial 允许 pre-existing unknown op 原样残留；full 要求包括 unknown 在内的所有 op 最终都 legal，否则失败。
2. **「partial 会用 unknown op 替代」表述不准确**。op 本身没有 legality action 就是 unknown，不是被「替代」成 unknown。
3. **「anchor 是 module 所以 func.func 必须可转换」逻辑不完整**。func.func 必须可转换的**真正原因**：它被 `addIllegalDialect<func::FuncDialect>()` **显式标记 illegal**，显式 illegal 在两种模式下都必须 legalize。anchor 只决定作用范围（整个 module），不决定某个 op 的合法性要求。
4. 正确的差异演示：unknown op（`tensor.empty`）放函数体内、签名用可转换类型（i32）。full → 报 `tensor.empty` 失败；partial → 保留 `tensor.empty`。

- 实验证据（两种模式）：
  - full：`error: failed to legalize operation 'tensor.empty'`（在 `%t = tensor.empty()` 处）
  - partial：`llvm.func` 内保留 `%0 = tensor.empty()`，arith 转成 `llvm.mlir.constant` / `llvm.return`
  - 额外观察：`llvm.add %arg0, 0` 被 fold 成 `%arg0`（x+0=x），conversion 过程会 fold 新 op

### Q7. 正例证明目标 illegal dialect 消失，负例诊断可验证？

提示：`CHECK-NOT: arith.` / `CHECK-NOT: func.func` / `CHECK: llvm.func`；`-verify-diagnostics` 的 expected-error 匹配。

- 我的理解：
没看懂问题

- 【Review 补充】
问题问的是：**你能否用测试「证明」conversion 做对了？**分正例、负例两半：
- **正例**：用 FileCheck 断言 illegal dialect **消失**。不能只查「出现了 llvm.add」——还要 `CHECK-NOT: arith.` / `CHECK-NOT: func.func` / `CHECK-NOT: cf.`，否则残留的 illegal op 没被发现，full conversion 其实没完成。
- **负例**：用 `-verify-diagnostics`，在 mlir 文件里写 `expected-error@+1 {{...}}`；运行时核对实际诊断与 expected 是否匹配，exit 0 = 全部匹配成功。

- 实验证据：`inputs/06-convert-invalid.mlir` 两条负例（tensor.empty 无法 legalize；签名不可转换导致 func.func 失败）都验证通过，exit 0。

### Q8. 为什么不能把 conversion failure 误认为任意外部状态自动 rollback？

提示：driver 对受控 rewrite 有回滚策略，但日志、计数器、外部容器不在其中；pattern 应在 match 阶段完成所有只读检查。

- 我的理解：

- 【参考答案】
conversion driver 对**受它控制的 IR rewrite** 有延迟/回滚策略，但不能推广到框架外：
- 不保证恢复的：日志文件、全局计数器、外部容器、pattern 自己做的直接 mutation。
- pattern 应把**所有 IR mutation 交给 `ConversionPatternRewriter`**，并在 match 阶段完成全部只读检查（类型、use、side effect），不要在 rewrite 中途发现条件不满足。
- 一句话：driver 只保证「IR 层面受控修改」的一致性，不保证框架外副作用。

---

## 追加思考（对应计划 Week 8 周日「写转换不变量」）

### A1. 哪些 dialect/op 在 pass 后必须全部消失？

- 【参考答案】`arith.*`、`func.*`（func.func 等）、`cf.*` 必须全部消失（被显式 illegal）；`LLVM.*` 出现；`builtin.module` 保留。注意是「这些 dialect 的所有 op 消失」，不是「所有非 LLVM 的东西消失」。

### A2. 容器 op（如 builtin.module）为什么可能保持 legal？

- 【参考答案】容器 op 是「承载者」，本身没有需要 lower 的语义，target IR 需要它作为根。`addLegalOp<ModuleOp>` 只表示它自己合法，**不代表内部 op 自动合法**——内部 illegal op 仍会被检查（除非 `markOpRecursivelyLegal`）。

### A3. TypeConverter 为什么不仅转换 result type？

- 【参考答案】类型转换是全局一致性问题。function signature、BlockArgument、operand、result、branch successor 必须形成一致的 SSA 图；TypeConverter 的映射规则对所有类型位置生效，不只是 result。

### A4. conversion failure 后能否假设 IR 自动 rollback？

- 【参考答案】不能。IR 层面受控 rewrite 可回滚，但框架外副作用（日志/计数/外部容器）不会自动恢复。详见 Q8。

---

## 正负例记录

### 正例（04-convert-to-llvm.mlir）

- before（关键 op）：`func.func @main(%a: i32, %b: i32) -> i32`；`arith.addi` / `arith.muli` / `arith.constant 1` / `arith.subi` / `return`
- after（关键 op）：`llvm.func @main(%arg0: i32, %arg1: i32) -> i32`；`llvm.add` / `llvm.mul` / `llvm.mlir.constant(1)` / `llvm.sub` / `llvm.return`
- 二次运行是否稳定（fixpoint）：稳定——在结果上再跑一次，diff 为空

### 负例 1（tensor.empty，full 模式）

- 诊断原文：`error: failed to legalize operation 'tensor.empty'`（`%t = tensor.empty() : tensor<4xf32>` 处）
- 为什么失败：tensor 方言无 legality action → unknown；full 模式要求 unknown 也必须 legalize，但没有 pattern/materialization 能转它

### 负例 2（签名不可转换）

- 诊断原文：`error: failed to legalize operation 'func.func'`（`func.func @unconvertible_signature() -> tensor<4xf32>` 处）
- 为什么失败：签名含 `tensor<4xf32>`，默认 LLVMTypeConverter 不处理 tensor → func.func pattern 转不了签名；func.func 被显式 illegal，partial/full 都必须 legalize → 无路可走

---

## 收获与卡点

- 这次实验学到最重要的 1 点：
- 卡住/还没完全理解的地方：
- 下一步想验证什么：
