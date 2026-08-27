# Phase 2 Exit Gate — 门禁问答

本文件是 Phase 2（MLIR Transformation Bridge,Week 5–10）的**通关门禁**。
与每章末尾的周 Exit Gate（能力清单）不同,这里是一组**必须当面讲清楚的问题**——
周 Gate 检查"你会不会做",本 Gate 检查"你是不是真的懂"。

## 使用规则

1. **闭卷**。回答时不得翻看任何笔记、源码、历史输出。
2. **限时**。每题 2–3 分钟,用嘴讲或用笔画出 IR,不能只说关键词。
3. **逐题评分**。每题附「通过标准」,答案必须覆盖其中**全部**要点,缺一项即视为该题不过。
4. **不过即回炉**。任何一题不过 → 回到对应章节重读,重做该周实验,一周后再来,不进入 Phase 3。
5. **真做,不做假**。现场验证任务必须实际执行命令,不能背输出。

## 评分表

| 组 | 题目数 | 通过 | 待补 | 结论 |
|---|---|---|---|---|
| Week 5 SSA/CFG/Dominance | 6 | ☐ | ☐ | 未通过 / 通过 |
| Week 6 Rewrite/Fold/Driver | 6 | ☐ | ☐ | 未通过 / 通过 |
| Week 7 Producer–Consumer Chain | 6 | ☐ | ☐ | 未通过 / 通过 |
| Week 8 DialectConversion | 6 | ☐ | ☐ | 未通过 / 通过 |
| Week 9 Transform 与 Tiling | 6 | ☐ | ☐ | 未通过 / 通过 |
| Week 10 参数化 Tiling | 6 | ☐ | ☐ | 未通过 / 通过 |
| 结项综合（现场演示） | 4 | ☐ | ☐ | 未通过 / 通过 |

**最终判定:七组全部通过** → 在下方签名,方可进入 [Phase 3: Triton Compiler Internals](../../Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md)。

---

## Week 5 — SSA、CFG、Dominance 与 Pass Anchor

对应章节 [01_SSA_CFG_Dominance_and_Pass.md](01_SSA_CFG_Dominance_and_Pass.md),实验 `passes/Phase2Inspection.cpp`。

- [ ] **Q5.1** 对着 `Demo/inputs/01-inspection.mlir`,逐一说出:每个 Value 的 owner、类型、defining op、users。
  - 通过标准:能区分 block argument 与 op result;能说出至少一个 multi-use 的 Value;不借助 inspection pass 输出也能指对。
- [ ] **Q5.2** OpResult 与 BlockArgument 分别由谁定义?它们各自可以属于哪个 Region/Block?
  - 通过标准:OpResult 由 Operation 定义、BlockArgument 由 Block 定义;能说出 BlockArgument 是进入 block 的入口,op 的 operand 可以来自二者任一。

- [ ] **Q5.3** `scf.if` 的 results、`scf.for` 的 iter_args / yield / results 之间如何对应?用真实 IR 举例。
  - 通过标准:能画出 scf.for 的 entry 到 yield 再到 result 的 SSA 路径;能解释为什么 scf.if 的 result 必须由所有分支 yield 统一。
- [ ] **Q5.4** CFG、SSA、dominance 各自回答什么问题?三者如何相互配合?
  - 通过标准:CFG 管控制流、SSA 管值关系、dominance 管"定义先于使用"的合法性;能说出 dominance 对 SSA 合法性的作用。
- [ ] **Q5.5** 什么是 pass anchor?FuncOp 锚定与 ModuleOp 锚定的 pass 有何区别?为什么 `phase2-inspection` 锚在 `func.func`?
  - 通过标准:能解释 anchor 决定 pass 在哪个 op 上运行、嵌套关系如何决定执行次数;能说清锚在 func.func 时每个函数各跑一次。
- [ ] **Q5.6** analysis invalidation 是什么?为什么 pass 修改 IR 后 analysis 必须失效?
  - 通过标准:能举出具体 analysis（如 dominance）被改坏的例子;能说出 pass manager 何时保存/失效分析缓存。

## Week 6 — Rewrite、Fold 与 Driver:到底是谁改了 IR

对应章节 [02_Rewrite_Fold_and_Driver.md](02_Rewrite_Fold_and_Driver.md),实验 `passes/Phase2AddZero.cpp`。

- [ ] **Q6.1** pattern、rewriter、driver 各承担什么?到底是谁在改 IR?
  - 通过标准:pattern 只负责匹配并决定改什么;rewriter 执行修改并维护 use-list;driver 调度 pattern 的遍历与迭代。
- [ ] **Q6.2** `replaceOp` 具体做了什么?DCE、fold 与 rewrite 的分工是什么?
  - 通过标准:replaceOp 重定向 uses、删除 op;fold 在常量/可折叠时替换;DCE 清除无 uses 的产物;能按时间线说出三者先后。
- [ ] **Q6.3** 怎么证明"是自定义 pattern 而不是其他机制"改的 IR?如何做 red/green 实验?
  - 通过标准:说出基线 green → 移除 pattern red → 恢复 green 的完整步骤;知道要固定 pipeline 与输入。
- [ ] **Q6.4** pattern 匹配成功但 rewrite 返回 failure,会发生什么?
  - 通过标准:该 pattern 视为未匹配,IR 不变,driver 继续尝试其他 pattern;不会留下半截修改。
- [ ] **Q6.5** PatternBenefit 如何影响调度?为什么 pass 运行两次不会产生额外变化?
  - 通过标准:能解释 greedy driver 按 benefit 排序与 worklist 迭代;用终止性/单调度量解释二次运行稳定。
- [ ] **Q6.6** 边界测试:multi-use、non-zero 常量、不同位宽、shaped value 各应该是什么结果?为什么?
  - 通过标准:能逐一说出测试预期与理由——multi-use 仍可替换但需重定向全部 uses;non-zero 不应被消掉;位宽不一致不能 fold;shaped 值 fold 需逐元素。

## Week 7 — Producer–Consumer Chain:先匹配完整,再修改

对应章节 [03_Producer_Consumer_Chain.md](03_Producer_Consumer_Chain.md),实验 `passes/Phase2FuseChain.cpp`。

- [ ] **Q7.1** 从 ReLU consumer 出发,如何沿 defining op 找到 bias add 与 matmul?
  - 通过标准:说出 `getDefiningOp` 回溯链;指出这条链的终点（没有可回溯的 producer）如何判断。
- [ ] **Q7.2** matcher 在 mutation 前必须检查哪些条件?为什么必须"先匹配完整,再修改"?
  - 通过标准:能列出 operand、maps、iterators、dtype、uses、effects 等检查;说明部分修改会导致 worklist 中 stale 状态。
- [ ] **Q7.3** 为什么正例只融合 elementwise bias + ReLU,而 matmul reduction 保持独立?
  - 通过标准:能解释 reduction dims 的依赖使 matmul 不能简单并入 elementwise 链;融合边界来自计算语义而非代码巧合。
- [ ] **Q7.4** multi-use、错误 layout/dtype、非 ReLU、side-effect 四类负例各自为什么必须拒绝?IR 保持原样吗?
  - 通过标准:每类说出一个具体原因（use 重定向、map 不对齐、语义不同、effect 不能乱动）;负例经测试确认原 IR 不变。
- [ ] **Q7.5** PatternBenefit 为什么不保证全局最优?
  - 通过标准:greedy 局部选择、匹配顺序依赖 worklist;能举例两个互斥 pattern 时结果取决于调度。
- [ ] **Q7.6** 用什么证明 rewrite 终止?单调度量是什么?
  - 通过标准:能说出度量（如 op 数量或结构复杂度）严格单调下降;说明为何不会无限循环。

## Week 8 — DialectConversion:Legality、Adaptor 与 Materialization

对应章节 [04_Dialect_Conversion_Bridge.md](04_Dialect_Conversion_Bridge.md),实验 `passes/Phase2ConvertToLLVM.cpp`。

- [ ] **Q8.1** legal / illegal / unknown 三种状态的区别?operation action 与 dialect action 的优先级?
  - 通过标准:能说出三者含义;specific（op 级）优先于 dialect 级;unknown 需 pattern 判定。
- [ ] **Q8.2** static、dynamic、recursive legality 如何配置?各自适用场景?
  - 通过标准:说出 `markLegal` / `markIllegal` / `addIllegalDialect` 与动态回调的用法;递归 legality 作用于 nested 内容。
- [ ] **Q8.3** 为什么 result、operand、BlockArgument、function signature 要一起转换?
  - 通过标准:指出类型不一致会导致 verifier 失败;signature 转换后 entry block arguments 也变;能画出转换的联动关系。
- [ ] **Q8.4** source / target / argument materialization 的区别?什么时候需要 materialize?
  - 通过标准:source 出现在类型不匹配的 uses 处、target 出现在 conversion 后需要转换回去的位置;能举出何时产生 materialization 的 IR 场景。
- [ ] **Q8.5** 用同一个 unknown op,如何演示 partial 与 full conversion 的差异?
  - 通过标准:partial 下 unknown op 保留并可能 materialize;full 下必须全部转换否则报错;能指出诊断输出的关键差异。
- [ ] **Q8.6** conversion failure 之后会发生什么?会不会自动回滚外部状态?
  - 通过标准:说清 conversion 框架回滚的是 IR 修改,不是任意外部副作用;不把 failure 误认为万能回滚。

## Week 9 — Transform Dialect 与 Tiling Interfaces

对应章节 [05_Transform_and_Tiling.md](05_Transform_and_Tiling.md),实验 `inputs/07-tile-payload.mlir` ~ `10-tile-boundary.mlir`。

- [ ] **Q9.1** payload IR、Transform IR、handle、TransformState 四者的关系?
  - 通过标准:能说出 transform script 是 IR、handle 指向 payload 中的 op;TransformState 是二者之间的绑定与追踪状态。
- [ ] **Q9.2** DestinationStyleOpInterface、TilingInterface、Reify 接口的分工?
  - 通过标准:destination-style 管输出即输入;tiling 接口回答"如何切分";reify 回答"形状如何计算";能各举一 op。
- [ ] **Q9.3** 为什么 tile 变换是"按接口重建 IR",而不是文本替换?
  - 通过标准:指出 tiling 按接口生成 loops、slices、tiled op 与 insert ops;结构由接口语义决定,输入文本只是载体。
- [ ] **Q9.4** 整除与 `127×251×509` 非整除场景的差异?非整除时 slices 如何生成?
  - 通过标准:说出非整除时最后一个 tile 变小、需要 `tensor.extract_slice` 的边界条件;指出边界 tile 大小不是均匀 tile。
- [ ] **Q9.5** named sequence 如何精确匹配目标 matmul?匹配失败会怎样?
  - 通过标准:match 条件（名称、rank、结构）;失败时给出明确诊断而不是静默不执行。
- [ ] **Q9.6** handle 失效是什么意思?什么时候 handle 会失效?
  - 通过标准:op 被改写/删除后旧 handle 不再指向有效对象;说出避免失效的写法（transform 顺序、使用返回值）。

## Week 10 — 参数化 Tiling 与 IR Evolution

对应章节 [06_Parameterized_Tiling_Evolution.md](06_Parameterized_Tiling_Evolution.md),实验 `passes/Phase2MatmulTiling.cpp`、`inputs/11-tile-even.mlir`、`12-tile-not-even.mlir`。

- [ ] **Q10.1** 从空文件实现带 `tile-m/n/k` 的 matmul tiling pass,讲出完整骨架（注册、选项、获取 target、调用 tiling、更新 uses）。
  - 通过标准:能无笔记写出 pass 骨架与关键调用（`tileUsingSCF`、`SCFTilingResult`）;指出 pass options 如何读取。
- [ ] **Q10.2** 为什么参数验证必须在任何 mutation 之前完成?`tile=0` 会怎样?
  - 通过标准:先验证避免失败时留下部分修改;tile=0 导致非法/无限循环或空 tile,必须在入口拒绝。
- [ ] **Q10.3** `SCFTilingResult` 各字段是什么?`replacements` 如何更新原 uses?
  - 通过标准:能列出 loops/slices/tiledOps/insertOps 并说出用途;replacements 把原结果替换为新 SSA 值。
- [ ] **Q10.4** transformation failure 是否可能留下部分 IR?如何避免?
  - 通过标准:指出不是所有失败都原子;验证参数 + 先匹配后改写降低风险;能说明发生部分修改时如何诊断。
- [ ] **Q10.5** folding / pattern rewrite / conversion / tiling / legality 五个概念,用一句话各回答什么问题?
  - 通过标准:五者各自指向"常量化简 / 结构改写 / 跨 dialect 下降 / 结构化切分 / 合法性判定",不得混为一谈。
- [ ] **Q10.6** tiling 对后续 bufferization / vectorization 有什么影响?哪些 analysis 被 invalidate?
  - 通过标准:说清 tiling 产生可 buffer 的 loop 结构、小 tiles 利于 vectorize;指出 dominance/loop-related analysis 等失效。

---

## 结项综合(现场演示)

对应官方计划 [Phase 2 Exit Gate](../../Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md)。

- [ ] **Q11.1 从零实现**:现场新建空文件,写出参数化 matmul tiling pass 并可编译运行。
  - 通过标准:不复制旧代码;最终命令、输出与 `tests/11-tile-even.check` 一致。
- [ ] **Q11.2 全量测试**:正例、负例、边界测试全部执行一遍,全部通过。
  - 通过标准:正例结构正确;tile=0、缺目标、非 rank-2、非整除负例按预期拒绝或保持原 IR;二次运行稳定。
- [ ] **Q11.3 概念辨析**:folding / pattern / conversion / tiling / legality 的区别,各用一个真实 IR 例子说明。
  - 通过标准:五例之间互不混淆,能说明各自在 Phase 2 中出现在哪一周。
- [ ] **Q11.4 演进链路**:沿保存的 IR snapshots 说明 tiling → bufferization → vectorization 的影响链。
  - 通过标准:能指出 tiling 后哪些结构变对 buffer 友好、哪些变对 vectorize 友好;能指出 invalidated analysis。

---

## 现场验证命令(必跑)

以下命令全部实际执行,输出需能当场解释:

```bash
cd ~/llvm-project
PLUGIN=./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/libPhase2Passes.so
IN=./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo

# 1. inspection:解释 SSA/CFG/dominance 输出
./build/bin/mlir-opt --load-pass-plugin=$PLUGIN \
  --pass-pipeline='builtin.module(func.func(phase2-inspection))' \
  $IN/inputs/01-inspection.mlir

# 2. add-zero:正例 + 负例(移除 pattern 后应 red)
./build/bin/mlir-opt --load-pass-plugin=$PLUGIN \
  --pass-pipeline='builtin.module(func.func(phase2-add-zero))' $IN/inputs/02-add-zero.mlir

# 3. fuse-chain:正例 + multi-use 负例
./build/bin/mlir-opt --load-pass-plugin=$PLUGIN \
  --pass-pipeline='builtin.module(func.func(phase2-fuse-chain))' $IN/inputs/03-producer-consumer.mlir

# 4. convert-to-llvm:partial 与 full
./build/bin/mlir-opt --load-pass-plugin=$PLUGIN \
  --pass-pipeline='builtin.module(func.func(phase2-convert-to-llvm))' \
  $IN/inputs/04-convert-partial-full.mlir

# 5. tiling:for / forall / boundary / even / not-even
./build/bin/mlir-opt --load-pass-plugin=$PLUGIN \
  --pass-pipeline='builtin.module(func.func(phase2-matmul-tiling{tile-m=2 tile-n=3 tile-k=4}))' \
  $IN/inputs/11-tile-even.mlir
```

---

## 结论

| 组 | 结果 |
|---|---|
| Week 5–10 全部题目 | ☐ 通过 ☐ 未通过 |
| 结项综合演示 | ☐ 通过 ☐ 未通过 |
| 现场验证命令 | ☐ 通过 ☐ 未通过 |

**门禁判定:☐ 通过 → 进入 Phase 3  ☐ 未通过 → 回炉(记录待补题号:_____)**

签名/日期:____________________

---

上一章:[← Week 10:参数化 Tiling 与 IR Evolution](06_Parameterized_Tiling_Evolution.md) ·
[查看原始 Phase 2 执行计划](../../Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md)
