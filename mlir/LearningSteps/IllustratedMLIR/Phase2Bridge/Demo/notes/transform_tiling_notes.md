# Week 9 复盘：Transform Dialect 与 Tiling Interfaces

> 填写说明：先自己回答，再对照实验证据（代码 / 输入 / 输出 / 诊断）核对。答不出的不要空着，写下「卡在哪、打算怎么查」。
> `【Review 补充】` 是复习后的校正/参考答案，理解后用自己的话复述。

## 实验概要

- 日期：
- 工具：`mlir-opt --transform-interpreter`（内置 pass，**不需要 C++ plugin**）
- 策略载体：Transform IR（`.mlir` 文件里的 `transform.named_sequence`）
- 涉及 dialect：`linalg` / `transform` / `scf` / `tensor` / `arith`
- 实验文件：`inputs/07-tile-payload.mlir` ~ `inputs/10-tile-boundary.mlir`，`tests/08~10-*.check`

## 复现命令

```bash
MLIR=/home/fuhao/llvm-project/build/bin
cd .../Phase2Bridge/Demo

# 上游 lit 测试（注意：指源树路径，不是 plan 文档里的 build 路径）
$MLIR/llvm-lit -sv \
  /home/fuhao/llvm-project/mlir/test/Interfaces/TilingInterface/tile-using-scfforall.mlir
$MLIR/llvm-lit -sv \
  /home/fuhao/llvm-project/mlir/test/Dialect/Linalg/transform-op-tile.mlir

# payload 验证（无 transform）
$MLIR/mlir-opt inputs/07-tile-payload.mlir

# for 版 tiling（3D，M/N/K = [64,64,32]）
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/08-tile-for.mlir

# forall 版 tiling（只 tile 并行维，K 传 0）
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/09-tile-forall.mlir

# 非整除边界（127×251×509）
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/10-tile-boundary.mlir

# FileCheck 测试（exit 0 = 通过）
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/08-tile-for.mlir \
  | $MLIR/FileCheck tests/08-tile-for.check
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/09-tile-forall.mlir \
  | $MLIR/FileCheck tests/09-tile-forall.check
$MLIR/mlir-opt --transform-interpreter --mlir-print-local-scope inputs/10-tile-boundary.mlir \
  | $MLIR/FileCheck tests/10-tile-boundary.check
```

---

## Exit Gate 9 问

### Q1. 能区分 payload IR、Transform IR、handle 和 TransformState 吗？

提示：payload 是"被优化的程序"、transform 是"优化策略"、handle 是策略里指向 payload 对象的 SSA value、TransformState 维护映射。用 08 文件的某一具体 handle 举例。

- 我的理解：
payload IR 是真正需要进行优化的对象，Transform IR 是用于编排如何进行优化的IR，handle主要表述了如何处理Transform IR中的SSA, TransformState 不明确。

- 【Review 补充】
payload / transform IR 的区分对。两个要精确化：
- **handle 本身就是 Transform IR 里的 SSA value**（类型 `!transform.any_op`），不是"如何处理 SSA"；它的**含义**是映射到 payload 的一个对象。例：08 里 `%matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1` —— `%matmul` 这个 handle 指向 payload 里那个 `linalg.matmul` op。
- **TransformState 是解释器运行时的簿记**：记录"哪个 handle → 哪些 payload op"的映射表（`MappedValue = PointerUnion<Operation*, Param, Value>`，`include/mlir/Dialect/Transform/Interfaces/TransformInterfaces.h:173`）。match 查询它（`getPayloadOps`），tile_using_for 更新它（`transformResults.set`）。它不在 .mlir 里，在运行时内存里。

### Q2. 能解释 handle 的绑定 / 失效生命周期吗？


提示：`structured.match` 产生 handle → `tile_using_for` 消费它（旧映射失效）→ 产生 `%tiled` / `%loops:3` 新 handle。`getEffects` 里 `consumesHandle` / `producesHandle` 对应什么。

- 我的理解：
Transform IR 中operation 生成的SSA为绑定handle，当handle消费完成后 handle生命周期失效。
- 【Review 补充】
方向对，精确化三步：
1. **绑定 = producesHandle**：transform op 的 result（新 SSA）在 TransformState 里登记指向哪些 payload op。match 产生 `%matmul`（指向 matmul）。
2. **失效 = consumesHandle**：`tile_using_for` 消费 `%matmul`——它把 `%matmul` 指向的 matmul op **替换成** tiled 结构（`rewriter.replaceOp`），旧映射不再指向有效对象，于是失效。
3. **不是 SSA 文本被删**：`%matmul` 文本还在 transform IR 里，失效的是 TransformState 里"handle→payload"这条映射的合法性。
另外：消费的同时会**产生新 handle**——同一 op 既 `consumesHandle(%matmul)` 又 `producesHandle(%tiled, %loops:3)`，映射到新建的 tiled matmul 和三个 scf.for。

### Q3. 能解释 DestinationStyle、Tiling、Reify 三个 interface 的分工吗？

提示：tiling 算法向 op 问三个问题——哪个 operand 是 init？迭代空间多大、怎么建 tile、结果插哪？动态 shape 怎么算？

- 我的理解：
destinationStyle 是用来的表述tiling 后的size，tiling 是真正执行分块的接口，reify 可能是用于分块后重新链接原始的dfg。
- 【Review 补充】
两处需要纠正：
- **DestinationStyleOpInterface 不是"表述 tiling 后的 size"**。它回答：这个 op 的 operands 里**哪些是 input、哪些是 init/destination**。对 08 的 matmul，它告诉算法 `%1`（fill 的结果）是 destination——结果 tile 要插回它。没有它算法不知道往哪写。
- **ReifyRankedShapedTypeOpInterface（及 InferShapedTypeOpInterface）不是"重新链接 DFG"**。它回答：**动态结果的每个维度由哪个 SSA value 决定**。10 里 `tensor<?x?xf32>` 的动态 shape 就是靠它把"tail size"和 SSA 对上。
- Tiling 的说法接近但精确化：**分块算法在 tileUsingSCF 里，TilingInterface 是算法向 op 问问题的接口**。算法问三个问题：`getIterationDomain`（迭代空间多大）、`getTiledImplementation`（给定 offsets/sizes 怎么建 tile）、`getResultTilePosition`（结果 tile 插哪）。分块逻辑不在 op 里，在通用算法里。

### Q4. 为什么 tiling 是"按接口重建 IR"而不是"文本替换"？

提示：源码调用链 `TileUsingForOp::apply → tileUsingSCF → getTiledImplementation`；算法不硬编码 linalg.matmul。

- 我的理解：
简单文本替换可能导致出现语义问题，按照接口重建可以基于语义匹配。
- 【Review 补充】
直觉方向对，补上具体证据让它可验证：
- **算法不认识 linalg.matmul**：`TileUsingForOp::apply`（LinalgTransformOps.cpp:3645）只调 `tileUsingSCF(rewriter, tilingInterface, options)` 再 `replaceOp`；它甚至要先 `dyn_cast<TilingInterface>(op)` 确认目标实现接口，否则直接报错。
- **"按接口"体现在**：迭代空间、tile 怎么建、结果插哪全部来自接口查询。换一个 op（如 conv）只要实现同样接口，同一算法照跑。
- **文本替换为什么不行**：tail 块（10 的 63/59/29）需要 `min(tileSize, bound-offset)` 动态算；operand 切片位置和 result 插回位置依赖 op 语义。这些是"语义"不是"文本模式"，只有接口能提供。

### Q5. 能区分 scf.for 和 scf.forall 的语义差异吗？为什么 K（reduction）维只能用 for tile？

提示：for = 顺序 + iter_args 部分和；forall = 逻辑并行 + shared_outs + in_parallel。参考 08 vs 09 输出。

- 我的理解：
scf.for 是按照维度依次进行遍历，scf.forall 可能可以利用数据局部行

- 【Review 补充】
for 对了一半；forall 的说法（"利用数据局部行"）需要重写：
- **scf.for**：顺序执行 + **iter_args 携带部分和**（08 的 %1→%arg3→%arg5→%arg7）。K 循环每轮把上轮结果当输入——这是 reduction 累加的正确载体。
- **scf.forall**：表达"逻辑并行迭代"，不是"数据局部性"。09 输出：单个 `scf.forall in (2,8)` + `shared_outs(%arg4 = %1)` + `in_parallel { parallel_insert_slice }`。它**没有 iter_args**，每个迭代独立算一个完整 C tile，各写各的不冲突。
- **K 为什么只能用 for tile**：K 是 reduction 维，同一个 C tile 上的累加有数据依赖（后一轮要用前一轮结果）。forall 的迭代之间必须无依赖才能并行；tile K 会引入对共享 C tile 的写冲突，所以 forall 只 tile 并行维（M/N）。这就是 09 传 `[64, 64, 0]` 的原因——K 传 0 表示不 tile。

### Q6. 尾块为什么要动态收缩？127×251×509 的三维尾块分别是什么？

提示：`size = min(tileSize, bound - offset)`；M/N/K 尾块分别 63/59/29。

- 我的理解：
有些时候输入的tensor的形状并不能完整的分块成为目标size，需要size数据才能计算尾块大小
- 【Review 补充】
直觉对。精确化：`size = min(tileSize, bound - offset)`，offset 是当前迭代起点。10 的验证：
- M：127 = 64 + 63 → 第二轮 size 63（`affine.min (-d0 + 127, 64)`）
- N：251 = 64×3 + 59 → 最后一轮 59
- K：509 = 32×15 + 29 → 最后一轮 29
运行时不写死，用 `affine.min` 动态算。不做收缩会越界或算错（slice 的 end = offset + size 超出 bound）。

### Q7. 整除与非整除的 tiling 输出有什么结构差异？

提示：整除版 tile size 是常量（`[64, 32]`），非整除版是 SSA value + `affine.min`，tile 类型变 `tensor<?x?xf32>`。

- 我的理解：

- 【Review 补充】
Q7 空着，补上（对照 10 vs 08）：
- **tile size 表达**：整除版是常量 `[64, 32]`（`extract_slice ... [64, 32]`）；非整除版是 SSA value `[%5, %7]`，前面有 `affine.min (-d0 + 127, 64)` 之类的动态计算。
- **tile 类型**：整除版 `tensor<64x32xf32>`（静态 shape）；非整除版 `tensor<?x?xf32>`（动态 shape）——size 运行时才知道。
- **副产品**：非整除版多出一堆 `affine.apply (d0) -> (d0 - 1)`（slice 展开算 offset+size-1 上界的辅助 op），整除版没有。
- 本质：整除时编译器能静态折叠成常量；非整除必须把 size 留成运行时值。

### Q8. FileCheck 的 CHECK / CHECK-SAME / CHECK-DAG / CHECK-NOT 各在什么时候用？

提示：顺序匹配 / 续同一行 / 顺序无关 / 断言不存在。你的 3 个测试各验证了什么？踩过哪两个坑？

- 我的理解：
CHECK 用于检查某些语句存在，CHECK-SAME 用于检查语句在同一行， CHECK-NOT用于检查CHECK-NOT 用于检查没有相关语句，CHECK-DAG 用于检查数据
- 【Review 补充】
CHECK / CHECK-SAME / CHECK-NOT 对（CHECK-NOT 有句重复 typo）。CHECK-DAG 要纠正：
- **CHECK-DAG 是"顺序无关匹配"**：多个 CHECK-DAG 之间可以任意顺序出现，用于**重排不影响语义**的结构（如 08 输出里一坨 `arith.constant 0/128/512/256/64/32` 的声明顺序可能变）。不是"检查数据"。
- 三个测试各验证：08 = for 版三层循环 + 切片 + insert_slice；09 = forall + in_parallel + parallel_insert_slice 且 `CHECK-NOT: scf.for` 断言没出现 for 版结构；10 = 三个 affine.min 尾块公式。
- 踩的两个坑：
  1. **CHECK 顺序 = 输出顺序**——09 把 parallel_insert_slice 写在 in_parallel 前面（输出里 in_parallel 先出现）→ 找不到。
  2. **CHECK-SAME 是"续同一行"**——10 用 CHECK-SAME 匹配三个不同行的 affine.min → 报错。

---

## 追加思考

### A1. tile_using_for 为什么返回 1+N 个 handle，tile_using_forall 为什么只返回 2 个？

因为**循环的表达方式不同**：
- `tile_using_for` 为每个维度生成一个独立 scf.for → 1 个 tiled op + N 个 loop = N+1 个 handle（matmul 3 维 → 4 个）。
- `tile_using_forall` 把并行维合成**一个** `scf.forall`（2D 迭代空间 `in (2, 8)`）→ 1 个 tiled op + 1 个 forall = 2 个 handle。
handle 数量由 loop 形态决定，不由维度数决定。

### A2. Transform 对比 C++ pass 的价值是什么？（策略变成数据）
可以不用改动上层工具逻辑， 通过dsl的方式编排优化的步骤，能够快速验证优化效果

正确。补一句：策略成为 IR 后还能被**工具化**（存文件、diff、版本管理、甚至用 transform op 去修改 transform 策略本身），C++ pass 做不到这点。
### A3. handle 失效后，后续 transform 应该用什么？
使用新的handle

对。用 `%tiled` / `%loops:3` 这些新 handle。这就是 plan 文档的 handle 生命周期：`%matmul ──consume──> tile_using_for ──> %tiled + %loops`。后续策略（如再 tile 或 vectorize）必须挂在新的 handle 上。

### A4. forall 的 shared_outs + in_parallel 为什么是并行写回的前提？

- **shared_outs**：所有并行迭代共享同一个 destination SSA（09 的 `shared_outs(%arg4 = %1)`）。没有它，各迭代写进各自局部，无法汇成一个结果 tensor。
- **in_parallel + parallel_insert_slice**：并行迭代各自把结果 tile 写回 destination 的**不相交**区域（`[%3, %4] [64, 64]`）。"不相交"是并行安全的前提——若两个迭代写同一位置就有 data race。
- 对比 for：for 用 iter_args + insert_slice + yield 顺序串起来，天然串行，不需要这层保证。

---

## 正负例记录

### 正例 1（08-tile-for.mlir，整除 for 版）

- 策略：`match linalg.matmul → tile_using_for [64, 64, 32]`
- 输出关键结构：三层嵌套 scf.for（M→N→K）、iter_args 部分和链、extract_slice（64×32 / 32×64 / 64×64）、内层 matmul、insert_slice

### 正例 2（09-tile-forall.mlir，forall 版）

- 策略：`tile_using_forall [64, 64, 0]`（K 不 tile）
- 输出关键结构：单个 `scf.forall in (2, 8)`、shared_outs、affine.apply 算偏移、A/B 全 K 切片、in_parallel + parallel_insert_slice

### 正例 3（10-tile-boundary.mlir，非整除 127×251×509）

- 输出关键结构：`affine.min (-d0 + 127, 64)` / `(-d0 + 251, 64)` / `(-d0 + 509, 32)`，tail size 63/59/29，tile 类型 `tensor<?x?xf32>`

### 负例记录（语法/测试坑）

- 08 L6 `!transform.any_opm` typo → `unknown type mnemonic`
- 09 只绑 1 个 handle 但 op 定义 2 个 result → `operation defines 2 results but was provided 1 to bind`
- 10 `%tiled,,` 双逗号 → `expected valid ssa identifier`
- 09-tile-forall.check CHECK 顺序反 → `expected string not found`
- 10-tile-boundary.check 用 CHECK-SAME 跨行 → `CHECK-SAME: is not on the same line`

---

## 收获与卡点

- 这次实验学到最重要的 1 点：
- 卡住/还没完全理解的地方：
- 下一步想验证什么：
