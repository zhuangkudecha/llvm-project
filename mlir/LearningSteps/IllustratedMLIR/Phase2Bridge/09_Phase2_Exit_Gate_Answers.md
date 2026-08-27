# Phase 2 Exit Gate — 参考答案

> **使用方式**:先闭卷自答(或口述),再对照本节。答案给的是**要点**,不是背诵稿——
> 判定标准仍是「能用自己的话讲清楚」,逐条覆盖对应「通过标准」即可。
> 每题答案都锚定 `Demo/passes/`、`Demo/inputs/` 与 `Demo/tests/` 中的真实实验。

---

## Week 5 — SSA、CFG、Dominance 与 Pass Anchor

### Q5.1 对着 `inputs/01-inspection.mlir` 说 Value 关系

```mlir
func.func @inspect(%arg0: i32, %cond: i1) -> i32 {
    %c1 = arith.constant 1 : i32
    cf.cond_br %cond, ^left, ^right
  ^left:   %left_value  = arith.addi %arg0, %c1 : i32
           cf.br ^merge(%left_value : i32)
  ^right:  %right_value = arith.addi %arg0, %c1 : i32
           cf.br ^merge(%right_value : i32)
  ^merge(%result: i32): return %result : i32
}
```

| Value | owner | 类型 | defining | users |
|---|---|---|---|---|
| `%arg0` | entry block | BlockArgument (i32) | 函数签名 | 2 个 `arith.addi`(left/right) |
| `%cond` | entry block | BlockArgument (i1) | 函数签名 | 1 个 `cf.cond_br` |
| `%c1` | `arith.constant` | OpResult (i32) | 常量 op | **2 个 addi —— multi-use** |
| `%left_value` | `arith.addi` | OpResult (i32) | 加法 op | 1 个 `cf.br`(作 merge 块参数) |
| `%right_value` | 同上 | OpResult (i32) | 加法 op | 1 个 `cf.br` |
| `%result` | ^merge 块 | BlockArgument (i32) | block argument | 1 个 `func.return` |

inspection pass 输出交叉印证:`operations=8`(func.func + constant + cond_br + 2×addi + 2×br + return)、`regions=1`、`blocks=4`;`block-dominance` 矩阵 0→1,2,3 为 1,1/2→3 为 1,1→2 与 2→1 为 0。

### Q5.2 OpResult 与 BlockArgument

- **OpResult**:由 Operation 定义,是 op 的产物;在 op 内部按序号区分(`getResultNumber()`),属于 op 所在的 block。
- **BlockArgument**:由 Block 定义,是该 block 的「入口」——值从哪里进来;等价于经典 SSA 的 φ 节点(如 `^merge(%result)` 接收 left/right 传进来的值)。
- op 的 operand **可以来自二者任一**:`arith.addi %arg0, %c1` 的左侧是 BlockArgument、右侧是 OpResult。
- inspection pass 分别打印 `op-result` 与 `block=… argument=…`,正是这两个集合。

### Q5.3 scf 结构对应关系

`scf.for`:`iter_args(%acc = %init)` 是进入循环的初始值;块参数 `^bb(%i, %acc)` 中 `%acc` 是「上一次迭代 yield 的值」;`scf.yield %acc_next` 既决定下一次迭代的块参数,也决定循环 results(最后一次 yield 就是结果)。

```mlir
%sum = scf.for %i = %c0 to %cN step %c1 iter_args(%acc = %c0) -> (i32) {
  %acc_next = arith.addi %acc, %c1 : i32
  scf.yield %acc_next : i32
} // %sum == 最后一次 yield 的 %acc_next
```

`scf.if`:results 必须由**所有分支** `scf.yield` 统一给出,类型一致——这正是 SSA 对「控制流汇合处只有一个定义」的处理方式,和 `cf.br ^merge(%v)` 传块参数是同一思想。

### Q5.4 CFG / SSA / dominance 各回答什么

- **CFG**:控制流——哪些块之间可以跳转(边),块是什么(节点)。回答「程序怎么走」。
- **SSA**:值关系——每个 Value 定义一次、use 指向定义。回答「值从哪来、被谁用」。
- **dominance**:合法性——定义必须支配所有 uses(定义所在块支配使用所在块)。回答「这个使用合不合法」;同时支撑依赖分析(如循环不变量)。

### Q5.5 Pass anchor

- anchor = pass 调度在哪个 op 上:`OperationPass<func::FuncOp>` 锚在 `func.func`,`OperationPass<ModuleOp>` 锚在 `module`。
- 锚在 func.func:每个函数**各跑一次**,pass 看到的是单个函数体;锚在 module:整个 module 跑一次,能跨函数看(如函数名、模块级属性)。
- `phase2-inspection` 锚在 `func.func`:它检查的是函数体内的块、块参数和函数内 dominance;dominance 是 per-function 的,跨函数无意义。

### Q5.6 Analysis invalidation

- 分析结果(如 dominance、liveness)是**缓存**;pass 修改 IR 后缓存变 stale。例:pass 删除了一个 addi,旧 dominance 信息不再反映真实 IR。
- PassManager 在 pass 之后自动失效分析,**除非 pass 声明保留**。inspection pass 末尾 `markAllAnalysesPreserved()`——因为它只读、不改 IR,所以所有分析可安全保留;只改 IR 的 pass(如 add-zero)则不能这么声明。
- 周实验里的一个可验证点:让某 pass 声明保留实际被改坏的 analysis,观察后续 pass 拿到错误结果。

---

## Week 6 — Rewrite、Fold 与 Driver

### Q6.1 谁在改 IR

- **pattern**:匹配(op + 条件)并**决定**改什么,代码在 `matchAndRewrite` 里,返回 success/failure。
- **rewriter**:真正执行修改(`replaceOp`/`eraseOp`),并维护 use-list、SSA 合法性——IR 的物理修改都由它完成。
- **driver**(greedy rewrite driver):调度——遍历 op、按 benefit 试 pattern、worklist 迭代直到 fixpoint。
- 一句话:pattern 出主意,rewriter 动手,driver 安排顺序。

### Q6.2 replaceOp / fold / DCE 分工

- `rewriter.replaceOp(op, value)`:把 op 所有 results 的 uses 重定向到 `value`,然后删除 op(AddZero 里把 `%result` 的 use 指向 `%arg0`,addi 消失)。
- **fold**:op 在 operand 是常量时直接换成计算后的常量(op 自身实现 fold 钩子),发生在 rewrite 之前。
- **DCE**:删除无 uses 的产物(如被替换后没人用的常量)。
- 时间线:fold/rewrite 改变 uses → 无用的 op 变成 dead → DCE 清除。AddZero pass 里 `config.enableFolding(false)` 正是为了**隔离**:证明变化来自自定义 pattern,而不是 arith 的 fold。

### Q6.3 red/green 实验

1. 固定输入 `inputs/02-add-zero.mlir` 和固定 pipeline,跑出基线输出并用 FileCheck 断言(`tests/02-add-zero.check` 要求 `CHECK-NOT: arith.constant` / `CHECK-NOT: arith.addi` 且 return 直接用 `%arg0`)→ **green**。
2. **移除** `patterns.add<AddZeroPattern>`(只留 driver 骨架)→ 再跑:addi 保留 → FileCheck 失败 → **red**。
3. **恢复** pattern → 重新 green。
- 关键:输入、pipeline、folding 开关全固定,唯一变量是 pattern 的注册,因此红绿差异只能由 pattern 引起。

### Q6.4 rewrite 返回 failure

- 该 pattern 视为「未匹配」:driver 继续尝试该 op 的其他 pattern、其他 op;IR 不变,不留下半截修改。
- 契约:pattern 要么不修改并返回 failure,要么修改并返回 success;先修改后返回 failure 是 pattern 的 bug。AddZero 的两个 `failure()`(非常量、非零)都发生在任何 mutation 之前。

### Q6.5 PatternBenefit 与终止性

- greedy driver 按 benefit 降序尝试 pattern;benefit 是**静态优先级**,匹配是**局部贪心**——结果依赖 worklist 顺序和 benefit,不是全局搜索,所以不保证全局最优(Q7.5 再展开)。
- 终止性:**每次成功 rewrite 都使一个良基度量严格下降**。AddZero 每次替换删除一个 addi,op 数 -1;度量有下界,不能无限下降。
- 二次运行稳定:第一轮把 `x+0` 全消掉,第二轮无 match → 无变化。

### Q6.6 边界测试预期

| case | 预期 | 理由 |
|---|---|---|
| multi-use | 替换成功,全部 uses 重定向 | `replaceOp` 走 use-list 重定向,一个不漏 |
| non-zero 常量 | 不匹配,IR 不变 | pattern 检查 `rhs.isZero()`,非零直接 failure |
| 位宽不同 | 不能跨类型 fold | operand 与常量类型由 verifier 保证一致;fold 结果类型必须匹配,不存在跨位宽复用 |
| shaped value | 不做逐元素 fold | `m_ConstantInt` 只匹配标量整数常量;shaped 常量要逐元素折叠,不在此 pattern 范围 |

---

## Week 7 — Producer–Consumer Chain

### Q7.1 回溯链

从 relu 出发:

```cpp
auto bias   = relu.getDpsInputs()[0].getDefiningOp<linalg::GenericOp>();  // relu ← bias
auto matmul = bias.getDpsInputs()[0].getDefiningOp<linalg::MatmulOp>();   // bias ← matmul
```

链的终点:`getDefiningOp()` 返回 null——operand 是 block argument / 函数参数(如 `%lhs`、`%rhs`、`%bias` 都是 func 参数,没有 defining op)。`Phase2FuseChain.cpp` 中每步都判空并 `notifyMatchFailure`,这就是「链中断」的检查点。

### Q7.2 先匹配完整,再修改

mutation 前检查(`Phase2FuseChain.cpp`):
- **链结构**:relu 的输入由 generic 定义、bias 的输入由 `linalg.matmul` 定义(逐级 `getDefiningOp` + 判空)。
- **body 形状**:`hasExpectedReluBody`(2 个块参数、constant 0.0 + `maximumf` + yield);`hasExpectedBiasBody`(3 个块参数、`arith.addf` + yield、f32)。
- **maps**:`hasExpectedMaps`——bias 三张 map 必须 [恒等, 广播到 d1, 恒等],relu 两张全恒等。
- **语义条件**:constant 是 0.0、max 的操作数顺序、yield 的就是 max/add 结果。
- 为什么先完整匹配:部分修改会让 worklist 里其他 pattern 拿到 stale 状态,失败时还可能留下半截 IR;完整匹配后一次 `fuseElementwiseOps` + `replaceOp` + `eraseOp` 原子完成。

### Q7.3 为什么 matmul 保持独立

- matmul 有 **reduction 维(K)**:它的迭代域是 (M,K,N) 三层且 K 是累加维;bias/relu 是纯 **parallel** 双循环。
- 把 matmul 并进 elementwise 链会改变迭代域语义,不能简单融合;matmul 的 result 是 bias 的输入——**融合边界就在那里**。
- 正例只融合 bias+relu 两个 elementwise generic(`fuseElementwiseOps` 只处理同域 elementwise 链),matmul 保持独立,这是计算语义决定的,不是代码巧合。

### Q7.4 四类负例为什么必须拒绝

| 负例 | 原因 |
|---|---|
| multi-use | `%biased` 若还有别的 user,融合后 erase bias 会悬空其他 use;必须拒绝,IR 保持原样 |
| 错误 layout/maps | maps 不对齐意味着不是 elementwise 同域,融合结果无定义 |
| 错误 dtype | body 检查 `isF32`,其他类型不匹配预期公式 |
| 非 ReLU / side-effect | 不是 `maximumf(·, 0.0)` 就不是 ReLU,融合改变语义;带 side-effect 的 op 不能乱动 |

负例一律 `notifyMatchFailure` → 无 mutation,原 IR 不变;`tests/03-producer-consumer.check` 覆盖正例结构,负例输入另行确认无变化。

### Q7.5 PatternBenefit 不保证全局最优

- benefit 是静态数字,driver 在每个 op 上**贪心**选择当前可用的最高 benefit pattern;
- 匹配顺序依赖 worklist 遍历顺序;两个互斥 pattern 都匹配时,谁先被试取决于调度,结果可能不同;
- 没有任何全局搜索/回溯,所以「局部最优」不是「全局最优」。

### Q7.6 终止性度量

- 单调度量:op 数量。一次成功融合 = 2 个 generic 变 1 个 + erase bias = op 数严格下降;度量有下界,必然终止。
- 二次运行稳定:融合后的 generic body 是 addf+maximumf,不再匹配 `hasExpectedBiasBody`/`hasExpectedReluBody` → 第二轮无 match。
- greedy driver 还有迭代上限兜底;终止性由「度量单调下降」证明,而非巧合。

---

## Week 8 — DialectConversion

### Q8.1 legal / illegal / unknown 与优先级

- **legal**:可以留在 IR 中(`addLegalDialect<LLVM>` / `addLegalOp<ModuleOp>`)。
- **illegal**:必须被 pattern 转换掉(`addIllegalDialect<arith/func/cf>`)。
- **unknown**:没有任何 action 指定——由 conversion 模式兜底(partial 允许残留,full 要求 legalize)。
- 优先级:operation 级 action > dialect 级 action > unknown-op 回调 > partial/full 兜底。例:`target.addLegalOp<ModuleOp>()` 让 module 即使属于无 action 的 builtin 也 legal;`tensor.empty` 无任何 action → unknown。

### Q8.2 static / dynamic / recursive legality

- **static**:`markLegal/markIllegal`(op 或 dialect),无条件判定。本实验全部用 static。
- **dynamic**:`addDynamicallyLegalOp<Op>(callback)`,按条件判定——如「只有 type 为 X 时 legal」。
- **recursive**:对容器 op 声明 legal 时连同其嵌套 region 内容一起判定(如声明某 op legal 时其 body 里的 op 也要 legal,否则降级为 illegal/unknown)。
- 适用场景:static 用于明确目标;dynamic 用于「大多数合法、少数非法」;recursive 用于容器内嵌套内容整体判定。

### Q8.3 为什么要一起转换

- 类型不一致会让 verifier 拒绝:`arith.addi` 若 operand 还是 `i32`、result 已变 `llvm.i32`,类型对不上。
- 联动:function signature 转换(`func.func @f(%a: i32)` → `llvm.func @f(%a: llvm.i32)`)→ entry block arguments 跟着转换 → 所有用到这些块参数/结果的 op 的 operand/result 一起转换 → uses 全部更新。任一处不转,链条断裂。
- 本实验 `populateFuncToLLVMConversionPatterns` 同时处理 signature、block arguments 和 `func.return`,`populateArithToLLVMConversionPatterns` 处理 arith ops,`populateControlFlowToLLVMConversionPatterns` 处理 cf。

### Q8.4 三种 materialization

| 类型 | 方向 | 出现场景 |
|---|---|---|
| target materialization | source → target | 旧 producer(未转换)连已转换的 consumer |
| source materialization | target → source | 新 producer(已转换)连未转换的 consumer |
| argument materialization | 在新块参数附近 | region signature 已转换、内部旧 user 还用旧类型 |

- 本实验**没出现 cast**:i32 等类型 1:1 转换,没有类型边界。若引入 `LLVMTypeConverter` 不认识的类型(如 tensor),转换交界处就会出现 materialization——这正是 `04-convert-partial-full.mlir` 里 `tensor.empty` 留在函数体内的意义。

### Q8.5 用同一个 unknown op 演示 partial/full

- `tensor.empty` 无任何 legality action → unknown;`04` 输入特意让 func 签名 `(i32) -> i32` 可转换,把 unknown 放在函数体内。
- **partial**(`{partial=true}`):unknown op 被允许原样保留——`llvm.func` 内保留 `%0 = tensor.empty()`,arith 转成 `llvm.mlir.constant` / `llvm.return`;conversion 成功。
- **full**:所有 op 必须 legal,`tensor.empty` 无 pattern → `error: failed to legalize operation 'tensor.empty'`,conversion 失败。
- 关键诊断差异:full 在 tensor.empty 处报错并 fail;partial 成功且残留 unknown。注意措辞:**op 没有 action 就是 unknown**,不是「被替代成 unknown」。

### Q8.6 failure 后会发生什么

- conversion 框架通过 undo 机制回滚**本次 conversion 对 IR 的修改**(已替换的 op、use-list 恢复)。
- **不会**回滚任意外部状态(文件、全局缓存、外部进程);pattern 若有外部副作用也无法回滚。
- 所以「conversion failure = 自动回滚」仅限 IR 范畴;不要假设失败后一切自动恢复原状。

---

## Week 9 — Transform 与 Tiling Interfaces

### Q9.1 payload / Transform IR / handle / TransformState

- **payload IR**:被优化的程序(08 文件里的 `func.func @matmul`)。
- **Transform IR**:优化策略脚本(同文件的 `transform.named_sequence`)——策略本身也是 IR,由 `--transform-interpreter` 执行。
- **handle**:Transform IR 里的 SSA value(类型 `!transform.any_op`),**含义**是映射到 payload 的对象。例:`%matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1`——`%matmul` 指向 payload 里那个 linalg.matmul op。
- **TransformState**:解释器运行时的簿记——记录「哪个 handle → 哪些 payload op」的映射表,不在 .mlir 里,在内存里;match 查询它(`getPayloadOps`),tile_using_for 更新它。

### Q9.2 三个 interface 分工

- **DestinationStyleOpInterface**:问「哪个 operand 是输出/init destination(outs)」——linalg 系列 op 都实现。
- **TilingInterface**:问「怎么切分」——迭代域多大、生成 tile 的实现(`getTiledImplementation` 等)。
- **ReifyRankedShapedTypeOpInterface**:问「动态形状怎么算」——需要时把 shape 计算出来。
- tiling 算法向 op 依次问这三个问题,答案决定了生成的 loops/slices/insert 结构。

> **补充:这三个 interface 在哪里实现?——native,不在 transform IR。**
>
> 1. 接口定义是 ODS 生成的 C++ 抽象类:`include/mlir/Interfaces/DestinationStyleOpInterface.td`、`TilingInterface.td`、`InferTypeOpInterface.td`(ReifyRankedShapedTypeOpInterface)。
> 2. **实现挂在 payload op 上**:linalg 基类 `LinalgOp` 声明 `DestinationStyleOpInterface, ReifyRankedShapedTypeOpInterface` 等(`LinalgStructuredOps.td`),方法在 `lib/Dialect/Linalg/IR/`(如 `getTiledImplementation`);`tensor.empty` 等声明 `ReifyRankedShapedTypeOpInterface`。
> 3. **transform op 只消费,不实现**:`transform.structured.tile_using_for` 的实现在 `lib/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp`,运行时从 handle 拿到 payload op 后 `dyn_cast<TilingInterface>(target)` 再调用方法,不是接口实现者:
>
> ```cpp
> auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
> if (!tilingInterfaceOp)
>   return transformOp->emitError("only TilingInterface ops are supported");
> ```
>
> 心智模型:策略(transform IR)→ handle → payload op → 运行时 `dyn_cast` → 调用 payload op 的 native 接口方法。自己的 `Phase2MatmulTiling.cpp` 里 `dyn_cast<TilingInterface>(matmul.getOperation())` 与 transform op 内部做的**是同一件事**——这也再次说明 tiling 结构是接口方法算出来的,不是文本替换。

### Q9.3 为什么是按接口重建,不是文本替换

tile_using_for 通过接口生成(`inputs/08-tile-for.mlir` 注释中的结构):
- M/N/K 三个 `scf.for`(reduction K 在最内层);
- `tensor.extract_slice` 切 A/B/C 的 tile(A 64×32、B 32×64、C 64×64);
- 内层 `linalg.matmul`(64×64)消费 slices;
- `tensor.insert_slice` 把结果插回 init,C 通过 iter_args 链部分和累加。

这个结构**由接口语义和算法决定**,输入文本只是载体——同样的输入文本,换 interface 实现产出的结构不同,所以不是文本替换。

### Q9.4 整除 vs 非整除

- 整除(128/256/512,tile 64/64/32):每维 tile 数 = dim/tile 整除法,M 2 次、N 8 次、K 8 次,所有 tile 等大。
- 非整除(127×251×509):`tensor.extract_slice` 的边界是 `min(offset+tile, dim)`,最后一排 tile 变小——127→64+63,251→64×3+59,509→32×15+29。
- 关键点:边界 tile 大小**不是均匀 tile**;slice 尺寸由边界计算得出,结构仍完整(loop 数 = ceil(dim/tile))。

> **补充:收缩边界 tile 是否违反分块乘法?——不违反,也不是补零。**
>
> 分块乘法只要求:① 每个维度的分块恰好全覆盖 `[0, dim)`(互不重叠、拼起来正好);② K 维分割在 A、B 两侧一致。最后一块更小(127→64+63)依然满足两者;K 维每次迭代对 A/B 取同一个区间,`Σ_区间 A·B` 累加不变——不需要零来凑。
>
> 证据:`tests/10-tile-boundary.check` 断言 `affine.min -d0 + 127, 64`——边界尺寸是**运行时**算出的 `min(dim - offset, tile)`,这正是动态 shape 下 IR 依然正确的机制。
>
> 补零是另一种技术(`transform.structured.pad`):把 shape 补到对齐倍数,多算元素再切回来,用于 SIMD 定长向量、硬件对齐;代价是多算 + 对齐约束。MLIR 默认 tiling 不补零。

### Q9.5 named sequence 匹配

- `transform.structured.match ops{["linalg.matmul"]}` 按 op 名称(可加约束)在 handle 的 payload 集合内查找;
- 匹配失败:transform op 报**明确诊断**(error 并中止解释),不会静默跳过——这保证策略「要么精确执行,要么明确失败」;
- 匹配到的结果成为新 handle,供后续 op 消费。

### Q9.6 handle 失效

- handle 的「有效性」是 TransformState 里 handle→payload 的映射;`tile_using_for` **消费**(consumesHandle)`%matmul` 时把 matmul 替换成 tiled 结构,旧映射不再指向有效对象 → 该 handle 失效。
- 失效的不是 transform IR 里的 SSA 文本,而是映射的合法性;消费同时会**产生**(producesHandle)新 handle:`%tiled`(新 matmul)、`%loops:3`(三个 scf.for)。
- 避免失效:先消费后使用返回的新 handle,不要在后面继续用已被消费的旧 handle。

---

## Week 10 — 参数化 Tiling 与 IR Evolution

### Q10.1 从空文件实现的骨架

```cpp
struct MatmulTilingPass : PassWrapper<MatmulTilingPass, OperationPass<func::FuncOp>> {
  Option<int64_t> tileM{*this, "tile-m", llvm::cl::init(64)};
  Option<int64_t> tileN{*this, "tile-n", llvm::cl::init(64)};
  Option<int64_t> tileK{*this, "tile-k", llvm::cl::init(32)};

  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<scf::SCFDialect, tensor::TensorDialect, arith::ArithDialect>();
  }

  void runOnOperation() final {
    // 1) 验证参数(任何 mutation 前)
    // 2) walk 收集满足约束的 linalg.matmul targets
    // 3) 对每个 target:
    //    IRRewriter rewriter(&getContext());
    //    rewriter.setInsertionPoint(matmul);
    //    SCFTilingOptions opts;  opts.setLoopType(ForOp).setTileSizes({M,N,K}).setReductionDims({2});
    //    FailureOr<SCFTilingResult> r = scf::tileUsingSCF(rewriter, tileable, opts);
    //    rewriter.replaceOp(matmul, r->replacements);
  }
};
void registerMatmulTilingPass() { PassRegistration<MatmulTilingPass>(); }
```

要点:options 通过 `Option<int64_t>` 暴露(命令行 `{tile-m=… tile-n=… tile-k=…}`);`tileUsingSCF` 需要 op 实现 `TilingInterface`(先 `dyn_cast`);`reductionDims({2})` 指定 K 维;依赖方言必须注册(`scf`/`tensor`/`arith`)。

### Q10.2 参数验证必须在 mutation 前 + tile=0

- `Phase2MatmulTiling.cpp` 明确分两阶段:Phase 1(验证 `tileM/N/K > 0`、targets 非空、每个 target 可 tile)→ Phase 2(才动 IR)。验证在**任何 mutation 之前**完成,失败时 IR 未被碰过。
- tile=0:0 尺寸 tile 会产生空循环/非法 slice,甚至死循环——必须在入口 `emitError` + `signalPassFailure` 拒绝。

### Q10.3 SCFTilingResult 字段

- `loops`:生成的 `scf.for`(本实验 3 个,reduction 在最内层);
- `tiledOps`:新的 tiled `linalg.matmul`(消费 slices);
- `slices`:`tensor.extract_slice` 们(operand 的 tile);
- `insertOps`:`tensor.insert_slice`(把 tile 结果写回 init);
- `replacements`:原结果 → 新值(insert/loop 的结果)的映射;`rewriter.replaceOp(matmul, tiled->replacements)` 把原 uses 全部重定向到新值。

### Q10.4 failure 是否留下部分 IR

- 不保证原子:自定义 pass 中失败位置不同,IR 状态不同;conversion 框架有 undo,`tileUsingSCF` 之外的手动 mutation 没有。
- 降低风险:验证参数先行 + 先匹配/检查后改写 + 单点失败即 `signalPassFailure`。
- 诊断:失败时保存 IR snapshot(`--mlir-print-ir-after-all` 或前后 dump),对比 before/after 定位部分修改。

### Q10.5 五个概念一句话

| 概念 | 回答的问题 | 出现周次 |
|---|---|---|
| folding | 常量能不能直接算出来?(op 内部化简) | Week 6 |
| pattern rewrite | 这个 op 结构能不能改写?(匹配+替换) | Week 6–7 |
| conversion | 跨 dialect 怎么降到合法终点?(legality 驱动) | Week 8 |
| tiling | 大 op 怎么切成循环+tile?(结构化分解) | Week 9–10 |
| legality | 这个 op 能不能留在目标 IR 里?(判定标准) | Week 8 |

### Q10.6 tiling 对后续 pipeline 的影响

- **bufferization**:tiling 产生边界清晰的 loop + slice 结构,每个 tile 的 buffer 大小已知、可复用(累加逻辑在 C tile 上),便于 `OneShotBufferization` 生成高效 memref。
- **vectorization**:小 tile(64×64 等)正好做 register blocking/vector 化,避免一次性向量化整个 matmul。
- **invalidated analyses**:dominance(新增 loop 嵌套)、loop 相关分析(scf 结构全变)、shape/依赖分析等全部失效;tiling pass 不应声明保留这些。

---

## 结项综合

### Q11.1 从零实现

即 Q10.1 骨架 + 现场编译。验证:`ninja -C Demo/build` 后跑 `phase2-matmul-tiling{tile-m=2 tile-n=3 tile-k=4}` 于 `inputs/11-tile-even.mlir`,输出结构对照 `tests/11-tile-even.check`(出现 3 层 scf.for、extract_slice、insert_slice,matmul 内层为 2×3)。

### Q11.2 全量测试

- 正例:`11-tile-even.mlir`(128×256×512)结构正确;
- 边界:`12-tile-not-even.mlir`(100×300×200,tile 若超出维度,边界 tile 变小);
- 负例:`tile=0` → 「must be positive」报错;缺 target(无 matmul)→ 「expected a supported linalg.matmul target」;非 rank-2/非静态 shape → 不被收集,报缺 target;
- 稳定性:同一命令跑两遍输出一致(二次运行无新变化)。

### Q11.3 概念辨析

即 Q10.5 表格,每个概念配一个真实例子:
- folding:`arith.addi %c1, %c2` → `%c3`;
- pattern:`x + 0 → x`(AddZero);
- conversion:arith/func/cf → LLVM 方言;
- tiling:matmul → 3 层 scf.for + slices;
- legality:`tensor.empty` 在 full conversion 下必须 legalize。

### Q11.4 演进链路

tiling 后(loop+slice+insert)→ bufferization 把 tensor 变 memref、slice 变 subview、循环可复用 buffer → vectorization 对小 tile 做 SIMD;每步 IR snapshot 保存(docs/notes),说明哪步 invalidate 了哪些 analysis、哪步结构变化是下一步的前提。

---

## 现场验证命令 — 预期要点

1. **inspection**:输出 `function=inspect operations=8 regions=1 blocks=4` + 块参数 users + 4×4 dominance 矩阵(0 支配 1/2/3,1/2 支配 3,1↔2 互不支配)。
2. **add-zero**:addi 和 constant 消失,return 直接用 `%arg0`(对应 `tests/02-add-zero.check`)。
3. **fuse-chain**:bias+relu 两个 generic 融合成一个,matmul 保留,`tests/03-producer-consumer.check` 结构匹配。
4. **convert-to-llvm**:full 模式 arith/func/cf 全部消失;`{partial=true}` 时 `tensor.empty` 残留;full 时 `04-convert-partial-full.mlir` 报 `failed to legalize operation 'tensor.empty'`。
5. **tiling**:3 层 scf.for + extract/insert_slice + 内层 2×3 matmul(tile 参数生效)。

---

[返回门禁问答](09_Phase2_Exit_Gate.md) · [查看原始 Phase 2 执行计划](../../Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md)
