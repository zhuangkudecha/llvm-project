# Phase 2 综合 Demo：Transformation Lab

## Demo 目标

这个 Demo 用同一个自定义 Pass 库和一组 MLIR 输入，把 Phase 2 的 Week 5–10 串成一条完整、
可运行、可测试的 transformation 主线：

```text
读取 IR
  → 理解 SSA / use-def / dominance
  → 完成第一次局部 Rewrite
  → 匹配并融合 Producer–Consumer 子图
  → 用 Legality 驱动 Dialect Conversion
  → 用 Transform IR 编排 Matmul Tiling
  → 把 Tiling 策略固化成参数化 C++ Pass
  → 保存 Tiled / Bufferized / Vectorized IR snapshots
```

不要强行让一个 `.mlir` 文件依次经过所有实验。AddZero、Producer–Consumer Fusion、
Dialect Conversion 和 Matmul Tiling 的输入合同不同。更合适的形式是：

> 一个 Demo 工程，五个独立小实验，加一条最终 Matmul IR evolution 主线。

Demo 最终包含下面五个自定义 Pass：

| Pass | 对应章节 | 展示内容 |
|---|---|---|
| `phase2-inspect` | Week 5 | SSA、use-def、Block、Region、Dominance |
| `phase2-add-zero` | Week 6 | Pattern、Rewriter、Greedy Driver、Fold 隔离 |
| `phase2-fuse-chain` | Week 7 | Producer–Consumer 链、完整匹配、single-use |
| `phase2-convert-to-llvm` | Week 8 | Legality、TypeConverter、DialectConversion |
| `phase2-matmul-tile` | Week 9–10 | TilingInterface、SCF loops、参数化 tiling |

另外增加一个 Transform Dialect 输入，用于对比：

```text
Transform IR 中的 tiling 策略
              vs
C++ 参数化 Tiling Pass
```

## 目录结构

根据 Phase 2 的统一实验约定，可编译真源放进 MLIR test support：

```text
test/lib/Transforms/
├── Phase2Inspection.cpp
├── Phase2AddZero.cpp
├── Phase2FuseChain.cpp
├── Phase2ConvertToLLVM.cpp
└── Phase2MatmulTiling.cpp
```

学习输入、测试、脚本及输出快照放在：

```text
LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/
├── README.md
├── inputs/
│   ├── 01-inspection.mlir
│   ├── 02-add-zero.mlir
│   ├── 03-producer-consumer.mlir
│   ├── 04-conversion.mlir
│   ├── 05-transform-matmul.mlir
│   ├── 06-matmul-tile.mlir
│   └── 07-matmul-tail.mlir
├── tests/
│   ├── inspection.mlir
│   ├── add-zero.mlir
│   ├── producer-consumer.mlir
│   ├── conversion.mlir
│   ├── transform-matmul.mlir
│   ├── matmul-tile.mlir
│   └── matmul-invalid.mlir
├── snapshots/
│   ├── 00-input.mlir
│   ├── 01-tiled.mlir
│   ├── 02-bufferized.mlir
│   └── 03-vectorized.mlir
└── scripts/
    ├── run-all.sh
    └── generate-snapshots.sh
```

约束：

- C++ 实现只有 `test/lib/Transforms` 中一份；
- Demo 目录只保存输入、测试、脚本和输出快照；
- 不要在 Demo 目录复制第二套 C++ 实现；
- 每个实验均保留正例、负例和可复现命令。

## 实验一：SSA、CFG 与 Dominance

### 输入

使用包含分支、BlockArgument 和多个 user 的函数：

```mlir
func.func @inspect(%arg0: i32, %cond: i1) -> i32 {
  %c1 = arith.constant 1 : i32
  cf.cond_br %cond, ^left, ^right

^left:
  %left_value = arith.addi %arg0, %c1 : i32
  cf.br ^merge(%left_value : i32)

^right:
  %right_value = arith.muli %arg0, %c1 : i32
  cf.br ^merge(%right_value : i32)

^merge(%result: i32):
  return %result : i32
}
```

### Pass 行为

`phase2-inspect` 是只读 Pass，输出稳定的聚合信息：

```text
function=inspect
operations=...
regions=...
blocks=4
block-argument %arg0 type=i32
block-argument %result type=i32
%c1 users=2
%left_value dominates left terminator=true
%left_value dominates right block=false
```

不要输出裸指针地址，也不要把 use-list 的迭代顺序当作语义顺序。

### 运行

```bash
$MLIR_BIN/mlir-opt \
  Demo/inputs/01-inspection.mlir \
  -pass-pipeline='builtin.module(func.func(phase2-inspect))'
```

还应准备 dominance 负例：

```mlir
// expected-error @+1 {{does not dominate this use}}
```

并运行：

```bash
$MLIR_BIN/mlir-opt invalid.mlir -verify-diagnostics
```

这个实验的完成标准是：

- 能区分 OpResult 和 BlockArgument；
- 能沿 use-def 找到 definition 和 users；
- 能查询 block/op/value dominance；
- 能用 verifier 负例证明非法 SSA 会被拒绝；
- 只读 Pass 正确声明 analysis preservation。

## 实验二：Add Zero Rewrite

### 输入

```mlir
func.func @add_zero(%arg0: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %result = arith.addi %arg0, %zero : i32
  return %result : i32
}
```

### Pattern

```cpp
struct AddZeroPattern : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      arith::AddIOp op,
      PatternRewriter &rewriter) const override {
    APInt value;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&value)) ||
        !value.isZero())
      return failure();

    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
```

### 预期变化

Before：

```mlir
%zero = arith.constant 0 : i32
%result = arith.addi %arg0, %zero : i32
return %result : i32
```

After：

```mlir
return %arg0 : i32
```

### 测试矩阵

至少覆盖：

| 输入 | 预期 |
|---|---|
| `x + 0` | 改写为 `x` |
| `0 + x` | 若合同不支持则保持不变 |
| `x + 1` | 保持不变 |
| `i32`、`i64` | 按合同处理 |
| zero 还有其他 user | zero 不得被错误删除 |
| shaped value | 不得错误匹配标量规则 |
| Pass 第二次运行 | IR 不再变化 |

为了证明变化来自自己的 Pattern，而不是 Fold，显式关闭 folding：

```cpp
GreedyRewriteConfig config;
config.enableFolding(false);
```

应完成一次 Red/Green 实验：

```text
有 Pattern       → 测试通过
移除 Pattern     → 测试失败
恢复 Pattern     → 测试重新通过
```

## 实验三：Producer–Consumer Chain

主线选择：

```text
matmul → bias add → ReLU
```

输入概念结构：

```mlir
%matmul = linalg.matmul ...
%bias = linalg.generic ...   // broadcast add
%relu = linalg.generic ...   // max(value, 0)
return %relu
```

Matcher 从最外层 consumer 反向查询：

```text
ReLU consumer
    ↓ operand.getDefiningOp()
Bias Add
    ↓ operand.getDefiningOp()
Matmul
```

修改前必须完成全部只读门禁：

```text
确实是 max(x, 0)
  → 输入确实来自 bias add
  → indexing maps 正确
  → iterator types 全是 parallel
  → shape 与 dtype 兼容
  → producer 满足 single-use
  → 没有不允许的副作用
  → 才开始 rewrite
```

负例至少包括：

- Bias Add 有第二个 user；
- ReLU 不是 `max(x, 0)`；
- indexing maps 不同；
- iterator types 不符合合同；
- dtype 不同；
- shape 不兼容；
- 中间存在不允许的副作用 Operation。

这个实验要证明的不是性能，而是：

> 修改多个 Operation 之前，必须先只读匹配完整子图；匹配失败不得留下部分修改。

还应检查：

- 正例中目标 elementwise Operation 数量减少；
- 负例中原链保持；
- Pass 第二次运行达到稳定状态；
- PatternBenefit 只表示同一 root 上的优先级，不代表全局最优。

## 实验四：Dialect Conversion

第一次 conversion 使用简单的标量/CFG 输入，不直接使用复杂 matmul：

```mlir
func.func @convert(%arg0: i32, %arg1: i32) -> i32 {
  %sum = arith.addi %arg0, %arg1 : i32
  return %sum : i32
}
```

### Conversion 配置

```cpp
LLVMTypeConverter typeConverter(&context);

arith::populateArithToLLVMConversionPatterns(
    typeConverter, patterns);
populateFuncToLLVMConversionPatterns(
    typeConverter, patterns);
populateControlFlowToLLVMConversionPatterns(
    typeConverter, patterns);
```

目标合法性：

```cpp
target.addLegalDialect<LLVM::LLVMDialect>();
target.addLegalOp<ModuleOp>();

target.addIllegalDialect<arith::ArithDialect>();
target.addIllegalDialect<func::FuncDialect>();
target.addIllegalDialect<cf::ControlFlowDialect>();
```

执行：

```cpp
if (failed(applyFullConversion(
        module, target, std::move(patterns))))
  signalPassFailure();
```

成功标准不是只生成一个 `llvm.add`，而是：

```text
func dialect 消失
arith dialect 消失
cf dialect 消失
函数签名完成类型转换
BlockArguments 完成类型转换
分支传值完成类型转换
所有剩余 Operation 对 ConversionTarget 都 legal
```

结构检查：

```text
CHECK: llvm.func
CHECK: llvm.add
CHECK-NOT: func.func
CHECK-NOT: arith.addi
```

还要提供一个没有 legalization path 的 illegal Operation，证明 full conversion 会失败。

## 实验五：Transform Dialect Matmul Tiling

### Payload IR

```mlir
func.func @matmul(
    %lhs: tensor<128x256xf32>,
    %rhs: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %init = linalg.fill ins(%zero : f32)
      outs(%empty : tensor<128x512xf32>)
      -> tensor<128x512xf32>

  %result = linalg.matmul
      ins(%lhs, %rhs
          : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init : tensor<128x512xf32>)
      -> tensor<128x512xf32>

  return %result : tensor<128x512xf32>
}
```

### Transform sequence

```mlir
transform.named_sequence @__transform_main(
    %root: !transform.any_op) {
  %matmul = transform.structured.match
      ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op

  %tiled, %loops:3 =
      transform.structured.tile_using_for %matmul
      tile_sizes [64, 64, 32]
      : (!transform.any_op)
        -> (!transform.any_op,
            !transform.any_op,
            !transform.any_op,
            !transform.any_op)

  transform.yield
}
```

重点观察：

- Transform handle 如何关联 payload Operation；
- 旧 `%matmul` handle 为什么在 tiling 后失效；
- 为什么返回 `%tiled` 和三个 loop handles；
- TilingInterface 如何提供 iteration domain 和 tiled implementation；
- DestinationStyleOpInterface 如何标识 inputs 与 init/destination；
- K reduction 如何通过 loop-carried value 累加。

输出至少检查：

```text
CHECK-COUNT-3: scf.for
CHECK: tensor.extract_slice
CHECK: linalg.matmul
CHECK: tensor.insert_slice
```

如果选择 `tile_using_forall`，相应检查：

```text
scf.forall
scf.forall.in_parallel
tensor.parallel_insert_slice
```

不能在同一个 case 中同时要求互斥的 `scf.for` 和 `scf.forall` 结构。

## 最终实验：参数化 C++ Matmul Tiling Pass

这是整个 Demo 的主线产物。

### Pass 参数

```text
tile-m
tile-n
tile-k
```

运行方式：

```bash
$MLIR_BIN/mlir-opt Demo/inputs/06-matmul-tile.mlir \
  -pass-pipeline='builtin.module(
    func.func(
      phase2-matmul-tile{
        tile-m=64
        tile-n=64
        tile-k=32
      }
    )
  )'
```

### 实现主线

```cpp
void Phase2MatmulTilingPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // 第一阶段：只读验证。
  validateOptions();
  collectAndValidateTargets();

  // 第二阶段：执行 mutation。
  IRRewriter rewriter(&getContext());

  for (linalg::MatmulOp matmul : targets) {
    auto tileable =
        cast<TilingInterface>(matmul.getOperation());

    SmallVector<OpFoldResult> sizes{
        rewriter.getIndexAttr(tileM),
        rewriter.getIndexAttr(tileN),
        rewriter.getIndexAttr(tileK)};

    scf::SCFTilingOptions options;
    options
        .setLoopType(
            scf::SCFTilingOptions::LoopType::ForOp)
        .setTileSizes(sizes)
        .setReductionDims({2});

    FailureOr<scf::SCFTilingResult> tiled =
        scf::tileUsingSCF(
            rewriter, tileable, options);

    if (failed(tiled))
      return signalPassFailure();

    rewriter.replaceOp(
        matmul, tiled->replacements);
  }
}
```

最终实现必须以当前 checkout 的
`include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h` API 为准。不要复制
`TileUsingInterface.cpp` 内部算法。

最关键的 replacement 是：

```cpp
rewriter.replaceOp(matmul, tiled->replacements);
```

不要使用：

```cpp
tiled->tiledOps.back()->getResults()
```

猜测原结果映射。`replacements` 才是 tiling API 明确给出的原 Operation results 替换值。

### 失败原子性

实现分成两阶段：

```text
第一遍：只读收集并验证所有目标
第二遍：执行 tiling
```

所有可预见的拒绝条件都应在第一次 mutation 前检查，避免：

```text
第一个 matmul 已经被 tile
第二个 matmul 验证失败
留下部分转换 IR
```

底层 tiling API 在 mutation 中途失败是否完全可回滚，必须按当前 API 合同判断，不能假设任意
外部状态都自动恢复。

## 参数化 Tiling 测试矩阵

### 整除正例

```text
M=128
N=512
K=256
tile=[64,64,32]
```

检查：

```text
CHECK-COUNT-3: scf.for
CHECK: arith.constant 64
CHECK: arith.constant 32
CHECK: tensor.extract_slice
CHECK: linalg.matmul
CHECK: tensor.insert_slice
```

### 非整除正例

```text
M=127
N=509
K=251
tile=[64,64,32]
```

尾块大小为：

```text
M tail = 127 - 64  = 63
N tail = 509 - 448 = 61
K tail = 251 - 224 = 27
```

检查重点：

```text
offset 沿 64 / 64 / 32 推进
remaining = bound - offset
size = min(tile, remaining)
extract_slice 使用尾块 size
insert_slice 使用相同 offset/size
```

不要写死 `%17` 之类的临时 SSA 名称。应捕获定义并复用：

```text
CHECK: %[[SIZE:.*]] = ...
CHECK: tensor.extract_slice {{.*}} [{{.*}}] [%[[SIZE]]]
```

### 负例

| 输入 | 预期 |
|---|---|
| `tile-m=0` | tile size 必须为正 |
| `tile-n=-1` | tile size 必须为正 |
| 函数中没有 matmul | 明确诊断缺少目标 |
| memref semantics matmul | 本 Demo 明确拒绝或单独支持 |
| 非 rank-2 operand/result | 明确诊断不支持的 rank |
| 动态 shape | 静态 shape 合同时明确拒绝 |
| 未实现预期 interface | 明确诊断 interface |

负例使用：

```bash
$MLIR_BIN/mlir-opt matmul-invalid.mlir \
  -pass-pipeline='builtin.module(
    func.func(
      phase2-matmul-tile{
        tile-m=0 tile-n=64 tile-k=32
      }))' \
  -verify-diagnostics
```

## 保存 IR Evolution

最终 Demo 不只保存一个 after 文件，而是保存：

```text
00-input.mlir
01-tiled.mlir
02-bufferized.mlir
03-vectorized.mlir
```

### `00-input.mlir`

主要结构：

```text
tensor linalg.matmul
```

### `01-tiled.mlir`

主要结构：

```text
scf.for
tensor.extract_slice
小 linalg.matmul
tensor.insert_slice
```

### `02-bufferized.mlir`

主要结构：

```text
memref / subview
buffer-based linalg
必要的 alloc / copy
```

### `03-vectorized.mlir`

主要结构：

```text
vector.transfer_read
vector.contract / vector.fma
vector.transfer_write
```

每个 snapshot 必须记录完整生成命令，不能只保存最终 IR 后反推变化来源。

记录时要回答：

- 哪个 Pass 创建了当前 Operation；
- 原始 Tensor slice 是否 bufferize 为 subview/alias；
- 是否产生额外 allocation/copy；
- tile shape 如何影响 vector shape；
- 边界 tile 是否需要 mask 或动态 transfer；
- 哪些旧 analysis 因 IR mutation 失效。

## 构建与注册

可编译真源接入流程：

```text
Phase2*.cpp
  → test/lib/Transforms/CMakeLists.txt
  → tools/mlir-opt/mlir-opt.cpp 中声明/注册
  → 构建 mlir-opt
  → mlir-opt --help 验证 Pass 可见
```

构建：

```bash
cmake --build /home/zhuangkudecha/llvm-project/build \
  --target mlir-opt \
  --parallel 1
```

本机内存有限，Release + assertions 构建大型 MLIR 翻译单元时应保守使用单路，避免
`cc1plus` 被 OOM killer 终止。

注册验证：

```bash
$MLIR_BIN/mlir-opt --help | rg 'phase2-|matmul-tile'
```

## 统一运行脚本

`Demo/scripts/run-all.sh` 可按顺序执行：

```bash
#!/usr/bin/env bash
set -euo pipefail

MLIR_BUILD=/home/zhuangkudecha/llvm-project/build
MLIR_OPT="$MLIR_BUILD/bin/mlir-opt"
FILECHECK="$MLIR_BUILD/bin/FileCheck"

"$MLIR_OPT" tests/inspection.mlir \
  -pass-pipeline='builtin.module(func.func(phase2-inspect))' \
  | "$FILECHECK" tests/inspection.mlir

"$MLIR_OPT" tests/add-zero.mlir \
  -pass-pipeline='builtin.module(func.func(phase2-add-zero))' \
  | "$FILECHECK" tests/add-zero.mlir

"$MLIR_OPT" tests/producer-consumer.mlir \
  -pass-pipeline='builtin.module(func.func(phase2-fuse-chain))' \
  | "$FILECHECK" tests/producer-consumer.mlir

"$MLIR_OPT" tests/conversion.mlir \
  -phase2-convert-to-llvm \
  | "$FILECHECK" tests/conversion.mlir

"$MLIR_OPT" tests/matmul-tile.mlir \
  -pass-pipeline='builtin.module(
    func.func(
      phase2-matmul-tile{
        tile-m=64 tile-n=64 tile-k=32
      }))' \
  | "$FILECHECK" tests/matmul-tile.mlir

"$MLIR_OPT" tests/matmul-invalid.mlir \
  -pass-pipeline='builtin.module(
    func.func(
      phase2-matmul-tile{
        tile-m=0 tile-n=64 tile-k=32
      }))' \
  -verify-diagnostics
```

脚本中的 Pass 名称、pipeline anchor 和测试路径必须与最终注册结果保持一致。

## 推荐实施顺序

不要一次写完五个 Pass。每一步都保持可构建、可运行、可测试：

1. 建立 Demo 目录、README 和最小输入。
2. 实现 `phase2-inspect`，确认注册与构建链。
3. 实现 `phase2-add-zero`，建立 Pattern/FileCheck 测试方式。
4. 实现 Producer–Consumer 的只读 matcher，再增加 rewrite。
5. 实现独立的 mini conversion Pass。
6. 先运行 Transform Dialect tiling，不写 C++ tiling。
7. 实现参数化 C++ tiling Pass。
8. 增加非整除和失败矩阵。
9. 保存 tiled/bufferized/vectorized snapshots。
10. 用 `run-all.sh` 作为最终验收入口。

## Demo 完成判据

最终完成标准不是“Demo 能运行一次”，而是：

```text
构建成功
  + Pass 能在 --help 中找到
  + 所有正例 FileCheck 通过
  + 所有负例 diagnostics 通过
  + 第二次运行达到预期稳定状态
  + 每阶段 IR 可复现
  + 能解释每个变化来自哪个机制
```

完整 Exit Gate：

- [ ] 能区分 Operation、Value、BlockArgument、Region 和 Block。
- [ ] Inspection Pass 能输出稳定的 use-def/dominance 证据。
- [ ] AddZero 的 Red/Green 实验证明变化来自自定义 Pattern。
- [ ] Producer–Consumer matcher 在全部门禁通过前不修改 IR。
- [ ] Fusion 正例减少目标 Operation，负例保持原结构。
- [ ] Full Conversion 后所有目标 illegal dialect 消失。
- [ ] Conversion 负例能验证缺少 legalization path。
- [ ] Transform sequence 能精确匹配并 tile 目标 matmul。
- [ ] 能解释 payload IR、Transform IR、handle 和 TransformState。
- [ ] 能从空文件实现带 `tile-m/n/k` 的 Matmul Tiling Pass。
- [ ] 参数和所有目标在 mutation 前完成验证。
- [ ] 使用 `SCFTilingResult::replacements` 更新原 results。
- [ ] 整除、非整除及参数/shape/目标负例测试通过。
- [ ] 能解释 tiling 新建的 loops、slices、tiled ops 和 insertion ops。
- [ ] 能解释 folding、pattern、conversion、tiling 与 legality 的区别。
- [ ] 能沿 snapshots 解释 tiling 对 bufferization/vectorization 的影响。
- [ ] 能说明 mutation 后哪些 analysis 必须失效或重新计算。

---

上一章：[← Week 10：参数化 Tiling 与 IR Evolution](06_Parameterized_Tiling_Evolution.md)  
返回：[Phase 2 阅读地图](00_Reading_Map.md)
