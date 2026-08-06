# Phase 2 Transformation Demo：Step-by-Step 实现指南

本文把 [07_Phase2_Transformation_Demo.md](./07_Phase2_Transformation_Demo.md) 中的设计拆成可以逐项完成、逐项验证的实现步骤。

本文不是完成后的代码，而是一份动手清单。每一步都包含：

1. 本步目标；
2. 需要修改的文件；
3. 关键实现；
4. 编译和运行命令；
5. 完成标准；
6. 常见失败及其原因。

与普通的“照抄代码教程”不同，本文还要求在每一步回答三个问题：

1. **Pass 在哪一层运行？** 是 `builtin.module`，还是嵌套的 `func.func`；
2. **它依赖哪一种 MLIR 机制？** 是只读 Analysis、Pattern Rewrite、Dialect Conversion，还是 Tiling Interface；
3. **如何证明它真的做了预期工作？** 不能只看命令返回 0，还要检查注册、诊断、IR 结构、负例和幂等性。

最终要得到的不是五段互不相关的代码，而是一条逐层增加能力的主线：

```text
读懂 IR
  │
  ├─ phase2-inspect            只读：Value / CFG / Dominance
  │
  ├─ phase2-add-zero           局部：一个 Operation 的等价替换
  │
  ├─ phase2-fuse-chain         图级：沿 use-def 链识别多个 Operation
  │
  ├─ phase2-convert-to-llvm    全局：用 legality 驱动 Dialect Conversion
  │
  └─ phase2-matmul-tile        结构化：通过 TilingInterface 生成循环与切片
```

## 阅读目标与当前仓库基线

本文中的“当前状态”和“完成后的目标状态”必须区分开。2026-08-06 对工作区实测得到：

| 项目 | 当前状态 | 本文目标 |
|---|---|---|
| Inspection | 已编译、已注册，命令名仍是 `phase2-inspection` | 统一成 `phase2-inspect`，修正稳定输出和负例 |
| AddZero | 已编译、已注册，基本正例可运行 | 补齐 TypeID、测试矩阵、幂等性和 Red/Green |
| FuseChain | `Phase2FuseChain.cpp` 是未接线的空骨架 | 完成严格 matcher、fusion 和正负测试 |
| ConvertToLLVM | 文件尚不存在 | 新增 Module Pass 和 full-conversion 测试 |
| MatmulTiling | 文件尚不存在 | 先验证 Transform Dialect，再实现参数化 C++ Pass |
| 自动化与快照 | 尚不存在 | 增加脚本、结构检查和可追溯快照 |

当前插件的真实可见 Pass 可用下面的命令重新确认：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  --help | rg 'phase2-'
```

当前预期只看到：

```text
--phase2-add-zero
--phase2-inspection
```

这张表也是进度表。每完成一个 Step，都应回到这里确认“源码存在”“完成四处接线”和“测试通过”是三件不同的事。

## 开始前先建立三个心智模型

### 心智模型 A：插件负责注册，PassManager 负责执行

```text
libPhase2Passes.so
  └─ mlirGetPassPluginInfo()
       └─ registerPassRegistryCallbacks()
            └─ registerPhase2XxxPass()
                 └─ PassRegistration<Phase2XxxPass>

命令行 pipeline 字符串
  └─ PassManager 解析 pass argument
       └─ 从 registry 创建 Pass 实例
            └─ 在匹配的 Operation 层级执行 runOnOperation()
```

因此，“`.so` 能被加载”不等于“Pass 已注册”，“Pass 出现在 `--help`”也不等于“pipeline 锚点写对了”。这三个边界要分别验证。

### 心智模型 B：Operation、Region、Block、Value 是不同层级

```text
func.func                     Operation
└── body                      Region
    ├── ^entry(%arg0, %cond)  Block + BlockArgument
    │   ├── arith.constant    Operation -> OpResult
    │   └── cf.cond_br        Operation
    ├── ^left                 Block
    ├── ^right                Block
    └── ^merge(%result)       Block + BlockArgument
        └── func.return       Operation
```

`OpResult` 的 owner 是 Operation；`BlockArgument` 的 owner 是 Block。它们都属于 `Value`，也都可以有 use-list，但来源、生命周期和支配规则不同。

### 心智模型 C：测试要分成四层

| 测试层 | 要证明什么 | 典型命令 |
|---|---|---|
| 加载层 | `.so` 能被动态加载 | `mlir-opt --load-pass-plugin=... --help` |
| 注册层 | argument 已进入 Pass registry | `--help \| rg 'phase2-'` |
| 语义层 | IR 确实按合同改变或保持 | `mlir-opt ... \| FileCheck` |
| 拒绝层 | 非法 IR/不支持输入得到明确诊断 | `-verify-diagnostics` |

后面的每个 Step 都按这四层组织验证，避免只用一个 happy-path 命令判断“已经完成”。

## 0. 先确定唯一的 C++ 真源位置

原设计要求把 Pass 真源放在 `test/lib/Transforms/`，但当前工作区已经存在一个可独立构建的插件工程：

```text
LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/
├── CMakeLists.txt
├── passes/
│   ├── Phase2Passes.h
│   ├── PluginEntry.cpp
│   ├── Phase2Inspection.cpp
│   └── Phase2AddZero.cpp
└── ...
```

为了让学习过程保持增量、避免每增加一个 Pass 都重编整个 `mlir-opt`，本文后续采用当前已有的动态插件路线：

```bash
mlir-opt --load-pass-plugin=Demo/libPhase2Passes.so ...
```

必须遵守一个原则：

> 后续 C++ 只放在 `Demo/passes/`，不要同时再向 `test/lib/Transforms/` 复制一份。

如果以后决定严格切回 upstream test-support 路线，应一次性迁移全部 Pass，并删除插件版真源。两条路线的业务实现相同，区别只在 CMake、注册入口和运行命令。

## 1. 建立统一工作变量

下面命令默认从 LLVM monorepo 根目录运行：

```bash
cd /home/fuhao/llvm-project

export MLIR_ROOT=/home/fuhao/llvm-project
export MLIR_BIN="$MLIR_ROOT/build/bin"
export PHASE2_DEMO="$MLIR_ROOT/mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo"
export PHASE2_PLUGIN="$PHASE2_DEMO/libPhase2Passes.so"
```

这些变量的职责不同：

- `MLIR_ROOT` 指向 monorepo 根目录，不是本文所在的 `mlir/` 子目录；
- `MLIR_BIN` 指向已经构建的 host tools，插件本身不会生成 `mlir-opt`；
- `PHASE2_DEMO` 是源码、测试、脚本和插件产物的根目录；
- `PHASE2_PLUGIN` 必须指向最终 `.so`，不能误指向 `build/` 下的对象文件。

建议在每个新终端先做一次路径预检：

```bash
test -x "$MLIR_BIN/mlir-opt"
test -x "$MLIR_BIN/FileCheck"
test -f "$PHASE2_DEMO/CMakeLists.txt"
printf 'mlir-opt=%s\nplugin=%s\n' \
  "$MLIR_BIN/mlir-opt" "$PHASE2_PLUGIN"
```

先确认工具与插件加载能力：

```bash
"$MLIR_BIN/mlir-opt" --version
"$MLIR_BIN/mlir-opt" --help | rg 'load-pass-plugin|transform-interpreter'
```

当前 checkout 应显示 LLVM/MLIR 23 开发版，并包含：

```text
--load-pass-plugin
--transform-interpreter
```

### 1.1 第一次配置与后续增量构建

第一次构建需要先生成 Ninja 工程：

```bash
cmake -G Ninja \
  -S "$PHASE2_DEMO" \
  -B "$PHASE2_DEMO/build" \
  -DMLIR_DIR="$MLIR_ROOT/build/lib/cmake/mlir" \
  -DLLVM_DIR="$MLIR_ROOT/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Debug

ninja -C "$PHASE2_DEMO/build"
```

后续只要 CMake 已包含新增源文件，直接运行：

```bash
ninja -C "$PHASE2_DEMO/build"
```

区分两类错误：

- `ninja: no work to do`：源文件没有变化，或者新文件根本没加入 `add_library`；
- 编译成功但 `--help` 找不到 Pass：通常是头文件声明、PluginEntry 调用或旧 `.so` 路径有问题。

### 1.2 为什么插件不直接链接 MLIR 静态库

这个 Demo 的 `.so` 只保存 Pass 的实现和注册入口。`mlir::registerPass`、Rewrite Driver、Dialect Conversion 等主体符号在加载时由 host `mlir-opt` 提供：

```text
编译期：Phase2Passes.so 允许保留未解析的 MLIR 符号
运行期：mlir-opt dlopen(.so)，动态链接器从 host 导出符号中解析它们
```

这就是 CMake 中 `MODULE`、`--allow-shlib-undefined` 和 host 的 `--export-dynamic` 必须配套的原因。若加载时报 `undefined symbol`，依次检查：

```bash
# 插件还缺哪些动态符号
nm -D -C "$PHASE2_PLUGIN" | rg ' U '

# host 是否导出了目标符号；把 Foo 换成错误消息里的符号关键字
nm -D -C "$MLIR_BIN/mlir-opt" | rg 'Foo'

# 实际 host 链接命令是否包含 export-dynamic
ninja -C "$MLIR_ROOT/build" -t commands mlir-opt \
  | tail -n 1 | rg 'export-dynamic'
```

不要只看旧的 `CMakeCache.txt` 推断 host 能力；最终应以真实链接命令和动态符号表为准。

## 2. 理解新增一个插件 Pass 的四处接线

每增加一个 `Phase2Xxx.cpp`，都要修改四处：

### 2.1 在头文件声明注册函数

文件：`Demo/passes/Phase2Passes.h`

```cpp
void registerPhase2XxxPass();
```

### 2.2 在唯一插件入口调用注册函数

文件：`Demo/passes/PluginEntry.cpp`

```cpp
[]() {
  registerPhase2InspectionPass();
  registerPhase2XxxPass();
}
```

### 2.3 把源文件加入插件 target

文件：`Demo/CMakeLists.txt`

```cmake
add_library(Phase2Passes MODULE
  passes/Phase2Inspection.cpp
  passes/Phase2Xxx.cpp
  passes/PluginEntry.cpp
)
```

### 2.4 在实现文件定义注册函数

```cpp
void registerPhase2XxxPass() {
  PassRegistration<Phase2XxxPass>();
}
```

编译：

```bash
ninja -C "$PHASE2_DEMO/build"
```

检查 Pass 是否真的进入 registry：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  --help | rg 'phase2-'
```

只写 `PassRegistration` 还不够。如果没有完成头文件、插件入口和 CMake 三处接线，Pass 不会出现在命令行中。

### 2.5 四处接线各自解决什么问题

| 位置 | 作用 | 漏掉后的现象 |
|---|---|---|
| `Phase2Xxx.cpp` | 定义 Pass 类和注册函数 | 没有实现，或链接时找不到注册函数 |
| `Phase2Passes.h` | 让入口文件看到注册函数声明 | `PluginEntry.cpp` 编译失败 |
| `PluginEntry.cpp` | 插件加载时真正调用注册函数 | `.so` 可加载，但 `--help` 没有该 Pass |
| `CMakeLists.txt` | 把实现编入当前 `.so` | 链接失败，或一直加载旧实现 |

建议每新增一个 Pass 后立刻执行下面这个最小闭环：

```bash
ninja -C "$PHASE2_DEMO/build"

test -f "$PHASE2_PLUGIN"

"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  --help | rg -- '--phase2-xxx'
```

最后一条只证明注册成功，还没有证明 Pass 能在正确的 IR 层级执行。

### 2.6 Pipeline 锚点为何必须匹配 Pass 类型

`OperationPass<func::FuncOp>` 只能在 `func.func` 上运行，所以它需要嵌套在 module pipeline 中：

```text
builtin.module(func.func(phase2-add-zero))
```

`OperationPass<ModuleOp>` 已经运行在 module 上，不应再塞进 `func.func(...)`：

```text
builtin.module(phase2-convert-to-llvm)
```

可以把 pipeline 看成一棵与 IR 结构对齐的执行树：

```text
builtin.module                         IR: module
└── func.func                          IR: func.func @foo
    ├── phase2-inspect                 在该函数上运行一次
    └── phase2-add-zero                在该函数上运行一次
```

当一个 module 有三个函数时，嵌套的 Function Pass 会创建/执行三次；Module Pass 只执行一次，但可以遍历整个 module。

## 3. Step 1：完成 `phase2-inspect`

### 3.1 先统一 Pass 名称

设计文档使用 `phase2-inspect`，当前原型使用 `phase2-inspection`。建议现在统一改成：

```cpp
StringRef getArgument() const final { return "phase2-inspect"; }
```

随后同步修改 README、测试命令和脚本。不要长期保留两个相近名字。

### 3.2 只输出稳定信息

文件：`Demo/passes/Phase2Inspection.cpp`

保留以下聚合信息：

```text
function=inspect
operations=<整数>
regions=<整数>
blocks=4
```

遍历 `OpResult` 时输出：

```text
op-result owner=arith.constant result=0 type=i32 users=2
```

遍历 `BlockArgument` 时输出：

```text
block=0 argument=0 type=i32 users=2
```

注意：

- 不输出 `Operation *`、`Block *` 等裸指针；
- 不依赖 use-list 的迭代顺序；
- Block 编号来自函数 body 的稳定顺序；
- 只读 Pass 最后调用 `markAllAnalysesPreserved()`。

#### 3.2.1 三种统计不能混为一谈

当前实现中的计数逻辑等价于：

```cpp
op->walk([&](Operation *nested) {
  ++operationCount;
  regionCount += nested->getNumRegions();
  for (Region &region : nested->getRegions())
    blockCount += std::distance(region.begin(), region.end());
});
```

对 `01-inspection.mlir` 来说：

- `operations=8` 包含 `func.func` 本身、常量、两个 `arith.addi`、三个 branch 和 return；
- `regions=1` 是 `func.func` 的 body region；
- `blocks=4` 是 entry、left、right、merge 四个 Block。

如果改写 walk 后数量变化，先确认回调是否包含根 Operation，不要立即修改 FileCheck 期待值来掩盖语义差异。

#### 3.2.2 为什么分别遍历 OpResult 和 BlockArgument

下面两段遍历回答的是不同问题：

```cpp
for (OpResult result : nested->getOpResults()) {
  // 谁定义了这个结果？它被使用几次？
}

for (BlockArgument argument : block.getArguments()) {
  // 这个值是函数入口参数，还是由 CFG 前驱传入？
}
```

以 `%result` 为例：

```mlir
^left:
  cf.br ^merge(%left_value : i32)
^right:
  cf.br ^merge(%right_value : i32)
^merge(%result: i32):
  return %result : i32
```

`%result` 不是某个 op 的计算结果，而是 merge block 的第 0 个参数。两条 `cf.br` edge 都向这个参数提供运行时值；但 `%result` 自身的 use-list 只有 `return` 这一个 use。因此，“有两个 incoming value”不等于“这个 BlockArgument 有两个 users”。

#### 3.2.3 DominanceInfo 查询的含义

`dom.dominates(&lhs, &rhs)` 表示：所有从函数入口到 `rhs` 的 CFG 路径都必须经过 `lhs`。当前 CFG 为：

```text
             ┌──> ^left ──┐
^entry ──────┤             ├──> ^merge
             └──> ^right ─┘
```

| 查询 | 结果 | 原因 |
|---|---:|---|
| entry → left | 1 | 到 left 必经 entry |
| entry → merge | 1 | 两条路径都从 entry 出发 |
| left → merge | 0 | 可以经 right 到 merge |
| right → merge | 0 | 可以经 left 到 merge |
| merge → merge | 1 | Block 支配自身 |

这个矩阵也解释了为什么 `%left_value` 不能直接在 merge 中使用：存在一条 `entry -> right -> merge` 路径没有执行它的定义。

#### 3.2.4 让输出可被测试稳定消费

当前源码仍输出 `users= 2`，等号后多了一个空格；目标格式是 `users=2`。建议统一成：

```cpp
llvm::errs() << "op-result"
             << " owner=" << nested->getName()
             << " result=" << result.getResultNumber()
             << " type=" << result.getType()
             << " users=" << std::distance(result.use_begin(),
                                             result.use_end())
             << '\n';
```

诊断信息建议走 `llvm::errs()`，转换后的 IR 走 stdout。测试使用 `2>&1` 显式合流，正常 IR pipeline 则不会把 inspection 文本混入输出文件。

### 3.3 修正 dominance 负例

当前 `tests/01-inspection-invalid.mlir` 需要同时声明 error 和 definition note：

```mlir
^left:
  // expected-note @+1 {{operand defined here}}
  %value = arith.constant 1 : i32
  cf.br ^merge

^merge:
  // expected-error @+1 {{operand #0 does not dominate this use}}
  return %value : i32
```

诊断匹配里不要在 `{{` 后人为添加空格；实际错误文本从 `operand` 开始。

这里的 `expected-note` 不是可选装饰。Verifier 通常先在错误使用点报告 “does not dominate this use”，再在定义点给出 “operand defined here”。`-verify-diagnostics` 要求这些诊断都能与源码注释对应。

调试时先不加 `-verify-diagnostics`，直接观察原始文本：

```bash
"$MLIR_BIN/mlir-opt" \
  "$PHASE2_DEMO/tests/01-inspection-invalid.mlir"
```

确认实际位置和措辞后再添加标注，最后运行 `-verify-diagnostics`。不要把 `2>&1 | FileCheck` 与 `-verify-diagnostics` 混为一谈：前者检查普通文本流，后者检查 MLIR diagnostic 与源码注释的对应关系。

### 3.4 验证

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -pass-pipeline='builtin.module(func.func(phase2-inspect))' \
  "$PHASE2_DEMO/tests/01-inspection.mlir" \
  2>&1 | tee /tmp/phase2-inspect.out

"$MLIR_BIN/FileCheck" \
  "$PHASE2_DEMO/tests/01-inspection.mlir" \
  < /tmp/phase2-inspect.out

"$MLIR_BIN/mlir-opt" \
  "$PHASE2_DEMO/tests/01-inspection-invalid.mlir" \
  -verify-diagnostics
```

### 3.5 完成标准

- [ ] 能区分 `OpResult` 和 `BlockArgument`；
- [ ] 能统计 users；
- [ ] 能查询 Block dominance；
- [ ] 非法 SSA 被 verifier 拒绝；
- [ ] `-verify-diagnostics` 返回 0；
- [ ] Pass 声明所有 analysis preserved。

### 3.6 Inspection 常见失败定位

| 现象 | 优先检查 | 原因 |
|---|---|---|
| `unknown pass 'phase2-inspect'` | `--help \| rg phase2` | 源码仍叫 `phase2-inspection` 或加载了旧 `.so` |
| `operations` 少 1 | walk 是否包含根 op | walk 起点或重载选择不同 |
| merge 参数 users 预期成 2 | 区分 incoming operands 与 uses | CFG 入边数不是 Value use 数 |
| dominance 负例只有 error 仍失败 | 原始输出中的 note | `-verify-diagnostics` 也匹配附属 note |
| FileCheck 找不到输出 | stdout/stderr 是否合流 | `llvm::errs()` 需要 `2>&1` |
| 错误 preserve analysis | Pass 是否真的只读 | 只有完全不改变 IR 才能 preserve all |

## 4. Step 2：完成 `phase2-add-zero`

当前 `Phase2AddZero.cpp` 已有 Pattern 雏形，并且已经进入 CMake、注册头和插件入口。当前正例实测可以把 `arith.addi %arg0, %zero` 化简为直接 `return %arg0`。本 Step 的重点不再是“让它第一次跑起来”，而是把实现合同、类型身份和测试证据补完整。

### 4.1 先修正 Pass 实现

文件：`Demo/passes/Phase2AddZero.cpp`

增加本地注册头：

```cpp
#include "Phase2Passes.h"
```

Pattern 只实现右侧零合同：

```cpp
struct AddZeroPattern : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      arith::AddIOp op,
      PatternRewriter &rewriter) const override {
    APInt rhs;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&rhs)) ||
        !rhs.isZero())
      return failure();

    rewriter.replaceOp(op, op.getLhs());
    return success();
  }
};
```

Pass 类需要稳定 TypeID 和标准覆写：

```cpp
struct Phase2AddZeroPass
    : PassWrapper<Phase2AddZeroPass,
                  OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Phase2AddZeroPass)

  StringRef getArgument() const final {
    return "phase2-add-zero";
  }

  StringRef getDescription() const final {
    return "Rewrite scalar integer x + 0 to x";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddZeroPattern>(&getContext());

    GreedyRewriteConfig config;
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(
            getOperation(), std::move(patterns), config)))
      signalPassFailure();
  }
};
```

文件末尾增加：

```cpp
void registerPhase2AddZeroPass() {
  PassRegistration<Phase2AddZeroPass>();
}
```

然后按第 2 节完成三处接线。

当前三处接线已经存在，修改后仍要逐项复核，避免重命名 Pass 类时只改实现文件：

```text
Phase2Passes.h    registerPhase2AddZeroPass 声明存在
PluginEntry.cpp   callback 中调用 registerPhase2AddZeroPass
CMakeLists.txt    add_library 中包含 Phase2AddZero.cpp
```

### 4.1.1 Pattern 的 match 与 rewrite 分别做什么

```text
arith.addi %lhs, %rhs
              │
              ├─ matchPattern(%rhs, m_ConstantInt(&rhsValue))
              │    └─ 沿定义找到整数常量，并提取 APInt
              │
              ├─ rhsValue.isZero()
              │    └─ 合同只接受右操作数为 0
              │
              └─ rewriter.replaceOp(op, %lhs)
                   ├─ 把 add 的所有 result use 改接到 %lhs
                   └─ 删除已无结果用途的 add op
```

`replaceOp` 不是简单文本替换。它维护 SSA use-def 链：先替换所有结果的 use，再安全擦除旧 Operation。不要手工遍历 user 并直接 `erase()`；那样很容易留下悬空 use。

### 4.1.2 为什么只匹配右侧零

虽然整数加法满足交换律，但此实验故意保持窄合同：

```mlir
%a = arith.addi %x, %zero : i32  // 改写
%b = arith.addi %zero, %x : i32  // 不改写
```

这样做能让负例真正验证 pattern 的 operand 方向，也能清楚区分：

- 自己写的 `AddZeroPattern`；
- `arith.addi` 自带的 canonicalization/folding；
- 另一个用于交换左右操作数的 pattern。

后续若决定支持左侧零，应新增独立 matcher 或明确扩展合同，并同步修改负例，而不是让测试模糊地接受两种结果。

### 4.1.3 为什么关闭 folding

`applyPatternsGreedily` 除了反复应用 pattern，也可能调用 fold。这里设置：

```cpp
GreedyRewriteConfig config;
config.enableFolding(false);
```

目的是建立因果证据：`x + 0 -> x` 必须由 `AddZeroPattern::matchAndRewrite` 触发。否则即使暂时删除自己的 pattern，通用 fold 仍可能得到相同 IR，测试会“绿”，却没有测试到本文的代码。

Greedy driver 会继续扫描受影响的 Operation，直到达到不动点。例如：

```mlir
%a = arith.addi %x, %zero : i32
%b = arith.addi %a, %zero : i32
```

同一次 Pass 应先后消除两层 add。幂等性测试则证明达到不动点后，再跑一遍不会继续改变 IR。

### 4.2 准备输入

文件：`Demo/inputs/02-add-zero.mlir`

```mlir
func.func @add_zero(%arg0: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %result = arith.addi %arg0, %zero : i32
  return %result : i32
}
```

### 4.3 建立测试矩阵

文件：`Demo/tests/02-add-zero.mlir`

至少包含六个函数：

| 函数 | 输入 | 预期 |
|---|---|---|
| `rhs_zero_i32` | `i32 x + 0` | 改写 |
| `rhs_zero_i64` | `i64 x + 0` | 改写 |
| `lhs_zero` | `0 + x` | 保持 |
| `non_zero` | `x + 1` | 保持 |
| `shared_zero` | zero 还有其他 user | 只删 add，不误删 zero |
| `shaped` | shaped/非 `arith.addi` | 不误匹配 |

关键检查示例：

```mlir
// CHECK-LABEL: func.func @rhs_zero_i32
// CHECK-NEXT: return %arg0 : i32

// CHECK-LABEL: func.func @lhs_zero
// CHECK: arith.addi
```

测试文件最好直接包含 `RUN` 行，使它既能手工执行，也能未来接入 lit：

```mlir
// RUN: mlir-opt --load-pass-plugin=%phase2_plugin \
// RUN:   -pass-pipeline='builtin.module(func.func(phase2-add-zero))' %s \
// RUN:   | FileCheck %s
```

当前仓库还是“输入文件 + 独立 `.check` 文件”的最小形态：

```text
inputs/02-add-zero.mlir
tests/02-add-zero.check
```

扩成测试矩阵时，应统一选一种布局。本文建议使用 `tests/02-add-zero.mlir`，把输入和 CHECK 放在同一文件中；否则每增加一个函数都要同时维护输入与检查文件，定位失败更绕。

`shared_zero` 需要特别注意：如果常量还有另一个 user，消除 add 后常量不能被误删；如果常量只服务于 add，greedy rewrite 后它是否消失取决于 driver 的 DCE 行为。测试应检查语义合同，不要把无关常量是否保留写成过度脆弱的约束。

### 4.4 证明幂等性

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -pass-pipeline='builtin.module(func.func(phase2-add-zero),func.func(phase2-add-zero))' \
  "$PHASE2_DEMO/inputs/02-add-zero.mlir" \
  > /tmp/add-zero-twice.mlir

"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -pass-pipeline='builtin.module(func.func(phase2-add-zero))' \
  /tmp/add-zero-twice.mlir \
  | diff -u /tmp/add-zero-twice.mlir -
```

`diff` 没有输出才表示第二次运行不再变化。

### 4.5 做一次 Red/Green

1. 保留 Pattern，测试应通过；
2. 临时注释 `patterns.add<AddZeroPattern>`，重新编译；
3. FileCheck 必须失败；
4. 恢复 Pattern，重新编译；
5. FileCheck 再次通过。

这一步用来证明结果来自自己的 Pattern，而不是 folding。

### 4.6 手工跟踪一次 Rewrite

调试 pattern 是否命中时，可以打开 rewrite 日志：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -debug-only=greedy-rewriter \
  -pass-pipeline='builtin.module(func.func(phase2-add-zero))' \
  "$PHASE2_DEMO/inputs/02-add-zero.mlir" \
  2>&1 | less
```

这要求当前 `mlir-opt` 构建保留 LLVM debug logging；若提示未知参数或没有日志，不代表 pattern 没执行，仍以输出 IR 和 Red/Green 为准。

还可以使用通用 Pass instrumentation 观察前后 IR：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -pass-pipeline='builtin.module(func.func(phase2-add-zero))' \
  -mlir-print-ir-before=phase2-add-zero \
  -mlir-print-ir-after=phase2-add-zero \
  "$PHASE2_DEMO/inputs/02-add-zero.mlir" \
  >/dev/null
```

### 4.7 AddZero 完成标准

- [ ] `--help` 中只有一个 `phase2-add-zero`；
- [ ] Pass 类包含显式 TypeID；
- [ ] 只改写右侧整数零；
- [ ] `i32`、`i64` 正例通过；
- [ ] 左零、非零和非目标 op 保持；
- [ ] folding 关闭后 Red/Green 仍能证明 pattern 生效；
- [ ] 连续运行两次输出无 diff。

## 5. Step 3：实现 `phase2-fuse-chain`

新建：`Demo/passes/Phase2FuseChain.cpp`

当前文件已经存在，但只是空骨架：`matchAndRewrite()` 无条件返回 success，`runOnOperation()` 为空，而且没有定义注册函数。这个状态不能接入插件：Pattern 声称 success 却没有修改 IR，会违反 RewritePattern 的进展约定，并可能让 greedy driver 反复尝试。

开始实现前，先把骨架修到“安全但不匹配”的状态：

```cpp
LogicalResult matchAndRewrite(
    linalg::GenericOp relu,
    PatternRewriter &rewriter) const override {
  return rewriter.notifyMatchFailure(relu, "not implemented yet");
}
```

然后再逐条加入 matcher。`success()` 只能表示本次调用确实完成了 rewrite。

本实验不尝试做通用图优化，只接受一个严格合同：

```text
linalg.matmul
  → linalg.generic broadcast bias add
  → linalg.generic maximumf(x, 0)
```

其中箭头表示 SSA use-def，不表示 Operation 在文本中恰好相邻：

```text
matmul.result #0
  └─ bias operand #0
       └─ bias.result #0
            └─ relu operand #0
                 └─ relu.result #0
```

从 ReLU 反向匹配的原因是它同时包含最终替换点和完整 consumer 上下文。若从 matmul 向 users 正向搜索，会遇到 user 顺序不稳定、多个 user 分支以及“哪个 generic 才是 bias”这些额外问题。

### 5.1 从最外层 ReLU 开始匹配

Pattern root：

```cpp
struct FuseChainPattern
    : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      linalg::GenericOp relu,
      PatternRewriter &rewriter) const override;
};
```

反向查询：

```cpp
auto bias = relu.getDpsInputs()[0]
                .getDefiningOp<linalg::GenericOp>();
auto matmul = bias.getDpsInputs()[0]
                  .getDefiningOp<linalg::MatmulOp>();
```

每次解引用前都要先检查对象是否存在。

建议不要写成一条很长的链式表达式。详细写法更适合定位失败：

```cpp
Value reluInput = relu.getDpsInputs()[0];
auto bias = reluInput.getDefiningOp<linalg::GenericOp>();
if (!bias)
  return rewriter.notifyMatchFailure(
      relu, "relu input is not defined by linalg.generic");

Value biasInput = bias.getDpsInputs()[0];
auto matmul = biasInput.getDefiningOp<linalg::MatmulOp>();
if (!matmul)
  return rewriter.notifyMatchFailure(
      relu, "bias input is not defined by linalg.matmul");
```

`getDefiningOp<T>()` 对 BlockArgument 返回空；因此函数参数直接送入 bias 时会自然拒绝，而不会解引用空指针。

### 5.2 第一阶段只读门禁

在创建或删除任何 Operation 前完成全部检查。

ReLU 门禁：

- pure tensor semantics；
- 一个 input、一个 init；
- 两个 parallel loop；
- input/output indexing map 都是 identity；
- region 中只有 `arith.constant 0.0`、`arith.maximumf`、`linalg.yield`；
- `maximumf` 必须是 `max(block_argument_0, 0.0)`。

Bias 门禁：

- pure tensor semantics；
- 两个 inputs、一个 init；
- 两个 parallel loop；
- maps 严格为 identity、`(d0,d1)->(d1)`、identity；
- region 中只有 `arith.addf` 和 `linalg.yield`；
- bias result 只有 ReLU 一个 user。

Matmul 门禁：

- 必须是 `linalg.matmul`；
- pure tensor semantics；
- result 只有 bias 一个 user；
- lhs、rhs、result 都是 rank-2 ranked tensor；
- bias 是 rank-1 tensor；
- element type 一致；
- bias 长度等于输出 N 维。

推荐把 region 检查拆成：

```cpp
static bool hasExpectedBiasBody(linalg::GenericOp op);
static bool hasExpectedReluBody(linalg::GenericOp op);
static bool hasExpectedMaps(linalg::GenericOp bias,
                            linalg::GenericOp relu);
```

#### 5.2.1 每类门禁在防止什么

| 门禁 | 防止的错误 |
|---|---|
| pure tensor semantics | buffer/memref op 可能有可观察写入，不能按纯 SSA result 随意替换 |
| 单结果、单 user | 删除 producer 时不能丢失其他数据流分支 |
| iterator types | reduction generic 与逐元素 generic 语义不同 |
| indexing maps | identity 与 broadcast 决定每个元素读哪个坐标 |
| region 精确结构 | 不能把任意 `linalg.generic` 猜成 bias 或 ReLU |
| rank/shape | 防止 bias 广播维与输出 N 维不一致 |
| element type | 防止融合后 region 参数或 arith op 类型不合法 |

检查 region 时不要只搜索“里面存在一个 `arith.addf`”。还要验证：

1. Block 参数数量与 inputs/outs 一致；
2. `arith.addf` 的 operand 是预期的 input block arguments；
3. `linalg.yield` 返回的正是该 add/max 结果；
4. 除允许的 constant/add/max/yield 外没有额外 Operation；
5. 0.0 常量的类型与被比较元素类型一致。

否则下面这种 region 也可能被错误接受：

```mlir
%sum = arith.addf %value, %bias : f32
%scaled = arith.mulf %sum, %c2 : f32
linalg.yield %scaled : f32
```

它已经不再是单纯 bias add，融合合同必须拒绝。

#### 5.2.2 为什么必须先完成全部只读检查

PatternRewriter 不是自动数据库事务。如果先融合 bias/ReLU，再发现 matmul shape 不合法并返回 failure，IR 可能已经部分改变，而 driver 会把这次调用当作“没有成功”。正确顺序是：

```text
定位候选链
  -> 检查 ReLU
  -> 检查 Bias
  -> 检查 Matmul
  -> 检查 use/shape/type/effect
  -> 所有门禁通过
  -> 第一次 mutation
```

可预见的失败都应发生在第一次 mutation 之前。公共 fusion API 自身失败时，则用 `notifyMatchFailure` 返回清楚原因。

### 5.3 第二阶段才执行 fusion

不要自己复制 `ElementwiseOpFusion.cpp` 的内部算法。严格匹配成功后调用当前 checkout 的公共 API：

```cpp
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

FailureOr<linalg::ElementwiseOpFusionResult> fused =
    linalg::fuseElementwiseOps(
        rewriter, &relu->getOpOperand(0));
if (failed(fused))
  return rewriter.notifyMatchFailure(
      relu, "elementwise fusion failed");

Value replacement =
    fused->replacements.lookup(relu.getResult(0));
if (!replacement)
  return failure();

rewriter.replaceOp(relu, replacement);
rewriter.eraseOp(bias);
return success();
```

这段 API 调用融合的是 bias generic 与 ReLU generic；matmul 仍作为融合后 generic 的 producer。最终结构预期是：

```text
变换前                         变换后
linalg.matmul                  linalg.matmul
  -> bias linalg.generic         -> fused linalg.generic
       -> relu linalg.generic         region: addf -> maximumf -> yield
```

因此正例的 `CHECK-COUNT-1: linalg.generic` 是在证明两个 elementwise generic 合为一个，不是在证明 matmul 也消失。

调用公共 API 后仍要通过 `fused->replacements` 查找原 ReLU result 的替代值，因为 fusion 结果可能包含多个 replacement，不能假定 `tiledOps.back()` 或某个固定 result 编号就是答案。

### 5.3.1 完成 Pass 外壳

Pattern 实现后，`runOnOperation()` 至少需要：

```cpp
void runOnOperation() override {
  RewritePatternSet patterns(&getContext());
  patterns.add<FuseChainPattern>(&getContext());

  GreedyRewriteConfig config;
  config.enableFolding(false);
  if (failed(applyPatternsGreedily(
          getOperation(), std::move(patterns), config)))
    signalPassFailure();
}
```

文件末尾还必须定义：

```cpp
void registerPhase2FuseChainPass() {
  PassRegistration<Phase2FuseChainPass>();
}
```

然后完成第 2 节的头文件、入口和 CMake 三处接线。编译前用下面的静态检查避免漏项：

```bash
rg -n 'registerPhase2FuseChainPass|Phase2FuseChain.cpp' \
  "$PHASE2_DEMO/passes" "$PHASE2_DEMO/CMakeLists.txt"
```

### 5.4 测试

准备：

```text
Demo/inputs/03-producer-consumer.mlir
Demo/tests/03-producer-consumer.mlir
```

正例检查：

```mlir
// CHECK-LABEL: func.func @fuses
// CHECK-COUNT-1: linalg.generic
// CHECK: arith.addf
// CHECK: arith.maximumf
```

负例至少建立独立函数：

- bias result 有第二个 user；
- ReLU 使用 `arith.addf`，不是 `maximumf`；
- bias map 不是 broadcast map；
- iterator 包含 reduction；
- bias dtype 不同；
- bias shape 不兼容。

每个负例都检查仍有两个 `linalg.generic`。最后重复运行 Pass，并用 `diff` 验证幂等性。

### 5.5 Fusion 调试顺序

如果正例没有融合，不要立刻放宽全部门禁。按 use-def 链逐层确认：

1. Pattern root 是否遍历到了 ReLU `linalg.generic`；
2. ReLU operand 0 的 defining op 是否为 bias generic；
3. Bias operand 0 的 defining op 是否为 matmul；
4. 哪一个 helper 首次返回 false；
5. 公共 `fuseElementwiseOps` 是否失败；
6. replacement map 是否包含原 ReLU result。

调试版可以使用 `notifyMatchFailure` 的具体文本配合 `-debug-only=greedy-rewriter`。最终保留有意义的失败原因，但测试不要依赖每个内部 debug 字符串，它们不是用户诊断合同。

### 5.6 Fusion 完成标准

- [ ] Pattern 失败时不修改 IR；
- [ ] Pattern 返回 success 时一定发生了 rewrite；
- [ ] 正例从两个 generic 变为一个；
- [ ] matmul 保留，并成为融合 generic 的 producer；
- [ ] 所有 use/shape/map/body 负例保持原状；
- [ ] 连续运行两次输出稳定；
- [ ] `mlir-opt --verify-each` 下通过。

## 6. Step 4：实现 `phase2-convert-to-llvm`

新建：`Demo/passes/Phase2ConvertToLLVM.cpp`

这一阶段与 Greedy Rewrite 的核心区别是：目标不是“看到某个局部形状就优化”，而是声明转换后的 IR 中什么合法、什么非法，并要求 conversion driver 为所有非法 Operation 找到合法化路径。

```text
Greedy Rewrite
  输入 op -> pattern 可匹配就改，不匹配可以原样保留

Full Dialect Conversion
  输入 op -> 若 target 判定 illegal，就必须被 pattern 完全合法化
           -> 任一 illegal op 留下，整个 Pass 失败
```

### 6.1 这是 Module Pass

`func.func` 本身也要被转换为 `llvm.func`，所以 Pass 不能锚定在 `func::FuncOp`：

```cpp
struct Phase2ConvertToLLVMPass
    : PassWrapper<Phase2ConvertToLLVMPass,
                  OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      Phase2ConvertToLLVMPass)
  // ...
};
```

运行 pipeline：

```text
builtin.module(phase2-convert-to-llvm)
```

不要写成：

```text
builtin.module(func.func(phase2-convert-to-llvm))
```

原因不仅是 pipeline 语法。转换会把 `func.func` Operation 本身替换成 `llvm.func`；如果 Pass 被锚定在这个即将被替换的 FunctionOp 上，它无法自然负责 module 内符号和所有函数边界的整体转换。

### 6.2 声明依赖 dialect

```cpp
void getDependentDialects(
    DialectRegistry &registry) const override {
  registry.insert<LLVM::LLVMDialect>();
}
```

### 6.3 建立 TypeConverter、patterns 和 legality

```cpp
LLVMTypeConverter typeConverter(&getContext());
RewritePatternSet patterns(&getContext());

arith::populateArithToLLVMConversionPatterns(
    typeConverter, patterns);
populateFuncToLLVMConversionPatterns(
    typeConverter, patterns);
cf::populateControlFlowToLLVMConversionPatterns(
    typeConverter, patterns);

ConversionTarget target(getContext());
target.addLegalDialect<LLVM::LLVMDialect>();
target.addLegalOp<ModuleOp>();
target.addIllegalDialect<arith::ArithDialect>();
target.addIllegalDialect<func::FuncDialect>();
target.addIllegalDialect<cf::ControlFlowDialect>();

if (failed(applyFullConversion(
        getOperation(), target, std::move(patterns))))
  signalPassFailure();
```

这段代码的三个对象职责要分清：

| 对象 | 职责 | 不负责什么 |
|---|---|---|
| `LLVMTypeConverter` | 把 builtin/function 类型映射到 LLVM dialect 类型 | 不主动遍历和替换 op |
| `RewritePatternSet` | 提供 `arith`、`func`、`cf` 到 LLVM 的改写方案 | 不决定哪些 op 必须消失 |
| `ConversionTarget` | 声明最终合法性合同 | 不提供如何转换的实现 |

`applyFullConversion` 把三者组合起来：遇到 illegal op 时搜索 pattern，转换 operands/results/types，并在结束时重新检查合法性。

#### 6.3.1 为什么 Module 仍要显式 legal

根 `ModuleOp` 不属于 LLVM dialect。若只写：

```cpp
target.addLegalDialect<LLVM::LLVMDialect>();
```

conversion target 并不会自动推断 module 应保留。显式 `target.addLegalOp<ModuleOp>()` 表示容器合法，但其内部非法 op 仍必须继续转换。

#### 6.3.2 Full、Partial、Analysis Conversion 的区别

| Driver | 允许目标中残留 illegal op | 是否修改 IR | 本 Demo 用途 |
|---|---:|---:|---|
| `applyFullConversion` | 否 | 是 | 最终 lower 到 LLVM |
| `applyPartialConversion` | 否，但未标记为 illegal 的 op 可保留 | 是 | 分阶段 lowering |
| `applyAnalysisConversion` | 只分析可否合法化 | 否 | 预检/调试 |

本实验选择 full conversion，是为了把“没有 source dialect 残留”变成可执行合同，而不只是 FileCheck 的愿望。

需要的主要头文件：

```cpp
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
```

### 6.4 输入必须覆盖 CFG 传值

文件：`Demo/inputs/04-conversion.mlir`

不要只测试一个直线 `arith.addi`。加入：

```text
cf.cond_br
  → 两个分支分别 add/sub
  → cf.br 向 merge block 传 i32
  → merge BlockArgument
```

这条输入同时覆盖四件事：

1. `func.func` signature 的类型转换；
2. `arith.addi/subi` 的 Operation 转换；
3. `cf.cond_br/cf.br` terminator 的转换；
4. successor operands 与 merge BlockArgument 的类型一致性。

只测试直线 add 会漏掉第 3、4 类问题，而 CFG 正是 Dialect Conversion 经常暴露类型映射错误的位置。

### 6.5 正负测试

正例检查：

```mlir
// CHECK: llvm.func
// CHECK: llvm.cond_br
// CHECK: llvm.add
// CHECK: llvm.sub
// CHECK: llvm.br
// CHECK-NOT: func.func
// CHECK-NOT: arith.
// CHECK-NOT: cf.
```

负例放一个没有 legalization pattern 的 op，例如：

```mlir
// expected-error @+1 {{failed to legalize operation 'math.absf'}}
%result = math.absf %arg0 : f32
```

运行：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  -pass-pipeline='builtin.module(phase2-convert-to-llvm)' \
  "$PHASE2_DEMO/tests/04-conversion.mlir" \
  | "$MLIR_BIN/FileCheck" \
      "$PHASE2_DEMO/tests/04-conversion.mlir"
```

建议额外打开逐 Pass verifier：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  --verify-each \
  -pass-pipeline='builtin.module(phase2-convert-to-llvm)' \
  "$PHASE2_DEMO/tests/04-conversion.mlir" \
  >/tmp/phase2-converted.mlir
```

`CHECK-NOT: arith.` 一类检查应放在合适的 `CHECK-LABEL` 范围内，避免它只检查文件末尾的一小段。转换成功还应再次 parse 输出：

```bash
"$MLIR_BIN/mlir-opt" /tmp/phase2-converted.mlir >/dev/null
```

### 6.6 Conversion 失败定位

| 诊断 | 含义 | 处理方式 |
|---|---|---|
| `failed to legalize operation 'X'` | X 被判 illegal，但没有成功 pattern | 补 conversion pattern 或调整明确的 legality 合同 |
| materialization/type mismatch | producer/consumer 类型转换边界断裂 | 检查 TypeConverter 和 block argument/signature conversion |
| unknown pass | 注册或 pipeline 锚点错误 | 先查 `--help`，再确认是 Module Pass |
| 插件 undefined symbol | host 未导出 conversion API | 用 `nm -D -C` 核验实际符号 |

不要为了让测试通过而把失败 op 一律 `addLegalDialect`。那会把“尚未 lower”伪装成“已经完成转换”。

### 6.7 Conversion 完成标准

- [ ] Pass 锚定 `ModuleOp`；
- [ ] target、patterns、type converter 职责清晰；
- [ ] 正例包含分支和 merge block argument；
- [ ] 输出不含 `func`、`arith`、`cf` source op；
- [ ] 未提供 pattern 的 illegal op 产生精确诊断；
- [ ] 输出能再次被 parser/verifier 接受。

## 7. Step 5：先用 Transform Dialect 做 Matmul Tiling

文件：`Demo/inputs/05-transform-matmul.mlir`

Payload 和 transform sequence 放在同一个文件。不要把 transform 文件错误地当成 Pass option 传入。

这一阶段先不写 C++，目的是把算法问题与 C++ API 接线问题拆开：

```text
Transform Dialect 先证明：
  当前 linalg.matmul 可被 tile
  tile size 的 M/N/K 顺序正确
  预期 loop/slice/insert 结构成立

C++ Pass 再证明：
  Option 能正确传入
  TilingInterface API 使用正确
  replacement 与失败处理正确
```

如果 Transform Dialect 都不能生成预期结构，直接写 C++ 只会同时面对 IR、算法、API 和插件四类变量。

### 7.1 Payload module

使用原设计中的 tensor `linalg.matmul`：

```mlir
module {
  func.func @matmul(
      %lhs: tensor<128x256xf32>,
      %rhs: tensor<256x512xf32>)
      -> tensor<128x512xf32> {
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
}
```

### 7.2 Transform module

```mlir
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {
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
}
```

这里存在两个不同世界：

- **Transform IR**：`transform.named_sequence`、handle、match、tile 指令；
- **Payload IR**：真正被变换的 `func.func`、`linalg.matmul`、tensor value。

`%matmul` 的类型是 `!transform.any_op`，它是对 payload operation 的句柄，不是 `tensor<128x512xf32>`。Transform IR 的 SSA 支配规则与 handle 生命周期用于编排变换；Payload IR 的 SSA 则描述程序数据流。

#### 7.2.1 tile size 与循环维的对应关系

对 matmul：

```text
C[M, N] += A[M, K] * B[K, N]
```

`tile_sizes [64, 64, 32]` 对应：

| 循环维 | 语义 | tile size |
|---|---|---:|
| d0 | M，输出行 | 64 |
| d1 | N，输出列 | 64 |
| d2 | K，reduction | 32 |

因此三层 loop 不只是数量为 3，还应分别控制 A/B/C slice 的 offset 与 size。K 维 tiling 还涉及 reduction 的初值与迭代累积，不能把它当作普通 parallel 维。

#### 7.2.2 handle 被消费后的规则

`tile_using_for` 会替换原 payload matmul。之后：

- 原 `%matmul` handle 指向的 payload op 已失效；
- `%tiled` 指向新生成的 tiled matmul；
- `%loops:3` 指向三层 `scf.for`；
- 后续 transform 必须使用返回的新 handle。

如果继续使用被消费/失效的 handle，应得到 Transform Dialect 的 silenceable 或 definite failure，而不是悄悄作用于旧 op。

### 7.3 运行和检查

```bash
"$MLIR_BIN/mlir-opt" \
  "$PHASE2_DEMO/inputs/05-transform-matmul.mlir" \
  --transform-interpreter \
  > /tmp/phase2-transform-tiled.mlir
```

测试：

```mlir
// CHECK-COUNT-3: scf.for
// CHECK: tensor.extract_slice
// CHECK: linalg.matmul
// CHECK: tensor.insert_slice
```

这一步要主动观察：

- `%matmul` 是 payload handle，不是 payload SSA value；
- tiling 消耗并替换旧 payload op 后，旧 handle 不能继续当作新 matmul；
- `%tiled` 关联新的 tiled matmul；
- `%loops:3` 分别关联 M、N、K 三层循环。

### 7.4 不要混用 `tile_using_for` 与 `tile_using_forall`

二者输出形态和并行语义不同：

| Transform op | 主要循环 | 写回结构 | 典型含义 |
|---|---|---|---|
| `tile_using_for` | `scf.for` | `tensor.insert_slice` | 顺序循环、可含 reduction tiling |
| `tile_using_forall` | `scf.forall` | `tensor.parallel_insert_slice` | 并行迭代空间 |

本文的 C++ 目标明确选择 `SCFTilingOptions::LoopType::ForOp`，所以基线测试也必须用 `tile_using_for`。不能拿已有 forall 示例的输出直接作为 C++ ForOp 的 FileCheck 期待。

### 7.5 Transform Tiling 完成标准

- [ ] match 只抓到预期 matmul；
- [ ] 返回 1 个 tiled handle 和 3 个 loop handle；
- [ ] 输出包含 3 层 `scf.for`；
- [ ] 输出包含 extract/insert slice；
- [ ] 内层仍有 tiled `linalg.matmul`；
- [ ] M/N/K tile size 与 slice 维度一致；
- [ ] 不复用已经失效的旧 handle。

## 8. Step 6：实现参数化 C++ Tiling Pass

新建：`Demo/passes/Phase2MatmulTiling.cpp`

### 8.1 定义参数

```cpp
Option<int64_t> tileM{
    *this, "tile-m",
    llvm::cl::desc("M dimension tile size"),
    llvm::cl::init(64)};
Option<int64_t> tileN{
    *this, "tile-n",
    llvm::cl::desc("N dimension tile size"),
    llvm::cl::init(64)};
Option<int64_t> tileK{
    *this, "tile-k",
    llvm::cl::desc("K reduction tile size"),
    llvm::cl::init(32)};
```

Pass 锚定 `func::FuncOp`：

```cpp
OperationPass<func::FuncOp>
```

命令行中的 option 必须放在 Pass 的 `{...}` 内，而不是作为全局参数：

```text
phase2-matmul-tile{tile-m=64 tile-n=64 tile-k=32}
```

解析后 `Option<int64_t>` 在每个 Pass 实例上保存配置；不要用全局变量保存 tile size，否则多条 pipeline 或并发 PassManager 会共享错误状态。

### 8.2 第一遍：只读收集和验证

先收集：

```cpp
SmallVector<linalg::MatmulOp> targets;
getOperation().walk(
    [&](linalg::MatmulOp op) { targets.push_back(op); });
```

mutation 前依次拒绝：

1. `tileM/tileN/tileK <= 0`；
2. 函数中没有 `linalg.matmul`；
3. 不是 pure tensor semantics；
4. lhs、rhs、init 或 result 不是 rank-2 ranked tensor；
5. 任一 tensor 是 dynamic shape；
6. op 未实现 `TilingInterface`；
7. result 数量不符合本 Demo 的单结果合同。

还应验证静态 shape 的 matmul 关系：

```text
lhs  = tensor<MxKxT>
rhs  = tensor<KxNxT>
init = tensor<MxNxT>
```

具体检查包括 lhs dim1 等于 rhs dim0、lhs dim0 等于 init dim0、rhs dim1 等于 init dim1，以及元素类型一致。虽然 `linalg.matmul` verifier 通常已经覆盖基本约束，Pass 自己的拒绝合同仍应说明它依赖 verifier 的哪些保证、额外限制了哪些输入。

诊断应锚定到具体函数或 matmul：

```cpp
return matmul.emitError(
    "phase2-matmul-tile requires static rank-2 tensor semantics");
```

当任一目标不合法时，调用 `signalPassFailure()` 并立即返回。此时 IR 尚未被修改。

“先收集再变换”还有一个生命周期原因：walk 中一边遍历一边把 matmul 替换成 loop nest，可能让 walk 进入刚生成的 IR 或使迭代器行为难以推断。先保存目标句柄并完成全量预检，可以把阶段边界固定下来。

### 8.3 第二遍：执行 tiling

以当前 checkout 的
`include/mlir/Dialect/SCF/Transforms/TileUsingInterface.h` 为准：

```cpp
IRRewriter rewriter(&getContext());

for (linalg::MatmulOp matmul : targets) {
  rewriter.setInsertionPoint(matmul);

  auto tileable =
      cast<TilingInterface>(matmul.getOperation());
  SmallVector<OpFoldResult> sizes{
      rewriter.getIndexAttr(tileM),
      rewriter.getIndexAttr(tileN),
      rewriter.getIndexAttr(tileK)};
  SmallVector<unsigned> reductionDims{2};

  scf::SCFTilingOptions options;
  options
      .setLoopType(
          scf::SCFTilingOptions::LoopType::ForOp)
      .setTileSizes(sizes)
      .setReductionDims(reductionDims);

  FailureOr<scf::SCFTilingResult> tiled =
      scf::tileUsingSCF(rewriter, tileable, options);
  if (failed(tiled)) {
    matmul.emitError("SCF tiling failed");
    signalPassFailure();
    return;
  }

  rewriter.replaceOp(matmul, tiled->replacements);
}
```

必须使用 `tiled->replacements`，不要猜测 `tiledOps.back()` 的 result 与原 matmul result 如何对应。

#### 8.3.1 每个 Tiling API 的角色

| API/对象 | 角色 |
|---|---|
| `IRRewriter` | 创建新 op、更新 use、删除旧 op |
| `TilingInterface` | 以统一接口暴露迭代域、切片与 tiled implementation |
| `SCFTilingOptions` | 选择 loop 类型、tile sizes、reduction dimensions |
| `tileUsingSCF` | 根据接口生成 loop nest、slice 和 tiled op |
| `SCFTilingResult::replacements` | 原 op 每个 result 对应的新 SSA value |

`rewriter.setInsertionPoint(matmul)` 必不可少：新循环必须插在旧 matmul 所在位置，才能支配原 result 的所有 users。插到函数尾部或某个 user 之后，会产生新的 dominance 错误。

#### 8.3.2 reductionDims `{2}` 的含义

Matmul 的迭代器通常是：

```text
[parallel, parallel, reduction]
     M         N         K
```

显式设置 `{2}` 告诉 SCF tiler 第三个维度是 reduction。K 分块时，每个 tile 的部分结果需要沿循环迭代累积，而不是像独立输出 tile 那样互不相关。

#### 8.3.3 `tileUsingSCF` 失败后的边界

前置验证可以排除可预测失败，但 `tileUsingSCF` 仍返回 `FailureOr`，调用者必须检查。不要写：

```cpp
auto tiled = *scf::tileUsingSCF(...); // 失败时直接解引用
```

也不要假设 failure 自动回滚此前所有 mutation。若函数包含多个 matmul，第一个已成功 tile、第二个 API 内部失败时，Pass 可能处于部分修改状态。Demo 的策略是尽量把失败提前到只读阶段，并把 API failure 视为 Pass 失败；若未来需要强事务语义，应在克隆 IR 或更高层 rollback 机制上单独设计。

### 8.4 运行

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  "$PHASE2_DEMO/inputs/06-matmul-tile.mlir" \
  -pass-pipeline='builtin.module(func.func(phase2-matmul-tile{tile-m=64 tile-n=64 tile-k=32}))'
```

先把输出保存下来再检查，不要直接在终端肉眼滚动：

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  --verify-each \
  "$PHASE2_DEMO/inputs/06-matmul-tile.mlir" \
  -pass-pipeline='builtin.module(func.func(phase2-matmul-tile{tile-m=64 tile-n=64 tile-k=32}))' \
  > /tmp/phase2-cpp-tiled.mlir

rg -n 'scf.for|extract_slice|linalg.matmul|insert_slice' \
  /tmp/phase2-cpp-tiled.mlir
```

### 8.5 与 Transform Dialect 基线比较

两条路径不要求 SSA 名称或 printer 排版逐字相同，但应比较这些语义结构：

| 结构 | Transform 路径 | C++ 路径 |
|---|---:|---:|
| loop 类型 | `scf.for` | `scf.for` |
| loop 层数 | 3 | 3 |
| tile 顺序 | M/N/K | M/N/K |
| slice | 有 | 有 |
| tiled matmul | 有 | 有 |
| result 写回 | `tensor.insert_slice` | `tensor.insert_slice` |

如果结构不同，先判断是 option、loop type、reduction dims 还是当前 API canonicalization 的差异，不要用整文件 `diff` 作为唯一正确性标准。

## 9. Step 7：参数化 Tiling 测试矩阵

### 9.1 整除输入

文件：`Demo/tests/06-matmul-tile.mlir`

尺寸：

```text
M=128, N=512, K=256
tile=[64,64,32]
```

检查：

```mlir
// CHECK-COUNT-3: scf.for
// CHECK: tensor.extract_slice
// CHECK: linalg.matmul
// CHECK: tensor.insert_slice
```

不要只检查常量，因为常量存在不等于 tiling 结构正确。

### 9.2 非整除输入

文件：`Demo/inputs/07-matmul-tail.mlir`

尺寸：

```text
M=127, N=509, K=251
tile=[64,64,32]
```

理论尾块：

```text
M tail = 63
N tail = 61
K tail = 27
```

检查动态尾块时捕获 SSA 定义，不要写死 `%17`：

```mlir
// CHECK: %[[REMAINING:.*]] = arith.subi
// CHECK: %[[SIZE:.*]] = arith.minsi
// CHECK: tensor.extract_slice {{.*}} [%[[SIZE]]]
```

实际输出可能经过 folding 而使用 `affine.min` 或其他等价形式。先查看当前 checkout 的真实输出，再把 FileCheck 锚定到语义稳定结构。

### 9.3 负例

文件：`Demo/tests/06-matmul-invalid.mlir`

分别验证：

- `tile-m=0`；
- `tile-n=-1`；
- 函数中没有 matmul；
- memref semantics matmul；
- 非 rank-2；
- dynamic shape。

每类拒绝应具有明确诊断，不能只依赖通用的 `failed to run pass`。

如果一个文件包含多个相互独立的负例，使用：

```text
// -----
```

分隔，并配合 `-split-input-file -verify-diagnostics`。不同 pass 参数对应的负例最好拆成独立文件，避免第一个错误遮蔽后续 case。

## 10. Step 8：保存 IR Evolution

只在前面所有正负测试通过后生成 snapshots：

```text
Demo/snapshots/
├── 00-input.mlir
├── 01-tiled.mlir
├── 02-bufferized.mlir
└── 03-vectorized.mlir
```

### 10.1 输入 snapshot

`00-input.mlir` 是经过 parser round-trip 的规范化输入：

```bash
"$MLIR_BIN/mlir-opt" \
  "$PHASE2_DEMO/inputs/06-matmul-tile.mlir" \
  > "$PHASE2_DEMO/snapshots/00-input.mlir"
```

### 10.2 Tiled snapshot

```bash
"$MLIR_BIN/mlir-opt" \
  --load-pass-plugin="$PHASE2_PLUGIN" \
  "$PHASE2_DEMO/snapshots/00-input.mlir" \
  -pass-pipeline='builtin.module(func.func(phase2-matmul-tile{tile-m=64 tile-n=64 tile-k=32}))' \
  > "$PHASE2_DEMO/snapshots/01-tiled.mlir"
```

### 10.3 Bufferized 与 Vectorized snapshot

这两步必须根据当前 checkout 实际可用的 Pass 组合确定，不要先写一个未经验证的长 pipeline。先查询：

```bash
"$MLIR_BIN/mlir-opt" --help \
  | rg 'one-shot-bufferize|linalg-vectorize|convert-linalg-to-loops'
```

然后分别在命令行中调通：

```text
01-tiled.mlir
  → one-shot-bufferize
  → 02-bufferized.mlir

01-tiled.mlir 或合适的 tiled/vectorizable 中间态
  → linalg/vector transform
  → 03-vectorized.mlir
```

保存前必须验证结构：

```bash
rg 'memref|subview|linalg' \
  "$PHASE2_DEMO/snapshots/02-bufferized.mlir"

rg 'vector.transfer_read|vector.contract|vector.fma|vector.transfer_write' \
  "$PHASE2_DEMO/snapshots/03-vectorized.mlir"
```

如果第二条没有匹配，不能把文件命名为 `03-vectorized.mlir`。

每个 snapshot 文件头部记录完整生成命令，例如：

```mlir
// Generated by:
// mlir-opt ... -pass-pipeline='...'
```

## 11. Step 9：编写自动化脚本

### 11.1 `scripts/run-all.sh`

脚本应：

1. 使用 `set -euo pipefail`；
2. 检查 `MLIR_BIN`、插件和 FileCheck 是否存在；
3. 重新构建插件；
4. 顺序执行 inspection、add-zero、fusion、conversion、Transform Dialect、C++ tiling；
5. 执行全部 verifier 负例；
6. 执行幂等性 diff；
7. 任一步失败立即以非零状态退出。

不要使用宽松的正则 alternation 把多个结构条件合成一个“任一命中即成功”的检查。

### 11.2 `scripts/generate-snapshots.sh`

脚本应：

1. 先调用 `run-all.sh`；
2. 使用临时目录生成结果；
3. 对每个临时结果执行结构检查；
4. 只有全部检查成功后才覆盖 `snapshots/`；
5. 打印每个 snapshot 的生成命令。

这样可以避免失败的中间结果覆盖上一份正确 snapshot。

## 12. 推荐实施顺序

严格按下面顺序推进，每一步绿灯后再进入下一步：

```text
1. 修正 inspection 正例与 dominance 负例
2. 接线并完成 add-zero
3. 完成 add-zero Red/Green 和幂等性
4. 准备 matmul/bias/ReLU 输入
5. 实现完整只读 matcher
6. 接入 elementwise fusion API
7. 完成 fusion 正负例和幂等性
8. 实现 module-level full conversion
9. 完成 conversion 正负例
10. 跑通 Transform Dialect tiling
11. 实现参数化 C++ tiling
12. 完成整除、尾块和拒绝合同
13. 生成并检查四个 snapshots
14. 用 run-all.sh 从干净构建状态复跑
```

不要同时写五个 Pass 后再统一编译。最小增量能让编译错误、注册错误、IR 语法错误和算法错误保持彼此隔离。

## 13. 最终验收表

### Pass 与注册

- [ ] `--help` 能看到五个 `phase2-*` Pass；
- [ ] 每个 Pass 只注册一次；
- [ ] 插件只有一个 `mlirGetPassPluginInfo`；
- [ ] `Demo/passes` 是唯一 C++ 真源。

### 语义

- [ ] inspection 不修改 IR；
- [ ] add-zero 的结果来自 Pattern，不是 Fold；
- [ ] fusion 在任何 mutation 前完成完整子图匹配；
- [ ] conversion 的剩余 op 全部 legal；
- [ ] tiling 使用 API 返回的 `replacements`。

### 测试

- [ ] 每个实验有正例；
- [ ] 每个拒绝合同有负例；
- [ ] rewrite/fusion/tiling 第二次运行稳定；
- [ ] dominance 与 conversion 负例使用 `-verify-diagnostics`；
- [ ] 非整除尾块不依赖临时 SSA 名称。

### IR Evolution

- [ ] `00-input.mlir` 包含 tensor matmul；
- [ ] `01-tiled.mlir` 包含三层 loop、slice 和 insert_slice；
- [ ] `02-bufferized.mlir` 确实包含 buffer-based 结构；
- [ ] `03-vectorized.mlir` 确实包含 vector 结构；
- [ ] 每个 snapshot 记录了完整生成命令。

完成以上清单后，这个 Demo 才真正把 Phase 2 的 SSA、Rewrite、子图匹配、Dialect Conversion、Transform Dialect 和参数化 Tiling 串成了一条可复现的 transformation 主线。
