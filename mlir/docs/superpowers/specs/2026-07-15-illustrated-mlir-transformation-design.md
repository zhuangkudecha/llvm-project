# 图解 MLIR Transformation 小册设计

## 目标

在 `LearningSteps/IllustratedMLIR/Transformation/` 下创建一套独立的中文图解小册，
系统解释 MLIR 中四个相互关联但职责不同的变换子系统：

1. Pass 基础设施与 `PassManager`；
2. `RewritePattern`、`PatternRewriter` 与 pattern driver；
3. Dialect Conversion、legality、类型转换与 materialization；
4. Transform Dialect、payload IR、Transform IR、handle 与 interpreter。

小册采用“子系统手册”结构：先分别讲清四个机制，再用同一个可运行的
`linalg.matmul` 示例完成端到端串联。它不是现有总结文档的改写，而是与
`LearningSteps/Transforms/*_Summary.md` 并存的图解叙事层。

## 目标读者

- 已经理解 `Operation`、`Value`、`Region` 和 `Block` 的基本关系；
- 能阅读简单 MLIR 文本，但尚不清楚不同变换机制的职责边界；
- 希望通过可运行命令、对象关系图和当前源码入口理解 MLIR 变换系统；
- 需要一套既能顺序学习、又能按子系统查阅的中文材料。

## 已确认的设计决定

- 内容同时覆盖 Pass、PatternRewriter、Dialect Conversion 和 Transform Dialect。
- 使用独立目录 `LearningSteps/IllustratedMLIR/Transformation/`，不直接续写现有
  `IllustratedMLIR` 的章节编号。
- 采用子系统手册结构，而不是单一旅程结构或双层参考手册结构。
- 使用新的 `linalg.matmul` 作为贯穿整册的共享 payload IR。
- 每章都提供当前 `build/bin/mlir-opt` 可运行的验证命令。
- 不编写或编译自定义 C++ Pass/Pattern 插件；局部改写通过当前构建已注册的机制展示。
- 读者可见的图中文字以中文落盘，精确技术标识符保持英文。

## 目录结构

```text
LearningSteps/IllustratedMLIR/Transformation/
├── 00_Reading_Map.md
├── 01_What_Transforms_IR.md
├── 02_Pass_Infrastructure.md
├── 03_Pattern_Rewriter.md
├── 04_Dialect_Conversion.md
├── 05_Transform_Dialect.md
├── 06_End_To_End_Matmul.md
└── examples/
    ├── matmul.mlir
    └── matmul_transform.mlir
```

仓库只保存输入真源。`mlir-opt` 规范化输出、各降层阶段的中间 IR 和 Mermaid
渲染产物写入 `/tmp`，不作为本阶段交付物提交。

## 统一心智模型

四个子系统都作用于 payload IR，但回答的问题不同：

| 子系统 | 核心问题 | 主要职责 | 不负责的内容 |
|---|---|---|---|
| Pass 基础设施 | 什么时候、在哪个 Operation 上运行什么变换？ | pipeline、锚点、嵌套执行、analysis 生命周期、失败传播 | 定义每一条局部改写规则 |
| Pattern Rewriting | 一条局部 IR 规则如何匹配并安全修改 IR？ | pattern、rewriter、driver、工作队列、改写约束 | 判断整个目标 IR 是否已经合法 |
| Dialect Conversion | 如何保证非法 IR 最终变成目标合法 IR？ | legality、conversion pattern、类型映射、materialization | 作为用户可编程的外部调度语言 |
| Transform Dialect | 如何用 IR 表达、选择和编排变换？ | payload/transform 双 IR、handle、interpreter、effects、失效规则 | 替代底层 Pass、Pattern 或 Conversion 的实现机制 |

主要控制与执行关系为：

```text
PassManager / pipeline
  -> 在指定 Operation 边界运行 Pass
      -> Pass 可调用 pattern driver
      -> Pass 可调用 conversion driver

Transform Dialect interpreter
  -> 解释 Transform IR
  -> 通过 handle 选择 payload Operation
  -> 调用具体变换实现修改 payload IR

所有路径最终观察同一个 payload IR 的状态变化。
```

## 共享示例与演化路径

### `examples/matmul.mlir`

保存纯 payload IR。示例应满足以下条件：

- 使用 tensor 语义的 `linalg.matmul`；
- 形状较小、文本紧凑，适合在章节中完整展示；
- 包含可由内置 canonicalization/CSE 处理的冗余结构；
- 可经过 Transform Dialect tile；
- 可继续经过 bufferization 和 Linalg/SCF conversion；
- 在当前构建中能够被 parser 和 verifier 接受。

### `examples/matmul_transform.mlir`

保存 payload IR 和可执行的 Transform sequence。该文件用于展示：

- payload IR 与 Transform IR 是两套不同的 IR；
- Transform handle 指向 payload Operation，而不是复制 payload；
- `transform-interpreter` 如何执行 sequence；
- tile 后 handle 如何关联新生成的 Operation；
- 消耗或替换 payload Operation 后，旧 handle 为什么可能失效。

### 端到端状态链

最终章节以当前构建实际可用的 pass 为准，验证下列逻辑状态链：

```text
原始 tensor linalg.matmul
  -> Transform Interpreter 执行 tile
  -> tiled Linalg + loop structure
  -> One-Shot Bufferize
  -> bufferized Linalg
  -> Linalg/SCF Conversion
  -> 更低层 loop/control-flow IR
```

具体 pass 顺序必须在实施阶段用当前 `mlir-opt --help` 和真实运行结果确认，不把旧版
教程中的命令未经验证地写入正文。

## 章节设计

### `00_Reading_Map.md`

提供整册入口：

- 四个子系统的职责地图；
- 推荐阅读顺序和按问题查阅路径；
- 图中颜色约定；
- 共享示例运行方法；
- 当前完成状态；
- 与现有 `Transforms` 总结文档的对应关系。

### `01_What_Transforms_IR.md`

回答“MLIR 中是谁在变换 IR”：

- 从同一个 before/after `linalg.matmul` 例子出发；
- 区分调度、局部改写、合法化和 Transform IR 编排；
- 解释“Pass 不等于 Pattern”“Transform Dialect 不等于 Pattern Rewriting”；
- 给出全册统一架构图和术语表；
- 预告后续四个子系统章节。

### `02_Pass_Infrastructure.md`

回答“变换何时、在哪里运行”：

- `PassManager`、`OpPassManager` 与嵌套 pipeline；
- Pass 的 Operation 锚点和 static filtering；
- pass 执行边界和并行约束；
- analysis 的按需计算、缓存、preserve 与 invalidation；
- pass failure 如何沿 pipeline 传播；
- 文本 pipeline 如何映射到嵌套 Operation；
- 使用当前已注册 pass 对共享示例运行真实 pipeline。

核心图：嵌套 pipeline 图、pass/analysis 生命周期图。

### `03_Pattern_Rewriter.md`

回答“一条局部规则如何安全修改 IR”：

- `RewritePattern` 的 root、benefit 和 `matchAndRewrite`；
- 为什么所有修改都必须通过 `PatternRewriter`；
- replace、erase、in-place update 的生命周期约束；
- success 必须对应真实修改；
- bounded recursion；
- greedy/walk/conversion driver 的差异；
- 工作队列如何让新 IR 再次进入匹配；
- 使用内置 canonicalization pattern 展示真实 before/after IR。

核心图：match-rewrite 状态机、greedy driver 工作队列图。

由于本阶段不构建自定义插件，本章可以给出精简 C++ 伪代码或源码片段来解释 API，
但“可运行验证”必须来自当前工具已经注册的 pattern/pass。

### `04_Dialect_Conversion.md`

回答“降层何时算完成”：

- partial、full 和 analysis conversion；
- `ConversionTarget` 的 legal、dynamic legal、illegal 与 unknown；
- conversion driver 如何寻找合法化路径；
- conversion pattern 与普通 rewrite pattern 的关系；
- adaptor/remapped operands 与原始 `op.getOperands()` 的区别；
- `TypeConverter`、region signature conversion 和 materialization；
- rollback/no-rollback 的观察边界；
- 通过共享示例运行真实 bufferization/conversion pipeline。

核心图：legality 决策图、原始 operand/adaptor/type materialization 数据流图。

### `05_Transform_Dialect.md`

回答“为什么把变换本身也表示成 IR”：

- payload IR 与 Transform IR 的双 IR 模型；
- transform sequence、interpreter 和入口 Operation；
- handle 是对 payload Operation 的映射，不是 payload 的 SSA 数据流；
- Transform Dialect 中的 effects；
- handle consumption、payload 替换与 invalidation；
- silenceable failure 与 definite failure；
- Transform Dialect 如何调用已有的底层变换，而不是替代它们；
- 使用 `transform-interpreter` 匹配并 tile 共享 `linalg.matmul`。

核心图：双 IR/handle 映射图、handle invalidation 时间线。

### `06_End_To_End_Matmul.md`

把四章重新串联：

- 冻结原始输入和实验命令；
- 展示每个阶段的关键 Operation 变化；
- 标注每个阶段由哪个控制机制调度、哪个底层机制修改 IR；
- 记录 tile、bufferize、conversion 后的结构断言；
- 给出失败定位路径：pipeline 锚点、pattern 未命中、legality 未满足、handle 失效；
- 用一张总时间线总结完整执行流程。

## 每章统一模板

每个机制章节严格使用以下阅读顺序：

1. 一个具体问题；
2. 共享示例的 before/after IR；
3. 职责与对象关系图；
4. 运行时执行时间线；
5. 与另外三个机制的边界及常见误区；
6. 当前 checkout 的源码入口；
7. 可运行命令、输出结构断言和验证记录；
8. 一张回顾图及下一章连接。

每张 Mermaid 图后必须有一段文字，明确告诉读者应该检查哪条边、哪个所有权关系或
哪个状态变化。图不能代替正文中的精确定义。

## 视觉契约

沿用 `IllustratedMLIR` 已建立的语义颜色，不按子系统重新分配颜色：

| 颜色 | 固定含义 |
|---|---|
| 蓝色 | `Operation` 与 payload IR |
| 红色 | `Value`、SSA 数据流和 Transform handle |
| 绿色 | `Region` |
| 黄色 | `Block` |
| 紫色 | `Dialect`、`Type` 与具体语义 |
| 灰色 | Pass、driver、工具、源码和其他控制上下文 |

Mermaid 使用 light `base` theme、白色背景、深色文字和稳定的 class definition。
读者可见的标题、节点、边、图例、caption 和 walkthrough 使用中文；
`Operation`、`PatternRewriter`、`ConversionTarget`、`transform.sequence` 等精确标识符
保留英文，并在首次出现时给出中文说明。

## 资料与源码依据

现有学习材料只作为内容输入和深入阅读链接，不作为唯一事实来源：

- `LearningSteps/Transforms/PassManagement_Summary.md`
- `LearningSteps/Transforms/PassManagement_OperationPass_Restrictions_Summary.md`
- `LearningSteps/Transforms/PatternRewriter_Summary.md`
- `LearningSteps/Transforms/DialectConversion_Summary.md`
- `LearningSteps/Transforms/Phase2_Core_Infrastructure_Summary.md`
- `LearningSteps/Transforms/Phase2_AI_Compiler_Beginner_Guide.md`

实施时还必须核对当前 checkout 中的官方资料与源码：

- `docs/PassManagement.md`
- `docs/PatternRewriter.md`
- `docs/DialectConversion.md`
- `docs/Dialects/Transform.md`
- `docs/Tutorials/transform/`
- `include/mlir/Pass/`
- `include/mlir/IR/PatternMatch.h`
- `include/mlir/Transforms/GreedyPatternRewriteDriver.h`
- `include/mlir/Transforms/DialectConversion.h`
- `include/mlir/Dialect/Transform/`
- 相应的 `lib/` 实现和 `test/` 回归用例。

所有源码链接以当前仓库相对路径落盘，避免依赖外部网页版本。

## 验证策略

### IR 验证

- 所有 `.mlir` 输入先单独通过 parser 和 verifier；
- 每章命令退出码必须为 0，且无未解释诊断；
- 使用 `rg` 对关键 Operation 的出现或消失做结构断言；
- 不把输出文本完全相等作为主要断言，避免 SSA 名称等非语义打印变化造成脆弱测试；
- 端到端章节保留每个阶段的 `/tmp` 输出，便于定位失败发生在哪一步。

### Mermaid 验证

- 使用当前 `/home/fuhao/.npm-global/bin/mmdc` 渲染每个 Markdown；
- 每个 Mermaid fence 必须生成一个 SVG；
- 渲染退出码为 0 且没有语法诊断；
- 渲染产物只用于验证，不提交到本小册目录。

### 内容验证

- 两个示例文件和七个 Markdown 文件全部存在；
- 所有章节含中文正文和中文图中说明；
- 不包含任何未完成标记或占位文本；
- 所有仓库内相对链接目标存在；
- `git diff --check` 通过；
- 手动确认图中的英文只用于精确技术标识符；
- 现有 `LearningSteps/Transforms` 文件保持不变。

## 失败处理

- 如果教程命令与当前构建不匹配，先用 `mlir-opt --help` 确认注册名，再核对当前源码
  和测试；正文记录当前可复现命令，不照搬旧命令。
- 如果共享示例无法穿过完整 pipeline，优先缩小或调整示例，同时保持 `linalg.matmul`
  主线；不通过加入自定义插件绕开问题。
- 如果某个 pattern 的内部调度无法从 Optimized build 直接打印，使用 before/after IR、
  当前源码和上游测试三者交叉说明，不伪造 debug log。
- 如果 Mermaid 图过密，将一张图拆成职责图和时间线图，而不是缩小字体或省略关键关系。
- 如果端到端 pipeline 某阶段失败，逐阶段运行并检查最后一个成功的 `/tmp` 输出，
  分别判断是锚点、pattern、legality、类型桥接还是 handle 生命周期问题。

## 非目标

- 不覆盖现有 `LearningSteps/Transforms` 总结文件；
- 不编写自定义 C++ Pass、Pattern 或 Transform Dialect extension；
- 不讲 GPU 映射、硬件调优或性能基准；
- 不穷举所有 Transform Dialect extension operation；
- 不把 canonicalization 描述为正确性依赖；
- 不把 `include/mlir/Transform/` 等其他组件与 Transform Dialect 或 Pattern Rewriting 混为一谈；
- 不提交 Mermaid 渲染产物或 `mlir-opt` 中间输出。

## 完成标准

本小册在同时满足以下条件时完成：

1. 七个章节和两个共享示例均已落盘；
2. 四个子系统各自的职责、执行模型、边界、源码入口和真实命令均已讲清；
3. 同一个 `linalg.matmul` 能贯穿独立章节并完成端到端串联；
4. 所有 IR、Mermaid、链接、中文落盘、未完成标记和空白检查通过；
5. 最终章节能够明确指出每个阶段“谁调度、谁修改、如何判断成功”；
6. 现有 `LearningSteps/Transforms` 材料没有被修改。
