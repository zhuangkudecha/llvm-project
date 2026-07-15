# RTX 4090 GPU CodeGen 执行计划分 Phase 设计

## 1. 背景

`LearningSteps/Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md` 当前同时承担全局规则、目录约定、24 周逐日任务、阶段门槛、参考资料和最终验收等职责，已经超过 2000 行。继续在单文件中细化每个 Step 的参考资料，会进一步增加查找和维护成本。

本次只拆分该执行计划。`LearningSteps/Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md` 保持单文件结构，除非链接新目录确有必要，否则不修改其正文。

## 2. 目标

将现有 24 周执行计划重构为一个总索引和七个 Phase 文档，同时做到：

- 保留全部 Week 1-24 任务、命令、Expected、负例和 Exit Gate；
- 为每个可执行 Step 配置直接相关的参考资料；
- 区分必读、实践参考和扩展阅读；
- 保持 24 周、264 小时以及现有阶段边界不变；
- 让学习者能够从总索引快速定位当前 Phase，并在单个 Phase 内完成连续学习。

## 3. 非目标

- 不改变 fused Linear + Bias + ReLU capstone workload；
- 不改变目标硬件 RTX 4090 或 Triton baseline；
- 不新增第八个 Phase；
- 不把每个 Week 拆成独立文件；
- 不在本次改动中实现计划里的 pass、kernel、benchmark 或 profiler 脚本；
- 不重写 `AI_Compiler_Engineer_Learning_Plan.md`。

## 4. 文档结构

```text
LearningSteps/
  Plans/
    GPU_CodeGen/
      RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
      Phases/
        Phase1_CUDA_Triton_Kernel_Fundamentals.md
        Phase2_MLIR_Transformation_Bridge.md
        Phase3_Triton_Compiler_Internals.md
        Phase4_Baseline_Freeze.md
        Phase5_Performance_Optimization.md
        Phase6_Generalization_Regression.md
        Phase7_Engineering_Delivery.md
```

### 4.1 总索引职责

总索引保留：

- 目标、Architecture、Tech Stack 和 Source plan；
- 时间盒、日历映射和完成定义；
- Stop/Go 规则；
- 环境变量、固定目录和 `progress.md` 格式；
- Phase 导航表；
- 每个 Phase 的周次、工时、输入、核心产出和 Exit Gate 摘要；
- 最终硬验收、benchmark schema、实验记录模板和调整规则；
- 官方参考入口。

总索引不再包含 Week 1-24 的逐日任务。目标长度为约 300-500 行；若完整保存全局验收模板需要略微超过 500 行，以避免信息丢失为优先。

### 4.2 Phase 文档职责

每个 Phase 文档包含：

- 返回总索引的相对链接；
- 阶段目标、时间、先决条件和 Exit Gate；
- 本阶段涉及的 Week；
- 每周 Files、工作日、周末和验证门槛；
- 每个 Step 对应的参考资料；
- 阶段产出清单和进入下一 Phase 的 Stop/Go 条件。

Phase 文档不复制全局环境变量、benchmark schema 和统一调整规则，而是链接回总索引中的对应章节。

## 5. Phase 映射

| Phase | 周次 | 工时 | 文件 |
|---|---:|---:|---|
| Phase 1：CUDA/Triton Kernel 基础 | Week 1-4 | 44h | `Phase1_CUDA_Triton_Kernel_Fundamentals.md` |
| Phase 2：MLIR Transformation 桥接 | Week 5-10 | 66h | `Phase2_MLIR_Transformation_Bridge.md` |
| Phase 3：Triton Compiler 内部 | Week 11-13 | 33h | `Phase3_Triton_Compiler_Internals.md` |
| Phase 4：冻结 Baseline | Week 14-16 | 33h | `Phase4_Baseline_Freeze.md` |
| Phase 5：性能优化实验 | Week 17-20 | 44h | `Phase5_Performance_Optimization.md` |
| Phase 6：泛化与回归 | Week 21-22 | 22h | `Phase6_Generalization_Regression.md` |
| Phase 7：工程化交付 | Week 23-24 | 22h | `Phase7_Engineering_Delivery.md` |

合计 24 周、264 小时。

## 6. Step 结构

原计划中每个带 checkbox 的周一至周日任务视为一个 Step。每个 Step 使用以下结构：

````markdown
- [ ] **周三：接入 LLVMTypeConverter 和 patterns**

  **目标：** 使用 upstream conversion components 完成最小 lowering，不重复实现已有转换。

  **产出：** `test/lib/Transforms/MiniArithToLLVMConversionPass.cpp`

  **验证：**

  ```bash
  cmake --build $MLIR_BUILD --target mlir-opt
  $MLIR_BIN/mlir-opt --help | rg mini-arith-to-llvm
  ```

  **Expected：** 构建成功并能查询到 pass 参数。

  **参考资料：**

  - 必读文档：`docs/DialectConversion.md`，重点阅读 Type Conversion 和 Rewrite Pattern 部分。
  - 必读源码：`include/mlir/Transforms/DialectConversion.h`，定位 `TypeConverter` 和 conversion driver API。
  - Upstream Test：对应 conversion test，观察 RUN、输入和 FileCheck 结构。
  - 实践参考：官方教程或可复现示例，说明它帮助完成的具体动作。
  - 扩展阅读：论文、博客或视频，说明它补充的设计背景或性能视角。
````

如果原 Step 已经包含目标、命令或 Expected，则迁移并规范格式，不重复表达。纯阅读 Step 的产出可以是指定笔记章节，验证可以是必须回答的问题或源码调用链；不得只用“读完”作为 Expected。

## 7. 参考资料策略

### 7.1 分级

每个 Step 按需使用以下类别：

1. 必读文档：完成任务前必须掌握的官方说明；
2. 必读源码：当前 checkout 中需要定位的类、函数、pass 或 interface；
3. Upstream Test：可直接验证 API 使用方式和边界的测试；
4. 实践参考：官方 tutorial 或高质量可复现实例；
5. 扩展阅读：论文、工程博客或公开视频。

每个 Step 至少包含一项必读资料，并优先给出源码或 upstream test。类别没有合适内容时可以省略，禁止为了填满五类而添加弱相关链接。

### 7.2 注释要求

每条资料必须说明本 Step 的阅读目的，例如：

```text
include/mlir/IR/PatternMatch.h
  重点：OpRewritePattern、PatternBenefit、notifyMatchFailure 的职责边界
```

不得只列资料标题或裸链接。源码引用应尽量精确到文件和符号；测试引用应给出文件路径以及要观察的 RUN 或 CHECK 行为。

### 7.3 来源优先级

```text
当前 checkout 源码和 tests
  > MLIR/LLVM/Triton/NVIDIA 官方文档
  > 原始论文和作者技术材料
  > 高质量工程博客或公开视频
```

外部资料需要在线核验链接和内容相关性。官方技术文档可能随版本变化时，以固定 Triton commit、当前 MLIR checkout 和文档中记录的 CUDA/Nsight 环境为准，并标注版本边界。

## 8. 链接规则

- 总索引使用相对链接进入 Phase 文档；
- 每个 Phase 顶部提供返回总索引的相对链接；
- Phase 之间不复制下一阶段正文，只提供“下一 Phase”链接；
- 本地源码使用仓库相对路径；
- 外部资料使用 Markdown 链接并保留可识别标题；
- 相同资料可在多个 Step 重复引用，但必须标明各 Step 不同的阅读重点。

## 9. 迁移方法

1. 以现有大文档为唯一迁移源，先按 Phase 边界抽取 Week 内容；
2. 在七个 Phase 文档中保持原 Week 顺序；
3. 为每个 Step 补齐目标、产出、验证、Expected 和分级参考资料；
4. 将原文档缩减为总索引并添加 Phase 导航；
5. 对照迁移前后的 Week 标题、checkbox 数量、命令和 Exit Gate；
6. 校验本地路径、外部链接、Markdown 和 Git diff。

迁移过程不得先删除原大文档内容再凭记忆重建。必须先生成 Phase 文档并验证覆盖，再缩减总索引。

## 10. 验收标准

### 10.1 结构完整性

- 七个 Phase 文件全部存在；
- 总索引链接到七个 Phase，七个 Phase 均能返回总索引；
- Week 1-24 在 Phase 文档中各出现一次；
- 周次与 Phase 映射符合第 5 节；
- 总工时仍为 264 小时。

### 10.2 内容完整性

- 原计划所有周一至周日 checkbox 均已迁移；
- 原计划中的命令、Expected、negative case、Exit Gate 和 Stop/Go 条件没有丢失；
- 每个 Step 至少有一项直接相关的必读资料；
- 每条资料说明阅读目的；
- 没有未决标记、跨项省略引用或无法执行的占位描述。

### 10.3 技术校验

- 本地文档、源码和 test 引用路径在当前 checkout 中存在；
- 外部链接经过在线核验；
- Markdown 代码围栏成对；
- 相对链接目标存在；
- `git diff --check` 返回成功；
- 只修改本设计覆盖的学习计划文档，不覆盖用户无关改动。

## 11. 风险与处理

### 11.1 参考资料导致文档再次膨胀

资料放在使用它的 Step 下，但每项只保留标题、路径或链接、阅读重点。长篇概念解释继续放在已有 `LearningSteps/*_Summary.md` 中，Phase 文档只链接，不复制正文。

### 11.2 重复资料产生维护成本

允许必要的重复引用，因为 Step 必须可独立执行；通过不同阅读重点说明重复的理由。全局安装和工具入口仍只保留在总索引。

### 11.3 upstream API 或路径变化

本地路径以当前 checkout 验证结果为准。Triton 和 CUDA 资料标明固定版本或记录核验日期；未来升级时单独更新引用，不改变历史实验基线。

### 11.4 拆分时遗漏任务

迁移前后比较 Week 标题、checkbox 数量、代码块中的命令和 Exit Gate 数量。任何计数下降都必须逐项解释，否则不通过验收。

## 12. 完成定义

只有在七个 Phase 文档、缩减后的总索引、逐 Step 参考资料和全部结构校验同时完成后，拆分任务才算完成。仅创建空 Phase 文件或仅把原正文机械移动而未补充资料，均不算完成。
