# MLIR LearningSteps

这里集中存放 MLIR 学习路线、专题笔记、实践代码和 GPU CodeGen 训练计划。目录按稳定主题组织；本文档是统一入口，具体学习进度以计划文件为准。

## 推荐阅读顺序

1. [MLIR 学习计划](Plans/MLIR_Learning_Plan.md)
2. 基础概念：[IR 结构](Foundations/Step1_IR_Structure_Summary.md) → [语言参考](Foundations/Step2_Language_Reference_Summary.md) → [mlir-opt 工具链](Foundations/Step3_mlir_opt_Toolchain_Summary.md)
3. Toy 教程：[Ch1-Ch6](Tutorials/Step4_Toy_Tutorial_Summary.md) → [Ch7 代码结构](Tutorials/Step4_Ch7_Code_Structure_Summary.md)
4. 图解 MLIR：[阅读地图](IllustratedMLIR/00_Reading_Map.md) → [为什么一切都是 Operation](IllustratedMLIR/01_Why_Everything_Is_An_Operation.md)
5. [Core IR Headers](Foundations/Step5_Core_IR_Headers_Summary.md)
6. ODS：[TableGen 与 ODS](ODS/TableGen_ODS_Notes.md) → [Operations](ODS/Operations_Summary.md) → [Toy ODS 实践](ODS/ODS_Toy_Dialect_Practical_Summary.md)
7. Dialect：[定义 Dialect](Dialects/DefiningDialects_Summary.md) → [Builtin](Dialects/Builtin_Dialect_Summary.md) → [Func 概念总结](Dialects/Func_Dialect_Summary.md) → [Func 源码导读](Dialects/FuncDialect_Summary.md)
8. 变换框架：[Phase 2 初学者导读](Transforms/Phase2_AI_Compiler_Beginner_Guide.md) → [Phase 2 基础设施总结](Transforms/Phase2_Core_Infrastructure_Summary.md) → [PatternRewriter](Transforms/PatternRewriter_Summary.md) → [DialectConversion](Transforms/DialectConversion_Summary.md) → [PassManagement](Transforms/PassManagement_Summary.md)
9. 测试与实践：[FileCheck](Tools/FileCheck_Summary.md) → [canonicalize/CSE 输入](Practices/practice_canonicalize_cse.mlir) → [rewrite pattern 示例](Practices/rewrite-pattern-addi-zero/)

## 主题索引

### 学习计划

- [MLIR Learning Plan](Plans/MLIR_Learning_Plan.md)
- [GPU Kernel/CodeGen Engineer Learning Plan](Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md)
- [RTX4090 GPU CodeGen 24 周执行计划](Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)
- [RTX4090 分阶段计划](Plans/GPU_CodeGen/Phases/)
  - [Phase 1：CUDA 与 Triton Kernel 基础](Plans/GPU_CodeGen/Phases/Phase1_CUDA_Triton_Kernel_Fundamentals.md)
  - [Phase 2：MLIR Transformation Bridge](Plans/GPU_CodeGen/Phases/Phase2_MLIR_Transformation_Bridge.md)
  - [Phase 3：Triton Compiler Internals](Plans/GPU_CodeGen/Phases/Phase3_Triton_Compiler_Internals.md)
  - [Phase 4：Baseline Freeze](Plans/GPU_CodeGen/Phases/Phase4_Baseline_Freeze.md)
  - [Phase 5：Performance Optimization](Plans/GPU_CodeGen/Phases/Phase5_Performance_Optimization.md)
  - [Phase 6：Generalization & Regression](Plans/GPU_CodeGen/Phases/Phase6_Generalization_Regression.md)
  - [Phase 7：Engineering Delivery](Plans/GPU_CodeGen/Phases/Phase7_Engineering_Delivery.md)

### Foundations

- [Step 1：IR 结构](Foundations/Step1_IR_Structure_Summary.md)
- [Step 2：语言参考](Foundations/Step2_Language_Reference_Summary.md)
- [Step 3：mlir-opt 工具链](Foundations/Step3_mlir_opt_Toolchain_Summary.md)
- [Step 5：Core IR Headers](Foundations/Step5_Core_IR_Headers_Summary.md)

### Dialects

- [Defining Dialects](Dialects/DefiningDialects_Summary.md)
- [Builtin Dialect](Dialects/Builtin_Dialect_Summary.md)
- [Func Dialect 概念总结](Dialects/Func_Dialect_Summary.md)：适合先建立函数 IR 心智模型。
- [Func Dialect 源码导读](Dialects/FuncDialect_Summary.md)：内容更完整，包含初始化、接口、pass 和源码关系。
- [Builtin IR 结构图（Mermaid）](Assets/Builtin_Dialect_IR_Overview.mmd)
- [Builtin IR 结构图（SVG）](Assets/Builtin_Dialect_IR_Overview.svg)

### ODS

- [Operations / ODS 详细总结](ODS/Operations_Summary.md)
- [TableGen 与 ODS 笔记](ODS/TableGen_ODS_Notes.md)
- [Toy Dialect ODS 实践](ODS/ODS_Toy_Dialect_Practical_Summary.md)

### Transforms 与 Passes

- [Phase 2 初学者导读](Transforms/Phase2_AI_Compiler_Beginner_Guide.md)
- [Phase 2 Core Infrastructure](Transforms/Phase2_Core_Infrastructure_Summary.md)
- [PatternRewriter](Transforms/PatternRewriter_Summary.md)
- [DialectConversion](Transforms/DialectConversion_Summary.md)
- [PassManagement](Transforms/PassManagement_Summary.md)
- [OperationPass 并行限制](Transforms/PassManagement_OperationPass_Restrictions_Summary.md)

### Tutorials 与 Tools

- [Toy Tutorial Ch1-Ch6](Tutorials/Step4_Toy_Tutorial_Summary.md)
- [Toy Tutorial Ch7 代码结构](Tutorials/Step4_Ch7_Code_Structure_Summary.md)
- [图解 MLIR 阅读地图](IllustratedMLIR/00_Reading_Map.md)
- [图解 MLIR 第 1 章：为什么一切都是 Operation](IllustratedMLIR/01_Why_Everything_Is_An_Operation.md)
- [图解 MLIR 共用示例](IllustratedMLIR/examples/accumulate.mlir)
- [FileCheck](Tools/FileCheck_Summary.md)

### Practices

- [SSA use-def 输入](Practices/ir_ssa_walk.mlir)与[分析笔记](Practices/mlir_ir_ssa_notes.md)
- [canonicalize/CSE 与 FileCheck 输入](Practices/practice_canonicalize_cse.mlir)
- [AddZeroPattern Pass 示例](Practices/rewrite-pattern-addi-zero/)
- [AddZeroPattern 构建说明](Practices/rewrite-pattern-addi-zero/CMakeLists_notes.md)

## 维护规则

- 新文档按主题放入对应目录，不再直接堆放到 `LearningSteps/` 根目录。
- 新增或移动文档时同步更新本索引和仓库内引用。
- 专题总结继续使用一个源文档对应一个 `*_Summary.md` 的方式；不同定位的文档应在索引中说明差异，未经比较不要直接合并。
- 实践输入、实现和说明放在 `Practices/`，图表资源放在 `Assets/`。
