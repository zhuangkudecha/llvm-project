# LearningSteps Topic Reorganization Design

**Date:** 2026-07-15

## Goal

Reorganize `LearningSteps` by stable subject area so that study plans, reference
notes, tutorials, tools, and runnable exercises no longer share one flat
directory. Add a single reading entry point and repair repository-local links
without rewriting the documents' technical content.

## Scope

This change only reorganizes files under `LearningSteps` and updates references
that point to their old locations. It includes the currently uncommitted RTX4090
execution-plan split and must preserve those edits byte-for-byte apart from link
changes required by the move.

The change does not:

- merge or delete either Func dialect summary;
- rewrite or shorten note content;
- modify upstream MLIR source code;
- create compatibility copies at the old paths;
- reorganize `docs/superpowers` beyond adding the design and implementation
  plan for this task.

## Target Structure

```text
LearningSteps/
|-- README.md
|-- Plans/
|   |-- MLIR_Learning_Plan.md
|   `-- GPU_CodeGen/
|       |-- AI_Compiler_Engineer_Learning_Plan.md
|       |-- RTX4090_GPU_CodeGen_24Week_Execution_Plan.md
|       `-- Phases/
|-- Foundations/
|-- Dialects/
|-- ODS/
|-- Transforms/
|-- Tutorials/
|-- Tools/
|-- Practices/
`-- Assets/
```

Directory names describe durable subject areas rather than the current study
sequence. Reading order lives in `README.md`, so adding a future topic does not
require renaming directories.

## File Mapping

### Plans

- `MLIR_Learning_Plan.md` -> `Plans/MLIR_Learning_Plan.md`
- `AI_Compiler_Engineer_Learning_Plan.md` ->
  `Plans/GPU_CodeGen/AI_Compiler_Engineer_Learning_Plan.md`
- `RTX4090_GPU_CodeGen_24Week_Execution_Plan.md` ->
  `Plans/GPU_CodeGen/RTX4090_GPU_CodeGen_24Week_Execution_Plan.md`
- `RTX4090_GPU_CodeGen_Phases/Phase*.md` ->
  `Plans/GPU_CodeGen/Phases/Phase*.md`

### Foundations

- `Step1_IR_Structure_Summary.md`
- `Step2_Language_Reference_Summary.md`
- `Step3_mlir_opt_Toolchain_Summary.md`
- `Step5_Core_IR_Headers_Summary.md`

### Dialects

- `Builtin_Dialect_Summary.md`
- `DefiningDialects_Summary.md`
- `FuncDialect_Summary.md`
- `Func_Dialect_Summary.md`

The two Func dialect documents remain separate. The index will distinguish the
larger source-oriented introduction from the shorter conceptual summary.

### ODS

- `Operations_Summary.md`
- `TableGen_ODS_Notes.md`
- `ODS_Toy_Dialect_Practical_Summary.md`

### Transforms

- `PatternRewriter_Summary.md`
- `DialectConversion_Summary.md`
- `PassManagement_Summary.md`
- `PassManagement_OperationPass_Restrictions_Summary.md`
- `Phase2_Core_Infrastructure_Summary.md`
- `Phase2_AI_Compiler_Beginner_Guide.md`

### Tutorials

- `Step4_Toy_Tutorial_Summary.md`
- `Step4_Ch7_Code_Structure_Summary.md`

### Tools

- `FileCheck_Summary.md`

### Practices

- `ir_ssa_walk.mlir`
- `mlir_ir_ssa_notes.md`
- `practice_canonicalize_cse.mlir`
- `rewrite-pattern-addi-zero/` and all files below it

### Assets

- `Builtin_Dialect_IR_Overview.mmd`
- `Builtin_Dialect_IR_Overview.svg`

## README Design

`LearningSteps/README.md` is the stable entry point. It contains:

1. a short statement of purpose;
2. a recommended beginner reading path;
3. a topic index covering every Markdown note;
4. a practice section linking examples to the notes they support;
5. a GPU code-generation plan section;
6. a note explaining the two Func dialect summaries.

Links in the index are relative so the repository remains portable.

## Link Migration Rules

- Update Markdown links and literal repository paths that refer to moved files.
- Update commands whose input paths moved, including `mlir-opt`, `FileCheck`,
  and practice-pass examples.
- Keep links to upstream `docs/`, `include/`, `lib/`, and `test/` unchanged.
- Update the new RTX phase documents and their index together, while preserving
  their existing uncommitted content.
- Update `docs/superpowers` references when they identify a moved
  `LearningSteps` path; historical prose that merely describes the former
  layout may remain unchanged only when it is clearly historical and not an
  actionable path.

No redirect or duplicate files remain at old paths. Git history is the source
of provenance, while the root README is the source of navigation.

## Safety and Verification

Before moving files, record the current working-tree status and checksums for
the modified/untracked RTX files. After migration:

1. verify every mapped source has exactly one destination;
2. verify no expected document remains loose at the `LearningSteps` root;
3. compare RTX file content after normalizing only intentional path edits;
4. scan tracked and untracked Markdown for stale actionable old paths;
5. validate every relative Markdown link whose target is local;
6. run `git diff --check` over all touched files;
7. inspect `git status --short` to confirm unrelated files were not changed.

MLIR compilation is unnecessary because this is a documentation and path-only
change. Runnable command examples receive static path validation.

## Completion Criteria

The reorganization is complete when the target structure exists, every current
document appears in the README index, repository-local references resolve, the
RTX4090 work remains intact, and the final diff contains no unrelated changes.
