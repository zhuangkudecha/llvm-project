# Illustrated MLIR Design

## 1. Goal

Create an independent Chinese tutorial series under `LearningSteps/IllustratedMLIR/`. The series adopts the
question-driven and diagram-led style of Xiao Lin Coding while remaining grounded in runnable MLIR examples and the
current MLIR source tree.

The first volume establishes the core IR mental model:

```text
Operation -> Value -> Region -> Block -> Dialect
```

It is an introductory narrative, not a replacement for the existing `*_Summary.md` reference notes.

## 2. Audience and Success Criteria

The target reader understands basic compiler concepts but cannot yet explain how the central MLIR IR objects work
together.

The first volume succeeds when the reader can:

- identify the Operation, Region, Block, and Value structure in a small `.mlir` program;
- distinguish an `OpResult` from a `BlockArgument`;
- trace a short SSA use-def chain;
- explain why operations from several dialects can coexist in one module;
- map the textual IR to the corresponding MLIR C++ object model;
- run the example through `mlir-opt` without parser or verifier errors.

## 3. Editorial Approach

Use a dual-track narrative:

1. A single runnable IR example evolves through the volume and provides continuity.
2. Each chapter pauses at one point in that example to explain a specific concept rigorously.

Every chapter follows this structure:

```text
Concrete question
  -> conclusion first
  -> Mermaid overview diagram
  -> progressive diagram walkthrough
  -> runnable MLIR example
  -> mapping to C++ objects and source definitions
  -> common misconceptions
  -> one-page recap diagram
```

The prose first builds an intuitive model, then states the precise MLIR definition. Each subsection answers one
question. Diagrams must be followed by explicit guidance about which arrows and relationships matter.

## 4. Volume Structure

Create the following files under `LearningSteps/IllustratedMLIR/`:

```text
00_Reading_Map.md
01_Why_Everything_Is_An_Operation.md
02_MLIR_Program_Structure.md
03_Value_And_SSA_Dataflow.md
04_Region_And_Block_Nesting.md
05_How_Dialects_Coexist.md
06_From_Textual_IR_To_Objects.md
```

The displayed titles, explanatory prose, diagram titles, Mermaid node labels, edge labels, legends, captions, and
diagram walkthroughs are written to disk in Chinese. Necessary MLIR identifiers such as `Operation`, `Region`, and
`Block` retain their original spelling and appear together with a Chinese explanation. ASCII filenames avoid tooling
and link portability issues.

Phase one creates the directory, the reading map, and chapter 1 only. The remaining chapters are expanded after the
first chapter establishes a validated template.

## 5. Anchor Example

The volume uses this `accumulate` function as its shared example:

```mlir
module {
  func.func @accumulate(%n: index) -> i32 {
    %c0 = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    %result = scf.for %i = %lb to %n step %step
        iter_args(%sum = %c0) -> i32 {
      %value = arith.index_cast %i : index to i32
      %next = arith.addi %sum, %value : i32
      scf.yield %next : i32
    }

    return %result : i32
  }
}
```

This example exposes all concepts required by the first volume:

- `builtin.module` and `func.func` demonstrate operations that own regions;
- the function body and loop body demonstrate blocks;
- `%n`, `%i`, and `%sum` demonstrate block arguments;
- constants, casts, additions, and the loop demonstrate operation results;
- `scf.for` demonstrates nested IR and loop-carried SSA values;
- `builtin`, `func`, `arith`, and `scf` demonstrate dialect coexistence.

The example is stored as a separate `.mlir` file so every chapter can link to one executable source of truth.

## 6. Diagram System

All first-volume diagrams use Mermaid with an explicitly configured light theme. Color has stable semantic meaning:

| Color | Meaning |
|---|---|
| Blue | Operation |
| Green | Region |
| Yellow | Block |
| Red | Value and SSA dataflow |
| Purple | Dialect or type-system concepts |
| Gray | Source code, tools, or supporting context |

The series uses three recurring diagram views:

- **Structure tree:** containment among Operation, Region, and Block.
- **SSA dataflow:** definition, use, and loop-carried Value edges.
- **Dialect layers:** semantic ownership of operations that coexist in one IR.

Complex diagrams are built progressively. The first view shows only the conclusion and outline; later views add
containment, dataflow, textual IR mapping, and finally C++ object mapping. Mermaid nodes and edges must remain readable
in a light Markdown preview without relying on color alone. All reader-facing text inside and around a diagram is
stored in Chinese; raw dialect names, operation names, SSA names, class names, and API names remain unchanged where
translation would make them technically inaccurate.

## 7. Relationship to Existing Notes

`IllustratedMLIR` is the learning layer. Existing files such as `Operations_Summary.md`, `Builtin_Dialect_Summary.md`,
and `Phase2_Core_Infrastructure_Summary.md` remain the reference layer.

Illustrated chapters link to the relevant summary when a reader needs source-level or API detail. Existing summaries
are not rewritten in phase one. The new series should avoid copying long sections from them; it should reuse verified
facts while presenting a new continuous narrative.

## 8. Source and Example Validation

Each completed chapter must pass three checks:

1. **IR validation:** run the shared example with the locally built `mlir-opt`; examples that demonstrate a pass also
   preserve the before and after IR.
2. **Source validation:** verify named C++ classes and methods against the current checkout and link to their repository
   paths.
3. **Diagram validation:** render or parse Mermaid diagrams and manually check node labels, edge directions, contrast,
   and agreement with the prose.

If the expected `mlir-opt` binary or Mermaid renderer is unavailable, record that limitation explicitly. Do not claim
the corresponding validation passed.

## 9. Phase-One Deliverables

Phase one produces:

- `LearningSteps/IllustratedMLIR/00_Reading_Map.md`;
- `LearningSteps/IllustratedMLIR/01_Why_Everything_Is_An_Operation.md`;
- `LearningSteps/IllustratedMLIR/examples/accumulate.mlir`;
- links from chapter 1 to the relevant existing reference notes;
- recorded IR and Mermaid verification results.

Passes, PatternRewriter, DialectConversion, and end-to-end lowering are explicitly outside phase one. They belong to a
later volume after the basic IR object model is established.
