# MLIR SSA Guided Exercise Design

**Date:** 2026-07-15

## Goal

Create `LearningSteps/Practices/mlir_ir_ssa_exercise.md` as a fill-in worksheet
derived from `mlir_ir_ssa_notes.md`. Keep the original answer document unchanged
and let the learner independently reconstruct the SSA use-def and dominance
analysis for `ir_ssa_walk.mlir`.

## Exercise Structure

The exercise keeps the source-file link and the five-topic progression of the
answer document:

1. classify MLIR `Value` sources;
2. inventory every SSA value;
3. reconstruct use-def relationships;
4. record the manual analysis process;
5. check scope and dominance.

## Retained Guidance

- Keep every SSA name from the original inventory in the first column so the
  learner does not accidentally omit a value.
- Keep table headers that state what must be discovered: category, owner, type,
  definition position, users, operand role, relationship kind, scope, and
  dominance result.
- Keep short task instructions and distinctions the learner must apply, such as
  `OpResult` versus `BlockArgument` and real uses versus SCF semantic mappings.
- Keep enough blank rows for every expected relationship and dominance case.

## Removed Answers

- Remove all filled classification, owner, type, defining-position, and user
  cells.
- Remove the completed Mermaid use-def graph.
- Remove worked conclusions for `%shared`, `%selected`, `%result`, `%acc`,
  `%base`, and `%then_value`.
- Remove prose that directly reveals whether a proposed dominance relationship
  is valid.

The worksheet may name the values or cases being investigated, but it must not
contain their solutions.

## Verification

- Confirm `mlir_ir_ssa_notes.md` is byte-for-byte unchanged.
- Confirm the exercise links to the existing `ir_ssa_walk.mlir` input.
- Confirm every original SSA value appears once in the inventory table.
- Confirm answer columns contain blank cells rather than copied solutions.
- Confirm no completed Mermaid graph or answer paragraphs remain.
- Run Markdown whitespace and local-link checks on the new file.
