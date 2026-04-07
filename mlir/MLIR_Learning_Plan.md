# MLIR Learning Plan

A structured course for learning the MLIR (Multi-Level Intermediate Representation) project.

---

## Phase 1: Foundations (1-2 weeks)

**Goal:** Understand what MLIR is and its IR structure.

| Step | Resource | Key Topics |
|------|----------|------------|
| 1 | `docs/Tutorials/UnderstandingTheIRStructure.md` | Operations, Blocks, Regions, Values |
| 2 | `docs/LangRef.md` | Full language reference — types, attributes, ops, dialects |
| 3 | Run `mlir-opt --help` to see available passes | Get a feel for the toolchain |
| 4 | Toy Tutorial (`docs/Tutorials/Toy/`) | Ch0-Ch4: Build a mini compiler end-to-end |

**Milestone:** You can read/write basic `.mlir` files and understand the IR structure.

---

## Phase 2: Core Infrastructure (2-3 weeks)

**Goal:** Understand the C++ infrastructure that powers MLIR.

| Step | Resource | Key Topics |
|------|----------|------------|
| 5 | `include/mlir/IR/` | `Operation.h`, `Builders.h`, `Types.h`, `Dialect.h`, `MLIRContext.h` |
| 6 | `include/mlir/Pass/` | Pass registration, pass manager, pass pipeline |
| 7 | `docs/PassManagement.md` | How passes are organized and executed |
| 8 | `docs/Canonicalization.md` | Canonical forms and simplification |
| 9 | `include/mlir/Transform/` | Pattern rewriting framework |
| 10 | `docs/PatternRewriter.md` | DialectConversion, rewrite patterns |

**Practice:** Build and run `examples/toy/` — trace through the C++ code.

---

## Phase 3: Dialects Deep Dive (2-3 weeks)

**Goal:** Understand how dialects work and how to create one.

| Step | Resource | Key Topics |
|------|----------|------------|
| 11 | `docs/Tutorials/CreatingADialect.md` | Step-by-step dialect creation |
| 12 | Simple dialects: `Arith`, `Func`, `ControlFlow` | ODS files in `include/mlir/Dialect/` |
| 13 | `SCF` and `Affine` dialects | Structured control flow and affine analysis |
| 14 | `Linalg`, `Tensor`, `MemRef` | Data computation and memory model |
| 15 | `docs/DialectConversion.md` | How to convert between dialects |

**Practice:** Create a simple custom dialect using TableGen/ODS.

---

## Phase 4: Transformation & Optimization (2-3 weeks)

**Goal:** Learn to write passes and transformations.

| Step | Resource | Key Topics |
|------|----------|------------|
| 16 | Transform Tutorial (`docs/Tutorials/transform/`) | Ch1-Ch7 covering transformation patterns |
| 17 | Conversion passes in `lib/Conversion/` | Real-world dialect lowering examples |
| 18 | Analysis in `include/mlir/Analysis/` | Data-flow analysis, alias analysis |
| 19 | `docs/Tutorials/QuickstartRewrites.md` | Hands-on rewrite patterns |
| 20 | `docs/Tutorials/DataFlowAnalysis.md` | Data-flow framework |

**Practice:** Write a custom pass that transforms a dialect (e.g., optimize arithmetic patterns).

---

## Phase 5: Real-World Application (2-4 weeks)

**Goal:** Apply MLIR to a target (LLVM IR, GPU, etc.).

| Step | Resource | Key Topics |
|------|----------|------------|
| 21 | `lib/Conversion/ToLLVM/` | Lowering to LLVM IR |
| 22 | `docs/TargetLLVMIR.md` | LLVM IR targeting |
| 23 | `GPU` and `SPIRV` dialects | GPU offloading pipeline |
| 24 | `python/` bindings | Python API for MLIR |
| 25 | `examples/standalone/` | Full standalone dialect + tool |

**Final Project:** Build a small compiler pipeline that lowers a custom dialect through several stages to LLVM IR or another target.

---

## Learning Tips

- **Read code alongside docs** — The implementations in `lib/` are the ground truth
- **Use `mlir-opt`** as your playground — write `.mlir` files and test passes
- **Use `--debug` flag** with `mlir-opt` to trace pass execution
- **Study test files** in `test/` — they show concise examples of each feature
- **Start small** — understand one dialect thoroughly before moving to the next

---

## Progress Tracker

### Phase 1: Foundations
- [ ] Step 1: Understanding the IR Structure
- [ ] Step 2: Language Reference
- [ ] Step 3: Explore `mlir-opt` toolchain
- [ ] Step 4: Toy Tutorial (Ch0-Ch4)

### Phase 2: Core Infrastructure
- [ ] Step 5: Core IR headers
- [ ] Step 6: Pass infrastructure
- [ ] Step 7: Pass Management docs
- [ ] Step 8: Canonicalization
- [ ] Step 9: Transform framework
- [ ] Step 10: Pattern Rewriter

### Phase 3: Dialects Deep Dive
- [ ] Step 11: Creating a dialect tutorial
- [ ] Step 12: Simple dialects (Arith, Func, CF)
- [ ] Step 13: SCF and Affine dialects
- [ ] Step 14: Linalg, Tensor, MemRef
- [ ] Step 15: Dialect Conversion

### Phase 4: Transformation & Optimization
- [ ] Step 16: Transform Tutorial (Ch1-Ch7)
- [ ] Step 17: Conversion passes
- [ ] Step 18: Analysis framework
- [ ] Step 19: Quickstart Rewrites
- [ ] Step 20: Data-flow Analysis

### Phase 5: Real-World Application
- [ ] Step 21: Lowering to LLVM IR
- [ ] Step 22: Target LLVM IR docs
- [ ] Step 23: GPU and SPIRV dialects
- [ ] Step 24: Python bindings
- [ ] Step 25: Standalone example
- [ ] Final Project: Custom compiler pipeline
