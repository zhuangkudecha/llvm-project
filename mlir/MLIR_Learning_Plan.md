# MLIR Learning Plan

A structured course for learning the MLIR (Multi-Level Intermediate Representation) project.

---

## Phase 0: AI Compiler Big Picture (2-3 days)

**Goal:** Build a mental map of where MLIR sits in an AI compiler stack before diving into source code details.

| Step | Resource | Key Topics |
|------|----------|------------|
| 0 | `Phase2_AI_Compiler_Beginner_Guide.md` and selected MLIR overview docs | Model graph, tensor programs, dialect layers, lowering pipeline |

**Core map:**

```text
Model graph / Tensor program
  -> High-level dialects      (tensor, linalg, stablehlo, tosa)
  -> Structured computation   (linalg, scf, affine, memref)
  -> Target dialects          (vector, gpu, nvvm, spirv, llvm)
  -> Target code / runtime calls
```

**Milestone:** You can explain why AI compilers need multiple IR levels, and how MLIR uses dialects and passes to move between those levels.

**Deliverable:** A short note or diagram that maps `Operation`, `Dialect`, `Pass`, and `Lowering` to an AI compiler pipeline.

---

## Phase 1: Foundations (1-2 weeks)

**Goal:** Understand what MLIR is and its IR structure.

| Step | Resource | Key Topics |
|------|----------|------------|
| 1 | `docs/Tutorials/UnderstandingTheIRStructure.md` | Operations, Blocks, Regions, Values |
| 2 | `docs/LangRef.md` | Full language reference — types, attributes, ops, dialects |
| 3 | Run `mlir-opt --help` to see available passes | Get a feel for the toolchain |
| 4 | Toy Tutorial (`docs/Tutorials/Toy/`) | Ch0-Ch7: Build a complete mini compiler end-to-end |
|   | Ch0-Ch4: Language, Dialect, Optimization, Interfaces | ODS, Pattern Rewriting, Shape Inference, Inlining |
|   | Ch5: Lowering to Affine + Loops | DialectConversion, ConversionTarget, TypeConverter |
|   | Ch6: Lowering to LLVM | Partial lowering, affine→SCF→CF→LLVM pipeline |
|   | Ch7: Extending Toy (Struct types, complete pipeline) | Custom types, full compilation to JIT execution |

**Toy-to-AI compiler mapping:**

| Toy Tutorial Concept | AI Compiler Analogy |
|----------------------|---------------------|
| Toy Dialect | High-level model/operator dialect |
| ShapeInference | Tensor shape inference / static shape refinement |
| LowerToAffine | Tensor computation lowered to loop-level IR |
| LowerToLLVM | Lowering toward target executable code |
| PatternRewriter | Graph/operator rewrite and local optimization |
| Pass Pipeline | Compiler pipeline from frontend IR to target IR |

**Milestone:** You can read/write basic `.mlir` files and understand the IR structure.

**Deliverable:** Hand-write at least three small `.mlir` examples, run `mlir-opt --canonicalize` and `mlir-opt --cse`, and save notes explaining how the IR changed.

---

## Phase 2: Core Infrastructure (2-3 weeks)

**Goal:** Understand the C++ infrastructure that powers MLIR.

| Step | Resource | Key Topics |
|------|----------|------------|
| 5 | `include/mlir/IR/` | `Operation.h`, `Builders.h`, `Types.h`, `Dialect.h`, `MLIRContext.h` |
| 6 | `include/mlir/Pass/` | Pass registration, pass manager, pass pipeline |
| 7 | `docs/PassManagement.md` | How passes are organized and executed |
| 8 | `docs/Canonicalization.md` | Canonical forms and simplification |
| 9 | `include/mlir/Transform/` | Transform infrastructure and scheduling perspective |
| 10 | `docs/PatternRewriter.md` | Pattern, PatternRewriter, GreedyPatternRewriteDriver |

**Practice:** Build and run `examples/toy/` — trace through the C++ code.

**Milestone:** You can explain how `Operation`, `Value`, `Region`, `Block`, `Dialect`, `PassManager`, and `PatternRewriter` cooperate during an IR transformation.

**Deliverable:** Maintain `Phase2_Core_Infrastructure_Summary.md` and `Phase2_AI_Compiler_Beginner_Guide.md`. Avoid spending too long on low-level implementation details such as `TrailingObjects` and `StorageUniquer` until they become necessary for a concrete debugging task.

---

## Phase 3: Dialects Deep Dive (2-3 weeks)

**Goal:** Understand the dialect layers used by AI compiler lowering pipelines.

| Step | Resource | Key Topics |
|------|----------|------------|
| 11 | `docs/Tutorials/CreatingADialect.md` | Step-by-step dialect creation |
| 12 | Builtin, `Func`, `Arith` | Basic module/function/scalar operation structure |
| 13 | `Tensor`, `MemRef` | Tensor value semantics and explicit memory model |
| 14 | `SCF`, `Affine` | Structured control flow and analyzable loop nests |
| 15 | `Linalg`, `Vector` | Structured tensor computation and vector-level lowering |
| 16 | `docs/DialectConversion.md` | Legality, ConversionTarget, TypeConverter, rewrite patterns |

**Recommended AI compiler reading order:**

```text
Builtin / Func / Arith
  -> Tensor / MemRef
  -> SCF / Affine
  -> Linalg
  -> Vector
  -> LLVM / GPU / SPIR-V
  -> DialectConversion
```

**Practice:** Create a simple custom dialect using TableGen/ODS, or fully trace how Toy Dialect operations are defined, verified, printed, canonicalized, and lowered.

**Milestone:** You can look at a mixed-dialect `.mlir` file and identify which abstraction level each operation belongs to.

**Deliverable:** A dialect map that explains how a tensor computation can move from `tensor/linalg` to `scf/affine/memref`, then toward `vector/gpu/llvm`.

---

## Phase 4: Transformation & Optimization (2-3 weeks)

**Goal:** Learn to write passes and transformations.

| Step | Resource | Key Topics |
|------|----------|------------|
| 17 | Transform Tutorial (`docs/Tutorials/transform/`) | Ch1-Ch7 covering transformation patterns |
| 18 | `docs/Tutorials/QuickstartRewrites.md` | Hands-on rewrite patterns |
| 19 | Conversion passes in `lib/Conversion/` | Real-world dialect lowering examples |
| 20 | Analysis in `include/mlir/Analysis/` | Data-flow analysis, alias analysis |
| 21 | `docs/Tutorials/DataFlowAnalysis.md` | Data-flow framework |

**Practice:** Write a custom pass that transforms a dialect (e.g., optimize arithmetic patterns).

**Milestone:** You can write a small rewrite pattern or pass, run it with `mlir-opt`, and explain before/after IR changes.

**Deliverable:** A minimal custom rewrite or pass, such as simplifying a redundant arithmetic pattern or rewriting a small tensor-style operation.

---

## Phase 5: Real-World Application (2-4 weeks)

**Goal:** Apply MLIR to a target (LLVM IR, GPU, etc.).

| Step | Resource | Key Topics |
|------|----------|------------|
| 22 | `lib/Conversion/ToLLVM/` | Lowering to LLVM IR |
| 23 | `docs/TargetLLVMIR.md` | LLVM IR targeting |
| 24 | `GPU`, `NVVM`, and `SPIRV` dialects | GPU offloading pipeline |
| 25 | `python/` bindings | Python API for MLIR |
| 26 | `examples/standalone/` | Full standalone dialect + tool |

**Final Project:** Build a small compiler pipeline that lowers a custom dialect through several stages to LLVM IR or another target.

**Suggested final project path:**

```text
Hand-written tensor/linalg-style .mlir
  -> canonicalize / cse
  -> lower to loops or affine/scf
  -> introduce memref/vector where appropriate
  -> lower toward llvm or gpu dialect
  -> document every IR snapshot
```

**Milestone:** You can build and explain a small end-to-end lowering pipeline.

**Deliverable:** A runnable script or documented command sequence that shows each lowering stage and its IR output.

---

## Learning Tips

- **Read code alongside docs** — The implementations in `lib/` are the ground truth
- **Use `mlir-opt`** as your playground — write `.mlir` files and test passes
- **Use `--debug` flag** with `mlir-opt` to trace pass execution
- **Study test files** in `test/` — they show concise examples of each feature
- **Start small** — understand one dialect thoroughly before moving to the next
- **Prefer artifacts over passive reading** — each phase should leave behind a note, `.mlir` example, pass, or command script
- **Keep an AI compiler through-line** — repeatedly ask how the concept helps lower tensor programs toward executable code

---

## Progress Tracker

### Phase 0: AI Compiler Big Picture
- [ ] Step 0: AI compiler overview and MLIR lowering map

### Phase 1: Foundations
- [x] Step 1: Understanding the IR Structure
- [x] Step 2: Language Reference
- [x] Step 3: Explore `mlir-opt` toolchain
- [x] Step 4: Toy Tutorial (Ch0-Ch7)

### Phase 2: Core Infrastructure
- [x] Step 5: Core IR headers
- [x] Step 6: Pass infrastructure
- [x] Step 7: Pass Management docs
- [x] Step 8: Canonicalization
- [x] Step 9: Transform framework
- [x] Step 10: Pattern Rewriter

### Phase 3: Dialects Deep Dive
- [ ] Step 11: Creating a dialect tutorial
- [ ] Step 12: Builtin, Func, Arith
- [ ] Step 13: Tensor and MemRef
- [ ] Step 14: SCF and Affine
- [ ] Step 15: Linalg and Vector
- [ ] Step 16: Dialect Conversion

### Phase 4: Transformation & Optimization
- [ ] Step 17: Transform Tutorial (Ch1-Ch7)
- [ ] Step 18: Quickstart Rewrites
- [ ] Step 19: Conversion passes
- [ ] Step 20: Analysis framework
- [ ] Step 21: Data-flow Analysis

### Phase 5: Real-World Application
- [ ] Step 22: Lowering to LLVM IR
- [ ] Step 23: Target LLVM IR docs
- [ ] Step 24: GPU, NVVM, and SPIRV dialects
- [ ] Step 25: Python bindings
- [ ] Step 26: Standalone example
- [ ] Final Project: Custom compiler pipeline
