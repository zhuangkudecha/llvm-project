# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure framework designed for building, analyzing, and transforming compiler intermediate representations. It's part of the LLVM monorepo and provides a hierarchical IR with extensible dialects.

## Building

MLIR is built as part of the LLVM monorepo using CMake. From the `llvm-project` directory:

```bash
# Configure (first time)
cmake -G Ninja -S . -B build -DLLVM_ENABLE_PROJECTS=mlir -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --target MLIR

# Or build specific targets
cmake --build build --target mlir-opt
cmake --build build --target mlir-translate
```

Common build targets:
- `MLIR` - Build all MLIR libraries
- `mlir-opt` - MLIR optimization/transformation tool
- `mlir-translate` - Translation tool (e.g., MLIR to LLVM IR)
- `mlir-tblgen` - TableGen-based operation definition generator
- `mlir-runner` - Execute MLIR with MLIR dialects
- `check-mlir` - Run all MLIR tests

## Testing

MLIR uses the lit test framework. Tests are located in `test/` and use `.mlir` files.

```bash
# Run all MLIR tests
cmake --build build --target check-mlir

# Run specific test directory
lit -v build/tools/mlir/test/mlir-opt

# Run specific test file
lit -v build/tools/mlir/test/mlir-opt/test-file.mlir
```

## Key Tools

- **mlir-opt**: Primary tool for running passes and transformations on MLIR
  ```bash
  build/bin/mlir-opt --pass-pipeline="builtin.module(pass-name)" input.mlir
  build/bin/mlir-opt --pass-pipeline="func.func(canonicalize,cse)" input.mlir
  ```

- **mlir-translate**: Translate MLIR to/from other formats
  ```bash
  build/bin/mlir-translate --mlir-to-llvmir input.mlir -o output.ll
  ```

- **mlir-tblgen**: Generates C++ code from TableGen operation definitions (.td files)

## Architecture

### Dialects

MLIR's core concept is extensible dialects, each defining a set of operations, types, and attributes for a specific domain.

**Core dialects** (`lib/Dialect/`):
- `Builtin` - Foundation types and operations
- `Func` - Function operations
- `SCF` (Structured Control Flow) - Loop and control flow
- `Arith` - Arithmetic operations
- `Affine` - Affine loop structures
- `MemRef` - Memory reference operations
- `Tensor` - Tensor operations
- `Vector` - Vector operations

**Target dialects**:
- `LLVMIR` - LLVM IR dialect
- `SPIRV` - SPIR-V GPU dialect
- `GPU` - GPU operations
- `NVGPU`, `AMDGPU`, `XeGPU` - Vendor-specific GPU dialects
- `ArmNEON`, `ArmSVE`, `ArmSME` - ARM vector dialects

**Compiler dialects**:
- `Linalg` - Linear algebra operations
- `Tosa` - Tensor Operator Set Architecture
- `Bufferization` - Buffer allocation/deallocation

### Operation Definition System (ODS)

Operations are defined using TableGen in `.td` files. This generates:
- Operation classes with verification
- Type inference
- Traits and interfaces

Typical ODS file structure:
```cpp
def MyOp : Op<MyDialect, "op_name",
    [SingleResult, ...Traits]> {
  let summary = "Description";
  let description = "Detailed docs";
  let arguments = (ins ...);
  let results = (outs ...);
}
```

### Pass Infrastructure

MLIR uses a hierarchical pass manager where passes can be anchored to specific operation types.

- **Rewrite passes**: Apply pattern-based transformations
- **Conversion passes**: Transform between dialects (dialect conversion framework)
- **Analysis passes**: Compute information about IR

Key files:
- `lib/Pass/` - Pass infrastructure
- `include/mlir/Pass/` - Pass headers
- `docs/PassManagement.md` - Pass architecture docs

### TableGen and Declarative Rewrites

- **mlir-tblgen**: Generates C++ from ODS files
- **DRR (Declarative Rewrite Rules)**: Declarative pattern matching and rewriting
- **PDLL**: Pattern Definition Language for advanced rewrite patterns

Key files:
- `lib/TableGen/` - TableGen backend
- `docs/DeclarativeRewrites.md` - DRR documentation
- `docs/PDLL.md` - PDLL documentation

## Directory Structure

```
mlir/
├── include/mlir/          # Public headers
│   ├── IR/              # Core IR (Operation, Value, Type, etc.)
│   ├── Dialect/         # Dialect headers
│   ├── Pass/            # Pass infrastructure
│   ├── Interfaces/      # Operation interfaces
│   └── Transforms/      # Transformation passes
├── lib/                 # Implementation
│   ├── IR/              # Core IR implementation
│   ├── Dialect/         # Dialect implementations
│   ├── Pass/            # Pass infrastructure
│   ├── Conversion/       # Dialect conversion passes
│   └── Transforms/      # Transformation passes
├── test/                # lit tests
│   ├── Dialect/         # Dialect tests
│   ├── mlir-opt/        # mlir-opt tool tests
│   └── Integration/     # Integration tests
├── tools/               # Command-line tools
│   ├── mlir-opt/
│   ├── mlir-translate/
│   └── mlir-tblgen/
└── docs/                # Documentation
    ├── Tutorials/        # Getting started tutorials
    ├── Dialects/         # Dialect documentation
    └── LangRef.md       # Language reference
```

## Common Development Tasks

### Adding a new operation

1. Add to dialect's `.td` file (e.g., `include/mlir/Dialect/MyDialect/MyDialect.td`)
2. Regenerate TableGen: `cmake --build build --target mlir-generic-headers`
3. Implement operation methods in `lib/Dialect/MyDialect/MyDialect.cpp`

### Adding a new dialect

1. Create `include/mlir/Dialect/NewDialect/` with `.td` files
2. Create `lib/Dialect/NewDialect/` with `.cpp` implementation
3. Add to `lib/Dialect/CMakeLists.txt`
4. Register in `mlir/lib/RegisterAllDialects.cpp`

### Adding a new pass

1. Create header in `include/mlir/Transforms/MyPass.h`
2. Create implementation in `lib/Transforms/MyPass.cpp`
3. Add to `lib/Transforms/CMakeLists.txt`
4. Register in `lib/RegisterAllPasses.cpp`

## Documentation

- [MLIR Homepage](https://mlir.llvm.org/)
- `docs/LangRef.md` - Complete language reference
- `docs/Tutorials/` - Step-by-step tutorials including "Toy" compiler
- `docs/DialectConversion.md` - Dialect conversion framework
- `docs/Interfaces.md` - Operation interfaces
- `docs/Traits/` - Trait definitions

## Python Bindings

MLIR has Python bindings in `python/`. Build with `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`.

```bash
python -c "import mlir"
```
