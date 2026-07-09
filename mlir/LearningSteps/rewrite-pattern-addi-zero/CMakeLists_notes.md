# AddZeroPatternPass 的 CMake 接入方式

这个练习文件在：

```text
mlir/LearningSteps/rewrite-pattern-addi-zero/AddZeroPatternPass.cpp
```

但是 `LearningSteps` 目录默认不属于 MLIR 的 CMake 构建树，所以只在这里新增
`CMakeLists.txt` 不会让 `mlir-opt` 自动看到这个 pass。

更直接的练习方式是：把这个 pass 临时接入 MLIR 已有的 test pass 库。

## 方案一：接到 test/lib/Transforms

推荐用于学习验证。

MLIR 的 `tools/mlir-opt/CMakeLists.txt` 在 `MLIR_INCLUDE_TESTS=ON` 时会链接
`MLIRTestTransforms`：

```cmake
if(MLIR_INCLUDE_TESTS)
  set(test_libs
    ...
    MLIRTestTransforms
    ...
  )
endif()
```

所以可以把 `AddZeroPatternPass.cpp` 加到：

```text
mlir/test/lib/Transforms/CMakeLists.txt
```

找到：

```cmake
add_mlir_library(MLIRTestTransforms
  TestCommutativityUtils.cpp
  TestCompositePass.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestInliningCallback.cpp
  TestMakeIsolatedFromAbove.cpp
  TestSingleFold.cpp
  TestTransformsOps.cpp
  ${MLIRTestTransformsPDLSrc}

  EXCLUDE_FROM_LIBMLIR
  ...
)
```

临时加一行：

```cmake
add_mlir_library(MLIRTestTransforms
  TestCommutativityUtils.cpp
  TestCompositePass.cpp
  TestControlFlowSink.cpp
  TestInlining.cpp
  TestInliningCallback.cpp
  TestMakeIsolatedFromAbove.cpp
  TestSingleFold.cpp
  TestTransformsOps.cpp
  AddZeroPatternPass.cpp
  ${MLIRTestTransformsPDLSrc}

  EXCLUDE_FROM_LIBMLIR
  ...
)
```

也就是说，最终只需要新增：

```cmake
  AddZeroPatternPass.cpp
```

## 源文件应该放哪里

不要直接从 `LearningSteps` 引用源码路径。更干净的方式是复制一份到：

```text
mlir/test/lib/Transforms/AddZeroPatternPass.cpp
```

然后 CMake 才能用相对路径：

```cmake
  AddZeroPatternPass.cpp
```

## 还需要链接哪些库

你的 pass 用到了：

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
```

`test/lib/Transforms/CMakeLists.txt` 里已经链接了：

```cmake
mlir_target_link_libraries(MLIRTestTransforms PUBLIC
  MLIRAnalysis
  MLIRFuncDialect
  MLIRInferIntRangeInterface
  MLIRTransforms
  MLIRTransformDialect
)
```

为了 `arith::AddIOp` 更明确，建议补上：

```cmake
  MLIRArithDialect
```

变成：

```cmake
mlir_target_link_libraries(MLIRTestTransforms PUBLIC
  MLIRAnalysis
  MLIRArithDialect
  MLIRFuncDialect
  MLIRInferIntRangeInterface
  MLIRTransforms
  MLIRTransformDialect
)
```

## pass 注册方式

如果 `AddZeroPatternPass.cpp` 里用了全局静态注册：

```cpp
static PassRegistration<MyRewritePass> pass;
```

那么只要这个 `.cpp` 被编译并链接进 `mlir-opt`，命令行里就能看到它。

pass 类里要提供：

```cpp
StringRef getArgument() const final { return "add-zero-pattern"; }

StringRef getDescription() const final {
  return "Rewrite arith.addi x, 0 and arith.addi 0, x to x";
}
```

## 重新构建

从 `llvm-project` 根目录执行：

```bash
cmake --build build --target mlir-opt
```

或者：

```bash
ninja -C build mlir-opt
```

具体 build 目录以你本地实际目录为准。

## 验证 pass 是否注册成功

```bash
./build/bin/mlir-opt --help | grep add-zero-pattern
```

能看到 `add-zero-pattern`，说明 CMake 和 pass 注册都通了。

## 运行 pipeline

因为当前 pass 是：

```cpp
OperationPass<func::FuncOp>
```

所以 pipeline 应写成：

```bash
./build/bin/mlir-opt input.mlir \
  -pass-pipeline='builtin.module(func.func(add-zero-pattern))'
```

如果写成：

```bash
./build/bin/mlir-opt input.mlir \
  -pass-pipeline='builtin.module(add-zero-pattern)'
```

会因为 pass anchor 不匹配而失败。

## 为什么不建议在 LearningSteps 里直接写 CMakeLists.txt

可以写，但默认不会被顶层 CMake 递归包含。

除非你再去改更上层的：

```text
mlir/CMakeLists.txt
```

加：

```cmake
add_subdirectory(LearningSteps/rewrite-pattern-addi-zero)
```

这会把学习目录变成正式构建目录，不适合这种临时实验。

所以学习阶段推荐：

1. `LearningSteps` 保存笔记和草稿。
2. 真正要编译的 `.cpp` 临时复制到 `test/lib/Transforms`。
3. 在 `test/lib/Transforms/CMakeLists.txt` 里加源文件。
4. 重新构建 `mlir-opt`。
