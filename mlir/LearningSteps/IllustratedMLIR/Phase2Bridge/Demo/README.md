# Phase2Bridge — MLIR Pass Plugin

基于 [MLIR Pass Plugin](https://mlir.llvm.org/doxygen/classmlir_1_1PassPlugin.html) 机制构建的自定义 pass 集合，通过 `--load-pass-plugin` 动态加载，无需修改 mlir-opt 上游代码。

## 目录结构

```
Demo/
├── CMakeLists.txt              # 插件构建配置
├── README.md                   # 本文件
├── build/                      # 构建目录（生成）
├── inputs/                     # 测试用 MLIR 输入
│   └── 01-inspection.mlir
├── libPhase2Passes.so    # 编译产物（生成）
└── passes/
    ├── Phase2Passes.h          # 注册函数声明
    ├── Phase2Inspection.cpp    # Pass: SSA use-def / CFG / 支配检查
    └── PluginEntry.cpp         # 唯一插件入口 mlirGetPassPluginInfo
```

## 前置条件

1. LLVM/MLIR monorepo 已编译完成（`~/llvm-project/build/`）
2. mlir-opt 需开启符号导出（`LLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON`）：

```bash
cd ~/llvm-project/build
cmake -G Ninja -DLLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON .
ninja mlir-opt
```

> 这一步只需要做一次。开启后 mlir-opt 会带 `--export-dynamic`，
> 插件运行时可以从 mlir-opt 解析 MLIR 符号。

## 构建

```bash
cd ~/llvm-project/mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo

cmake -G Ninja -S . -B build \
    -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm

ninja -C build
```

产物：`libPhase2Passes.so`

## 使用

```bash
cd ~/llvm-project

./build/bin/mlir-opt \
    --load-pass-plugin=./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/libPhase2Passes.so \
    --pass-pipeline='builtin.module(func.func(phase2-inspection))' \
    ./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/inputs/01-inspection.mlir
```

### 输出示例

```
function=inspect
operations=8
regions=1
blocks=4
block=0 argument=0 type=i32 users=2
block=0 argument=1 type=i1 users=1
block-dominance 0->0=1
block-dominance 0->1=1
block-dominance 0->2=1
block-dominance 0->3=1
block-dominance 1->0=0
block-dominance 1->1=1
...
```

## 已注册的 Pass

| Pass 名 | 描述 | 作用域 |
|---------|------|--------|
| `phase2-inspection` | 检查 SSA use-def、CFG 块、支配关系 | `func.func` |

## 添加新 Pass

多文件结构下新增一个 pass 的完整流程：

1. 新建 `passes/Phase2Xxx.cpp`：定义 pass 类（匿名 namespace）+ 导出注册函数
2. 在 `passes/Phase2Passes.h` 加注册函数声明
3. 在 `passes/PluginEntry.cpp` 的 lambda 里加一行调用
4. 在 `CMakeLists.txt` 的 `add_library` 加源文件

```cpp
// passes/Phase2ConvertToLLVM.cpp
#include "Phase2Passes.h"
// ...

namespace {
struct Phase2ConvertToLLVMPass
    : PassWrapper<Phase2ConvertToLLVMPass, OperationPass<func::FuncOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Phase2ConvertToLLVMPass)
    StringRef getArgument() const final { return "phase2-convert-to-llvm"; }
    StringRef getDescription() const final { return "..."; }
    void runOnOperation() override { /* ... */ }
};
} // namespace

void registerPhase2ConvertToLLVMPass() {
    PassRegistration<Phase2ConvertToLLVMPass>();
}
```

```cpp
// passes/Phase2Passes.h —— 加一行
void registerPhase2ConvertToLLVMPass();
```

```cpp
// passes/PluginEntry.cpp —— lambda 里加一行
[]() {
    registerPhase2InspectionPass();
    registerPhase2ConvertToLLVMPass();  // ← 新增
}
```

5. 重新编译：

```bash
ninja -C build  # 只编译 .so，秒级
```

## 原理

```
mlir-opt (带 --export-dynamic)
    │
    │  --load-pass-plugin=libPhase2Passes.so
    │         │
    │         ▼
    │  dlopen("libPhase2Passes.so")
    │  查找 mlirGetPassPluginInfo 符号
    │  调用 → 返回 PassPluginLibraryInfo
    │  调用 registerPassRegistryCallbacks()
    │         │
    │         ▼
    │  PassRegistration<Phase2InspectionPass>()
    │  → 调用 mlir::registerPass()
    │    (符号从 mlir-opt 动态解析)
    │  → 注册到 mlir-opt 的全局 pass registry
    │
    │  --pass-pipeline='...func.func(phase2-inspection)...'
    │  → pipeline parser 查找已注册的 pass
    │  → 找到 phase2-inspection，创建 pass 实例
    ▼
  执行 pass
```

关键点：
- 插件编译为 MODULE（不链接 MLIR 静态库）
- MLIR 符号在运行时从 mlir-opt 动态解析
- `--allow-shlib-undefined` 允许链接时有未解析符号
- `LLVM_EXPORT_SYMBOLS_FOR_PLUGINS=ON` 让 mlir-opt 导出所有符号

## 对比

| 方式 | 改上游代码 | 重编 mlir-opt | 增删 pass |
|------|-----------|--------------|----------|
| 改 mlir-opt.cpp | 是 | 每次 | 重编 mlir-opt |
| **Pass Plugin（本方案）** | **否** | **一次** | **只重编 .so** |
| 自建工具 | 否 | N/A | 重编工具 |
