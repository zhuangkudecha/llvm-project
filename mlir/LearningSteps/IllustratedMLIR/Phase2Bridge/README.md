# Phase2Bridge — MLIR 自定义 Pass

基于 MLIR Pass Plugin 机制构建的插件式自定义 pass 集合，通过 `--load-pass-plugin` 动态加载，**不修改任何 MLIR 上游文件**（`tools/mlir-opt/mlir-opt.cpp` 已恢复原样）。

## 目录结构

```
Phase2Bridge/
├── README.md                        # 本文件
└── Demo/
    ├── CMakeLists.txt               # 插件构建配置（目标 libPhase2Passes.so）
    ├── inputs/
    │   └── 01-inspection.mlir       # 测试输入
    ├── libPhase2Passes.so           # 编译产物（生成）
    └── passes/
        ├── Phase2Passes.h           # 注册函数声明
        ├── Phase2Inspection.cpp     # Pass: SSA use-def / CFG / 支配检查
        └── PluginEntry.cpp          # 唯一插件入口 mlirGetPassPluginInfo
```

## 构建插件

```bash
cd ~/llvm-project/mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo

# 首次配置
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=$HOME/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm

# 编译 / 重编（改 pass 后只需这一步）
ninja -C build   # 产物：libPhase2Passes.so
```

改 pass 只需重编 `.so`，`mlir-opt` 无需重编。前置条件：build 开启 `LLVM_ENABLE_PLUGINS=ON` 且 mlir-opt 导出符号。

## 使用

```bash
cd ~/llvm-project

./build/bin/mlir-opt \
    --load-pass-plugin=./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/libPhase2Passes.so \
    --pass-pipeline='builtin.module(func.func(phase2-inspection))' \
    ./mlir/LearningSteps/IllustratedMLIR/Phase2Bridge/Demo/inputs/01-inspection.mlir
```

## 已注册的 Pass

| Pass 名 | 描述 | 作用域 |
|---------|------|--------|
| `phase2-inspection` | 检查 SSA use-def、CFG 块、支配关系 | `func.func` |

## 添加新 Pass

详见 `Demo/README.md` 的「添加新 Pass」：每个 pass 一个文件 + 导出注册函数，声明加进 `Phase2Passes.h`，入口统一在 `PluginEntry.cpp` 调用，源文件加进 `CMakeLists.txt` 的 `add_library`。

## 三种注册方式对比

| 方式 | 改上游 | 重编 mlir-opt | 增删 pass | 适用场景 |
|------|-------|--------------|----------|---------|
| in-tree（已弃用） | 是（改 mlir-opt.cpp） | 是 | 重编 mlir-opt | 学习/开发 |
| Pass Plugin（当前方案） | 否 | 一次（开 PLUGINS） | 只重编 .so | 扩展/研究 |
| 自建工具 | 否 | N/A | 重编工具 | 完整工具链 |

## 备注

- 旧 in-tree 改动（`mlir/test/lib/Transforms/CMakeLists.txt`、`mlir/test/lib/Transforms/Phase2Inspection.cpp`）仍保留在工作区，但 `mlir-opt.cpp` 已恢复上游，注册函数不再被调用。如需零上游改动，可一并还原这两个文件。
- 插件 `.so` 为 MODULE 库，不链接 MLIR 静态库，运行时从 mlir-opt 进程解析 MLIR 符号；因此 `.so` 不能脱离 mlir-opt 单独使用，也不要在 `LD_LIBRARY_PATH` 里混入 MLIR 库目录（避免加载两份符号）。
