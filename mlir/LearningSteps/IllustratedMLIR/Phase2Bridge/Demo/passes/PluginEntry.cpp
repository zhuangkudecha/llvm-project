//===----------------------------------------------------------------------===//
// PluginEntry - 唯一的 pass 插件入口
//===----------------------------------------------------------------------===//
//
// 一个 .so 只能有一个 mlirGetPassPluginInfo() 入口。新增 pass 时：
//   1. 新建 passes/Phase2Xxx.cpp（pass 类 + registerPhase2XxxPass()）
//   2. 在 Phase2Passes.h 加声明
//   3. 在这里的 lambda 里加一行 registerPhase2XxxPass()
//   4. 在 CMakeLists.txt 的 add_library 里加源文件
// 然后只重编 .so 即可，mlir-opt 无需改动。
//
//===----------------------------------------------------------------------===//

#include "Phase2Passes.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/Compiler.h"

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "Phase2Passes",
        "v0.1",
        []() {
            registerPhase2InspectionPass();
            registerPhase2AddZeroPass();
            registerFuseChainPass();
        },
    };
}
