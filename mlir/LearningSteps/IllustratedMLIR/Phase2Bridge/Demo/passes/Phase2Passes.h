//===----------------------------------------------------------------------===//
// Phase2Passes.h - 各 pass 的注册函数声明
//===----------------------------------------------------------------------===//
//
// 每个 pass 实现文件（Phase2Xxx.cpp）定义自己的 pass 类（匿名 namespace）
// 并导出一个注册函数。本头文件把注册函数声明集中起来，供唯一的插件入口
// PluginEntry.cpp 统一调用。
//
// 新增 pass 时：
//   1. 新建 passes/Phase2Xxx.cpp（pass 类 + registerPhase2XxxPass()）
//   2. 在此头文件加声明
//   3. 在 PluginEntry.cpp 的 lambda 里加一行 registerPhase2XxxPass()
//   4. 在 CMakeLists.txt 的 add_library 里加源文件
//   5. ninja -C build 只重编 .so
//
//===----------------------------------------------------------------------===//

#ifndef PHASE2_BRIDGE_PASSES_H
#define PHASE2_BRIDGE_PASSES_H

void registerPhase2InspectionPass();
void registerPhase2AddZeroPass();
void registerFuseChainPass();
#endif // PHASE2_BRIDGE_PASSES_H

