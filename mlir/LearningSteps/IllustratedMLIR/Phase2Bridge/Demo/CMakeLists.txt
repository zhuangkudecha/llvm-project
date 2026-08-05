cmake_minimum_required(VERSION 3.20.0)
project(Phase2Passes LANGUAGES CXX)

# ── 1. 找 MLIR / LLVM 包（只用头文件路径，不链接静态库） ──────────────────
find_package(MLIR CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "MLIR_INCLUDE_DIRS = ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS = ${LLVM_INCLUDE_DIRS}")

# ── 2. 编译插件 ─────────────────────────────────────────────────────────────
# 关键：MODULE + 不链接 MLIR 静态库 + 允许未解析符号
# 运行时 MLIR 符号从 mlir-opt 二进制中动态解析
# mlir-opt 已导出 mlir::registerPass 等符号
add_library(Phase2Passes MODULE
    passes/Phase2Inspection.cpp
    passes/PluginEntry.cpp
)

target_include_directories(Phase2Passes PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/passes
)

target_compile_options(Phase2Passes PRIVATE
    -fno-rtti
    -fno-exceptions
    -fvisibility-inlines-hidden
    -ffunction-sections
    -fdata-sections
)

# 允许未解析符号 — 运行时由 mlir-opt 提供
target_link_options(Phase2Passes PRIVATE
    "-Wl,--allow-shlib-undefined"
)

# 输出到项目根目录
set_target_properties(Phase2Passes PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "lib"
)
