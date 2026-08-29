cmake_minimum_required(VERSION 3.18)
project(transformer_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# RTX 4090 是 sm_89（Ada）。在 4090 实例上构建时取消注释：
# set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(transformer_block transformer.cu)

# -lineinfo 供 ncu 做 source-level 归因（对应计划的 profiling 纪律）
target_compile_options(transformer_block PRIVATE -O3 -lineinfo)
