cmake_minimum_required(VERSION 3.15.4)
project(SimpleCNNCompiler)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found MLIR: ${MLIR_DIR}")

# Add MLIR include directories
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

# TableGen for operation definitions
set(LLVM_TARGET_DEFINITIONS mlir/CNNOps.td)
mlir_tablegen(CNNOps.inc -gen-op-defs)
add_public_tablegen_target(CNNOpsIncGen)

# Subdirectories
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

message(STATUS "SimpleCNNCompiler configured successfully")
