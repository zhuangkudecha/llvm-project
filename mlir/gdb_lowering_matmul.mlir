// Minimal example for debugging Linalg lowering with gdb.
//
// Try:
//   /Users/yuermei/Documents/llvm-project/build/bin/mlir-opt \
//     /Users/yuermei/Documents/llvm-project/mlir/gdb_lowering_matmul.mlir \
//     -convert-linalg-to-loops \
//     -mlir-print-ir-before-all \
//     -mlir-print-ir-after-all \
//     -mlir-disable-threading

func.func @matmul(%A: memref<2x3xf32>, %B: memref<3x4xf32>, %C: memref<2x4xf32>) {
  linalg.matmul
    ins(%A, %B : memref<2x3xf32>, memref<3x4xf32>)
    outs(%C : memref<2x4xf32>)
  return
}
