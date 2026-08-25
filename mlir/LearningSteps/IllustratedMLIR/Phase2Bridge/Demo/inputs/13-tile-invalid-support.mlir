// ============================================================
// Week 10 · Test 13：支持检查负例（合法 MLIR，但 contract 拒绝）
// pass: phase2-matmul-tiling{tile-m=64 tile-n=64 tile-k=32}
// 运行：
//   mlir-opt --load-pass-plugin=./libPhase2Passes.so \
//     -pass-pipeline='builtin.module(func.func(phase2-matmul-tiling{tile-m=64 tile-n=64 tile-k=32}))' \
//     inputs/13-tile-invalid-support.mlir -verify-diagnostics
// 关键观察：
//   - 两个输入都是合法 MLIR（能正常 parse）
//   - 拒绝来自 pass 的 isSupportedStaticRank2TensorMatmul 检查
//   - func1: memref operand → 不是 RankedTensorType → 拒
//   - func2: 动态 shape → hasStaticShape() == false → 拒
//   - signalPassFailure() 不会中断后续 func，两个错误都会报出
// ============================================================
// expected-error @+1 {{expected a supported linalg.matmul target}}
func.func @memref_matmul(%a: memref<128x256xf32>, %b: memref<256x512xf32>,
                         %c: memref<128x512xf32>) {
  linalg.matmul ins(%a, %b : memref<128x256xf32>, memref<256x512xf32>)
      outs(%c : memref<128x512xf32>)
  return
}

// expected-error @+1 {{expected a supported linalg.matmul target}}
func.func @dynshape_matmul(%a: tensor<?x256xf32>, %b: tensor<256x512xf32>,
                           %M: index) -> tensor<?x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty(%M) : tensor<?x512xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<?x512xf32>) -> tensor<?x512xf32>
  %r = linalg.matmul ins(%a, %b : tensor<?x256xf32>, tensor<256x512xf32>)
      outs(%init: tensor<?x512xf32>) -> tensor<?x512xf32>
  return %r : tensor<?x512xf32>
}
