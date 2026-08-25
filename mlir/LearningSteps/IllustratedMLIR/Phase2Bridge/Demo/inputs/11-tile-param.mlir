// ============================================================
// Week 10 · Test 11：参数化 tiling 整除正例
// pass: phase2-matmul-tiling{tile-m=64 tile-n=64 tile-k=32}
// 运行：
//   mlir-opt --load-pass-plugin=./libPhase2Passes.so \
//     -pass-pipeline='builtin.module(func.func(phase2-matmul-tiling{tile-m=64 tile-n=64 tile-k=32}))' \
//     inputs/11-tile-param.mlir | FileCheck tests/11-tile-param.check
// 关键观察：
//   - 3 层 scf.for（M→N→K），step = 64/64/32
//   - extract_slice 的 tile 类型是静态 tensor<64x32xf32> / 32x64 / 64x64
//     （静态 shape = 整除路径，没有 affine.min）
// ============================================================
func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<128x512xf32>) -> tensor<128x512xf32>
  %result = linalg.matmul ins(%arg0, %arg1: tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
