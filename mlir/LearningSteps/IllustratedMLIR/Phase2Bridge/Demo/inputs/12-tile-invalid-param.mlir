// ============================================================
// Week 10 · Test 12：参数校验负例（tile-m=0）
// pass: phase2-matmul-tiling{tile-m=0 tile-n=64 tile-k=32}
// 运行：
//   mlir-opt --load-pass-plugin=./libPhase2Passes.so \
//     -pass-pipeline='builtin.module(func.func(phase2-matmul-tiling{tile-m=0 tile-n=64 tile-k=32}))' \
//     inputs/12-tile-invalid-param.mlir -verify-diagnostics
// 关键观察：
//   - 输入是合法 payload（能正常 parse / 正常 tile）
//   - 失败来自 pass 的参数校验，不是 MLIR 语法错误
//   - 校验发生在任何 mutation 之前（只读阶段）
// ============================================================
// expected-error @+1 {{tile-m, tile-n and tile-k must be positive}}
func.func @matmul(%arg0: tensor<128x256xf32>, %arg1: tensor<256x512xf32>) -> tensor<128x512xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x512xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<128x512xf32>) -> tensor<128x512xf32>
  %result = linalg.matmul ins(%arg0, %arg1: tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>
  return %result : tensor<128x512xf32>
}
