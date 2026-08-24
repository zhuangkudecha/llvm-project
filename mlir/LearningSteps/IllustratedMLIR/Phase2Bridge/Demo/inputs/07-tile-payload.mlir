// ============================================================
// Week 9 · Step 1：最小 Payload IR
// 形状：A(128×256) × B(256×512) → C(128×512)
//   - lhs = M×K = 128×256
//   - rhs = K×N = 256×512
//   - C   = M×N = 128×512
// 关键点：
//   - %init 是 destination/init（linalg.fill 生成），不是可省略的装饰参数。
//   - tensor 语义的 matmul 返回新 SSA value；tiling 后各结果 tile 要插回 %init。
// 运行：mlir-opt 07-tile-payload.mlir（无 transform，通过 verifier 即可）
// 输出：原样打印，无 tiling 结构。
// ============================================================
func.func @matmul(%lhs: tensor<128x256xf32>, %rhs: tensor<256x512xf32>) -> tensor<128x512xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x512xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<128x512xf32>) -> tensor<128x512xf32>
    %result = linalg.matmul ins(%lhs, %rhs: tensor<128x256xf32>, tensor<256x512xf32>)
        outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>
    
    return %result : tensor<128x512xf32>
}