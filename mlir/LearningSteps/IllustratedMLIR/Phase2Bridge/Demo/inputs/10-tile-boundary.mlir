  // ============================================================
  // Week 9 · Step 4：非整除边界尾块收缩
  // 形状：M×N×K = 127×251×509，tile_sizes [64, 64, 32]
  // 运行：mlir-opt --transform-interpreter --mlir-print-local-scope 10-tile-boundary.mlir
  // 预期尾块（size = min(tileSize, bound - offset)）：
  //   M：127 = 64 + 63     → 第二轮 size 收缩为 63
  //   N：251 = 64*3 + 59   → 最后一轮 size 收缩为 59
  //   K：509 = 32*15 + 29  → 最后一轮 size 收缩为 29
  // 重点观察：
  //   - 尾块 extract_slice 的 size 是否变成动态计算（affine.min / arith.min*）
  //   - insert_slice 是否同样收缩（否则写越界）
  //
  // 实际输出中的尾块 size（已验证，与预期 63/59/29 吻合）：
  //   %5 = affine.min affine_map<(d0) -> (-d0 + 127, 64)>(%arg2)   // M: min(127-m, 64)
  //   %6 = affine.min affine_map<(d0) -> (-d0 + 251, 64)>(%arg4)   // N: min(251-n, 64)
  //   %7 = affine.min affine_map<(d0) -> (-d0 + 509, 32)>(%arg6)   // K: min(509-k, 32)
  //   → 尾块运行时计算：M 63（m=64）、N 59（n=192）、K 29（k=480）
  // 结构变化（对比 08 整除版）：
  //   - tile size 变成 SSA value（%5/%6/%7），tile 类型变 tensor<?x?xf32>（动态 shape）
  //   - 大量 affine.apply (d0) -> (d0 - 1)：slice 展开算 offset+size-1 上界，非新 tiling 语义
  //   - insert_slice 同样用动态 [%5, %6]，结果 tile 不写越界
  // ============================================================

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
            : (!transform.any_op) -> !transform.any_op
        
        %tiled, %loops:3 = transform.structured.tile_using_for %matmul tile_sizes [64, 64, 32]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}

func.func @matmul_boundary(%lhs: tensor<127x509xf32>, %rhs: tensor<509x251xf32>) -> tensor<127x251xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<127x251xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<127x251xf32>) -> tensor<127x251xf32>
    %result = linalg.matmul ins(%lhs, %rhs: tensor<127x509xf32>, tensor<509x251xf32>)
        outs(%init: tensor<127x251xf32>) -> tensor<127x251xf32>
    return %result : tensor<127x251xf32>
  }