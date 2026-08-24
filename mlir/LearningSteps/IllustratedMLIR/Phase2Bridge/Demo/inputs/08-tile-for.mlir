// ============================================================
// Week 9 · Step 2：tile_using_for（scf.for 版，3D tiling）
// 策略：match linalg.matmul → tile_using_for tile_sizes [64, 64, 32]（M/N/K）
// 运行：mlir-opt --transform-interpreter --mlir-print-local-scope 08-tile-for.mlir
//
// 实际输出（payload func.func 部分）：
//   scf.for %arg2 = %c0 to %c128 step %c64              // M 循环（128/64=2 次）
//     scf.for %arg4 = %c0_0 to %c512 step %c64_2        // N 循环（512/64=8 次）
//       scf.for %arg6 = %c0_1 to %c256 step %c32        // K 循环（256/32=8 次，reduction 最内层）
//         %e0 = tensor.extract_slice %arg0[%arg2, %arg6] [64, 32]    // A[m:m+64, k:k+32]
//         %e1 = tensor.extract_slice %arg1[%arg6, %arg4] [32, 64]    // B[k:k+32, n:n+64]
//         %e2 = tensor.extract_slice %arg7[%arg2, %arg4] [64, 64]    // C[m:m+64, n:n+64]（来自 iter_args）
//         %5  = linalg.matmul ins(%e0, %e1 : 64x32, 32x64) outs(%e2 : 64x64) -> 64x64
//         %ins = tensor.insert_slice %5 into %arg7[%arg2, %arg4] [64, 64]  // 插回 C[m,n]
//         scf.yield %ins
// 关键观察：
//   - offset 三维语义：m 决定 A/C 行；n 决定 B/C 列；k 决定 A 列 / B 行。
//   - iter_args 链 %1→%arg3→%arg5→%arg7 = 部分和累加（plan 里的 %acc）：
//     K 循环不产生新输出位置，只在同一 C tile 上累加。
//   - tile 形状：A 64×32、B 32×64、内层 matmul 64×64，与 plan 图一致。
//   - handle：%tiled → 新 matmul；%loops:3 → M/N/K 三个 scf.for。
//   - tile_using_for 对 3 维 op 返回 1 + 3 = 4 个 handle。
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


func.func @matmul(%lhs: tensor<128x256xf32>, %rhs: tensor<256x512xf32>) -> tensor<128x512xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x512xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty: tensor<128x512xf32>) -> tensor<128x512xf32>
    %result = linalg.matmul ins(%lhs, %rhs: tensor<128x256xf32>, tensor<256x512xf32>)
        outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>

    return %result : tensor<128x512xf32>
}