// ============================================================
// Week 9 · Step 3：tile_using_forall（scf.forall 版，只 tile 并行维）
// 策略：match linalg.matmul → tile_using_forall tile_sizes [64, 64, 0]（M/N/K）
//       K 传 0 = 不 tile reduction 维（forall 只 tile 并行维，tile K 会警告并行不安全）
// 运行：mlir-opt --transform-interpreter --mlir-print-local-scope 09-tile-forall.mlir
//
// 实际输出（payload func.func 部分）：
//   scf.forall (%arg2, %arg3) in (2, 8) shared_outs(%arg4 = %1) -> (tensor<128x512xf32>) {
//     %3 = affine.apply (d0) -> (d0 * 64)(%arg2)          // M offset = iv0 × 64
//     %4 = affine.apply (d0) -> (d0 * 64)(%arg3)          // N offset = iv1 × 64
//     %e0 = tensor.extract_slice %arg0[%3, 0] [64, 256]   // A tile 64×256（K 全量）
//     %e1 = tensor.extract_slice %arg1[0, %4] [256, 64]   // B tile 256×64（K 全量）
//     %e2 = tensor.extract_slice %arg4[%3, %4] [64, 64]   // C tile 64×64
//     %5  = linalg.matmul ins(64x256, 256x64) outs(64x64) -> 64x64
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %5 into %arg4[%3, %4] [64, 64]   // 并行写回
//     }
//   }
//
// 与 for 版（08）的 5 个关键差异：
//   - 迭代结构：单个 forall 的 2D 逻辑空间 in (2, 8)，不是三层嵌套 for
//   - K 维：无内层 K 循环，每块做全 K（256）reduction
//   - 部分和：无 iter_args 链，每块独立算完整 C tile
//   - 偏移：affine.apply 把 iteration index → offset（for 版直接用循环 IV）
//   - 写回：in_parallel + parallel_insert_slice（并行、无顺序），vs insert_slice + yield
// 要点：
//   - in (2, 8) 表达"16 个可并行迭代"，不等同 CUDA thread，需后续 mapping 才映射到硬件
//   - shared_outs：所有并行迭代共享的 destination，各迭代写不相交 region 是并行化前提
//   - handle：tile_using_forall 只返回 2 个 handle（tiled op + forall op）
// ============================================================
module attributes {transform.with_named_sequence}  {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1
            : (!transform.any_op) -> !transform.any_op

        %tiled, %forall = transform.structured.tile_using_forall %matmul tile_sizes [64, 64, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        
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