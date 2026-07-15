module attributes {transform.with_named_sequence} {
  func.func @matmul(%lhs: tensor<8x16xf32>, %rhs: tensor<16x4xf32>)
      -> tensor<8x4xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<8x4xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<16x4xf32>)
        outs(%filled : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }

  transform.named_sequence @__transform_main(
      %root: !transform.any_op) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_forall %matmul
        tile_sizes [4, 2]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
