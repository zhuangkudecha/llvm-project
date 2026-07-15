module {
  func.func @matmul(%lhs: tensor<8x16xf32>, %rhs: tensor<16x4xf32>)
      -> tensor<8x4xf32> {
    %zero = arith.constant 0.0 : f32
    %dead = arith.constant 1 : index
    %init = tensor.empty() : tensor<8x4xf32>
    %filled = linalg.fill ins(%zero : f32)
        outs(%init : tensor<8x4xf32>) -> tensor<8x4xf32>
    %result = linalg.matmul
        ins(%lhs, %rhs : tensor<8x16xf32>, tensor<16x4xf32>)
        outs(%filled : tensor<8x4xf32>) -> tensor<8x4xf32>
    return %result : tensor<8x4xf32>
  }
}
