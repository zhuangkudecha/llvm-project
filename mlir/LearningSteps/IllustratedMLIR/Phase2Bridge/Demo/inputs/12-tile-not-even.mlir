func.func @matmul(%a: tensor<100x300xf32>, %b: tensor<300x200xf32>)-> tensor<100x200xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<100x200xf32>
    %init = linalg.fill ins(%zero: f32) outs(%empty: tensor<100x200xf32>) -> tensor<100x200xf32>
    %r = linalg.matmul ins(%a, %b: tensor<100x300xf32>, tensor<300x200xf32>) outs(%init: tensor<100x200xf32>) -> tensor<100x200xf32>

    return %r: tensor<100x200xf32>
}