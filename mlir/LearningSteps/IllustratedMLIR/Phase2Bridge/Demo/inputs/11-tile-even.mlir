func.func @matmul(%a: tensor<128x256xf32>, %b: tensor<256x512xf32>)-> tensor<128x512xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<128x512xf32>
    %init = linalg.fill ins(%zero: f32) outs(%empty: tensor<128x512xf32>) -> tensor<128x512xf32>
    %r = linalg.matmul ins(%a, %b: tensor<128x256xf32>, tensor<256x512xf32>) outs(%init: tensor<128x512xf32>) -> tensor<128x512xf32>

    return %r: tensor<128x512xf32>
}