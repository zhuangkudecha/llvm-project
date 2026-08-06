
module {
    func.func @fuses(
        %lhs: tensor<4x8xf32>,
        %rhs: tensor<8x16xf32>,
        %bias: tensor<16xf32>,
        %matmulInit: tensor<4x16xf32>,
        %biasInit: tensor<4x16xf32>,
        %reluInit: tensor<4x16xf32>
    ) -> tensor<4x16xf32> {

        %matmul = linalg.matmul ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>) outs(%matmulInit : tensor<4x16xf32>) -> tensor<4x16xf32>

        %biased = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1) -> (d0, d1)>,
                affine_map<(d0, d1) -> (d1)>,
                affine_map<(d0, d1) -> (d0, d1)>
            ], 
            iterator_types = ["parallel", "parallel"]
        }
        ins(%matmul, %bias : tensor<4x16xf32>, tensor<16xf32>)
        outs(%biasInit : tensor<4x16xf32>) {
            ^bb0(%value: f32, %biasValue: f32, %out: f32): 
            %sum = arith.addf %value, %biasValue : f32
            linalg.yield %sum : f32
        } -> tensor<4x16xf32>

        %relu = linalg.generic {
            indexing_maps = [
                affine_map<(d0, d1) -> (d0, d1)>,
                affine_map<(d0, d1) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel"]
        }
        ins(%biased: tensor<4x16xf32>)
        outs(%reluInit: tensor<4x16xf32>) {
            ^bb0(%value: f32, %out: f32):
            %zero = arith.constant 0.0 : f32
            %result = arith.maximumf %value, %zero : f32
            linalg.yield %result : f32
        } -> tensor<4x16xf32>


        return %relu : tensor<4x16xf32>
    }
}