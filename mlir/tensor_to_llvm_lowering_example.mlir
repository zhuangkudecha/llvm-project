// Minimal tensor-to-LLVM lowering example.
//
// High-level input:
//   tensor.empty + linalg.generic on tensor values
//
// Try the full lowering pipeline:
//   /Users/yuermei/Documents/llvm-project/build/bin/mlir-opt \
//     /Users/yuermei/Documents/llvm-project/mlir/tensor_to_llvm_lowering_example.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -convert-linalg-to-loops \
//     -convert-scf-to-cf \
//     -expand-strided-metadata \
//     -lower-affine \
//     -convert-arith-to-llvm \
//     -finalize-memref-to-llvm \
//     -convert-func-to-llvm \
//     -convert-cf-to-llvm \
//     -reconcile-unrealized-casts

func.func @tensor_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
    outs(%out : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %old: f32):
    %sum = arith.addf %a, %b : f32
    linalg.yield %sum : f32
  } -> tensor<4xf32>
  return %result : tensor<4xf32>
}
