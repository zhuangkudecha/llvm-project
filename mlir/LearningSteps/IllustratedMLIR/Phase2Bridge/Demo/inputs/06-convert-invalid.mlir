// 负例 1：full 模式下，unknown op（tensor.empty）无法 legalize → 报错
func.func @unknown_op_full(%a: i32) -> i32 {
  // expected-error@+1 {{failed to legalize operation 'tensor.empty'}}
  %t = tensor.empty() : tensor<4xf32>
  %c = arith.addi %a, %a : i32
  return %c : i32
}

// -----

// 负例 2：函数签名含 tensor<4xf32>，LLVMTypeConverter 无法转换 → func.func 无法 legalize
// expected-error@+1 {{failed to legalize operation 'func.func'}}
func.func @unconvertible_signature() -> tensor<4xf32> {
  %t = tensor.empty() : tensor<4xf32>
  return %t : tensor<4xf32>
}
