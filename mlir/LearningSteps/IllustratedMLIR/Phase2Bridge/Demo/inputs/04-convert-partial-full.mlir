// partial/full 差异演示：
// - func.func 签名 (i32) -> i32 可被 LLVMTypeConverter 转换 ✓
// - tensor 方言没有配置任何 legality action → tensor.empty 是 unknown op
//   → partial 模式：允许原样保留
//   → full 模式：必须被 legalize，否则 conversion 失败
func.func @with_unknown(%a: i32) -> i32 {
    %t = tensor.empty() : tensor<4xf32>
    %c = arith.constant 0 : i32
    %d = arith.addi %a, %c : i32
    return %d : i32
}
