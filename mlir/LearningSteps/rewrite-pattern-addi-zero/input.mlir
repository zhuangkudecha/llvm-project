module {
  func.func @add_zero_rhs(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = arith.addi %arg0, %c0 : i32
    return %0 : i32
  }

  func.func @add_zero_lhs(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = arith.addi %c0, %arg0 : i32
    return %0 : i32
  }

  func.func @add_non_zero(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %0 = arith.addi %arg0, %c1 : i32
    return %0 : i32
  }
}
