builtin.module {
  func.func @multi_region_block(%cond: i1, %n: index) -> i32 {
    %base = arith.constant 10 : i32
    %shared = arith.constant 1 : i32

    %selected = scf.if %cond -> (i32) {
      %then_value = arith.addi  %base, %shared : i32
      scf.yield %then_value : i32
    } else {
      %else_value = arith.addi %base, %shared : i32
      scf.yield %else_value : i32
    }

    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index

    %result = scf.for %iv = %zero to %n step %one 
        iter_args(%acc = %selected) -> (i32) {
          %next = arith.addi %acc, %base : i32
          scf.yield %next : i32
        }

    return %result : i32
  }
}