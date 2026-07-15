module {
  func.func @accumulate(%n: index) -> i32 {
    %c0 = arith.constant 0 : i32
    %lb = arith.constant 0 : index
    %step = arith.constant 1 : index

    %result = scf.for %i = %lb to %n step %step
        iter_args(%sum = %c0) -> i32 {
      %value = arith.index_cast %i : index to i32
      %next = arith.addi %sum, %value : i32
      scf.yield %next : i32
    }

    return %result : i32
  }
}
