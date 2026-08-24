func.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c100 = arith.constant 100 : i32
  %r = scf.parallel (%i) = (%c0) to (%c100) step (%c1) init(%c0) -> (i32) {
    scf.reduce(%c1) : i32 {
    ^bb0(%lhs: i32, %rhs: i32):
      %s = arith.addi %lhs, %rhs : i32
      scf.reduce.return %s : i32
    }
  }
  return %r : i32
}
