func.func @main(%a: i32, %b: i32) -> i32 {
    %c = arith.addi %a, %b : i32
    %d = arith.muli %c, %b : i32
    %e = arith.constant 1 : i32
    %f = arith.subi %d, %e : i32
    return %f : i32
}