func.func @inspect(%arg0: i32, %cond: i1) -> i32 {
    %c1 = arith.constant 1 : i32
    cf.cond_br %cond, ^left, ^right

    ^left:
        %left_value = arith.addi %arg0, %c1: i32
        cf.br ^merge(%left_value: i32)
    
    ^right:
        %right_value = arith.addi %arg0, %c1: i32
        cf.br ^merge(%right_value: i32)

    ^merge(%result: i32): 
        return %result: i32
}