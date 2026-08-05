// CHECK: op-result owner=arith.constant result=0 type=i32
// CHECK: op-result owner=arith.addi result=0 type=i32
// CHECK: function=inspect
// CHECK: operations=8
// CHECK: regions=1
// CHECK: blocks=4
// CHECK: block=0 argument=0 type=i32 users=2
// CHECK: block=3 argument=0 type=i32 users=1
// CHECK: block-dominance 0->3=1

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