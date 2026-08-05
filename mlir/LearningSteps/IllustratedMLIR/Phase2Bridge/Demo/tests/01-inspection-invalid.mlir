func.func @invalid_dominance(%cond: i1) -> i32 {
    cf.cond_br %cond, ^left, ^right

    ^left:
        %value = arith.constant 1 : i32
        cf.br ^merge

    ^right:
        cf.br ^merge

    ^merge:
        // expected-error @+1 {{ operand #0 does not dominate this use}}
        return %value: i32
}