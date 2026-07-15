// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(add-zero-pattern))' | FileCheck %s
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
// CHECK-LABEL: func.func @add_zero_rhs
// CHECK: %[[ARG0:.*]]: i32
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.constant 0
// CHECK: return %[[ARG0]] : i32
// CHECK-LABEL: func.func @add_zero_lhs
// CHECK: %[[ARG0:.*]]: i32
// CHECK-NOT: arith.addi
// CHECK-NOT: arith.constant 0
// CHECK: return %[[ARG0]] : i32
// CHECK-LABEL: func.func @add_non_zero
// CHECK: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[SUM:.*]] = arith.addi %[[ARG0:.*]], %[[C1]] : i32
// CHECK: return %[[SUM]] : i32