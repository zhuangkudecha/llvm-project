// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse,canonicalize))' | FileCheck %s

// This file is intentionally small. It is useful for:
// 1. Running canonicalize and cse with mlir-opt.
// 2. Setting a debugger breakpoint in CSE or Canonicalizer pass execution.
// 3. Testing a simple rewrite pattern that removes x + 0 or x * 1.
// 4. Writing FileCheck assertions for before/after IR.

module {
  func.func @canonicalize_and_cse(%arg0: i32, %arg1: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32

    // CSE target: these two operations compute the same expression.
    %sum0 = arith.addi %arg0, %arg1 : i32
    %sum1 = arith.addi %arg0, %arg1 : i32

    // Canonicalize target: x + 0 -> x.
    %plus_zero = arith.addi %sum0, %c0 : i32

    // Canonicalize target: x * 1 -> x.
    %times_one = arith.muli %sum1, %c1 : i32

    return %plus_zero, %times_one : i32, i32
  }
}

// CHECK-LABEL: func.func @canonicalize_and_cse
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
// CHECK: %[[SUM:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.constant 0
// CHECK-NOT: arith.constant 1
// CHECK: return %[[SUM]], %[[SUM]] : i32, i32
