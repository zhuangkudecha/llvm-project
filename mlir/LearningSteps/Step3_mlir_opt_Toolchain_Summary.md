# MLIR Step 3: mlir-opt 工具链 - 学习总结

## 一、mlir-opt 是什么

`mlir-opt` 是 MLIR 的**核心工具**，相当于编译器的"瑞士军刀"。它的作用是：

- **读取** `.mlir` 文件
- **运行**一组 pass（变换/优化/降级）
- **输出**变换后的 `.mlir`

```
输入 .mlir → mlir-opt [passes] → 输出 .mlir
```

它不做代码生成（那是 `mlir-translate` 的工作），只做 IR 层面的变换。

---

## 二、基本用法

### 2.1 单个 Pass

```bash
mlir-opt input.mlir --canonicalize        # 运行一个pass
mlir-opt input.mlir --cse                  # 运行另一个pass
```

### 2.2 多个 Pass 串联（Pipeline）

Pass 按命令行顺序**依次执行**：

```bash
mlir-opt input.mlir --cse --canonicalize --inline
```

**实际演示 — 多 Pass 串联**：
```mlir
// 输入：重复计算 + 冗余运算
func.func @pipeline_demo(%a: i32, %b: i32, %cond: i1) -> i32 {
  %0 = arith.addi %a, %b : i32       // 第一次计算
  %1 = arith.addi %a, %b : i32       // 完全相同（CSE可消除）
  %2 = arith.muli %0, %1 : i32
  %c0 = arith.constant 0 : i32
  %3 = arith.addi %2, %c0 : i32      // x + 0（canonicalize可消除）
  func.return %3 : i32
}

// mlir-opt --cse --canonicalize 输出：
func.func @pipeline_demo(%arg0: i32, %arg1: i32, %arg2: i1) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.muli %0, %0 : i32       // 消除了重复计算和冗余加0
  return %1 : i32
}
```

### 2.3 使用 --pass-pipeline 语法

更精确地控制 pass 作用的层级：

```bash
# 在 module 级别运行，func 内部执行 cse + canonicalize
mlir-opt input.mlir --pass-pipeline="builtin.module(func.func(cse,canonicalize))"
```

**实际演示 — 层级 pipeline**：
```mlir
// 输入同上

// mlir-opt --pass-pipeline="builtin.module(func.func(cse,canonicalize))" 输出：
func.func @pipeline_demo(%arg0: i32, %arg1: i32, %arg2: i1) -> i32 {
  %0 = arith.addi %arg0, %arg1 : i32
  %1 = arith.muli %0, %0 : i32
  return %1 : i32
}
```

---

## 三、Pass 分类总览

当前版本共有 **~480 个 pass**（其中 ~289 个非测试 pass），分为以下几大类：

### 3.1 优化 Pass — 简化和消除冗余

#### `--canonicalize` — 规范化简化

将 IR 转换为规范形式，消除代数恒等式。

```mlir
// 输入：
%0 = arith.constant 0 : i32
%1 = arith.addi %0, %arg0 : i32     // 0 + x → x
%2 = arith.constant 1 : i32
%3 = arith.muli %1, %2 : i32        // x * 1 → x
%4 = arith.addi %3, %0 : i32        // x + 0 → x
func.return %4 : i32

// 输出：
func.return %arg0 : i32              // 全部消除，直接返回 x
```

#### `--cse` — 公共子表达式消除

消除完全相同的重复计算。

```mlir
// 输入：
%0 = arith.addi %a, %b : i32        // 第一次计算
%1 = arith.addi %a, %b : i32        // 完全相同的计算
%2 = arith.muli %0, %1 : i32

// 输出：
%0 = arith.addi %a, %b : i32        // 只保留一次
%1 = arith.muli %0, %0 : i32        // 重用结果
```

#### `--sccp` — 稀疏条件常量传播

传播常量值，折叠已知条件分支，消除死代码。

```mlir
// 输入：
%a = arith.constant 5 : i32
%b = arith.constant 3 : i32
%c = arith.addi %a, %b : i32           // 5 + 3 = 8，可常量折叠
%cond = arith.constant true
%result = scf.if %cond -> i32 {        // 条件恒为 true，else 分支是死代码
  scf.yield %c : i32
} else {
  %d = arith.constant 0 : i32          // 死代码
  scf.yield %d : i32
}
func.return %result : i32

// 输出：
%c8_i32 = arith.constant 8 : i32       // 直接折叠为 8
return %c8_i32 : i32                    // 消除了 if/else 和死代码
```

#### `--inline` — 函数内联

将函数调用展开为函数体，消除调用开销。

```mlir
// 输入：
func.func @add_one(%x: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %result = arith.addi %x, %c1 : i32
  func.return %result : i32
}
func.func @caller(%val: i32) -> i32 {
  %r = func.call @add_one(%val) : (i32) -> i32   // 调用被内联
  func.return %r : i32
}

// 输出：
func.func @caller(%arg0: i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32               // @add_one 的体被展开进来
  %0 = arith.addi %arg0, %c1_i32 : i32
  return %0 : i32
}
```

#### `--mem2reg` — 内存到寄存器提升

消除不必要的 alloca/store/load，直接用 SSA 值。

```mlir
// 输入：
func.func @mem2reg_demo(%arg0: i32) -> i32 {
  %alloc = memref.alloca() : memref<i32>
  memref.store %arg0, %alloc[] : memref<i32>     // 存入临时内存
  %loaded = memref.load %alloc[] : memref<i32>   // 再读出来
  func.return %loaded : i32
}

// 输出：
func.func @mem2reg_demo(%arg0: i32) -> i32 {
  return %arg0 : i32       // 消除了 alloca/store/load，直接返回
}
```

#### `--symbol-dce` — 符号死代码消除

移除未被引用的函数和符号。

```mlir
// 输入：
func.func @used(%x: i32) -> i32 { func.return %x : i32 }
func.func @unused(%x: i32) -> i32 { func.return %x : i32 }  // 从未被调用
func.func @main() -> i32 {
  %c = arith.constant 42 : i32
  func.return %c : i32
}

// 输出（注：需配合标记，否则保守保留）：
// 默认情况下 symbol-dce 保留所有 func.func（因为可能被外部调用）
// 配合 `sym_visibility = "private"` 才会移除真正无用的私有函数
```

#### `--reconcile-unrealized-casts` — 消除冗余类型转换

移除可以互相抵消的 `unrealized_conversion_cast` 链。

```mlir
// 输入：
%0 = builtin.unrealized_conversion_cast %x : i32 to i64    // i32 → i64
%1 = builtin.unrealized_conversion_cast %0 : i64 to i32    // i64 → i32，互相抵消
func.return %1 : i32

// 输出：
return %x : i32       // 直接消除两步转换
```

---

### 3.2 转换/降级 Pass — convert-*

这是最大的一类，共 **~55 个**，将高层方言逐步降级到底层：

#### `--convert-arith-to-llvm` — 算术运算降级

将 arith 操作转换为 llvm 操作。

```mlir
// 输入：
%0 = arith.addi %arg0, %arg1 : i32
%1 = arith.muli %0, %arg0 : i32

// 输出：
%0 = llvm.add %arg0, %arg1 : i32     // arith.addi → llvm.add
%1 = llvm.mul %0, %arg0 : i32        // arith.muli → llvm.mul
```

#### `--convert-func-to-llvm` — 函数降级

将 func 方言转为 llvm 方言的函数。

```mlir
// 输入：
func.func @add(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  func.return %r : i32
}

// 输出：
llvm.func @add(%arg0: i32, %arg1: i32) -> i32 {  // func.func → llvm.func
  %0 = arith.addi %arg0, %arg1 : i32
  llvm.return %0 : i32                             // func.return → llvm.return
}
```

#### `--convert-cf-to-llvm` — 控制流降级

将 cf 分支操作转换为 llvm 分支。

```mlir
// 输入：
cf.cond_br %cond, ^then(%x : i32), ^else(%y : i32)
^then(%a: i32):
  cf.br ^merge(%a : i32)
^else(%b: i32):
  cf.br ^merge(%b : i32)

// 输出：
llvm.cond_br %arg0, ^bb1(%arg1 : i32), ^bb2(%arg2 : i32)  // cf.cond_br → llvm.cond_br
^bb1(%0: i32):
  llvm.br ^bb3(%0 : i32)                                    // cf.br → llvm.br
^bb2(%1: i32):
  llvm.br ^bb3(%1 : i32)
```

#### `--convert-math-to-llvm` — 数学函数降级

将 math 内建函数转换为 LLVM intrinsics。

```mlir
// 输入：
%0 = math.sqrt %x : f32
%1 = math.sin %0 : f32

// 输出：
%0 = llvm.intr.sqrt(%arg0) : (f32) -> f32    // math.sqrt → llvm.intr.sqrt
%1 = llvm.intr.sin(%0) : (f32) -> f32        // math.sin → llvm.intr.sin
```

#### `--finalize-memref-to-llvm` — MemRef 降级

将 memref 转为 LLVM 结构体指针操作。

```mlir
// 输入：
%0 = memref.load %mem[%idx] : memref<10xf32>

// 输出：
%0 = builtin.unrealized_conversion_cast %idx : index to i64
%1 = builtin.unrealized_conversion_cast %mem : memref<10xf32>
       to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
%2 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, ...)>    // 取出 aligned ptr
%3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32  // 计算偏移
%4 = llvm.load %3 : !llvm.ptr -> f32                                  // 加载值
```

#### `--convert-scf-to-cf` — 结构化控制流降级

将 scf.if/scf.for 转为底层 cf.cond_br/cf.br。

```mlir
// 输入：
%result = scf.if %cond -> i32 {
  %then = arith.constant 1 : i32
  scf.yield %then : i32
} else {
  %else = arith.constant 0 : i32
  scf.yield %else : i32
}

// 输出：
cf.cond_br %cond, ^bb1, ^bb2               // scf.if → cf.cond_br
^bb1:
  %c1 = arith.constant 1 : i32
  cf.br ^bb3(%c1 : i32)                    // scf.yield → cf.br
^bb2:
  %c0 = arith.constant 0 : i32
  cf.br ^bb3(%c0 : i32)
^bb3(%0: i32):                              // 结果通过 block argument 传递
  ...
```

#### `--lower-affine` — Affine 方言降级

将 affine.for/affine.load 等转为 scf.for + memref.load。

```mlir
// 输入：
affine.for %i = 0 to 100 {
  %v = affine.load %A[%i] : memref<100xf32>
  affine.store %v, %B[%i] : memref<100xf32>
}

// 输出：
%c0 = arith.constant 0 : index
%c100 = arith.constant 100 : index
%c1 = arith.constant 1 : index
scf.for %arg2 = %c0 to %c100 step %c1 {     // affine.for → scf.for
  %0 = memref.load %arg0[%arg2] : memref<100xf32>   // affine.load → memref.load
  memref.store %0, %arg1[%arg2] : memref<100xf32>   // affine.store → memref.store
}
```

#### `--linalg-generalize-named-ops` — Linalg 命名操作泛化

将 `linalg.matmul` 等命名操作转为通用的 `linalg.generic`。

```mlir
// 输入：
%result = linalg.matmul ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
                        outs(%C : tensor<4x4xf32>) -> tensor<4x4xf32>

// 输出：
#map = affine_map<(d0, d1, d2) -> (d0, d2)>       // A[m,k]
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>      // B[k,n]
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>      // C[m,n]
%0 = linalg.generic {
  indexing_maps = [#map, #map1, #map2],
  iterator_types = ["parallel", "parallel", "reduction"]
} ins(%A, %B : ...) outs(%C : ...) {
^bb0(%in: f32, %in_0: f32, %out: f32):
  %1 = arith.mulf %in, %in_0 : f32               // 乘法
  %2 = arith.addf %out, %1 : f32                  // 累加
  linalg.yield %2 : f32
} -> tensor<4x4xf32>
```

---

### 3.3 方言特定优化 Pass

#### Affine 方言（多面体优化）

##### `--affine-loop-tile` — 循环分块

将大循环拆成小块，提高缓存命中率。

```mlir
// 输入：一个 100x100 的双层循环
affine.for %i = 0 to 100 {
  affine.for %j = 0 to 100 {
    %v = affine.load %A[%i, %j] : memref<100x100xf32>
    affine.store %v, %B[%i, %j] : memref<100x100xf32>
  }
}

// mlir-opt --affine-loop-tile="tile-size=32" 输出：
// 外层按 32 分块，内层在块内遍历
affine.for %arg2 = 0 to 100 step 32 {        // 外层：步长 32
  affine.for %arg3 = 0 to 100 step 32 {
    affine.for %arg4 = #map(%arg2) to min #map1(%arg2) {   // 块内循环
      affine.for %arg5 = #map(%arg3) to min #map1(%arg3) {
        %0 = affine.load %arg0[%arg4, %arg5] : ...
        affine.store %0, %arg1[%arg4, %arg5] : ...
      }
    }
  }
}
```

##### `--affine-loop-unroll` — 循环展开

将循环体复制多份，减少循环开销。需要通过 `--pass-pipeline` 锚定到 func。

```mlir
// 输入：长度为 8 的循环
affine.for %i = 0 to 8 {
  %v = affine.load %A[%i] : memref<8xf32>
  affine.store %v, %B[%i] : memref<8xf32>
}

// mlir-opt --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))" 输出：
// 步长变为 4，循环体被复制 4 次
affine.for %arg2 = 0 to 8 step 4 {
  %0 = affine.load %arg0[%arg2] : memref<8xf32>           // 第 1 份
  affine.store %0, %arg1[%arg2] : memref<8xf32>
  %1 = affine.apply #map(%arg2)                            // %arg2 + 1
  %2 = affine.load %arg0[%1] : memref<8xf32>              // 第 2 份
  affine.store %2, %arg1[%1] : memref<8xf32>
  %3 = affine.apply #map1(%arg2)                           // %arg2 + 2
  %4 = affine.load %arg0[%3] : memref<8xf32>              // 第 3 份
  affine.store %4, %arg1[%3] : memref<8xf32>
  %5 = affine.apply #map2(%arg2)                           // %arg2 + 3
  %6 = affine.load %arg0[%5] : memref<8xf32>              // 第 4 份
  affine.store %6, %arg1[%5] : memref<8xf32>
}
```

##### `--affine-loop-invariant-code-motion` — 循环不变量外提

将循环内不变的计算移到循环外。

```mlir
// 输入：%c 在循环内不变但每次都重新运算（如果涉及 load 等）
// 注：对于纯算术运算，canonicalize 通常已处理
affine.for %i = 0 to 100 {
  %v = affine.load %A[%i] : memref<100xf32>
  %mul = arith.mulf %v, %c : f32       // 如果 %c 是循环不变的 load 结果
  affine.store %mul, %B[%i] : memref<100xf32>
}
// 此 pass 会分析 affine 循环内的 memref 依赖，将安全的不变量外提
```

#### SCF 方言（结构化控制流优化）

##### `--scf-for-loop-peeling` — 循环剥离

将循环的最后一个（或前几个）迭代从主循环中分离出来，便于主循环向量化。

```mlir
// 输入：
scf.for %i = %c0 to %c100 step %c1 {
  %v = memref.load %A[%i] : memref<100xf32>
  memref.store %v, %B[%i] : memref<100xf32>
}

// 输出（当迭代数与向量化宽度不整除时更有意义）：
// 主循环处理整除部分 + 剩余部分单独处理
```

#### GPU 方言

##### `--gpu-kernel-outlining` — GPU Kernel 外联

将 `gpu.launch` 的体提取为独立的 `gpu.func`。

```mlir
// 输入：
func.func @main() {
  %gDim = arith.constant 1 : index
  %bDim = arith.constant 256 : index
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %gDim, ...)
             threads(%tx, %ty, %tz) in (%block_x = %bDim, ...) {
    gpu.terminator
  }
  func.return
}

// 输出：
module attributes {gpu.container_module} {
  func.func @main() {
    gpu.launch_func @main_kernel::@main_kernel     // gpu.launch → gpu.launch_func
      blocks in (%c1, %c1, %c1) threads in (%c256, %c1, %c1)
    return
  }
  gpu.module @main_kernel {                         // 提取为独立模块
    gpu.func @main_kernel() kernel {
      gpu.return
    }
  }
}
```

#### 其他优化 Pass

##### `--control-flow-sink` — 控制流下沉

将操作移入最接近使用位置的分支中，减少不必要路径上的计算。

```mlir
// 输入：%c1 在 if 外部定义
%c1 = arith.constant 1 : i32
%result = scf.if %cond -> i32 {
  %r = arith.addi %x, %c1 : i32
  scf.yield %r : i32
} else {
  %r = arith.muli %x, %c1 : i32
  scf.yield %r : i32
}

// 输出：%c1 仍在 if 外部（此处 constant 太轻量不值得移动，对于更重的操作更有效）
```

##### `--loop-invariant-code-motion` — SCF 循环不变量外提

将 scf.for 中不变的操作移到循环外。

```mlir
// 输入：
%result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %x) -> i32 {
  %sum = arith.addi %acc, %y : i32    // %y 在循环中不变
  scf.yield %sum : i32
}

// 输出：如果 %y 的计算涉及更重的操作（如 load），会被外提
// 对于简单的算术，此 pass 不做额外优化
```

---

### 3.4 诊断/分析 Pass

#### `--print-op-stats` — 操作统计

统计 IR 中每种操作的数量。

```mlir
// 输入：
func.func @stats(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  %1 = arith.muli %0, %a : i32
  %2 = arith.addi %1, %b : i32
  func.return %2 : i32
}

// mlir-opt --print-op-stats 输出：
Operations encountered:
-----------------------
    arith.addi   , 2      // 两个 addi
    arith.muli   , 1      // 一个 muli
  builtin.module , 1
     func.func   , 1
     func.return , 1
```

#### `--strip-debuginfo` — 移除调试信息

移除所有 `loc(...)` 位置信息，减小输出大小。

```mlir
// 输入（带位置信息）：
func.func @debug(%x: i32) -> i32 loc("test.mlir":1:1) {
  %c1 = arith.constant 1 : i32 loc("test.mlir":2:5)
  func.return %c1 : i32 loc("test.mlir":3:3)
}

// 输出（位置信息被移除）：
func.func @debug(%arg0: i32) -> i32 {
  %c1_i32 = arith.constant 1 : i32
  return %c1_i32 : i32
}
```

---

## 四、调试选项

### 4.1 `--mlir-print-ir-after-all` — 查看每步变换

```bash
mlir-opt input.mlir --cse --canonicalize --mlir-print-ir-after-all
```

输出示例：
```mlir
// -----// IR Dump After CSE (cse) //----- //
module {
  func.func @pipeline_demo(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %0 : i32        // CSE 已消除重复 addi
    %c0_i32 = arith.constant 0 : i32
    %2 = arith.addi %1, %c0_i32 : i32   // canonicalize 会继续消除 x+0
    return %2 : i32
  }
}

// -----// IR Dump After Canonicalizer (canonicalize) //----- //
module {
  func.func @pipeline_demo(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %0 : i32        // x+0 已被消除
    return %1 : i32
  }
}
```

### 4.2 `--mlir-timing` — 查看耗时

```bash
mlir-opt input.mlir --cse --canonicalize --mlir-timing
```

输出：
```
===-------------------------------------------------------------------------===
                         ... Execution time report ...
===-------------------------------------------------------------------------===
  ----Wall Time----  ----Name----
    0.0004 ( 68.4%)  Parser          // 解析输入
    0.0000 (  4.0%)  CSE             // CSE 耗时
    0.0000 (  8.0%)  Canonicalizer   // 规范化耗时
    0.0000 (  6.8%)  Output          // 输出结果
    0.0001 ( 12.7%)  Rest
    0.0005 (100.0%)  Total
```

### 4.3 `--dump-pass-pipeline` — 查看 Pipeline 结构

```bash
mlir-opt input.mlir --cse --canonicalize --dump-pass-pipeline
```

输出：
```
Pass Manager with 2 passes:
builtin.module(
  cse,
  canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal ...}
)
```

### 4.4 `--verify-each` — 每个 Pass 后验证

```bash
mlir-opt input.mlir --cse --canonicalize --verify-each
```

每个 pass 执行后自动运行 IR 验证器，确保 IR 合法。如果某个 pass 产生了非法 IR，会立即报错。

### 4.5 `--mlir-elide-elementsattrs-if-larger=N` — 省略大型常量

超过 N 个元素的 dense 属性用 `dense_resource<__elided__>` 代替。

```mlir
// 输入：20 个元素的常量张量
%cst = arith.constant dense<[1.0, 2.0, 3.0, ..., 10.0]> : tensor<20xf32>

// mlir-opt --mlir-elide-elementsattrs-if-larger=5 输出：
%cst = arith.constant dense_resource<__elided__> : tensor<20xf32>  // 超过5个，省略显示
```

### 4.6 `--split-input-file` — 分割输入文件

用一个文件包含多个独立测试用例，用 `// -----` 分隔。

```mlir
// 第一个测试用例
module {
  func.func @test1(%x: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %r = arith.addi %x, %c0 : i32
    func.return %r : i32
  }
}
// -----

// 第二个测试用例（独立处理）
module {
  func.func @test2(%a: i32, %b: i32) -> i32 {
    %r = arith.addi %a, %b : i32       // 第一次
    %r2 = arith.addi %a, %b : i32      // 重复
    func.return %r : i32
  }
}
```

```bash
# 每个片段独立经过 --canonicalize --cse 处理
mlir-opt input.mlir --canonicalize --cse --split-input-file
```

输出：
```mlir
// 第一个结果
module {
  func.func @test1(%arg0: i32) -> i32 {
    return %arg0 : i32                 // x + 0 被消除
  }
}
// -----
module {
  func.func @test2(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32 // 重复的 addi 被消除
    return %0 : i32
  }
}
```

### 4.7 `--mlir-print-debuginfo` — 显示位置信息

在输出中保留 `loc(...)` 位置信息（默认不显示）。

### 4.8 `--allow-unregistered-dialect` — 允许未注册方言

允许使用未注册的自定义操作，用于测试。

```bash
mlir-opt input.mlir --allow-unregistered-dialect
```

### 4.9 `--verify-diagnostics` — 验证诊断信息

用于 lit 测试，检查是否产生了预期的错误/警告信息。

---

## 五、已注册的方言

当前版本支持 **47 个方言**：

```
acc, affine, amdgpu, amx, arith, arm_neon, arm_sme, arm_sve,
async, bufferization, builtin, cf, complex, dlti, emitc, func,
gpu, index, irdl, linalg, llvm, math, memref, ml_program, mpi,
nvgpu, nvvm, omp, pdl, pdl_interp, ptr, quant, rocdl, scf,
shape, shard, smt, sparse_tensor, spirv, tensor, test, test_dyn,
test_irdl_to_cpp, tosa, transform, ub, vector, wasmssa,
x86vector, xegpu, xevm
```

查看命令：
```bash
mlir-opt --show-dialects
```

---

## 六、典型编译 Pipeline 示例

### 6.1 完整 CPU 降级路径（Affine → LLVM）

```mlir
// 输入：一个简单的仿射循环
module {
  func.func @add_arrays(%A: memref<100xf32>, %B: memref<100xf32>, %C: memref<100xf32>) {
    affine.for %i = 0 to 100 {
      %a = affine.load %A[%i] : memref<100xf32>
      %b = affine.load %B[%i] : memref<100xf32>
      %sum = arith.addf %a, %b : f32
      affine.store %sum, %C[%i] : memref<100xf32>
    }
    func.return
  }
}
```

```bash
# 完整降级流水线
mlir-opt input.mlir \
  --lower-affine \                    # affine.for → scf.for, affine.load → memref.load
  --convert-scf-to-cf \               # scf.for → cf.br + cf.cond_br
  --canonicalize \                     # 简化
  --cse \                              # 消除重复
  --convert-arith-to-llvm \            # arith.addf → llvm.fadd
  --convert-func-to-llvm \             # func.func → llvm.func
  --convert-cf-to-llvm \               # cf.cond_br → llvm.cond_br
  --finalize-memref-to-llvm \          # memref → llvm struct/ptr
  --convert-index-to-llvm \            # index → i64
  --reconcile-unrealized-casts         # 消除残留的转换操作
```

### 6.2 通用 GPU 路径

```bash
mlir-opt input.mlir \
  --gpu-kernel-outlining \             # 提取 gpu.launch 体为 gpu.func
  --convert-gpu-to-nvvm \              # gpu 方言 → nvvm 方言（NVIDIA）
  --canonicalize
```

### 6.3 简单优化 Pipeline

```bash
mlir-opt input.mlir \
  --inline \                           # 内联函数调用
  --canonicalize \                     # 代数简化
  --cse \                              # 消除重复计算
  --loop-invariant-code-motion         # 循环不变量外提
```

---

## 七、Pass Pipeline 语法详解

### 7.1 层级结构

```
builtin.module(          # 顶层 module
  func.func(             # 每个 function
    cse,                 #   先做 CSE
    canonicalize         #   再做规范化
  )
)
```

### 7.2 嵌套管线

```bash
# 在 module 级别运行 inline，在 func 级别运行 cse + canonicalize
mlir-opt input.mlir --pass-pipeline="builtin.module(inline, func.func(cse, canonicalize))"
```

### 7.3 带参数的 Pass

```bash
# affine-loop-unroll 必须锚定到 func.func
mlir-opt input.mlir --pass-pipeline="builtin.module(func.func(affine-loop-unroll{unroll-factor=4}))"

# affine-loop-tile 可以直接在 module 级别运行
mlir-opt input.mlir --affine-loop-tile="tile-size=32"
```

用 `--dump-pass-pipeline` 查看参数默认值：
```
canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}
```

---

## 八、关键要点

### 8.1 mlir-opt 的核心概念

| 概念 | 说明 |
|------|------|
| **Pass** | 一个 IR 变换单元（优化、降级、分析等） |
| **Pipeline** | 多个 Pass 按顺序执行的管线 |
| **锚点** | Pass 作用的操作类型（module、func 等） |
| **验证** | 每个 Pass 后检查 IR 合法性 |

### 8.2 Pass 的三大类别

| 类别 | 数量 | 目的 | 示例 |
|------|------|------|------|
| **优化 Pass** | ~100+ | 简化/优化 IR | canonicalize, cse, inline, sccp |
| **转换 Pass** | ~55+ | 方言降级 | convert-*-to-* |
| **分析/工具 Pass** | ~30+ | 调试和分析 | print-op-stats, print-ir |

### 8.3 实践要点

1. **始终从 `--canonicalize` 和 `--cse` 开始** — 最常用的优化
2. **用 `--mlir-print-ir-after-all` 调试** — 看清每步变换
3. **用 `--dump-pass-pipeline` 确认** — 确保 pipeline 符合预期
4. **用 `--verify-each` 保证正确性** — 每个 pass 后验证 IR
5. **降级顺序很重要** — 高层到底层：affine → scf → cf → llvm
6. **某些 pass 需要 `--pass-pipeline`** — 如 `affine-loop-unroll` 需要锚定到 `func.func`
7. **`test/` 目录是最好的参考** — 每个功能都有对应的 lit 测试

### 8.4 与其他工具的关系

```
mlir-tblgen    → 从 .td 生成 C++ 代码（操作定义）
mlir-opt       → IR 变换和优化的主要工具
mlir-translate → MLIR ↔ 其他格式（如 LLVM IR）的翻译
mlir-runner    → 执行 MLIR JIT
```

---

## 九、下一步学习

**Step 4: Toy Tutorial**
- 通过 Toy 教程（Ch0-Ch4）构建一个完整的 mini 编译器
- 理解从源语言 → MLIR → 代码生成的完整流程
