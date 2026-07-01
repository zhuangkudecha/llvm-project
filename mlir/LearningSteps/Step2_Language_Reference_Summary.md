# MLIR Step 2: 语言参考（Language Reference）- 学习总结

## 一、类型系统（Type System）

MLIR支持丰富而可扩展的类型系统，类型是Value的核心组成部分。

---

### 1.1 基本类型（Primitive Types）

#### 整数类型
```mlir
// 有符号整数
%0 = arith.constant 42 : i1      // 1位整数（布尔）
%1 = arith.constant 127 : i8     // 8位有符号整数
%2 = arith.constant 32767 : i16   // 16位有符号整数
%3 = arith.constant 2147483647 : i32  // 32位有符号整数
%4 = arith.constant 9223372036854775807 : i64  // 64位有符号整数

// 无符号整数（存储时仍是有符号，但语义上当作无符号）
%5 = arith.constant 255 : i8     // 存储为255，语义可解释为无符号
```

#### 浮点类型
```mlir
// 16位浮点
%1 = arith.constant 1.5 : f16    // IEEE 754 half-precision
%2 = arith.constant 2.5 : bf16   // BFloat16 (Brain Float 16)

// 32位浮点（最常用）
%3 = arith.constant 3.1415926 : f32

// 64位浮点
%4 = arith.constant 2.718281828 : f64

// 扩展精度浮点
%5 = arith.constant 1.0 : f80    // 80位扩展精度
%6 = arith.constant 1.0 : f128   // 128位四精度
```

#### 布尔类型
```mlir
%true = arith.constant true : i1
%false = arith.constant false : i1

// 布尔运算
%result = arith.andi %true, %false : i1
```

---

### 1.2 复合类型（Composite Types）

#### Tensor类型（张量）
Tensor是**不可变**的高级数据结构，用于表示张量运算。

```mlir
// 1. 静态形状张量
%1 = tensor.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
%2 = tensor.constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32>
%3 = tensor.constant dense<1.0> : tensor<f32>

// 2. 动态维度（用?表示）
%4 = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>

// 3. 混合静态和动态维度
%5 = tensor.empty(%dim) : tensor<100x?xf32>
%6 = tensor.empty(%dim1, %dim2) : tensor<?x50xf32>

// 4. 高维张量
%7 = tensor.empty() : tensor<10x20x30x40xi64>

// 5. 空张量（0维）
%8 = tensor.constant dense<5.0> : tensor<f32>
```

**Tensor操作示例**:
```mlir
// 创建空张量
%empty = tensor.empty() : tensor<10x20xf32>

// 提取切片
%slice = tensor.extract_slice %tensor[0, 0] [5, 10] [1, 1]
    : tensor<10x20xf32kt -> tensor<5x10xf32>

// 插入切片
%updated = tensor.insert_slice %slice into %tensor[0, 0] [5, 10] [1, 1]
    : tensor<5x10xf32kt -> tensor<10x20xf32>

// 重塑
%reshaped = tensor.reshape %tensor(%new_shape)
    : tensor<10x20xf32kt -> tensor<200xf32>
```

#### MemRef类型（内存引用）
MemRef是**可变**的内存区域，用于表示可变缓冲区。

```mlir
// 1. 静态形状内存引用
%1 = memref.alloc() : memref<10x20xf32>

// 2. 动态形状
%2 = memref.alloc(%dim1, %dim2) : memref<?x?xf32>

// 3. 混合形状
%3 = memref.alloc(%dim) : memref<100x?xf32>

// 4. 带仿射映射的MemRef（表示非连续内存）
#map = affine_map<(i, j) -> (i * stride + j)>
%4 = memref.alloc() : memref<10x20xf32, #map>
```

**MemRef操作示例**:
```mlir
// 分配内存
%mem = memref.alloc() : memref<10x20xf32>

// 存储值
store %value, %mem[3, 5] : memref<10x20xf32>

// 加载值
%loaded = load %mem[3, 5] : memref<10x20xf32>

// 获取维度
%d0 = memref.dim %mem, 0 : memref<10x20xf32>  // 返回10
%d1 = memref.dim %mem, 1 : memref<10x20xf32>  // 返回20

// 复制
%copy = memref.alloc() : memref<10x20xf32>
memref.copy %mem, %copy : memref<10x20xf32> to memref<10x20xf32>

// 释放
memref.dealloc %mem : memref<10x20xf32>
```

#### Vector类型（向量）
Vector用于表示**SIMD向量操作**，形状必须是静态的。

```mlir
// 1. 一维向量
%1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>
%2 = arith.constant dense<[1, 2, 3, 4, 5, 6, 7, 8]> : vector<8xi32>

// 2. 多维向量
%3 = arith.constant dense<[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]> : vector<3x3xi32>

// 3. 向量操作
%sum = arith.addv %v1, %v2 : vector<4xf32>
%product = arith.mulf %v1, %v2 : vector<4xf32>

// 4. 向量归约
%total = vector.reduction <add>, %v : vector<4xf32> into f32
```

#### Tuple类型（元组）
Tuple可以包含不同类型的值。

```mlir
// 创建元组
%1 = "make_tuple"(%int_val, %float_val) : tuple<i32, f32>

// 提取元素
%int = "extract_tuple" %1[0] : tuple<i32, f32> -> i32
%float = "extract_tuple" %1[1] : tuple<i32, f32> -> f32

// 嵌套元组
%2 = "make_tuple"(%1, %bool_val) : tuple<tuple<i32, f32>, i1>
```

#### None类型
```mlir
// 表示空结果或无返回值
func.func @void_function(%x: i32) -> () {
  return
}
```

---

### 1.3 类型继承关系

```
Type (基类)
│
├── IntegerType (i1, i8, i16, i32, i64, i128, ...)
│   └── 位宽: 1, 8, 16, 32, 64, 128...
│
├── FloatType (f16, bf16, f32, f64, f80, f128)
│   i── Float16Type (半精度)
│   ├── BFloat16Type (Brain Float)
│   ├── Float32Type (单精度)
│   ├── Float64Type (双精度)
│   ├── Float80Type (扩展精度)
│   └── Float128Type (四精度)
│
├── ShapedType (有形状的类型)
│   ├── TensorType (tensor<...>, 不可变)
│   ├── MemRefType (memref<...>, 可变)
│   └── VectorType (vector<...>, SIMD)
│
├── TupleType (tuple<T1, T2, ...>)
│
├── NoneType (空类型)
│
└── FunctionType ((T1, T2, ...) -> (R1, R2, ...))
```

---

### 1.4 类型判断和转换

```cpp
// C++中的类型判断和转换

if (auto intType = dyn_cast<IntegerType>(value.getType())) {
  unsigned width = intType.getWidth();  // 获取位宽
  llvm::outs() << "Integer type with width: " << width << "\n";
}

if (auto floatType = dyn_cast<FloatType>(value.getType())) {
  if (floatType.isF32()) {
    llvm::outs() << "32-bit float\n";
  }
}

if (auto shapedType = dyn_cast<ShapedType>(value.getType())) {
  ArrayRef<int64_t> shape = shapedType.getShape();
  bool hasDynamicShape = shapedType.hasDynamicShape();
}
```

---

## 二、属性系统（Attributes）

属性用于存储**编译时常量**，不影响运行时语义。

---

### 2.1 基本属性类型

#### 整数属性
```mlir
// 有符号整数
"some.op"() {
  value = 42 : i64,
  count = -10 : i32,
  size = 255 : i8
} : () -> ()

// 无符号整数（存储为有符号）
"some.op"() {
  max_value = 4294967295 : i64  // 解释为无符号32位最大值
} : () -> ()
```

#### 浮点属性
```mlir
"some.op"() {
  learning_rate = 0.001 : f64,
  threshold = 1.5 : f32,
  epsilon = 1e-6 : f64
} : () -> ()
```

#### 布尔属性
```mlir
// 标准形式
"some.op.op"() {
  enabled = true,
  is_const = false
} : () -> ()

// 单位属性（布尔属性的特殊形式）
"some.op.op"() {unit} : () -> ()  // 等同于 {unit = true}
```

#### 字符串属性
```mlir
"some.op"() {
  name = "my_operation",
  path = "/path/to/file",
  description = "This is a long string with spaces"
} : () -> ()
```

---

### 2.2 数组属性

#### 整数数组
```mlir
"some.op"() {
  dims = [10, 20, 30] : vector<3xi64>,
  strides = [1, 10, 200] : vector<3xi64>,
  indices = [0, 1, 2, 3] : vector<4xi32>
} : () -> ()
```

#### 字符串数组
```mlir
"some.op"() {
  names = ["input", "hidden", "output"],
  paths = ["/path1", "/path2", "/path3"]
} : () -> ()
```

#### 浮点数组
```mlir
"some.op"() {
  weights = [0.1, 0.2, 0.3, 0.4] : vector<4xf32>,
  biases = [1.0, 2.0, 3.0] : vector<3xf64>
} : () -> ()
```

---

### 2.3 嵌套字典属性

```mlir
"some.op"() {
  config = {
    batch_size = 32 : i32,
    learning_rate = 0.001 : f64,
    use_bias = true,
    activation = "relu"
  }
} : () -> ()
```

---

### 2.4 类型属性

```mlir
"some.op"() {
  // 简单类型
  element_type = f32 : type,

  // 复杂类型
  tensor_type = tensor<10x20xf32> : type,
  memref_type = memref<100xf32> : type,
  func_type = (i32, f32) -> i32 : type
} : () -> ()
```

---

### 2.5 仿射映射属性

```mlir
"some.op"() {
  // 仿射映射定义索引转换
  map = affine_map<(d0, d1) -> (d0, d1)>,  // 恒等映射
  transpose = affine_map<(i, j) -> (j, i)>,  // 转置
  strided = affine_map<(i) -> (i * 2 + 1)>   // 带步长的访问
} : () -> ()
```

---

### 2.6 数组属性（Dense Elements Attribute）

```mlir
"some.op"() {
  // 常量数组
  int_array = dense<[1, 2, 3, 4, 5]> : tensor<5xi32>,
  float_matrix = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>,

  // 广播填充
  filled = dense<5.0> : tensor<10x20xf32>,  // 所有元素都是5.0

  // 稀疏表示
  sparse_data = dense<[
    [1.0, 0.0, 2.0],
    [0.0, 3.0, 0.0],
    [4.0, 0.0, 5.0]
  ]> : tensor<3x3xf32>
} : () -> ()
```

---

### 2.7 属性的作用

| 用途 | 说明 |
|------|------|
| **编译时常量** | 循环边界、数组大小等 |
| **配置参数** | 算法参数、优化选项 |
| **元数据** | 调试信息、注释 |
| **布局信息** | 内存布局、数据排列 |
| **语义标记** | 标记特殊属性（如pure、commutative）|

---

### 2.8 属性与操作数的区别

```mlir
// 操作数：运行时值（Value）
%input = arith.constant dense<[1, 2, 3]> : tensor<3xf32>
%weight = arith.constant dense<[0.1, 0.2, 0.3]> : tensor<3xf32>

// 属性：编译时常量
"linalg.matmul"(%input, %weight, %output) ({
  ^bb0(%arg: f32):
    arith.mulf %arg, %arg : f32
    linalg.yield %arg : f32
}) {
  indexing_maps = [
    {affine_map<(m, n, k) -> (m, k)>},
    {affine_map<(m, n, k) -> (k, n)>},
    {affine_map<(m, n, k) -> (m, n)>}
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
} : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) -> (tensor<3x3xf32>)
#          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#          这些是属性，编译时确定，不会改变
```

---

## 三、方言（Dialects）

**方言**是MLIR的核心抽象机制，每个方言定义一组相关的操作、类型和属性。方言使得MLIR可以表示不同抽象级别的IR。

---

### 3.1 方言命名约定

```
操作名格式: "dialect.operation_name"
或自定义形式: dialect.operation_name
```

---

### 3.2 核心方言详解

#### Builtin方言（builtin）
MLIR的内置方言，提供最基础的类型和操作。

```mlir
// Module操作 - 所有MLIR程序的根
module {
  // 所有内容在module内
}

// 函数类型
func.func @example() -> (i32, f32) {
  // 函数类型: () -> (i32, f32)
}

// 函数调用
%results = call @func(%args...) : (i32) -> i32

// 泛型转换操作
%casted = builtin.unrealized_cast %input
  : tensor<10xf32> to memref<10xf32>
```

#### Func方言（func）
函数定义和调用。

```mlir
// 函数定义
func.func @add(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  return %result : i32
}

// 函数调用
%sum = func.call @add(%x, %y) : (i32, i32) -> i32

// 间接调用
%func_ptr = func.constref @add : () -> (i32, i32) -> i32
%result = func.call_indirect %func_ptr(%arg1, %arg2) : (i32, i32) -> i32

// 返回操作
return %value : i32
```

#### Arith方言（arith）
基础算术运算。

```mlir
// 整数运算
%add = arith.addi %a, %b : i32      // 加法
%sub = arith.subi %a, %b : i32      // 减法
%mul = arith.muli %a, %b : i32      // 乘法
%div = arith.divsi %a, %b : i32     // 有符号除法
%udiv = arith.divui %a, %b : i32    // 无符号除法
%rem = arith.remsi %a, %b : i32     // 有符号余数
%urem = arith.remui %a, %b : i32    // 无符号余数

// 浮点运算
%fadd = arith.addf %a, %b : f32
%fsub = arith.subf %a, %b : f32
%fmul = arith.mulf %a, %b : f32
%fdiv = arith.divf %a, %b : f32

// 比较运算
%cmp = arith.cmpi slt, %a, %b : i32  // 有符号小于
%fcmp = arith.cmpf olt, %a, %b : f32  // 浮点小于

// 比较谓词: eq, ne, lt, le, gt, ge
// 通用前缀: s (signed), u (unsigned), o (ordered), uo (unordered)

// 扩展和截断
%ext = arith.extsi %value : i8 to i32     // 符号扩展
%trunc = arith.trunci %value : i32 to i8   // 截断

// 常量
%const = arith.constant 42 : i32
%fconst = arith.constant 3.14 : f32

// 位运算
%and = arith.andi %a, %b : i32
%or = arith.ori %a, %b : i32
%xor = arith.xori %a, %b : i32
%not = arith.xori %a, -1 : i32
%shl = arith.shli %a, %shift : i32  // 左移
%shr = arith.shrsi %a, %shift : i32 // 右移（有符号）
%ushr = arith.shrui %a, %shift : i32 // 右移（无符号）
```

#### MemRef方言（memref）
内存分配和访问。

```mlir
// 内存分配
%mem1 = memref.alloc() : memref<10x20xf32>
%mem2 = memref.alloc(%dim) : memref<?xf32>
%mem3 = memref.alloc(%d1, %d2, %d3) : memref<?x?x?xi32>

// 堆栈分配（函数结束自动释放）
%stack = memref.alloca() : memref<100xi32>

// 内存访问
store %value, %mem[%i, %j] : memref<10x20xf32>
%loaded = load %mem[%i, %j] : memref<10x20xf32>

// 维度查询
%d0 = memref.dim %mem, 0 : memref<10x20xf32>  // 返回10

// 复制内存
memref.copy %src, %dst : memref<10xf32> to memref<10xf32>

// Tensor到MemRef转换（bufferization）
%ref = memref.buffer_cast %tensor : tensor<10xf32> to memref<10xf32>
%tensor = memref.tensor_load %ref : memref<10xf32> to tensor<10xf32>

// 释放
memref.dealloc %mem : memref<10x20xf32>
```

#### Affine方言（affine）
多面体优化和循环结构。

```mlir
// 循环结构
affine.for %i = 0 to 100 {
  // 循环体
}

affine.for %i = 0 to 10 {
  affine.for %j = 0 to 20 {
    affine.for %k = 0 to 30 {
      // 嵌套循环
    }
  }
}

// 带步长的循环
affine.for %i = 0 to 100 step 2 {
  // %i 取值: 0, 2, 4, ..., 98
}

// 带仿射条件的循环
affine.for %i = 0 to 10 {
  affine.for %j = 0 to 10 {
    // 只执行当 i + j < 15
    affine.if #condition(%i, %j) {
      // 循环体
    }
  }
}

// 条件执行
affine.if #condition(%i, %j) {
  // 分支1
}

// Apply操作（用于映射循环到内存）
affine.apply #map(%i, %j)
```

#### SCF方言（scf）
结构化控制流。

```mlir
// 条件语句
%result = scf.if %cond -> i32 {
  ^bb0:
    %then_val = arith.addi %x, %x : i32
    scf.yield %then_val : i32  // 带值的if
} else {
  ^bb0:
    %else_val = arith.muli %x, %x : i32
    scf.yield %else_val : i32
}

// 无值的if
scf.if %cond {
  ^bb0:
    %1 = arith.addi %x, %x : i32
    store %1, %mem[%idx] : memref<10xi32>
}

// For循环（单迭代区域）
scf.for %i = 0 to %upper iter_args(%acc = %initial) -> i32 {
  %new_acc = arith.addi %acc, %acc : i32
  scf.yield %new_acc : i32  // 返回累加值
}

// While循环
%final = scf.while (%arg = %init) : (i32) -> (i32) {
  // 条件区域
  ^bb0(%arg: i32):
    %cond = arith.cmpi slt, %arg, %max : i32
    scf.condition(%cond) : (i32)

  // 循环体区域
  ^bb1(%arg: i32):
    %next = arith.addi %arg, %inc : i32
    scf.yield %next : i32
}

// 并行循环
scf.parallel (%i, %j) = (0, 0) to (10, 20) step (1, 1) {
  // 并行执行
}
```

#### Tensor方言（tensor）
高级张量操作。

```mlir
// 创建空张量
%empty = tensor.empty() : tensor<10x20xf32>

// 常量张量
%const = tensor.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>

// 提取切片
%slice = tensor.extract_slice %tensor[0, 0] [5, 10] [1, 1]
    : tensor<10x20xf32> to tensor<5x10xf32>

// 插入切片
%updated = tensor.insert_slice %slice into %tensor[0, 0] [5, 10] [1, 1]
    : tensor<5x10xf32> into tensor<10x20xf32>

// 重塑
%reshape = tensor.reshape %tensor(%new_shape)
    : tensor<10x20xf32> to tensor<200xf32>

// Cast
%casted = tensor.cast %tensor
    : tensor<10xf32> to tensor<?xf32>

// 扩展/收缩
%expanded = tensor.expand_shape %tensor [[0, 1], [2]]
    : tensor<10x20xf32> to tensor<2x5x20xf32>
```

#### Linalg方言（linalg）
线性代数和高阶操作。

```mlir
// 矩阵乘法
%result = linalg.matmul(%A, %B, %C_init)
    : (tensor<mxkxf32>, tensor<kxnf32>, tensor<mxnf32>)
      -> tensor<mxnf32>

// 通用循环嵌套
linalg.generic {
  indexing_maps = [
    affine_map<(i, j) -> (i, j)>,  // 输入A的访问
    affine_map<(i, j) -> (i, j)>,  // 输入B的访问
    affine_map<(i, j) -> (i, j)>   // 输出的访问
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%input1, %input2: tensor<10x20xf32>, tensor<10x20xf32>)
   outs(%output: tensor<10x20xf32>) {
  ^bb0(%a: f32, %b: f32):
    %result = arith.addf %a, %b : f32
    linalg.yield %result : f32
}

// 点积
%dot = linalg.dot(%vec1, %vec2, %init)
    : (tensor<100xf32>, tensor<100xf32>, f32) -> f32

// 向量化卷积
%conv = linalg.conv_2d(%input, %filter, %init)
    : (tensor<NxHxWxCxf32>,
       tensor<KHxKWxCFCxf32>,
       tensor<NxOHxOWxFCxf32>)
      -> tensor<NxOHxOWxFCxf32>
```

#### GPU方言（gpu）
GPU代码生成和操作。

```mlir
// GPU kernel启动
gpu.launch blocks(%bx, %by, %bz) in (%grid_x, %grid_y, %grid_z)
               threads(%tx, %ty, %tz) in (%block_x, %block_y, %block_z) {
  // GPU kernel代码
  %tid_x = gpu.block_id x : i32
  %x = gpu.thread_id x : i32
  // ...
}

// GPU内存操作
%gpu_mem = gpu.alloc %size : memref<1024xf32, 3>
gpu.memcpy %host_mem, %gpu_mem : memref<1024xf32>, memref<1024xf32, 3>
gpu.dealloc %gpu_mem : memref<1024xf32, 3>

// Barrier
gpu.barrier

// 共享内存
%shared = gpu.alloc %size : memref<256xf32, 5>
```

#### SPIRV方言（spirv）
Vulkan/SPIR-V操作。

```mlir
// SPIR-V模块
spirv.module "GLSL450" "SPIR-V" {
  // 全局变量
  spirv.GlobalVariable @var : !spirv.ptr<f32, StorageClass>
    {binding = 0 : i32, descriptor_set = 0 : i32}

  // 函数
  spirv.func @kernel(%in: !spirv.ptr<f32, StorageClass>) "None" {
    // SPIR-V操作
  }
}
```

#### LLVMIR方言（llvm）
LLVM IR方言，用于代码生成。

```mlir
// LLVM类型
%int_ptr = llvm.ptr<i32>
%func_type = llvm.func<i32 (i32)>

// LLVM操作
%addr = llvm.inttoptr %int_val : i64 to !llvm.ptr<i32>
%value = llvm.load %addr : !llvm.ptr<i32>
llvm.store %value, %addr : !llvm.ptr<i32>

// LLVM调用
%result = llvm.call @func(%arg) : (i32) -> i32

// LLVM控制流
llvm.cond_br %cond, ^bb1, ^bb2
```

---

### 3.3 方言体系结构

```
方言层次（从高层到低层）：

高层抽象：
├── TensorFlow (tf.*)
├── Tosa (tosa.*)
├── ONNX (onnx.*)
├── PyTorch (torch.*)
│
中层抽象：
├── Tensor (tensor.*)
├── Linalg (linalg.*)
├── SCF (scf.*)
│
中层优化：
├── Affine (affine.*)
├── MemRef (memref.*)
├── Bufferization (bufferization.*)
│
基础抽象：
├── Func (func.*)
├── Arith (arith.*)
├── CF (cf.*)        - 控制流
├── Math (math.*)    - 数学函数
│
底层抽象：
├── Vector (vector.*)
├── GPU (gpu.*)
├── LLVMIR (llvm.*)
└── NVGPU / AMDGPU / XeGPU
```

---

## 四、操作语法详解

---

### 4.1 通用操作语法（EBNF）

```
operation ::= op-result-list? generic-or-custom-op trailing-location?

generic-op ::= string-literal '(' value-use-list? ')' successor-list?
              dictionary-properties? region-list? dictionary-attribute?
              ':' function-type

custom-op ::= bare-id custom-operation-format

op-result-list ::= op-result (',' op-result)* '='
op-result ::= value-id (':' integer-literal)?

successor-list ::= '[' successor (',' successor)* ']'
successor ::= caret-id (':' block-arg-list)?

dictionary-properties ::= '<' dictionary-attribute '>'

region-list ::= '(' region (',' region)* ')'

dictionary-attribute ::= '{' (attribute-entry (',' attribute-entry)*)? '}'

trailing-location ::= 'loc' '(' location ')'
```

---

### 4.2 操作组成部分详细示例

```mlir
// 完整的操作分解
%a, %b = "multi.result.op"(%x, %y, %z)
#    └──────┬────┘          └──┬──┘  操作数
#         结果列表

    {attr1 = 10 : i32, attr2 = "value"}
#   └──────────────┬──────────────┘
#           属性字典

    ({
      ^bb0(%arg: i32):
        scf.yield %arg : i32
    })
#   └─┬─┘
#    Region列表

    : (i32, i32, i32) -> (i32, i32)
#   └─────────────┬──────────────┘
#            函数类型

    loc("file.mlir":10:5)
#   └─────────────────┬─────────────────┘
#            位置信息
```

---

### 4.3 通用形式 vs 自定义形式

```mlir
// 通用形式（所有方言都支持）
%result = "arith.addi"(%a, %b) : (i32, i32) -> i32
%cond = "arith.cmpi"(%a, %b) {cmpPredicate = 0 : i64} : (i32, i32) -> i1

// 自定义形式（方言注册后的漂亮打印）
%result = arith.addi %a, %b : i32
%cond = arith.cmpi slt, %a, %b : i32
```

---

### 4.4 终止操作

在SSACFG Region中，Block必须以终止操作结束。

```mlir
// return - 函数返回
return %value : i32
return : ()  // 无返回值

// cf.br - 无条件跳转
cf.br ^bb1(%arg: i32)

// cf.cond_br - 条件跳转
cf.cond_br %cond, ^bb1, ^bb2
cf.cond_br %cond, ^bb1(%arg1: i32), ^bb2(%arg2: i32)

// cf.switch - 多路跳转
cf.switch %index : i32, [
  default: ^bb_default,
  0: ^bb0,
  1: ^bb1(%arg: i32),
  2: ^bb2
]

// scf.yield - scf.if/for的值传递
scf.yield %value : i32
scf.yield %val1, %val2 : i32, f32

// func.call_indirect - 间接调用
func.call_indirect %func_ptr(%args...) : (i32) -> i32

// rethrow - 异常重抛
rethrow %exception

// unreachable - 不可达点
unreachable
```

---

### 4.5 属性字典格式

```mlir
// 简单属性字典
"op"() {
  attr1 = 42 : i32,
  attr2 = "value"
} : () -> ()

// 嵌套属性字典
"op"() {
  config = {
    size = 10 : i32,
    nested = {
      value = 1.0 : f32,
      enabled = true
    }
  }
} : () -> ()

// 数组属性
"op"() {
  array = [1, 2, 3] : vector<3xi32>,
  matrix = [[1, 2], [3, 4]] : tensor<2x2xi32>
} : () -> ()
```

---

## 五、标识符和命名规则

---

### 5.1 Value标识符

```mlir
// 百分号开头的标识符
%1 = arith.constant 10 : i32
%my_value = arith.addi %1, %1 : i32
%result = arith.muli %my_value, %my_value : i32

// 多结果访问
%multi:2 = "multi.result.op"() : () -> (i32, i32)
// 使用 %multi#0 和 %multi#1
```

---

### 5.2 Block标识符

```mlir
// 脱字符开头的标识符
^bb0(%arg: i32):
  // ...

^entry(%arg1: i32, %arg2: f32):
  // ...

^then_block:
  // ...

^else_block:
  // ...
```

---

### 5.3 函数标识符

```mlir
// @开头
func.func @my_function(%x: i32) -> i32 {
  // ...
}

func.func @compute_result(%a: i32, %b: f32) -> (i32, f32) {
  // ...
}
```

---

### 5.4 类型别名

```mlir
// 定义类型别名
!my_tensor = tensor<10x20xf32>
!matrix_3x3 = tensor<3x3xf32>
!dynamic_mem = memref<?x?xf32>
!int_ptr = memref<i32>

// 使用别名
%1 = arith.constant ... : !my_tensor
%2 = memref.alloc() : !dynamic_mem
```

---

### 5.5 属性别名

```mlir
// 定义属性别名
#identity_map = affine_map<(i, j) -> (i, j)>
#transpose_map = affine_map<(i, j) -> (j, i)>
#config = {
  batch_size = 32 : i32,
  learning_rate = 0.001 : f64
}

// 使用别名
"op"() {
  map = #identity_map
} : () -> ()

"op"() {
  config = #config
} : () -> ()
```

---

## 六、位置信息（Location）

位置信息用于调试和错误报告，可以通过`--mlir-print-debuginfo`选项显示。

---

### 6.1 内联位置

```mlir
// 格式: "file:line:column"
%1 = arith.constant 10 : i32 loc("my_file.mlir":10:5)
%2 = arith.addi %1, %1 : i32 loc("my_file.mlir":11:8)
```

---

### 6.2 命名位置

```mlir
// 格式: "name"
%1 = arith.constant 20 : i32 loc("my_constant")
%2 = arith.addi %1, %1 : i32 loc("addition_op")
```

---

### 6.3 未知位置

```mlir
// 表示位置未知或不存在
%1 = arith.constant 30 : i32 loc(unknown)
```

---

### 6.4 调用栈位置

```mlir
// 表示调用栈信息
%1 = arith.constant 40 : i32 loc(callsite("file.mlir":10:5) at "inlined_from")
```

---

## 七、完整的MLIR程序示例

```mlir
// ==================== 别名定义 ====================
!matrix = tensor<100x100xf32>
!mem = memref<100x100xf32>

#map0 = affine_map<(i, j) -> (i, j)>
#map1 = affine_map<(i, j) -> (j, i)>

// ==================== Module ====================
module {
  // ==================== 函数定义 ====================
  func.func @matrix_multiply(%A: !matrix, %B: !matrix) -> !matrix {
    // 分配输出内存
    %C = memref.alloc() : !mem

    // ==================== 循环嵌套（Affine方言） ====================
    affine.for %i = 0 to 100 {
      affine.for %j = 0 to 100 {
        // 初始化累加器
        %zero = arith.constant 0.0 : f32
        store %zero, %C[%i, %j] : !mem

        // 内循环（矩阵乘法）
        affine.for %k = 0 to 100 {
          // 加载元素
          %a = load %A[%i, %k] : !mem
          %b = load %B[%k, %j] : !mem
          %c = load %C[%i, %j] : !mem

          // 计算乘积和累加
          %prod = arith.mulf %a, %b : f32
          %sum = arith.addf %c, %prod : f32

          // 存储结果
          store %sum, %C[%i, %j] : !mem
        }
      }
    }

    // ==================== 转换回Tensor ====================
    %result = memref.tensor_load %C : !mem

    // 返回
    return %result : !matrix
  }

  // ==================== 另一个函数示例 ====================
  func.func @conditional_sum(%x: i32, %y: i32, %cond: i1) -> i32 {
    %result = scf.if %cond -> i32 {
      ^bb0:
        %then = arith.addi %x, %y : i32
        scf.yield %then : i32
    } else {
      ^bb0:
        %else = arith.muli %x, %y : i32
        scf.yield %else : i32
    }
    return %result : i32
  }
}
```

---

## 八、关键要点总结

### 8.1 类型系统

| 类型类别 | 示例 | 特点 |
|----------|------|------|
| **整数** | i1, i8, i16, i32, i64 | 可表示布尔和整数值 |
| **浮点** | f16, bf16, f32, f64 | IEEE 754格式 |
| **张量** | tensor<10x20xf32> | 不可变，高级抽象 |
| **MemRef** | memref<10x20xf32> | 可变，用于缓冲区 |
| **向量** | vector<4xf32> | SIMD操作 |
| **元组** | tuple<i32, f32> | 混合类型容器 |

### 8.2 属性系统

| 属性类型 | 示例 | 用途 |
|----------|------|------|
| **整数属性** | {size = 10 : i32} | 编译时常量 |
| **浮点属性** | {lr = 0.001 : f64} | 学习率等参数 |
| **布尔属性** | {enabled = true} | 开关选项 |
| **字符串属性** | {name = "op"} | 元数据 |
| **数组属性** | {dims = [10, 20]} | 多维常量 |
| **类型属性** | {type = f32} | 类型信息 |
| **仿射映射** | {map = affine_map<...>} | 索引转换 |

### 8.3 方言体系

| 抽象层次 | 方言 | 用途 |
|----------|------|------|
| **高层** | tf, tosa, onnx | 框架IR |
| **中层** | tensor, linalg, scf | 优化IR |
| **中层优化** | affine, memref | 循环优化 |
| **基础** | func, arith, cf | 基础操作 |
| **底层** | vector, gpu, llvm | 代码生成 |

### 8.4 操作语法

| 组件 | 说明 | 示例 |
|------|------|------|
| **结果列表** | 操作的输出 | `%1, %2 =` |
| **操作数** | 操作的输入 | `(%a, %b)` |
| **后继块** | 控制流目标 | `[^bb1, ^bb2]` |
| **属性字典** | 编译时常量 | `{attr = 10}` |
| **Region列表** | 嵌套代码块 | `({^bb0...})` |
| **函数类型** | 类型签名 | `(i32) -> i32` |
| **位置信息** | 调试信息 | `loc("file":1:1)` |

### 8.5 标识符规则

| 标识符类型 | 前缀 | 示例 |
|------------|------|------|
| **Value** | `%` | `%value`, `%1` |
| **Block** | `^` | `^bb0`, `^entry` |
| **Function** | `@` | `@myfunc` |
| **类型别名** | `!` | `!my_type` |
| **属性别名** | `#` | `#config` |

---

## 九、下一步学习

**Step 3: mlir-opt工具链**
- mlir-opt的基本使用
- 常用passes介绍
- pass pipeline的构建
- 调试选项
