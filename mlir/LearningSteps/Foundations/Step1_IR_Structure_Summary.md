# MLIR Step 1: 理解IR结构 - 学习总结

## 核心概念

MLIR IR是一个**递归嵌套**的数据结构，由三个核心组件组成：

---

## 一、Operation（操作）

### 定义
操作是MLIR的基本计算单元，具有以下属性：
- **唯一标识**: 如 `"func.func"`, `"arith.addi"`, `"dialect.op_name"`
- **操作数**: 0个或多个输入值（Use）
- **结果**: 0个或多个输出值（Def）
- **属性**: 键值对字典 `{key = value}`
- **区域**: 嵌套的代码块（Region列表）

### 语法示例
```mlir
// 通用形式
%result = "dialect.op"(%input1, %input2) {attr = 42 : i32} : (i32, i32) -> i32

// 自定义形式（dialect注册后）
%1, %2 = arith.addi %a, %b : f32
```

### 关键特征
- 每个操作的结果都是新的**Value**
- 操作可以包含多个**Region**（嵌套代码）
- 属性用于编译时信息，不参与运行时

---

## 二、Region（区域）

### 定义
- Region是**Block的有序列表**
- Region是**操作**的容器
- 语义由包含它的**操作**定义

### 语法示例
```mlir
"my.op"() ({  // Region开始
  // Block内容
}) : () -> ()
```

### 关键特征
- Region没有名称或地址，只有其中的Block有
- Region没有类型或属性
- 第一个Block叫"entry block"，其参数即区域参数

---

## 三、Block（块）

### 定义
- Block是**操作的有序列表**
- Block可以有**参数**（类似函数参数）
- 在SSACFG Region中，Block最后必须是****终止操作**

### 语法示例
```mlir
^bb0(%arg0: i32, %arg1: f32):  // 块标签和参数
  %1 = arith.addi %arg0, %arg0 : i32
  return %1 : i32  // 终止操作
```

### 关键特征
| 特征 | 说明 |
|------|------|
| **标签** | 如 `^bb0`, `^bb1`，用于引用和跳转 |
| **参数** | Block开始时可接收的值 |
| **操作列表** | Block内按顺序执行的操作 |
| **终止操作** | Block的最后操作，决定控制流 |

### Region内的Block类型

#### 1. 单Block Region（最常见）
```mlir
func.func @single_block(%x: i32) -> i32 {
  ^bb0(%x: i32):
    %1 = arith.addi %x, %x : i32
    return %1 : i32
}
```

#### 2. 多Block Region（有控制流）
```mlir
func.func @multi_block(%x: i32, %cond: i1) -> i32 {
  ^bb0(%x: i32, %cond: i1):
    cf.cond_br %cond, ^bb1, ^bb2

  ^bb1:
    %1 = arith.addi %x, %x : i32
    return %1 : i32

  ^bb2:
    %2 = arith.muli %x, %x : i32
    return %2 : i32
}
```

---

## 四、Value（值）

### 定义
Value是IR中传递的数据，分为两类：
1. **Block Argument** - Block的参数
2. **Operation Result** - Operation的结果

### Value的产生和使用
```mlir
%a = arith.constant 5 : i32      // %a是Value（Operation结果）
%b = arith.addi %a, %a : i32      // %a被使用（Use），%b是新Value
%c = arith.muli %b, %b : i32      // %b被使用，%c是新Value
```

---

## 五、层次结构图

```
ModuleOp (builtin.module)
│
└── Region 1
    │
    └── Block 0
        ├── func.func @myfunc [Operation]
        │   │
        │   └── Region 1 (func的主体)
        │       │
        │       └── Block 0 (%args: ...) [Block Arguments作为Value]
        │           ├── %1 = arith.constant ... [Operation]
        │           │   └── Value %1 [结果]
        │           │
        │           ├── %2 = arith.addi %1, %1 [Operation]
        │           │   ├── Value %1 [操作数 - Use]
        │           │   └── Value %2 [结果 - Def]
        │           │
        │           ├── %3 = scf.if %cond [Operation - 有嵌套Region]
        │           │   ├── Value %cond [操作数 - Use]
        │           │   │
        │           │   ├── Region #1 (then分支)
        │           │   │   └── Block ^bb0
        │           │   │       └── ...操作...
        │           │   │
        │           │   └── Region #2 (else分支)
        │           │       └── Block ^bb0
        │           │           └── ...操作...
        │           │
        │           └── return %3 [Operation - 终止操作]
        │               └── Value %3 [操作数 - Use]
        │
        └── func.func @anotherfunc
```

---

## 六、Def-Use链（定义-使用链）

### 定义
- **Def（定义）**: 值的创建者（Operation或Block Argument）
- **Use（使用）**: 引用该值的操作

### 如何确定Def和Use

#### 判断规则
```
左赋值 = Def
右使用 = Use
Block参数 = Def
操作数 = Use
```

#### 示例分析
```mlir
%const = arithic.constant 42 : i32
#          ↑Def (定义 %const)

%add = arith.addi %const, %x : i32
#      ↑Def (定义 %add)    ↑Use    ↑Use
```

### 完整例子分析
```mlir
func.func @example(%x: i32) -> i32 {
  %1 = arith.constant 10 : i32    // %1的Def: arith.constant
  %2 = arith.addi %x, %1 : i32    // %x的Use, %1的Use, %2的Def
  %3 = arith.muli %2, %1 : i32    // %2的Use, %1的Use, %3的Def
  return %3 : i32                  // %3的Use
}
```

| 值 | Def来源 | Use位置 |
|----|---------|---------|
| `%x` | Block参数 | `arith.addi` |
| `%1` | `arith.constant` | `arith.addi`, `arith.muli` |
| `%2` | `arith.addi` | `arith.muli` |
| `%3` | `arith.muli` | `return` |

### Def-Use链图
```
%1 (arith.constant)
   │
   ├──→ %2 (arith.addi) ← %x (Block Arg)
   │       │
   │       └──→ %3 (arith.muli) ← %1
   │               │
   │               └──→ return
```

---

## 七、Def-Use链的作用

| 用途 | 说明 |
|------|------|
| **死代码消除** | 检测 `value.use_empty()`，删除无用值 |
| **常量传播** | 遍历 `value.getUsers()`，优化使用点 |
| **值替换** | `oldValue.replaceAllUsesWith(newValue)` |
| **依赖分析** | 构建操作依赖图，确定并行性 |
| **活跃变量** | 反向追踪使用链，分析值生命周期 |
| **调试** | 打印数据流关系，理解IR结构 |

---

## 八、C++遍历API

### 遍历操作
```cpp
// 递归遍历操作及其所有嵌套区域
op->walk([&](Operation *nestedOp) {
  // 处理每个操作
});

// 只遍历特定类型的操作
op->walk([](arith::AddIOp addOp) {
  // 只处理加法操作
});
```

### 获取Def
```cpp
Value value = ...;
if (Operation *defOp = value.getDefiningOp()) {
  // Value是Operation的结果
  llvm::outs() << "Defined by: " << defOp->getName() << "\n";
} else {
  // Value是Block Argument
  auto blockArg = cast<BlockArgument>(value);
  llvm::outs() << "Block argument #" << blockArg.getArgNumber() << "\n";
}
```

### 获取Use
```cpp
// 获取Value的所有使用者
for (Operation *user : value.getUsers()) {
  llvm::outs() << "Used by: " << user->getName() << "\n";
}

// 获取操作的所有操作数（Use）
for (Value operand : op->getOperands()) {
  // operand是一个Use
}

// 获取操作的所有结果（Def）
for (Value result : op->getResults()) {
  // result是一个Def
}
```

---

## 九、完整示例

```mlir
module {
  func.func @example(%x: i32, %y: i32) -> i32 {
    // Block ^bb0 with arguments %x, %y
    %1 = arith.constant 42 : i32
    %2 = arith.addi %x, %1 : i32

    %3 = scf.if %y -> i32 {
      // Region #1 (then branch)
      ^bb0:
        %4 = arith.muli %2, %2 : i32
        scf.yield %4 : i32
    } else {
      // Region #2 (else branch)
      ^bb0:
        %5 = arith.subi %2, %1 : i32
        scf.yield %5 : i32
    }

    return %3 : i32
  }
}
```

---

## 十、关键要点总结

1. **Operation** = 基本计算单元，有操作数、结果、属性、区域
2. **Region** = Block的容器，由操作定义其语义
3. **Block** = 操作的列表，可以有参数，有终止操作（SSACFG）
4. **Value** = Block Argument或Operation Result
5. **Def-Use链** = 追踪值的定义和使用关系，编译器优化的核心

---

## 下一步学习

Step 2: 语言参考（LangRef.md）
- 完整的类型系统
- 详细属性系统
- 方言（Dialect）体系
