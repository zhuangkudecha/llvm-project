# LLVM/MLIR FileCheck

## 概述

`FileCheck` 是 LLVM/MLIR 测试中最常用的文本输出验证工具。它读取两份内容：

```text
1. 标准输入 stdin
   被检查的真实输出，例如 mlir-opt 输出的 IR

2. 命令行参数 match-filename
   检查规则文件，里面通常写着 // CHECK: ...
```

一句话理解：

> FileCheck 用测试文件里的 `CHECK` 规则，验证某个工具的文本输出是否符合预期。

在 MLIR 中，它常用于测试 pass：

```text
input.mlir
  -> mlir-opt 跑 pass
      -> 输出转换后的 IR
          -> FileCheck 检查 IR 是否包含/不包含某些模式
```

---

## 一、FileCheck 解决什么问题

假设你写了一个 pass，希望验证：

```text
重复 add 被 CSE 消掉
x + 0 被 canonicalize 消掉
x * 1 被 canonicalize 消掉
```

你不想人工每次看 `mlir-opt` 输出，而是希望自动测试：

```bash
mlir-opt input.mlir -pass-pipeline='...' | FileCheck input.mlir
```

FileCheck 就负责检查输出里是否有你预期的文本。

它比 `grep` 更适合编译器测试，因为：

- 可以检查多个模式按顺序出现
- 可以检查某些内容不能出现
- 可以捕获 SSA 名字并在后面复用
- 可以把一个文件拆成多个逻辑检查块
- 可以支持多个 check prefix 测不同配置

---

## 二、最基本用法

格式：

```bash
producer-command | FileCheck check-pattern-file
```

例如：

```bash
./build/bin/mlir-opt ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  | ./build/bin/FileCheck ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir
```

含义：

```text
左边 mlir-opt:
  产生要检查的 IR 输出

右边 FileCheck 文件:
  提供 CHECK 规则

管道:
  把左边输出送给 FileCheck 的 stdin
```

### 2.1 为什么单独运行 FileCheck 会像卡死

这个命令不完整：

```bash
FileCheck input.mlir
```

它只告诉 FileCheck：

```text
请从 input.mlir 读取 CHECK 规则
```

但没有给它被检查的输入，所以它会继续等待 stdin。

如果已经卡住，按：

```text
Ctrl-C
```

退出。

### 2.2 用临时文件分两步运行

也可以不用 pipe：

```bash
./build/bin/mlir-opt ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir \
  -pass-pipeline='builtin.module(func.func(cse,canonicalize))' \
  > /tmp/out.mlir

./build/bin/FileCheck ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir \
  < /tmp/out.mlir
```

或者：

```bash
./build/bin/FileCheck ./mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir \
  --input-file=/tmp/out.mlir
```

---

## 三、RUN 行和 lit

LLVM/MLIR 的自动测试通常由 `lit` 运行。`lit` 会读取测试文件里的 `RUN:` 注释。

例如：

```mlir
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse,canonicalize))' | FileCheck %s
```

含义：

```text
%s
  当前测试文件路径

mlir-opt %s ...
  用当前文件作为输入跑 mlir-opt

| FileCheck %s
  用当前文件里的 CHECK 规则检查 mlir-opt 输出
```

`RUN:` 只对 `lit` 自动测试有意义。手动运行时可以没有 `RUN:`，只要你自己把命令写完整即可。

如果一个 lit 测试文件没有 `RUN:`，`lit` 会报错；但你手动运行 `mlir-opt | FileCheck` 时不需要 `RUN:`。

---

## 四、一个 MLIR 示例

输入文件：

[practice_canonicalize_cse.mlir](/home/fuhao/llvm-project/mlir/LearningSteps/Practices/practice_canonicalize_cse.mlir:1)

核心 IR：

```mlir
module {
  func.func @canonicalize_and_cse(%arg0: i32, %arg1: i32) -> (i32, i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32

    %sum0 = arith.addi %arg0, %arg1 : i32
    %sum1 = arith.addi %arg0, %arg1 : i32

    %plus_zero = arith.addi %sum0, %c0 : i32
    %times_one = arith.muli %sum1, %c1 : i32

    return %plus_zero, %times_one : i32, i32
  }
}
```

经过：

```bash
mlir-opt -pass-pipeline='builtin.module(func.func(cse,canonicalize))'
```

预期变成：

```mlir
%0 = arith.addi %arg0, %arg1 : i32
return %0, %0 : i32, i32
```

对应 FileCheck：

```mlir
// CHECK-LABEL: func.func @canonicalize_and_cse
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
// CHECK: %[[SUM:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.constant 0
// CHECK-NOT: arith.constant 1
// CHECK: return %[[SUM]], %[[SUM]] : i32, i32
```

---

## 五、常用 CHECK 指令

### 5.1 CHECK

```mlir
// CHECK: arith.addi
```

表示后续输出中必须出现 `arith.addi`。

多个 `CHECK:` 按顺序匹配：

```mlir
// CHECK: func.func @foo
// CHECK: arith.addi
// CHECK: return
```

要求 `arith.addi` 在 `func.func @foo` 后面，`return` 在 `arith.addi` 后面。

### 5.2 CHECK-LABEL

```mlir
// CHECK-LABEL: func.func @foo
```

用于标记一个逻辑检查块的开始。

常用于函数、module、独立 test case：

```mlir
// CHECK-LABEL: func.func @case0
// CHECK: ...

// CHECK-LABEL: func.func @case1
// CHECK: ...
```

它能避免 `case0` 的 `CHECK` 意外匹配到 `case1` 的输出。

### 5.3 CHECK-SAME

```mlir
// CHECK: func.func @foo
// CHECK-SAME: (%arg0: i32)
```

表示必须和上一条匹配在同一行。

MLIR 中常用于检查函数签名：

```mlir
// CHECK-LABEL: func.func @foo
// CHECK-SAME: (%[[ARG0:.*]]: i32)
```

### 5.4 CHECK-NEXT

```mlir
// CHECK: arith.addi
// CHECK-NEXT: return
```

表示下一条匹配必须出现在紧接着的下一行，中间不能有其他行。

适合检查严格相邻的输出，但 MLIR IR 有时会因为打印格式变化而插入额外内容，所以不要滥用。

### 5.5 CHECK-NOT

```mlir
// CHECK-NOT: arith.muli
```

表示在两个正向检查点之间不能出现该模式。

例如：

```mlir
// CHECK-LABEL: func.func @foo
// CHECK-NOT: arith.muli
// CHECK: return
```

含义：

```text
从 @foo 到 return 之间不能出现 arith.muli
```

### 5.6 CHECK-DAG

```mlir
// CHECK-DAG: arith.addi
// CHECK-DAG: arith.muli
```

表示这些模式都要出现，但顺序不固定。

适合检查：

- pass 输出顺序不稳定
- 多个独立 declaration
- 多个并列属性/符号

不要用 `CHECK-DAG` 检查有强数据依赖顺序的内容。

### 5.7 CHECK-EMPTY

```mlir
// CHECK: foo
// CHECK-EMPTY:
// CHECK-NEXT: bar
```

表示下一行必须是空行。

### 5.8 CHECK-COUNT

```mlir
// CHECK-COUNT-3: arith.addi
```

表示 `arith.addi` 必须连续匹配 3 次。

---

## 六、正则表达式

FileCheck 默认是字符串匹配。如果需要正则，用双大括号：

```mlir
// CHECK: arith.constant {{[0-9]+}} : i32
```

常见正则：

```text
{{.*}}
  任意内容

{{[0-9]+}}
  一个或多个数字

{{$}}
  行尾

{{^}}
  行首
```

MLIR 中常见写法：

```mlir
// CHECK: %{{.*}} = arith.addi
```

但更推荐使用 FileCheck 变量捕获 SSA 名字。

---

## 七、变量捕获和复用

MLIR 输出里的 SSA 编号不稳定：

```mlir
%0 = arith.addi ...
```

也可能变成：

```mlir
%1 = arith.addi ...
```

所以不要硬写 `%0`。应该写：

```mlir
// CHECK: %[[SUM:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK: return %[[SUM]], %[[SUM]] : i32, i32
```

语义：

```text
%[[SUM:.*]]
  定义一个 FileCheck 变量 SUM，匹配任意 SSA 名字

%[[SUM]]
  后面引用同一个变量，要求是同一个 SSA value 名字
```

参数也可以捕获：

```mlir
// CHECK-SAME: (%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32)
```

这样后面可以写：

```mlir
// CHECK: arith.addi %[[ARG0]], %[[ARG1]] : i32
```

---

## 八、多个 check prefix

默认 prefix 是：

```text
CHECK
```

可以用：

```bash
FileCheck %s --check-prefix=BEFORE
FileCheck %s --check-prefix=AFTER
```

或多个：

```bash
FileCheck %s --check-prefixes=COMMON,AFTER
```

文件中：

```mlir
// COMMON-LABEL: func.func @foo
// BEFORE: arith.muli
// AFTER-NOT: arith.muli
```

适合一个输入文件同时测试多个配置：

```mlir
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s --check-prefix=CSE
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse,canonicalize))' | FileCheck %s --check-prefix=CANON
```

---

## 九、COM 注释

如果你想在测试文件里写一行注释，里面提到 `CHECK:`，但不想让 FileCheck 识别它，可以用：

```mlir
// COM: 下面只是说明，不是 FileCheck 规则：
// COM: CHECK: arith.addi
```

FileCheck 默认把 `COM:` 和 `RUN:` 当成 comment prefix。

---

## 十、常见选项

### 10.1 --input-file

不用 stdin，直接指定被检查输出：

```bash
FileCheck check.mlir --input-file=/tmp/out.mlir
```

### 10.2 --dump-input

失败时显示输入和诊断：

```bash
FileCheck check.mlir --dump-input=fail
```

总是显示：

```bash
FileCheck check.mlir --dump-input=always
```

### 10.3 -v / -vv

显示匹配诊断：

```bash
FileCheck check.mlir -v
FileCheck check.mlir -vv
```

### 10.4 --strict-whitespace

默认 FileCheck 会规范化水平空白，空格和 tab 的差异通常不敏感。

如果需要严格空白：

```bash
FileCheck check.mlir --strict-whitespace
```

MLIR 测试一般不建议依赖过细空白。

### 10.5 --implicit-check-not

在正向检查之间隐式插入 NOT：

```bash
FileCheck check.mlir --implicit-check-not='warning:'
```

适合诊断测试，确保没有未预期 warning。

---

## 十一、FileCheck 和 lit 的关系

```text
lit
  测试执行器，读取 RUN 行并执行命令

FileCheck
  文本检查器，验证命令输出
```

一个典型 MLIR lit 测试：

```mlir
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(cse))' | FileCheck %s

module {
  func.func @foo(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.addi %arg0, %arg1 : i32
    return %1 : i32
  }
}

// CHECK-LABEL: func.func @foo
// CHECK: %[[SUM:.*]] = arith.addi
// CHECK: return %[[SUM]] : i32
```

用 `lit` 跑时：

```bash
llvm-lit path/to/test.mlir
```

用手动命令跑时：

```bash
mlir-opt path/to/test.mlir -pass-pipeline='...' | FileCheck path/to/test.mlir
```

---

## 十二、写 MLIR FileCheck 的实践建议

### 12.1 优先使用 CHECK-LABEL

每个函数或测试块前写：

```mlir
// CHECK-LABEL: func.func @foo
```

避免跨函数误匹配。

### 12.2 不要硬编码 SSA 编号

不推荐：

```mlir
// CHECK: %0 = arith.addi
// CHECK: return %0
```

推荐：

```mlir
// CHECK: %[[SUM:.*]] = arith.addi
// CHECK: return %[[SUM]]
```

### 12.3 CHECK-NOT 要放在明确范围内

更好：

```mlir
// CHECK-LABEL: func.func @foo
// CHECK-NOT: arith.muli
// CHECK: return
```

避免孤立写：

```mlir
// CHECK-NOT: arith.muli
```

因为它的作用范围可能比你想象的大。

### 12.4 不要检查无关细节

FileCheck 应该检查 pass 语义相关输出，不要过度检查所有行。过度严格会让测试很脆。

例如测试 CSE 时，重点是：

```text
重复 op 被合并
return 使用同一个 value
```

不是检查每个空格、每个 SSA 编号。

### 12.5 用 CHECK-DAG 处理无序输出

如果两个 declaration 顺序不重要：

```mlir
// CHECK-DAG: func.func private @runtime_a
// CHECK-DAG: func.func private @runtime_b
```

不要强制顺序。

---

## 十三、排错 checklist

### 13.1 FileCheck 像卡住

通常是没有输入 stdin：

```bash
FileCheck check.mlir
```

正确：

```bash
producer | FileCheck check.mlir
```

### 13.2 FileCheck 后面忘了文件

错误：

```bash
mlir-opt input.mlir | FileCheck
```

正确：

```bash
mlir-opt input.mlir | FileCheck input.mlir
```

### 13.3 没有 CHECK 规则

如果 check file 没有 `CHECK:`，FileCheck 没有实际检查内容。

### 13.4 RUN 行不是手动命令

`RUN:` 是给 `lit` 用的。手动 shell 里要把 `%s` 换成真实路径。

### 13.5 检查失败时看真实输出

先单独跑 producer：

```bash
mlir-opt input.mlir -pass-pipeline='...'
```

确认真实输出长什么样，再改 `CHECK`。

---

## 总结

FileCheck 的核心模型是：

```text
tool output from stdin
  +
CHECK patterns from file
  ->
ordered textual verification
```

在 MLIR pass 测试中，最常见模式是：

```bash
mlir-opt input.mlir -pass-pipeline='...' | FileCheck input.mlir
```

最常用指令：

```text
CHECK-LABEL
  定位函数/测试块

CHECK
  按顺序匹配必须出现的内容

CHECK-SAME / CHECK-NEXT
  约束同一行或下一行

CHECK-NOT
  约束某段范围内不能出现

CHECK-DAG
  约束无序出现

%[[NAME:.*]]
  捕获不稳定 SSA 名字并复用
```

掌握这些就足够写大多数 MLIR pass、canonicalize、cse、conversion、lowering 的回归测试。

