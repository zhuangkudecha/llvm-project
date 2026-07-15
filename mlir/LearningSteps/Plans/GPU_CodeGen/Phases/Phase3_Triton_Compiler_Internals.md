# Phase 3：Triton Compiler Internals

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 11–13（2026-09-23 至 2026-10-13）
- 计划工时：33h

## 前置条件

[Phase 2 Exit Gate](./Phase2_MLIR_Transformation_Bridge.md#phase-2-exit-gate) 已通过。

## 可验证阶段目标

- 固定并复现 Triton 源码构建环境与基线 commit。
- 能把同一 kernel 的 TTIR、TTGIR、LLVM IR、PTX 串成可解释的 lowering 链。
- 完成一个可开关、带测试、能生成预期 IR 差异的 compiler 改动。

## 本阶段产出

- Triton 源码版本、构建步骤与复现记录。
- TTIR/TTGIR/LLVM IR/PTX 对照 artifact 和 pipeline 笔记。
- 第一个 Triton compiler patch、测试与 reproducer。

## 阅读与源码版本边界

- 本页外链已于 **2026-07-14** 在线核验；它们是官方文档或官方仓库的直接页面，但 `main` 会继续演进。
- 一切实现结论以 `$CAPSTONE/compiler/baseline_commit.txt` 记录的 40 位 commit 为准。完成 Week 11 周一前，文中的 `$TRITON_ROOT/...` 仅是待核验候选路径，**不表示本机当前存在**。
- 固定 commit 后先执行以下检查；路径变化时，在 `environment/triton-build.md` 记录该 commit 的等价路径，再阅读和引用。GitHub 阅读链接也应把 `main` 替换为该 commit：`https://github.com/triton-lang/triton/blob/<commit>/<path>`。

```bash
TRITON_COMMIT=$(tr -d '[:space:]' < $CAPSTONE/compiler/baseline_commit.txt)
test "$(git -C $TRITON_ROOT rev-parse HEAD)" = "$TRITON_COMMIT"
test -e $TRITON_ROOT/python/triton/compiler/compiler.py
test -e $TRITON_ROOT/lib/Dialect/Triton/Transforms
test -e $TRITON_ROOT/lib/Dialect/TritonGPU/Transforms
rg -n 'add_stages|make_ttir|make_ttgir|make_llir|make_ptx' \
  $TRITON_ROOT/python/triton/compiler $TRITON_ROOT/third_party/nvidia
```

## Week 11：固定 Triton 源码与构建

**Files:**

- Create: `compiler/baseline_commit.txt`
- Create: `environment/triton-build.md`
- Update: `README.md`

### 工作日

- [ ] **周一：clone Triton 并固定 commit**

```bash
mkdir -p $HOME/workspace
git clone https://github.com/triton-lang/triton.git $TRITON_ROOT
git -C $TRITON_ROOT rev-parse HEAD | tee $CAPSTONE/compiler/baseline_commit.txt
```

  - **必读（目的：先理解官方源码入口和构建前提）：** [Triton 官方仓库 README / Install from source](https://github.com/triton-lang/triton/blob/main/README.md#install-from-source)。
  - **核验：** 写入后运行 `git -C $TRITON_ROOT cat-file -e "$(cat $CAPSTONE/compiler/baseline_commit.txt)^{commit}"`；后续所有源码阅读均切到这个 commit。

- [ ] **周二：创建独立虚拟环境**

```bash
python3 -m venv $TRITON_VENV
source $TRITON_VENV/bin/activate
python -m pip install --upgrade pip
test -e $TRITON_ROOT/python/requirements.txt
test -e $TRITON_ROOT/python/test-requirements.txt
python -m pip install -r $TRITON_ROOT/python/requirements.txt
python -m pip install -r $TRITON_ROOT/python/test-requirements.txt
# 从 PyTorch 官方 selector 复制与本机驱动/目标 CUDA 兼容的命令并执行：
python -m pip install torch --index-url <PYTORCH_INDEX_URL>
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
```

  - **必读（目的：从 Triton 自身声明安装完整 build/test 依赖）：** 固定 commit 的 `README.md`、`python/requirements.txt`、`python/test-requirements.txt` 和 `Makefile` 的 `dev-install-requires`；当前官方入口可对照 [`test-requirements.txt`](https://github.com/triton-lang/triton/blob/main/python/test-requirements.txt) 与 [`Makefile`](https://github.com/triton-lang/triton/blob/main/Makefile)，但执行前必须对两个 requirements 文件分别 `test -e`。
  - **必读（目的：选择与 RTX 4090 主机驱动和目标 CUDA 兼容的官方 PyTorch wheel）：** [PyTorch Start Locally](https://pytorch.org/get-started/locally/)。不得原样保留 `<PYTORCH_INDEX_URL>`；把 selector 生成的实际命令、index URL、`torch.__version__`、`torch.version.cuda`、driver 版本和 `torch.cuda.is_available()` 写入 `environment/triton-build.md`。
  - **官方补充：** [Python `venv` 文档](https://docs.python.org/3/library/venv.html)，用于理解隔离环境及 activation 的边界。后续 build/test 必须复用此 venv，不能混用系统 Python。

- [ ] **周三：从源码构建 editable install**

```bash
source $TRITON_VENV/bin/activate
cd $TRITON_ROOT
TRITON_BUILD_WITH_CLANG_LLD=true \
TRITON_BUILD_WITH_CCACHE=true \
python -m pip install -e . --no-build-isolation
```

  - **必读（目的：核对 editable build、LLVM 下载/复用和 build flags）：** [官方 Install from source](https://github.com/triton-lang/triton/blob/main/README.md#install-from-source)，并在固定 commit 上运行 `rg -n 'TRITON_BUILD_WITH|LLVM|pip install -e' $TRITON_ROOT/README.md $TRITON_ROOT/CMakeLists.txt $TRITON_ROOT/python`。
  - **Expected：** `pip` 成功生成可编辑安装；实际 compiler、linker、LLVM hash 与 flags 写入 `environment/triton-build.md`。

- [ ] **周四：验证 Python package 与解释器**

```bash
python -c 'import triton; print(triton.__version__); print(triton.__file__)'
TRITON_INTERPRET=1 python $CAPSTONE/triton_kernels/vector_add.py
```

  - **必读（目的：区分 interpreter 调试路径与 GPU 编译/执行路径）：** [官方 Debugging Triton / Interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html#using-the-interpreter)。
  - **Expected：** `triton.__file__` 指向该 editable checkout；interpreter 完成 vector add 数值检查。记录 interpreter 不支持项，不把它的通过等同于 PTX backend 通过。

- [ ] **周五：定位 triton-opt 和 pass pipeline**

```bash
find $TRITON_ROOT -type f -name triton-opt -perm -111
rg -n 'add_stages|make_ttir|make_ttgir|make_llir|make_ptx' \
  $TRITON_ROOT/third_party/nvidia $TRITON_ROOT/python/triton/compiler
```

  - **必读（目的：从 Python compile 入口定位 backend stages，而非凭阶段名猜 pipeline）：** [官方 `python/triton/compiler/compiler.py`](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)；在固定 commit 执行 `test -e` 后阅读 `compile`、stage 字典和 metadata/artifact 处理。
  - **必读（目的：定位 pass 实现集合）：** [官方 Triton transforms](https://github.com/triton-lang/triton/tree/main/lib/Dialect/Triton/Transforms) 与 [TritonGPU transforms](https://github.com/triton-lang/triton/tree/main/lib/Dialect/TritonGPU/Transforms)，但目录/文件名须由固定 commit 的 `test -e/rg` 再确认。

### 周末

- [ ] **周六：运行 Triton 单测子集**

先运行无 GPU 测试入口，再在 RTX 4090 上运行两个官方 tutorial smoke：

```bash
source $TRITON_VENV/bin/activate
export PYTHON=$TRITON_VENV/bin/python
test "$(command -v python)" = "$TRITON_VENV/bin/python"
make -C $TRITON_ROOT test-nogpu
python $TRITON_ROOT/python/tutorials/01-vector-add.py
python $TRITON_ROOT/python/tutorials/03-matrix-multiplication.py
```

若固定 commit 的 Make target 或 tutorial 路径发生变化，使用该 commit 的 README 中“Running tests”命令替换，并把最终实际命令、测试数和耗时写入 `triton-build.md`。

  - **必读（目的：理解两个 smoke 分别覆盖基础 load/store 与 dot/matmul）：** [官方 Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 和 [Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
  - **核验：** 先 `test -e` 两个固定 commit tutorial 文件并用 `rg -n 'PYTHON \?=|BUILD_DIR|test-nogpu|Running tests' $TRITON_ROOT/README.md $TRITON_ROOT/Makefile` 核实入口；`PYTHON=$TRITON_VENV/bin/python` 使 Make 的 build-dir 计算、lit/C++ build 与 pytest 都绑定同一解释器。不存在时按上段规则替换，不静默跳过。

- [ ] **周日：记录构建复现信息**

保存 Triton commit、其 LLVM hash、Python、compiler、CMake、Ninja 和 build flags。不得只写“使用最新版”。

  - **必读（目的：找到 Triton 固定 LLVM revision 的一手来源）：** 当前官方路径是 [`cmake/llvm-info.json`](https://github.com/triton-lang/triton/blob/main/cmake/llvm-info.json) 的 `llvm_hash` 字段。切到固定 commit 后先执行 `test -e $TRITON_ROOT/cmake/llvm-info.json && rg -n '"llvm_hash"' $TRITON_ROOT/cmake/llvm-info.json`；若该历史 commit 的布局不同，再用 `rg -n 'llvm_hash|llvm-info\.json|LLVM.*revision|LLVM_SYSPATH' $TRITON_ROOT/cmake $TRITON_ROOT` 定位并记录等价一手来源。
  - **Expected：** `environment/triton-build.md` 含 Triton SHA、LLVM SHA/来源、`python --version`、C/C++ compiler、`cmake --version`、`ninja --version`、环境变量和从冷环境复现命令。

### Week 11 Exit Gate

```text
editable Triton 从源码 import
triton-opt 路径已定位
baseline commit 和 LLVM revision 已记录
至少一组 compiler/kernel smoke test 通过
```

---

## Week 12：TTIR、TTGIR、LLVM IR、PTX 对照

**Files:**

- Create: `scripts/dump_triton_ir.sh`
- Create: `docs/triton_compiler_pipeline.md`
- Populate: `ir/ttir`, `ir/ttgir`, `ir/llvm`, `ir/ptx`

### 工作日

- [ ] **周一：dump vector add**

```bash
source $TRITON_VENV/bin/activate
TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=$CAPSTONE/results/experiments/week12-vector-add-dump \
python $CAPSTONE/triton_kernels/vector_add.py
```

  - **必读（目的：理解 `tl.program_id`、mask、load/store 在最小 kernel 中的语义）：** [官方 Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 与 [Triton language API](https://triton-lang.org/main/python-api/triton.language.html)。
  - **核验：** 在固定 commit 用 `rg -n 'MLIR_ENABLE_DUMP|TRITON_KERNEL_DUMP|TRITON_DUMP_DIR|TRITON_ALWAYS_COMPILE' $TRITON_ROOT` 确认变量仍生效；记录生成文件名到 `scripts/dump_triton_ir.sh`。

- [ ] **周二：dump fused Linear**

用相同环境变量运行 `fused_linear.py`，固定一个小 shape 避免 dump 混入多个 autotune candidate。

  - **必读（目的：识别 `tl.dot` 的输入/累加器约束，避免把多个 autotune config 的 IR 混为一条链）：** [官方 `triton.language.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) 与 [官方 Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
  - **Expected：** 记录 shape、dtype、`num_warps`、`num_stages` 和唯一 kernel specialization key；artifact 能与 vector add 分目录区分。

- [ ] **周三：标注 TTIR**

对 `tl.load/tl.dot/bias/relu/tl.store` 各找出对应 op；写明 shape、type 和 SSA data flow。

  - **必读（目的：用 op 定义确认 TTIR 的 operand/result/attribute 语义）：** [官方 `tt` Dialect](https://triton-lang.org/main/dialects/TritonDialect.html) 与 [TritonOps](https://triton-lang.org/main/dialects/TritonOps.html)。
  - **源码核验：** 固定 commit 后用 `rg -n 'def .*Load|def .*Store|def .*Dot' $TRITON_ROOT/include $TRITON_ROOT/lib` 定位 `.td` op 定义；文档标注必须能回指具体 op 名和 SSA 值。

- [ ] **周四：标注 TTGIR**

找出 module target、layout encoding、CTA/warp mapping、dot operand 和 layout conversion。

  - **必读（目的：理解 TTGIR 中 GPU layout encoding 和 conversion op 的契约）：** [官方 TritonGPUOps](https://triton-lang.org/main/dialects/TritonGPUOps.html)；另以 [官方 Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html) 辅助建立 thread/warp/CTA 映射直觉。
  - **源码核验：** 在固定 commit 执行 `rg -n 'BlockedEncoding|DotOperandEncoding|ConvertLayout|CTALayout' $TRITON_ROOT/include $TRITON_ROOT/lib/Dialect/TritonGPU`，以 `.td`/interface 实现校正文档随 `main` 产生的差异。

- [ ] **周五：标注 LLVM/PTX**

找出 address space、barrier、shared-memory、load/store 和 MMA 相关片段；不能只粘贴整份 IR。

  - **必读（目的：把 LLVM address space/NVVM intrinsic 映射到 PTX 概念）：** [LLVM NVPTX Back-end User Guide](https://llvm.org/docs/NVPTXUsage.html) 与 [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。
  - **Expected：** 每段摘录同时写“上游 SSA/op → LLVM/NVVM 形式 → PTX 指令/状态空间”，并注明 RTX 4090 的实际 target/`sm_89` 证据，不能仅按硬件型号推断。

### 周末

- [ ] **周六：画出一条 op lowering 链**

至少跟踪一个 `tl.dot` 和一个 bias broadcast，从 TTIR 一直跟到 PTX，并记录每一层丢失/增加的信息。

  - **必读（目的：从真实 stage 构造顺序确定层间边界）：** 固定 commit 的 `python/triton/compiler/compiler.py` 及 NVIDIA backend stage registration；先运行 `test -e` 和 `rg -n 'add_stages|make_ttir|make_ttgir|make_llir|make_ptx' $TRITON_ROOT/python/triton/compiler $TRITON_ROOT/third_party/nvidia`。
  - **官方语义对照：** [`tl.dot`](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)、[TritonGPUOps](https://triton-lang.org/main/dialects/TritonGPUOps.html)、[LLVM NVPTX guide](https://llvm.org/docs/NVPTXUsage.html)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

- [ ] **周日：验证 dump 可复现**

清理 Triton cache 后重新运行一次；确认脚本能重新生成同类 artifact，并记录 cache 对 dump 的影响。

  - **必读（目的：从官方编译入口理解 cache key、metadata 与输出 artifact 的关系）：** [官方 `compiler.py`](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)；在固定 commit 用 `rg -n 'cache|hash|metadata|asm' $TRITON_ROOT/python/triton/compiler/compiler.py` 锁定实际实现。
  - **Expected：** 清 cache 前后保存命令、文件清单和内容 hash；允许临时名/地址变化，但须解释结构差异，不能只写“看起来一样”。

### Week 12 Exit Gate

```text
四级 IR/PTX artifact 可复现
能解释 TTIR 与 TTGIR 职责差异
能跟踪至少一个 op 的完整 lowering 链
```

---

## Week 13：第一个 Triton Compiler 改动

**Files:**

- Create: `compiler/reproducers/week13-fused-linear.mlir`
- Create in Triton checkout: `$TRITON_ROOT/test/TritonGPU/phase3-local-transform.mlir`
- Mirror for experiment records: `compiler/tests/week13-local-transform.mlir`
- Create: `compiler/patches/week13-local-transform.patch`
- Update: `docs/triton_compiler_pipeline.md`

本周每一步都必须填写同一条可审计阅读链，且全部以固定 commit 为边界：

```text
pass registration → implementation → op/interface definition → test → TTGIR/PTX artifact
```

候选根目录为 `lib/Dialect/Triton/Transforms/`、`lib/Dialect/TritonGPU/Transforms/` 与 `third_party/nvidia/`；先 `test -e/rg`，再把实际文件、symbol 和行号写入 `docs/triton_compiler_pipeline.md`。官方滚动目录仅用于导航：[Triton transforms](https://github.com/triton-lang/triton/tree/main/lib/Dialect/Triton/Transforms)、[TritonGPU transforms](https://github.com/triton-lang/triton/tree/main/lib/Dialect/TritonGPU/Transforms)。

本阶段默认选择作用于 TTGIR 的低风险 TritonGPU local transform，因此真实回归测试 suite 是固定 commit 的 `$TRITON_ROOT/test/TritonGPU/`（当前官方 [test 根目录](https://github.com/triton-lang/triton/tree/main/test) 与 [TritonGPU suite](https://github.com/triton-lang/triton/tree/main/test/TritonGPU) 已于 2026-07-14 核验）。开始改动前必须执行 `test -d $TRITON_ROOT/test/TritonGPU`，并从目标 pass 的相邻 `.mlir` 测试复制 `RUN:`/FileCheck 约定。`$CAPSTONE/compiler/tests/` 只保存镜像与运行记录，不是 Triton CI 输入，不能代替 in-tree test。

### 工作日

- [ ] **周一：选择一个低风险 transformation**

使用以下命令从固定 commit 选择只影响局部 IR、已有测试充分的 pass：

```bash
rg -n 'RemoveLayout|Coalesce|Canonicalize|combine|fold' \
  $TRITON_ROOT/lib $TRITON_ROOT/third_party/nvidia/lib | head -80
test -d $TRITON_ROOT/test/TritonGPU
rg -n 'RemoveLayout|Coalesce|Canonicalize|combine|fold' \
  $TRITON_ROOT/test/TritonGPU | head -80
```

选择标准写入文档：输入/输出 IR 明确、无需改变 ABI、可用单个 reproducer、预期不改变数值语义。第一项改动不选择 MMA lowering 或 shared-memory allocator。

  - **必读（目的：完成候选 pass 的五段链并评估改动面）：** 固定 commit 上依次阅读 **pass registration**（`rg -n 'register.*Pass|GEN_PASS_REGISTRATION|Passes.td'`）→ **implementation**（候选 `Transforms/*.cpp`）→ **op/interface definition**（`include/**.td` 及 interface 实现）→ **test**（`test/` 中该 pass 名/FileCheck）→ 运行 pipeline 导出 **TTGIR/PTX artifact**。
  - **本步五段链：** 上述五段每一段至少记录一个固定 commit 下核验过的真实路径/symbol，并保存候选 pass 的 baseline TTGIR/PTX artifact。
  - **官方方法参考：** [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/) 与 [FileCheck documentation](https://llvm.org/docs/CommandGuide/FileCheck.html)；它们解释 registration/pipeline 与稳定断言的用法，不替代 Triton 固定 commit 源码。

- [ ] **周二：从 kernel 捕获 reproducer**

```bash
TRITON_REPRODUCER_PATH=$CAPSTONE/compiler/reproducers/week13-fused-linear.mlir \
TRITON_ALWAYS_COMPILE=1 \
python $CAPSTONE/triton_kernels/fused_linear.py
```

  - **必读（目的：确认 reproducer 环境变量的实现和捕获点）：** 在固定 commit 用 `rg -n 'TRITON_REPRODUCER_PATH|Reproducer' $TRITON_ROOT` 找到一手实现。
  - **本步五段链：** 从捕获点回溯 **pass registration** → 目标 pass **implementation** → reproducer 中目标 **op/interface definition** → 相邻官方 **test** → 保存 baseline **TTGIR/PTX artifact**；若任一段找不到，暂停改动并在文档记录缺口。

- [ ] **周三：先写 compiler test**

从 reproducer 最小化出一个只覆盖目标 transformation 的 test，运行 baseline 并保存 FileCheck output。

  - **必读（目的：写抗无关打印变化的检查）：** [FileCheck 官方文档](https://llvm.org/docs/CommandGuide/FileCheck.html)，重点读 `CHECK-LABEL`、变量和 `CHECK-NOT`；再读固定 commit 中目标 pass 的最近邻 `test/` 用例。
  - **真实测试路径：** 先 `test -d $TRITON_ROOT/test/TritonGPU`，再创建 `$TRITON_ROOT/test/TritonGPU/phase3-local-transform.mlir`；测试名在选择 pass 后应改成目标 pass 的语义名。把同一文件复制到 `$CAPSTONE/compiler/tests/week13-local-transform.mlir` 仅作 artifact 镜像，CI 证据必须来自 in-tree 文件。
  - **本步五段链：** 由 in-tree 测试 `RUN:` 中的 **pass registration/pipeline name** → 对应 **implementation** → 测试输入的 **op/interface definition** → 新增最小 **test** → baseline 运行并保存 **TTGIR/PTX artifact**。预期 baseline 对“新行为”检查先 FAIL，已有行为检查仍 PASS。

- [ ] **周四：实现一个行为可见的小改动**

要求：生成 IR 有稳定差异、语义不变、改动不超过一个 pass/utility 的职责范围。

  - **必读（目的：实现前确认 rewrite/fold 的合法性与失败语义）：** [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)；同时必须阅读固定 commit 的目标 op `.td`、interface verifier/trait 和相邻 transform 实现。
  - **本步五段链：** 修改或核对 **pass registration** → 限定一个 pass/utility 的 **implementation** → 依据 **op/interface definition** 保证 verifier/type/layout 契约 → 编译并运行目标 **test** → 对同一输入保存变更后的 **TTGIR/PTX artifact**。若 PTX 不应变化，也要用 hash/diff 给出证据并解释为何 TTGIR 可见而 PTX 等价。

- [ ] **周五：运行目标 compiler test**

使用 Triton 仓库实际 test harness；记录命令、PASS 数和耗时。

```bash
source $TRITON_VENV/bin/activate
export PYTHON=$TRITON_VENV/bin/python
test -e $TRITON_ROOT/test/TritonGPU/phase3-local-transform.mlir
make -C $TRITON_ROOT test-lit
```

  - **必读（目的：使用固定 commit 真正接入 CI 的 harness，而不是自创命令）：** 固定 commit 的 `README.md`、`Makefile`/CMake test 配置和目标测试的 `RUN:` 行；用 `rg -n 'lit|check-triton|pytest|Running tests' $TRITON_ROOT/README.md $TRITON_ROOT/Makefile $TRITON_ROOT/CMakeLists.txt $TRITON_ROOT/test` 核验。
  - **本步五段链：** harness 调用的 **pass registration** → 被测 **implementation** → 输入/输出 **op/interface definition** → 记录 in-tree **test** 路径、精准命令、PASS 数、耗时 → 从该次执行保存/比对 **TTGIR/PTX artifact**，避免测试与 artifact 来自不同 Python/build dir。

### 周末

- [ ] **周六：做 red/green 回退验证**

```text
修改存在：test PASS
临时撤销修改：新增检查 FAIL
恢复修改：test PASS
```

保存三次结果摘要。

  - **必读（目的：证明新增检查确实观测到改动而不是恒真）：** [FileCheck 官方文档](https://llvm.org/docs/CommandGuide/FileCheck.html) 的匹配语义，并复读新测试的 `RUN:`/check prefix。
  - **本步五段链：** 三次运行使用完全相同的 **pass registration** → 仅切换目标 **implementation** → 保持 **op/interface definition** 与输入不变 → 保存 red/green **test** 输出 → 对三次 **TTGIR/PTX artifact** 做 diff；回退仅临时用于验证，不覆盖用户其他工作。

- [ ] **周日：导出 patch 并复盘**

```bash
TRITON_TEST=test/TritonGPU/phase3-local-transform.mlir
git -C $TRITON_ROOT add -N $TRITON_TEST
git -C $TRITON_ROOT diff --binary -- \
  lib include third_party/nvidia/lib third_party/nvidia/include $TRITON_TEST \
  > $CAPSTONE/compiler/patches/week13-local-transform.patch
git -C $TRITON_ROOT diff --cached --quiet -- $TRITON_TEST
```

解释 pass anchor、匹配条件、生成 op、失败条件和测试边界。

  - **必读（目的：让 patch 复盘可由源码与产物双向验证）：** 固定 commit 的目标 pass registration、实现、op/interface `.td`、新增/相邻测试；官方辅助阅读 [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/) 和 [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)。
  - `git add -N` 只给新测试设置 **intent-to-add**，让普通 `git diff` 纳入未跟踪文件；它不提交内容。紧随其后的 `git diff --cached --quiet` 必须成功，证明 index 中没有 staged content。导出路径集合必须覆盖实际 compiler source/header 与 in-tree test；若改动落在列表外，先显式补入路径，不得导出残缺 patch。
  - **可应用性验证：** 在同一 baseline commit 的干净临时 clone/worktree 中执行 `git apply --check $CAPSTONE/compiler/patches/week13-local-transform.patch`；记录 exit code。验证环境不得含 patch 中的修改，也不要用会覆盖现有工作的破坏性清理命令。
  - **本步五段链：** 在复盘中逐项列出 **pass registration** 文件/symbol → **implementation** anchor/匹配/生成/失败条件 → **op/interface definition** 契约 → in-tree **test** 边界与 red/green 证据 → baseline/patch 后 **TTGIR/PTX artifact** 路径和 diff 摘要；`week13-local-transform.patch` 必须包含 compiler 改动和新测试。

### Phase 3 Exit Gate

```text
能够从源码构建和调试 Triton
能够生成 reproducer
完成一个真实 compiler transformation 改动
新增 test 经 red/green 验证
```

[进入 Phase 4：Baseline Freeze →](./Phase4_Baseline_Freeze.md)
