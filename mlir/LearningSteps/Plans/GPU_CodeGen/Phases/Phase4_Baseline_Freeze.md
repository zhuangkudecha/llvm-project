# Phase 4：Baseline Freeze

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 14–16（2026-10-14 至 2026-11-03）
- 计划工时：33h

## 前置条件

[Phase 3 Exit Gate](./Phase3_Triton_Compiler_Internals.md#phase-3-exit-gate) 已通过。

## 可验证阶段目标

- 冻结 fused Linear + Bias + ReLU 的标准 shape、dtype、数值参考与 correctness 判据。
- 建立可重复的 autotune、benchmark 与 profiler 流程。
- 固化 baseline commit、配置、IR/PTX/SASS、性能数据与环境元数据。

## 本阶段产出

- 标准 shape 集与 correctness harness。
- autotune 配置、benchmark harness 与统一结果格式。
- ncu/nsys profile、IR/PTX/SASS artifact 与 baseline report。
- 可供后续实验复用的冻结 baseline。

## 资料边界与冻结契约

本页外链已于 **2026-07-14** 在线核验。Correctness 与 performance 使用不同证据链：前者回答“结果是否满足冻结的数值语义”，后者回答“在固定 workload 和 GPU 状态下延迟与硬件指标是什么”。Profiler 结果不能替代 correctness，数值 PASS 也不能证明性能测量有效。

### Correctness 必读资料

- [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)：理解浮点运算顺序、设备和版本均可能造成非 bitwise-identical 结果，因而必须冻结容差而非要求逐 bit 相等。
- [PyTorch CUDA semantics / TF32](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) 与 [CUDA environment variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)：确认 Ampere+ 的 FP32 matmul 可能使用 TF32、`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1` 会覆盖 precision 设置，以及 full FP32 reference 所需的控制；旧 `allow_tf32` API 正在弃用，因此必须同时记录实际 PyTorch 版本和读回值。
- [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)：以 `|actual - expected| <= atol + rtol * |expected|` 定义 PASS，并显式处理 dtype 与 NaN。
- [Triton `triton.testing`](https://triton-lang.org/main/python-api/triton.testing.html)：只把其中的 `assert_close` 用作辅助交叉检查；正式 JSON 仍记录 PyTorch 判据和误差统计。
- [Compute Sanitizer / Memcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool)：检查 global/local/shared memory 越界与 misaligned access；它是内存安全检查，不是性能工具。

冻结的 FP16 correctness contract 为：FP32 输入提升、FP32 GEMM 与 bias、FP32 ReLU，最后 cast FP16；调用 `torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2, equal_nan=False)`。同时单独要求输出全为 finite，并记录 `max_abs/max_rel`。若实测表明容差需改变，只能在首次完整 baseline 前基于误差分布写出理由并重新冻结，后续实验不得为使失败通过而放宽容差。

### Performance 必读资料

- [Triton `triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)：`warmup`、`rep` 的单位是毫秒；`return_mode="all"` 返回逐次运行的毫秒列表，可用于自行计算并审计 P20/P50/P80；不要把 100/500 误解成迭代次数。
- [Triton `Benchmark`/`perf_report`](https://triton-lang.org/main/python-api/triton.testing.html)：用于组织 shape/provider 维度和 machine-readable benchmark；不改变本阶段的三轮协议。
- [CUDA Events timing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams)：用同一 stream 中的 start/stop event 和同步后的 elapsed time 测 GPU 工作，不能用未同步的 host wall clock 代替。
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：正式测量前后以可脚本化 query 记录 UUID、P-state、温度、功耗、graphics/memory clock 与活跃 compute process。
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli)：用 CUDA/NVTX timeline 验证 capture range、kernel launch 数与额外 epilogue kernel。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 与 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：先查询本机可用 section/metric，再采集 SpeedOfLight、Occupancy、memory、scheduler 等指标；报告 metric 全名，不能只写 UI 标签。
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)：`cuobjdump` 可从 host binary/fatbin/cubin 提取 PTX/SASS，`nvdisasm` 接收 cubin 并提供更丰富的反汇编/控制流输出。

所有正式 steady-state latency 使用同一协议：预热 **100 ms**、测量窗口 **500 ms**、独立运行 **3 轮**，每轮保存 P20/P50/P80，以 P50 为主统计量。每轮前后记录 GPU 状态；出现 thermal/power throttling、clock 明显漂移或其他 compute process 时废弃该轮但保留原记录和原因。compile/autotune 时间单列，不混入 CUDA Events 所包围的 steady-state kernel 区间。

```bash
nvidia-smi --query-gpu=timestamp,uuid,pstate,temperature.gpu,power.draw,clocks.gr,clocks.mem \
  --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

## Week 14：标准 Shape 与 Correctness Harness

**Files:**

- Create: `benchmark/shapes.py`
- Create: `benchmark/correctness.py`
- Create: `scripts/run_correctness.sh`
- Create: `results/baseline/correctness.json`

### 工作日

- [ ] **周一：编码六组冻结 shape**

`shapes.py` 只包含：

```python
SHAPES = [
    (32, 4096, 4096),
    (128, 4096, 4096),
    (512, 4096, 4096),
    (512, 16384, 4096),
    (512, 4096, 16384),
    (4096, 4096, 4096),
]
```

  - **必读（目的：理解浮点结果为何不能按逐 bit 等价定义）：** [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。本步把 shape 顺序也视为冻结接口，JSON 与 CSV 必须使用同一顺序。

- [ ] **周二：实现 FP32 reference**

固定 `torch.matmul(a.float(), b.float()) + bias.float()`，ReLU 后 cast FP16。

```python
import os

assert os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") != "1"
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.get_float32_matmul_precision() == "highest"

a_fp32 = a.to(torch.float32)
b_fp32 = b.to(torch.float32)
bias_fp32 = bias.to(torch.float32)
reference = torch.relu(
    torch.matmul(a_fp32, b_fp32) + bias_fp32
).to(torch.float16)
```

  - **必读（目的：将 full FP32 reference 和“接近”都变成显式契约）：** [PyTorch CUDA semantics / TF32](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) 与 [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。调用时显式传入冻结的 `rtol=1e-2, atol=1e-2, equal_nan=False`，不得依赖版本可能变化的默认值；设置、读回值和 PyTorch 版本都写入 correctness JSON。

- [ ] **周三：实现十个固定 seed**

每个 shape 至少覆盖两个 seed；完整十 seed 可在分批运行中完成，结果必须记录 seed。十个 seed 的具体整数列表只定义一次并写入 `correctness.py`，运行顺序变化不能改变输入生成算法。

  - **必读（目的：区分随机输入可复现与浮点结果 bitwise 可复现）：** [PyTorch Reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)。记录 PyTorch/CUDA/Triton 版本和 seed，但不把设 seed 误当成跨版本 bitwise 保证。

- [ ] **周四：实现边界输入**

加入全负 bias、零附近输入、非整除 shape `127x251x509` 和 NaN/Inf 检查。

  - **必读（目的：核对 NaN/Inf 与 dtype 检查的默认语义）：** [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。除 `equal_nan=False` 外显式执行 `torch.isfinite(actual).all()`；非整除 shape 必须实际走 mask/boundary tile，不能 pad 后绕过。

- [ ] **周五：输出 machine-readable JSON**

至少包含：shape、seed、max_abs、max_rel、pass/fail、dtype、commit；并新增 `rtol`、`atol`、case 名、reference expression、PyTorch/Triton/CUDA 版本、`allow_tf32`、`float32_matmul_precision`，使容差、FP32 reference 设置和运行环境可审计。

  - **必读（目的：确保 JSON 中的 pass/fail 能回算）：** [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。保存最大误差的 index 与对应 actual/reference；禁止只保存布尔值。

### 周末

- [ ] **周六：RTX 4090 完整 correctness**

```bash
bash $CAPSTONE/scripts/run_correctness.sh
```

Expected：全部 case PASS；任何失败先修 correctness，不进入 Week 15。

  - **必读（目的：解释硬件/版本间允许容差但不允许静默 NaN）：** [PyTorch Numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) 与 [`assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)。
  - **核验：** 六组冻结 shape × 十个 seed 的结果均可在 JSON 中查询，边界 case 单独列出；脚本遇到任一 FAIL 返回非零。

- [ ] **周日：运行 compute-sanitizer/边界复盘**

对可独立 launch 的最小 kernel 执行 memcheck，记录 mask 和 boundary tile 行为。

```bash
compute-sanitizer --tool memcheck \
  python $CAPSTONE/benchmark/correctness.py --shape 127 251 509 --seed 0
```

  - **必读（目的：理解 memcheck 能精确报告的越界/misaligned 类别及 source attribution 条件）：** [Compute Sanitizer / Memcheck](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool)。若 Python launcher 无法被工具稳定隔离，增加单 shape/seed 子命令，而不是跳过检查。
  - **Expected：** `ERROR SUMMARY: 0 errors`，并保存命令、tool/CUDA 版本、退出码和日志；sanitizer 数据不进入性能 CSV。

### Week 14 Exit Gate

```text
六组 shape、十组 seed 和边界 case 全部正确
结果可由 JSON 审计
无 NaN/Inf 和越界
```

---

## Week 15：Autotune 与 Benchmark Harness

**Files:**

- Create: `benchmark/baseline_configs.py`
- Create: `benchmark/benchmark.py`
- Create: `scripts/run_benchmark.sh`
- Create: `results/baseline/raw.csv`
- Populate: `results/baseline/raw-samples/`
- Create: `results/baseline/compile-input-roots.txt`
- Create: `results/baseline/compiler-tree-excludes.txt`
- Create: `results/baseline/compiler-tree-files.sha256`
- Create: `results/baseline/configs.sha256`
- Create: `results/baseline/configs/{M32_N4096_K4096,M128_N4096_K4096,M512_N4096_K4096,M512_N16384_K4096,M512_N4096_K16384,M4096_N4096_K4096}.json`

### 工作日

- [ ] **周一：冻结 autotune search space**

写明 BLOCK_M/N/K、GROUP_M、num_warps、num_stages 候选和过滤条件。每组 shape 的选择只在 autotune 阶段发生；正式三轮 latency 复用已选 config。

完成候选测量后把六组 shape 的 selected config 各冻结成一个 JSON。每个文件至少包含 `shape`、`dtype`、`BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`GROUP_M`、`num_warps`、`num_stages`、`kernel_name`、`baseline_commit`、`triton_checkout_commit` 和 `compiler_tree_hash`。`baseline_commit` 来自 `compiler/baseline_commit.txt`，必须等于 `$TRITON_ROOT` 的 HEAD；`compiler_tree_hash` 不能只 hash 一个 patch 文件，而要覆盖固定 commit 下所有可能影响 Python JIT/autotune、Triton compiler/dialect/pass、NVIDIA backend 与实际回归测试的输入：整个 `python/triton`、`lib`、`include`、`third_party/nvidia` 和实际 suite `test/TritonGPU`。每个 root 先 `test -d`，再对其中 tracked、modified、untracked 的普通文件按相对路径排序生成逐文件 SHA-256 manifest。

排除项只能是非源码状态：`.git`、build/cmake-build、cache/`__pycache__`、venv，以及 `.pyc/.pyo/.o/.a/.so` 和本计划生成的 TTIR/TTGIR/LLVM/PTX/cubin/SASS artifact。root 清单和排除清单分别冻结到文本文件，并与逐文件 manifest 一起进入最终 `compiler_tree_hash`；后续若增删 root 或排除规则，identity 必然变化，不能静默扩大排除。

`benchmark.py --validate-source-identity` 必须在内存或独立临时文件中重算同一算法、比较六个 JSON 字段，并验证 editable import 的 resolved `triton.__file__` 位于 resolved `$TRITON_ROOT` 下；该验证为只读，不得重写冻结的 tree/config manifest。仅 `import triton` 成功不算通过。

```bash
CONFIG_DIR=$CAPSTONE/results/baseline/configs
mkdir -p "$CONFIG_DIR"
ROOTS_FILE=$CAPSTONE/results/baseline/compile-input-roots.txt
EXCLUDES_FILE=$CAPSTONE/results/baseline/compiler-tree-excludes.txt
BASELINE_COMMIT=$(tr -d "[:space:]" < $CAPSTONE/compiler/baseline_commit.txt)
TRITON_CHECKOUT_COMMIT=$(git -C "$TRITON_ROOT" rev-parse HEAD)
test "$BASELINE_COMMIT" = "$TRITON_CHECKOUT_COMMIT"

printf "%s\n" python/triton lib include third_party/nvidia test/TritonGPU \
  > "$ROOTS_FILE"
printf "%s\n" \
  "*/.git/*" "*/build/*" "*/cmake-build-*/*" "*/.cache/*" \
  "*/__pycache__/*" "*/.venv/*" "*/venv/*" \
  "*.pyc" "*.pyo" "*.o" "*.a" "*.so" \
  "*.ttir" "*.ttgir" "*.ll" "*.ptx" "*.cubin" "*.sass" \
  > "$EXCLUDES_FILE"
while IFS= read -r ROOT; do
  test -d "$TRITON_ROOT/$ROOT"
done < "$ROOTS_FILE"

(cd "$TRITON_ROOT" && \
  while IFS= read -r ROOT; do
    find "$ROOT" -type f \
      ! -path "*/.git/*" ! -path "*/build/*" ! -path "*/cmake-build-*/*" \
      ! -path "*/.cache/*" ! -path "*/__pycache__/*" \
      ! -path "*/.venv/*" ! -path "*/venv/*" \
      ! -name "*.pyc" ! -name "*.pyo" ! -name "*.o" ! -name "*.a" ! -name "*.so" \
      ! -name "*.ttir" ! -name "*.ttgir" ! -name "*.ll" ! -name "*.ptx" \
      ! -name "*.cubin" ! -name "*.sass" -print0
  done < "$ROOTS_FILE" | sort -zu | xargs -0 sha256sum) \
  > $CAPSTONE/results/baseline/compiler-tree-files.sha256
COMPILER_TREE_HASH=$(
  cd $CAPSTONE/results/baseline
  sha256sum compile-input-roots.txt compiler-tree-excludes.txt \
    compiler-tree-files.sha256 | sha256sum | cut -d " " -f1
)

TRITON_IMPORT=$($TRITON_VENV/bin/python -c "import triton; print(triton.__file__)")
case "$(realpath "$TRITON_IMPORT")" in
  "$(realpath "$TRITON_ROOT")"/*) ;;
  *) echo "triton import is not from editable checkout: $TRITON_IMPORT" >&2; exit 1 ;;
esac

python $CAPSTONE/benchmark/benchmark.py \
  --provider baseline --freeze-selected-configs --config-dir "$CONFIG_DIR" \
  --baseline-commit "$BASELINE_COMMIT" \
  --triton-checkout-commit "$TRITON_CHECKOUT_COMMIT" \
  --compiler-tree-hash "$COMPILER_TREE_HASH"

for SHAPE_TAG in \
  M32_N4096_K4096 M128_N4096_K4096 M512_N4096_K4096 \
  M512_N16384_K4096 M512_N4096_K16384 M4096_N4096_K4096; do
  test -s "$CONFIG_DIR/$SHAPE_TAG.json"
done

python $CAPSTONE/benchmark/benchmark.py \
  --validate-configs "$CONFIG_DIR" --expect-count 6 \
  --require-fields shape,dtype,BLOCK_M,BLOCK_N,BLOCK_K,GROUP_M,num_warps,num_stages,kernel_name,baseline_commit,triton_checkout_commit,compiler_tree_hash
python $CAPSTONE/benchmark/benchmark.py \
  --validate-source-identity --config-dir "$CONFIG_DIR" \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots "$ROOTS_FILE" --compiler-tree-excludes "$EXCLUDES_FILE" \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
(cd $CAPSTONE/results/baseline && \
  sha256sum configs/*.json > configs.sha256 && \
  sha256sum -c configs.sha256)
test "$(wc -l < $CAPSTONE/results/baseline/configs.sha256)" -eq 6
```

  - **必读（目的：理解 Triton autotune 的 config、key、reset 与 benchmark hook 契约）：** [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)。
  - **性能依据：** 候选比较使用本页冻结的 100 ms warm-up、500 ms measurement 和 P50；每个候选前后记录 `nvidia-smi` 状态，依据 [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 与 [NVSMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

- [ ] **周二：实现 timing protocol**

固定 warmup 100、measurement 500、三轮完整测量，输出 median/P20/P80。固定 Triton commit 后先检查 `do_bench` 签名是否含 `return_mode`，并运行 `return_mode="all"` smoke；100/500 的单位均为毫秒。正式协议保存每次 raw sample，再自行计算分位数，禁止只保存聚合值。

`do_bench` 返回毫秒，而冻结 CSV 的三列以 `_us` 结尾；写 CSV 前必须统一转换，禁止只改列名：

```python
import inspect
import numpy as np

assert "return_mode" in inspect.signature(triton.testing.do_bench).parameters
raw_ms = triton.testing.do_bench(
    fn, warmup=100, rep=500, return_mode="all"
)
assert isinstance(raw_ms, list) and len(raw_ms) > 0
raw_us = [sample_ms * 1000.0 for sample_ms in raw_ms]
p20_ms, median_ms, p80_ms = np.quantile(
    raw_ms, [0.2, 0.5, 0.8], method="linear"
)
p20_us, median_us, p80_us = (
    p20_ms * 1000.0,
    median_ms * 1000.0,
    p80_ms * 1000.0,
)
assert 0.0 < p20_us <= median_us <= p80_us
assert abs(median_us / 1000.0 - median_ms) < 1e-12
```

每个 `SHAPE_TAG + round` 保存一个 `results/baseline/raw-samples/<tag>-round<R>.json`，同时含 `raw_ms`、`raw_us`、sample count、timing backend、warm-up/measurement 毫秒数和 linear quantile method；summary CSV 只保存该文件路径及复算结果。

若固定 commit 不支持 `return_mode="all"`，记录该 API 缺口并改用同一 stream 的 CUDA Events 循环：先运行至累计 warm-up event time ≥100 ms，再逐次保存 start/stop elapsed time，直到 measurement 累计 ≥500 ms；raw ms/us 与分位数 schema 保持不变。不得退回只返回 quantile 的路径。

  - **必读（目的：核对参数单位、quantile 顺序和返回单位）：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)。
  - **性能依据：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 明确 `return_mode="all"` 返回每次运行的毫秒列表；timing 必须由同一 CUDA stream 的 GPU event 覆盖，并在 stop event 完成后读取，依据 [CUDA Events timing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams)。每轮必须保存 raw ms、换算后的 raw us、sample count 与从 raw 值复算的 P20/P50/P80。

- [ ] **周三：分离 compile/autotune 与 steady state**

第一次运行记录 compile/autotune 时间；正式 latency 使用已编译 cache，二者写入不同列。

  - **必读（目的：确认 benchmark API 只包围被测 callable，而非 Python 初始化和编译阶段）：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 与 [CUDA Events timing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams)。
  - **性能依据：** compile/autotune 单独用 host timer 记 `compile_ms`；steady state 仍执行 100 ms warm-up、500 ms GPU-event measurement、三轮 P20/P50/P80，并保存每轮前后 GPU clocks/temperature/power/P-state。

- [ ] **周四：计算 effective TFLOPS**

主 GEMM 使用：

```text
2 * M * N * K / latency_seconds / 1e12
```

同时明确该数值未计 bias/ReLU FLOPs，仅用于和相同语义 baseline 比较。

  - **必读（目的：确保 latency 单位从 `do_bench` 的毫秒正确换算为秒）：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)。
  - **性能依据：** TFLOPS 使用每轮 P50 计算，再报告三轮 P50 的分布；不得拿 P20 latency 计算“最好”TFLOPS 作为主结果。每行关联该轮 GPU 状态记录。

- [ ] **周五：固定 CSV schema**

基础 18 列、字段定义、单位和顺序以[总索引：全局 benchmark CSV schema](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md#global-benchmark-csv-schema)为唯一权威定义。Phase 4 不复制或重定义基础 schema，只允许在基础列之后依次追加：

```text
sample_count,raw_samples_path,timing_backend
```

这三个扩展字段分别记录本轮原始样本数、可从结果根目录解析的 raw-sample 文件路径，以及实际 timing backend。基础 18 列不得删除、重命名或改变顺序，扩展列也不得替代基础 latency/compile 字段。

  - **必读（目的：让字段与 benchmark API 的 statistic 对齐）：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 与 [`Benchmark`/`perf_report`](https://triton-lang.org/main/python-api/triton.testing.html)。
  - **单位契约与测试：** `raw_samples_us` 必须由每个 raw ms 乘 `1000.0` 得到，`median_us/p20_us/p80_us` 再从 raw samples 复算；在 `benchmark.py --self-test-units` 中用 `[0.8, 1.0, 1.2] ms → [800.0, 1000.0, 1200.0] us` 检查逐样本转换、sample count 和 linear P20/P50/P80，并执行 `python $CAPSTONE/benchmark/benchmark.py --self-test-units`。若单位/统计断言失败，脚本必须非零退出且不得写 CSV。
  - **性能依据：** 保留总索引的基础 schema 和上述三列扩展，另用同 basename 的 `*-gpu-state.csv` 关联 round 的测量前后 NVSMI 数据；依据 [NVSMI query/CSV](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。正式行只接受 warm-up=100 ms、rep=500 ms、P20/P50/P80 完整、毫秒到微秒测试通过且 GPU 状态无异常的轮次。

### 周末

- [ ] **周六：运行三轮完整 baseline**

```bash
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
python $CAPSTONE/benchmark/benchmark.py --validate-source-identity \
  --config-dir $CAPSTONE/results/baseline/configs \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
bash $CAPSTONE/scripts/run_benchmark.sh --provider baseline --rounds 3
```

  - **必读（目的：用官方 timing contract 审计三轮是否同口径）：** [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)、[CUDA Events timing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams) 和 [NVSMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
  - **Expected：** 六组 shape × 三轮均有 100 ms warm-up、500 ms measurement、完整 raw ms/us samples、sample count、从 raw 复算的 P20/P50/P80、冻结 config、compile_ms 和前后 GPU 状态；正式运行期间无其他 compute process。

- [ ] **周日：计算噪声和异常值**

报告每个 shape 三轮 median 的最大相对偏差；如果温度、时钟或共享实例负载导致异常，重新测量并保留被废弃记录及原因。

  - **必读（目的：用一手 GPU telemetry 判定而非凭目测删除异常值）：** [NVSMI query、loop 与 CSV 格式](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。
  - **性能依据：** 主统计量固定为三轮 P50；同时保留每轮 P20/P80。只允许基于 thermal/power/clock/P-state/其他 process 的记录废弃，不能因为结果“不够快”删除；重测仍使用 100/500 ms 协议。

### Week 15 Exit Gate

```text
autotune search space 已冻结
三轮 baseline 数据完整
compile time 与 steady-state latency 分离
测量噪声有量化结果
```

---

## Week 16：Profiler、IR/SASS Artifact 与 Baseline Freeze

**Files:**

- Create: `scripts/run_ncu.sh`
- Update: `scripts/dump_triton_ir.sh`
- Populate: `profiles/ncu/baseline/`
- Populate: `profiles/nsys/baseline/`
- Populate: `ir/*/baseline/`
- Create: `results/baseline/profiler-summary.csv`
- Create: `results/baseline/profiler-map.csv`
- Create: `docs/baseline_report.md`
- Create: `results/baseline/MANIFEST.md`
- Create: `results/baseline/MANIFEST.sha256`
- Populate: `results/baseline/cold-replay/`
- Update: `compiler/baseline_commit.txt`
- Update: `README.md`

### 工作日

- [ ] **周一：为每组 shape 找到最终 autotune config**

保存 BLOCK、warps、stages 和 kernel name，后续 ncu 不重新触发全量 autotune。

在 profiler 启动前重新验证 Week 15 的六个 config JSON，且对每个文件计算 SHA-256；任一文件缺失、为空、数量不是六、schema 不完整或 commit/hash 与当前 baseline 不一致时立即停止 Week 16：

```bash
CONFIG_DIR=$CAPSTONE/results/baseline/configs
python $CAPSTONE/benchmark/benchmark.py \
  --validate-configs "$CONFIG_DIR" --expect-count 6 \
  --require-fields shape,dtype,BLOCK_M,BLOCK_N,BLOCK_K,GROUP_M,num_warps,num_stages,kernel_name,baseline_commit,triton_checkout_commit,compiler_tree_hash
python $CAPSTONE/benchmark/benchmark.py \
  --validate-source-identity --config-dir "$CONFIG_DIR" \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
for CONFIG_JSON in "$CONFIG_DIR"/*.json; do
  test -s "$CONFIG_JSON"
done
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
test "$(wc -l < $CAPSTONE/results/baseline/configs.sha256)" -eq 6
```

  - **必读（目的：确认 config/key 如何决定一次 autotune 选择）：** [`triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)。
  - **性能依据：** config 来自 Week 15 的 100 ms warm-up、500 ms measurement、三轮 P50，并关联 GPU 状态；固定后 profiler 只 launch 该 config，防止 replay 混入其他候选。

- [ ] **周二：采集 nsys timeline**

确认 fused workload 只有目标 kernel launch，没有额外 bias/ReLU kernel。

```bash
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
python $CAPSTONE/benchmark/benchmark.py --validate-source-identity \
  --config-dir $CAPSTONE/results/baseline/configs \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256

for CONFIG_JSON in $CAPSTONE/results/baseline/configs/*.json; do
  SHAPE_TAG=$(basename "$CONFIG_JSON" .json)
  CONFIG_HASH=$(sha256sum "$CONFIG_JSON" | cut -c1-12)
  KERNEL_NAME=$($TRITON_VENV/bin/python -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['kernel_name'])" "$CONFIG_JSON")
  OUTPUT_TAG=${SHAPE_TAG}_${CONFIG_HASH}_${KERNEL_NAME}
  nsys profile --trace=cuda,nvtx --sample=none \
    --capture-range=cudaProfilerApi --force-overwrite=true \
    --output=$CAPSTONE/profiles/nsys/baseline/$OUTPUT_TAG \
    $TRITON_VENV/bin/python $CAPSTONE/benchmark/benchmark.py \
      --provider baseline --shape-tag "$SHAPE_TAG" \
      --config-json "$CONFIG_JSON" --disable-autotune \
      --warmup-ms 100 --profile-iterations 1 --capture-profiler-api \
      --profile-metadata $CAPSTONE/profiles/nsys/baseline/$OUTPUT_TAG.metadata.json
done
```

  - **必读（目的：限定 CUDA/NVTX capture 并从 timeline 核对 launch 边界）：** [Nsight Systems focused profiling 与 CLI](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#profiling-from-the-cli)。
  - **性能依据：** `--disable-autotune` 要求 harness 直接实例化 JSON 中的 selected config；`--profile-iterations 1 --capture-profiler-api` 的契约是先完成 100 ms warm-up，再用 `cudaProfilerStart/Stop` 只包围一次目标 launch。六组 shape 全部遍历，metadata 必须记录 shape/config hash/kernel/launch count/stream；timeline 用于边界公平性，不以 profiler duration 替代 Week 15 P50。

- [ ] **周三：采集 ncu SpeedOfLight/Occupancy**

先使用较小 section set，记录 duration、occupancy、register、shared memory。

`scripts/run_ncu.sh` 必须显式接受 `--shape-tag`、`--config-json`、`--disable-autotune`、`--warmup-ms`、`--profile-iterations`、`--sections`/`--set`、`--kernel-name` 和 `--output`：先验证 config/source hash，再把 selected block/warps/stages 直接传给 harness，禁止调用 autotuner；warm-up 后只 launch 指定次数，并用 `--kernel-name` 过滤 report。脚本同时写 `<output>.metadata.json`，至少含 shape tag、config hash、完整 config、kernel、report hash、ncu/CUDA 版本和 replay count。

```bash
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
ncu --list-sections
python $CAPSTONE/benchmark/benchmark.py --validate-source-identity \
  --config-dir $CAPSTONE/results/baseline/configs \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
for CONFIG_JSON in $CAPSTONE/results/baseline/configs/*.json; do
  SHAPE_TAG=$(basename "$CONFIG_JSON" .json)
  CONFIG_HASH=$(sha256sum "$CONFIG_JSON" | cut -c1-12)
  KERNEL_NAME=$($TRITON_VENV/bin/python -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['kernel_name'])" "$CONFIG_JSON")
  bash $CAPSTONE/scripts/run_ncu.sh \
    --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" \
    --disable-autotune --warmup-ms 100 --profile-iterations 1 \
    --sections SpeedOfLight,Occupancy --kernel-name "$KERNEL_NAME" \
    --output $CAPSTONE/profiles/ncu/baseline/${SHAPE_TAG}_${CONFIG_HASH}_${KERNEL_NAME}
done
```

  - **必读（目的：理解 section 集、kernel replay 与 report export）：** [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)；[Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) 用于解释吞吐率、occupancy 与 metric collection/replay。
  - **性能依据：** 100 ms warm-up 后只 profile 冻结 config；先运行 `ncu --list-sections`/`--query-metrics` 核对本机名称。CSV 保存 duration、achieved occupancy、register/thread、static/dynamic shared memory 的**实际 metric 全名与单位**，不得把推测值写成测量值。

- [ ] **周四：对重点 shape 采集 full metrics**

重点选择小 M、大 K 和大型方阵三组，避免对六组全部使用高开销 full replay。

```bash
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
python $CAPSTONE/benchmark/benchmark.py --validate-source-identity \
  --config-dir $CAPSTONE/results/baseline/configs \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
for SHAPE_TAG in M32_N4096_K4096 M512_N4096_K16384 M4096_N4096_K4096; do
  CONFIG_JSON=$CAPSTONE/results/baseline/configs/$SHAPE_TAG.json
  test -s "$CONFIG_JSON"
  CONFIG_HASH=$(sha256sum "$CONFIG_JSON" | cut -c1-12)
  KERNEL_NAME=$($TRITON_VENV/bin/python -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['kernel_name'])" "$CONFIG_JSON")
  bash $CAPSTONE/scripts/run_ncu.sh \
    --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" \
    --disable-autotune --warmup-ms 100 --profile-iterations 1 \
    --set full --kernel-name "$KERNEL_NAME" \
    --output $CAPSTONE/profiles/ncu/baseline/${SHAPE_TAG}_${CONFIG_HASH}_${KERNEL_NAME}_full
done

python $CAPSTONE/benchmark/benchmark.py --validate-profiler-map \
  --config-dir $CAPSTONE/results/baseline/configs \
  --nsys-dir $CAPSTONE/profiles/nsys/baseline \
  --ncu-dir $CAPSTONE/profiles/ncu/baseline \
  --output $CAPSTONE/results/baseline/profiler-map.csv \
  --require-nsys-shapes all --require-ncu-basic-shapes all \
  --require-ncu-full-shapes M32_N4096_K4096,M512_N4096_K16384,M4096_N4096_K4096
```

  - **必读（目的：理解 metric collection、replay overhead 及 SpeedOfLight/Memory/Scheduler 指标语义）：** [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
  - **性能依据：** 三个重点 shape 分别代表小 M、大 K 和大型方阵；每个在 100 ms warm-up 后 profile JSON 强制选择的单一 config，记录 config hash、kernel、ncu/CUDA 版本、replay 次数和 metric 全名。`--validate-profiler-map` 要求每个 report metadata 与 `SHAPE_TAG + CONFIG_HASH + kernel name` 一一对应，无额外/缺失，并生成供 `MANIFEST.md` 原样纳入的 `profiler-map.csv`；full-set duration 不混入三轮 latency。

- [ ] **周五：导出 PTX/SASS 并写 baseline report**

保存与 profiler 中 kernel 对应的 PTX/SASS，并记录 hash；核对 workload、environment 和 autotune contract，开始填写 baseline report。

先扩展 Phase 3 的 `scripts/dump_triton_ir.sh`：接受 `--shape`、`--config-json`、`--config-dir`、`--all-configs`、`--output-root`、`--artifact-tag`、`--emit-all`、`--manifest` 和 `--sha256-manifest`，内部继续使用 Phase 3 已核验的 `TRITON_ALWAYS_COMPILE=1`、`MLIR_ENABLE_DUMP=1`、`TRITON_KERNEL_DUMP=1`、`TRITON_DUMP_DIR` 机制。`--emit-all` 必须把**同一次**冻结 shape/config 编译产生的 TTIR、TTGIR、LLVM IR、PTX、cubin/SASS 分别保存到 `ir/{ttir,ttgir,llvm,ptx,sass}/baseline/`；并对每个 cubin 在脚本内部同时执行 `cuobjdump --dump-sass` 与 `nvdisasm`，以同一 `$ARTIFACT_TAG.{cuobjdump,nvdisasm}.sass` basename contract 落盘。baseline 和 cold replay 必须调用同一个 `--emit-all` 实现；找不到任一级或任一反汇编产物时非零退出，不能拿另一次编译的文件补齐。`--all-configs --config-dir DIR` 必须先验证 `configs.sha256`，再按六个 JSON 内的 shape 与 config 逐一执行同一流程。

下面以 `512x4096x4096` 展开完整命令；对六组冻结 shape 各执行一次，只替换 `SHAPE_TAG`、三个 shape 参数及其冻结 `CONFIG_JSON`。全部六组完成后再生成一次总 hash 文件。

```bash
SHAPE_TAG=M512_N4096_K4096
CONFIG_JSON=$CAPSTONE/results/baseline/configs/$SHAPE_TAG.json
(cd $CAPSTONE/results/baseline && sha256sum -c configs.sha256)
python $CAPSTONE/benchmark/benchmark.py --validate-source-identity \
  --config-dir $CAPSTONE/results/baseline/configs \
  --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
  --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
  --compile-input-roots $CAPSTONE/results/baseline/compile-input-roots.txt \
  --compiler-tree-excludes $CAPSTONE/results/baseline/compiler-tree-excludes.txt \
  --compiler-tree-manifest $CAPSTONE/results/baseline/compiler-tree-files.sha256
test -s "$CONFIG_JSON"
CONFIG_HASH=$(sha256sum "$CONFIG_JSON" | cut -c1-12)
ARTIFACT_TAG=${SHAPE_TAG}_${CONFIG_HASH}

TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=$CAPSTONE/results/baseline/dump/$ARTIFACT_TAG \
bash $CAPSTONE/scripts/dump_triton_ir.sh \
  --kernel fused_linear --shape 512 4096 4096 \
  --config-json "$CONFIG_JSON" --artifact-tag "$ARTIFACT_TAG" \
  --output-root $CAPSTONE/ir --emit-all \
  --manifest $CAPSTONE/results/baseline/MANIFEST.md \
  --sha256-manifest $CAPSTONE/results/baseline/MANIFEST.sha256

test -s $CAPSTONE/ir/ttir/baseline/$ARTIFACT_TAG.ttir
test -s $CAPSTONE/ir/ttgir/baseline/$ARTIFACT_TAG.ttgir
test -s $CAPSTONE/ir/llvm/baseline/$ARTIFACT_TAG.ll
test -s $CAPSTONE/ir/ptx/baseline/$ARTIFACT_TAG.ptx
test -s $CAPSTONE/ir/sass/baseline/$ARTIFACT_TAG.cubin

cuobjdump --dump-sass \
  $CAPSTONE/ir/sass/baseline/$ARTIFACT_TAG.cubin \
  > $CAPSTONE/ir/sass/baseline/$ARTIFACT_TAG.cuobjdump.sass
nvdisasm $CAPSTONE/ir/sass/baseline/$ARTIFACT_TAG.cubin \
  > $CAPSTONE/ir/sass/baseline/$ARTIFACT_TAG.nvdisasm.sass

# 在所有反汇编输出落盘后一次性冻结五级 artifact 的 hash；不得提前写 hash。
(cd $CAPSTONE && \
  find ir/ttir/baseline ir/ttgir/baseline ir/llvm/baseline \
       ir/ptx/baseline ir/sass/baseline \
       -type f -print0 | \
  sort -z | xargs -0 sha256sum > results/baseline/MANIFEST.sha256)
(cd $CAPSTONE && sha256sum -c results/baseline/MANIFEST.sha256)
```

  - **必读（目的：根据输入 artifact 选择 `cuobjdump` 或 `nvdisasm` 并保存可复核反汇编）：** [CUDA Binary Utilities：cuobjdump 与 nvdisasm](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#differences-between-cuobjdump-and-nvdisasm)。
  - **artifact 映射：** `MANIFEST.md` 每行记录 shape、完整 config、config JSON hash、kernel name、`ARTIFACT_TAG`、五级 IR/binary 路径、ncu/nsys report 路径和 SHA-256；`MANIFEST.sha256` 使用相对 `$CAPSTONE` 的路径。五级产物与 profiler 必须共享 `SHAPE_TAG + CONFIG_HASH + kernel name`，从而证明来自同一 shape/config，而非只按相似文件名关联。
  - **性能依据：** 用 profiler report 中的 kernel name、register/shared-memory metric 与 binary symbol 对齐；报告仍引用 Week 15 的 100/500 ms、三轮 P50/P20/P80 和 GPU 状态，不能从 SASS 目测推断性能收益。

### 周末

- [ ] **周六：建立瓶颈矩阵并做冷启动复现**

每个 shape 填写：

```text
CTA count
waves per SM
occupancy
register/thread
shared memory/CTA
DRAM throughput
L2 hit rate
Tensor Core utilization
top stall reason
```

在新 shell 中仅依据 README 重跑六组 correctness 和一轮 benchmark，与 Week 15 三轮数据比较。

用显式最小环境启动真正的 cold shell；`CAPSTONE`、`TRITON_ROOT`、`TRITON_VENV` 在外层 shell 必须已是绝对路径。`run_correctness.sh` 必须实现 `--output-dir/--output-json`，`run_benchmark.sh` 必须实现 `--output-dir/--output-csv/--raw-samples-dir/--gpu-state-output`；给出这些参数后不得写 `results/baseline/` 的冻结结果。六组 correctness 和一轮 benchmark 全跑；ncu 只重跑代表 shape `512x4096x4096` 的 SpeedOfLight/Occupancy，因为 cold-shell gate 验证的是 profiler 命令、artifact 映射和环境可重建性，不重复 Week 16 已完成的三组高开销 full replay。

```bash
test "${CAPSTONE#/}" != "$CAPSTONE"
test "${TRITON_ROOT#/}" != "$TRITON_ROOT"
test "${TRITON_VENV#/}" != "$TRITON_VENV"

env -i \
  HOME="$HOME" \
  PATH="$TRITON_VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
  LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
  CUDA_VISIBLE_DEVICES=0 \
  CAPSTONE="$CAPSTONE" \
  TRITON_ROOT="$TRITON_ROOT" \
  TRITON_VENV="$TRITON_VENV" \
  bash --noprofile --norc -c '
    set -euo pipefail
    source "$TRITON_VENV/bin/activate"

    BASELINE="$CAPSTONE/results/baseline"
    RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
    COLD_ROOT="$BASELINE/cold-replay/$RUN_ID"
    test ! -e "$COLD_ROOT"
    mkdir -p "$COLD_ROOT/results/raw-samples" "$COLD_ROOT/profiles/ncu"

    (cd "$BASELINE" && sha256sum -c configs.sha256)
    (cd "$CAPSTONE" && sha256sum -c results/baseline/MANIFEST.sha256)

    TRITON_IMPORT=$($TRITON_VENV/bin/python -c \
      "import triton; print(triton.__file__)")
    case "$(realpath "$TRITON_IMPORT")" in
      "$(realpath "$TRITON_ROOT")"/*) ;;
      *) echo "cold shell imported non-editable Triton: $TRITON_IMPORT" >&2; exit 1 ;;
    esac
    $TRITON_VENV/bin/python $CAPSTONE/benchmark/benchmark.py \
      --validate-source-identity --config-dir "$BASELINE/configs" \
      --baseline-commit-file $CAPSTONE/compiler/baseline_commit.txt \
      --triton-root "$TRITON_ROOT" --venv-python $TRITON_VENV/bin/python \
      --compile-input-roots "$BASELINE/compile-input-roots.txt" \
      --compiler-tree-excludes "$BASELINE/compiler-tree-excludes.txt" \
      --compiler-tree-manifest "$BASELINE/compiler-tree-files.sha256"

    bash "$CAPSTONE/scripts/run_correctness.sh" \
      --output-dir "$COLD_ROOT/results" \
      --output-json "$COLD_ROOT/results/correctness.json"
    $TRITON_VENV/bin/python $CAPSTONE/benchmark/correctness.py \
      --validate-json "$COLD_ROOT/results/correctness.json" \
      --expect-shapes 6 --expect-all-pass \
      --reference-settings-from "$BASELINE/correctness.json" \
      --reference-manifest "$BASELINE/MANIFEST.md" \
      --require-allow-tf32 false --require-float32-matmul-precision highest

    bash "$CAPSTONE/scripts/run_benchmark.sh" \
      --provider baseline --rounds 1 --disable-autotune \
      --config-dir "$BASELINE/configs" \
      --output-dir "$COLD_ROOT/results" \
      --output-csv "$COLD_ROOT/results/raw.csv" \
      --raw-samples-dir "$COLD_ROOT/results/raw-samples" \
      --gpu-state-output "$COLD_ROOT/results/gpu-state.csv"
    $TRITON_VENV/bin/python $CAPSTONE/benchmark/benchmark.py \
      --compare-baseline "$BASELINE/raw.csv" \
      --candidate-csv "$COLD_ROOT/results/raw.csv" \
      --shape-set "$BASELINE/configs" --metric median_us \
      --baseline-aggregate median-of-three-round-medians \
      --max-relative-delta 0.03 \
      --output "$COLD_ROOT/results/comparison.csv"

    CONFIG_JSON="$BASELINE/configs/M512_N4096_K4096.json"
    test -s "$CONFIG_JSON"
    CONFIG_HASH=$(sha256sum "$CONFIG_JSON" | cut -c1-12)
    ARTIFACT_TAG=M512_N4096_K4096_${CONFIG_HASH}
    rg -Fq "$ARTIFACT_TAG" "$BASELINE/MANIFEST.md"
    KERNEL_NAME=$($TRITON_VENV/bin/python -c \
      "import json,sys; print(json.load(open(sys.argv[1]))['kernel_name'])" "$CONFIG_JSON")

    bash "$CAPSTONE/scripts/run_ncu.sh" \
      --shape-tag M512_N4096_K4096 --config-json "$CONFIG_JSON" \
      --disable-autotune --warmup-ms 100 --profile-iterations 1 \
      --sections SpeedOfLight,Occupancy --kernel-name "$KERNEL_NAME" \
      --output "$COLD_ROOT/profiles/ncu/${ARTIFACT_TAG}_${KERNEL_NAME}"

    TRITON_ALWAYS_COMPILE=1 \
    MLIR_ENABLE_DUMP=1 \
    TRITON_KERNEL_DUMP=1 \
    TRITON_DUMP_DIR="$COLD_ROOT/raw-dump" \
    bash "$CAPSTONE/scripts/dump_triton_ir.sh" \
      --kernel fused_linear --config-dir "$BASELINE/configs" --all-configs \
      --output-root "$COLD_ROOT/ir" --emit-all

    # MANIFEST.sha256 的路径相对 CAPSTONE；cold root 必须生成完全相同的路径集合。
    sed -E "s/^[0-9a-f]{64} [ *]//" "$BASELINE/MANIFEST.sha256" \
      | sort > "$COLD_ROOT/expected.paths"
    (cd "$COLD_ROOT" && find ir -type f -print | sort) \
      > "$COLD_ROOT/actual.paths"
    cmp "$COLD_ROOT/expected.paths" "$COLD_ROOT/actual.paths"

    # 文本 IR/PTX/SASS 要求逐字相同；cubin 容器先 cmp，失败时只允许规范化
    # 为 cuobjdump SASS 后相同，并把“容器 hash 不同”明确记录，不能宣称 binary hash 相同。
    mkdir -p "$COLD_ROOT/compare"
    : > "$COLD_ROOT/binary-container-differences.tsv"
    while read -r FROZEN_HASH REL_PATH; do
      test -n "$REL_PATH"
      test -s "$CAPSTONE/$REL_PATH"
      test -s "$COLD_ROOT/$REL_PATH"
      case "$REL_PATH" in
        *.cubin)
          if ! cmp -s "$CAPSTONE/$REL_PATH" "$COLD_ROOT/$REL_PATH"; then
            KEY=$(printf "%s" "$REL_PATH" | tr "/" "_")
            cuobjdump --dump-sass "$CAPSTONE/$REL_PATH" \
              > "$COLD_ROOT/compare/$KEY.frozen.sass"
            cuobjdump --dump-sass "$COLD_ROOT/$REL_PATH" \
              > "$COLD_ROOT/compare/$KEY.cold.sass"
            cmp "$COLD_ROOT/compare/$KEY.frozen.sass" \
                "$COLD_ROOT/compare/$KEY.cold.sass"
            printf "%s\\t%s\\t%s\\n" "$REL_PATH" "$FROZEN_HASH" \
              "$(sha256sum "$COLD_ROOT/$REL_PATH" | cut -d " " -f1)" \
              >> "$COLD_ROOT/binary-container-differences.tsv"
          fi
          ;;
        *)
          cmp "$CAPSTONE/$REL_PATH" "$COLD_ROOT/$REL_PATH"
          ;;
      esac
    done < "$BASELINE/MANIFEST.sha256"
  '
```

  - **必读（目的：按工具定义解释指标而不是把相关性写成因果）：** [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)；用 [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#preparing-your-application-for-profiling) 再确认 capture 只覆盖目标 workload。
  - **性能依据：** 矩阵每格写 ncu metric 全名、单位和 report 路径；cold-shell benchmark 同样执行 100 ms warm-up、500 ms measurement 并报告 P20/P50/P80，记录前后 GPU 状态。`comparison.csv` 对六组 shape 分别用 cold P50 与 Week 15“三轮 P50 的中位数”计算 `abs(cold-baseline)/baseline`，每组必须 ≤ **3%**，缺 shape、重复 shape、raw sample 缺失或超阈值均非零退出；correctness JSON 在 benchmark 前独立验证，不能用性能阈值掩盖 correctness 失败。cold run 必须先校验冻结 config/source/reference hash，再在独立 root 重新生成六组五级 artifact/binary；路径数量与集合必须相等，文本 artifact 必须 `cmp` 相同。cubin 若因容器元数据导致 raw hash 不同，只能在 baseline/cold 同一 `--emit-all` 产生的 `cuobjdump --dump-sass` 与 `nvdisasm` 输出均完全一致且差异 hash 已记录时判定指令语义一致；报告必须保留“binary container hash 不同”，不得写成二进制可复现。

- [ ] **周日：冻结 baseline 并写三个候选假设**

在 `MANIFEST.md` 写入 baseline commit、checkout commit、compiler tree hash/算法、editable Triton resolved path、环境、配置、命令和 artifact hash；同时写入 PyTorch 版本、`allow_tf32=False`、`float32_matmul_precision=highest` 的设置与读回值、reference expression，并纳入 `profiler-map.csv` 的一一映射。cold correctness 必须同时与冻结 correctness JSON 和 manifest 逐字段对比这些 reference 设置。每个候选假设必须包含 profiler 证据和预期指标变化。此周禁止提前实现优化。

```bash
python $CAPSTONE/benchmark/benchmark.py --validate-freeze-manifest \
  --manifest $CAPSTONE/results/baseline/MANIFEST.md \
  --configs-sha256 $CAPSTONE/results/baseline/configs.sha256 \
  --artifacts-sha256 $CAPSTONE/results/baseline/MANIFEST.sha256 \
  --profiler-map $CAPSTONE/results/baseline/profiler-map.csv \
  --correctness-json $CAPSTONE/results/baseline/correctness.json \
  --require-reference allow_tf32=false,float32_matmul_precision=highest \
  --require-profiler-key shape_tag,config_hash,kernel_name \
  --require-one-to-one
```

  - **必读（目的：让每个假设绑定可复查的 section/metric 而非 profiler UI 印象）：** [Nsight Compute CLI report/export](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 与 [Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。
  - **性能依据：** manifest 冻结 100/500 ms 协议、三轮统计、GPU 状态、autotune config、ncu/nsys 版本与 metric 名；每个假设引用具体 report/metric 当前值和预期变化方向，禁止用带 profiler overhead 的 duration 替换 baseline P50。
  - **Expected：** 在全新 shell 中只按 README 可重建六组 correctness、一轮 benchmark、profiler/binary 对应关系，并由 hash 验证 artifact 未漂移。

### Phase 4 Exit Gate

```text
六组 correctness PASS
三轮 benchmark 可复现
baseline kernel launch 边界确认公平
重点 shape 有 ncu report 和对应 binary artifact
baseline workload/config/environment 已冻结
baseline report 和 manifest 完整
候选假设由指标触发，而非凭直觉选择
```

[进入 Phase 5：Performance Optimization →](./Phase5_Performance_Optimization.md)
