# Phase 5：Profiler 驱动的 Compiler 优化

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 17–20（2026-11-04 至 2026-12-01）
- 计划工时：44h

## 前置条件

[Phase 4 Exit Gate](./Phase4_Baseline_Freeze.md#phase-4-exit-gate) 已通过。

## 可验证阶段目标

- 用 profiler 证据选择唯一主瓶颈并提出可证伪假设。
- 正常计划完成三次预注册、可开关、可测试、可 ablation 的完整 compiler 实验：Exp01、Exp02、Exp03。
- 用冻结 baseline 和统一统计方法确认或拒绝性能归因。

## 本阶段产出

- optimization design 与主瓶颈证据。
- Experiment 01–03 patch、测试、benchmark、profile 与 ablation 结果。
- failed experiments 记录和最终优化候选。
- 候选方案相对 baseline 的 correctness 与性能结论。

## 证据契约与资料边界

本页外链已于 **2026-07-14** 在线核验。Phase 5 不重新定义 workload、数值容差、autotune config、warmup/measurement、机器状态过滤或 profiler sections；这些都继承 [Phase 4 的冻结契约](./Phase4_Baseline_Freeze.md#资料边界与冻结契约)。每个性能结论必须同时具备：冻结 baseline 对照、Phase 4 correctness 全通过、同一 experiment binary 的 feature-flag ablation，以及 benchmark 与 profiler 的双重证据。任何一项缺失只能记为 `INCONCLUSIVE`，不能冻结为候选。

### 实验数量、预注册与状态契约

正常计划必须完成 **Exp01、Exp02、Exp03 三次完整 experiment**。每次 experiment 都必须在看到该次 correctness、benchmark 或 profiler 结果前，在对应 `results/experiments/expNN/README.md` 预注册 hypothesis、唯一主要变量、目标/回退 shape、机制 profiler shape、paired 顺序、判定阈值和预期 IR/PTX/硬件指标变化。一次“完整 experiment”必须同时具有 identity、compiler red/green test、六 shape correctness、flag-off drift、三轮 paired benchmark、机制 profiler、同 binary flag-off/on ablation 和 `SUPPORTED/REJECTED/INCONCLUSIVE` 结论；失败或被拒绝的 experiment 只要证据完整，仍计入三次，但不得删除。

唯一例外是总索引[“落后 2-3 周”](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md#落后-2-3-周)定义的 Reduced 模式。它必须在看到 **任何** Phase 5 experiment 结果前写入 `progress.md` 和 `docs/optimization_design.md`；看到结果后不得启用、补记或用它改写失败。Reduced 模式可把计划缩为至少一次完整 experiment，但不降低上述单次实验的证据标准，且 Phase 5 只能标记 `PASS-REDUCED`，不能标记正常 `PASS`。

### Baseline identity 与 experiment identity

两层身份不得混为一谈：

- immutable baseline identity：Phase 4 冻结的 `baseline_commit`、`compiler_tree_hash`、六个 config 及 `configs.sha256`、compile-input roots/excludes 和逐文件 manifest。Phase 5 只读引用，绝不重写。
- experiment identity：`baseline_commit + exact_compiler_patch_sha256 + experiment_tree_hash + build_hash`。`experiment_tree_hash` 必须覆盖 baseline roots 中 tracked、modified、dirty 和 untracked 普通文件并沿用 Phase 4 exclusions；`build_hash` 覆盖实际被加载的 compiler/runtime binary、shared objects 与 build metadata manifest。
- 每个 `expNN/README.md` 同时记录两套 identity、patch 应用关系、build manifest，以及 `flag-off`/`flag-on` 到同一 experiment identity、同一 binary 的映射。off/on 唯一允许的源码行为差异是明确记录的 `TRITON_EXP_FLAG=off|on`；不能分别 build，不能要求 experiment tree 等于 baseline tree。
- source/build identity 相同不等于 compile cache 可共享。每个实验固定 `$EXP_ROOT/cache/{off,on}`，运行前设置并核验固定 commit 实际使用的 `TRITON_CACHE_DIR`（若该 commit 名称不同，记录等价 cache env/symbol）；off/on 首次 compile 与 warmup 分别完成后才进入 paired timing，compile time 永不进入 latency。
- 正式 paired 测量前，先用 experiment binary 的 flag-off 对 Phase 4 frozen baseline 做逐 shape drift gate。drift 使用本页从 Phase 4 三轮 frozen P50 重算的 `baseline_center` 与 `noise_floor_pct`；超阈值则整轮实验身份无效，先定位 build/environment/cache drift，不能继续解释 flag-on。

```text
baseline_identity = baseline_commit + baseline_tree_hash + configs_sha256 + manifests_hash
experiment_identity = baseline_commit + exact_compiler_patch_sha256 + experiment_tree_hash + build_hash
flag_off_identity = experiment_identity + TRITON_EXP_FLAG=off
flag_on_identity  = experiment_identity + TRITON_EXP_FLAG=on
```

每个实验在 build 完成、任何测试开始前冻结 identity。tree manifest 必须复用 Phase 4 的 roots/exclusions 与排序算法；其输入覆盖 tracked、modified、dirty 和 untracked 普通文件，不能退化成只枚举 `git ls-files`。build manifest 的 roots 由 editable import 和进程实际加载路径反查后预注册，至少覆盖 compiler executable、Python/native extension、shared libraries、CMake cache/build metadata。

```bash
EXP_ROOT=$CAPSTONE/results/experiments/expNN
mkdir -p "$EXP_ROOT/identity" "$EXP_ROOT/paired" "$EXP_ROOT/off" "$EXP_ROOT/on"
BASELINE_COMMIT=$(tr -d "[:space:]" < "$CAPSTONE/compiler/baseline_commit.txt")
test "$BASELINE_COMMIT" = "$(git -C "$TRITON_ROOT" rev-parse HEAD)"
sha256sum "$CAPSTONE/results/baseline/compiler-tree-files.sha256" \
  "$CAPSTONE/results/baseline/configs.sha256" \
  "$CAPSTONE/results/baseline/compile-input-roots.txt" \
  "$CAPSTONE/results/baseline/compiler-tree-excludes.txt" \
  > "$EXP_ROOT/identity/baseline-manifests.sha256"
git -C "$TRITON_ROOT" diff --binary "$BASELINE_COMMIT" -- \
  python/triton lib include third_party/nvidia test/TritonGPU \
  > "$EXP_ROOT/identity/exact-compiler.patch"
(cd "$TRITON_ROOT" && \
  git ls-files --others --exclude-standard -z -- \
    python/triton lib include third_party/nvidia test/TritonGPU | \
  while IFS= read -r -d '' FILE; do
    git diff --no-index --binary /dev/null "$FILE" || test "$?" -eq 1
  done) >> "$EXP_ROOT/identity/exact-compiler.patch"
sha256sum "$EXP_ROOT/identity/exact-compiler.patch" \
  > "$EXP_ROOT/identity/exact-compiler-patch.sha256"

# 这两个脚本由 Week 17 identity 工作实现；tree 脚本复用 Phase 4 算法并纳入 dirty/untracked。
bash $CAPSTONE/scripts/hash_compiler_tree.sh \
  --triton-root "$TRITON_ROOT" \
  --roots-file "$CAPSTONE/results/baseline/compile-input-roots.txt" \
  --excludes-file "$CAPSTONE/results/baseline/compiler-tree-excludes.txt" \
  --output "$EXP_ROOT/identity/experiment-tree-files.sha256"
sha256sum "$EXP_ROOT/identity/experiment-tree-files.sha256" \
  > "$EXP_ROOT/identity/experiment-tree.sha256"
bash $CAPSTONE/scripts/hash_loaded_build.sh \
  --triton-root "$TRITON_ROOT" --venv-python "$TRITON_VENV/bin/python" \
  --manifest "$EXP_ROOT/identity/build-files.sha256" \
  --output "$EXP_ROOT/identity/build.sha256"
```

`hash_compiler_tree.sh` 必须用 modified/untracked fixture 自测；`hash_loaded_build.sh` 必须验证 recorded path 存在且被当前 editable import/build 实际使用。README 保存两套 identity、四个 experiment 组成 hash、flag 映射和生成命令；任何源码/build 文件变化都产生新 experiment identity 并重跑 off drift。

compile 前在固定 commit 核验真实 cache 接口，并冻结 variant/cache/artifact 映射：

```bash
rg -n 'TRITON_CACHE_DIR|CACHE_DIR|cache.*key|CacheManager' \
  "$TRITON_ROOT/python/triton" "$TRITON_ROOT/lib"
mkdir -p "$EXP_ROOT/cache/off" "$EXP_ROOT/cache/on" \
  "$EXP_ROOT/off/artifacts" "$EXP_ROOT/on/artifacts"
for VARIANT in off on; do
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_benchmark.sh \
      --config-dir "$CAPSTONE/results/baseline/configs" \
      --variant "$VARIANT" --cache-dir "$EXP_ROOT/cache/$VARIANT" \
      --precompile-only --rounds 3 \
      --output-dir "$EXP_ROOT/$VARIANT/precompile" \
      --output-csv "$EXP_ROOT/$VARIANT/precompile/compile.csv" \
      --raw-samples-dir "$EXP_ROOT/$VARIANT/precompile/raw-samples" \
      --gpu-state-output "$EXP_ROOT/$VARIANT/precompile/gpu-state.csv"
done
```

`--precompile-only` 的冻结语义是：对 config-dir 六个 config 在该 variant 空 cache 中完成首次 compile，再执行至少一次不计时 warmup，保存 compile time/artifact metadata 后退出，不写正式 latency。每个 artifact metadata 必须 assert `variant`、`TRITON_EXP_FLAG`、cache root、cache key、experiment identity、config hash 和 binary hash；off metadata 不能指向 on cache，反之亦然。保存 off/on TTIR/TTGIR/LLVM/PTX/cubin/SASS hash 与 flag/cache-key 映射。若优化预期改变 codegen，则相关 artifact hash 应不同；若 hash 相同，只能结合 IR diff 与 profiler 解释为“change 未物化”或等价 codegen，不能把 cache 命中当作证据，更不能让两 variant 串 cache。

### 预注册 shape、paired 顺序与冻结统计

每轮实验默认预注册全部六个 Phase 4 shape；若 hypothesis 只对明确子集作单 shape/子集结论，仍必须跑六 shape correctness、off drift 和 paired benchmark，非目标 shape 用于报告回退，不能事后删除。`results/experiments/expNN/README.md` 在运行前写死目标 shape、机制 profiler 代表 shape、三轮顺序和判定阈值。

```text
M32_N4096_K4096
M128_N4096_K4096
M512_N4096_K4096
M512_N16384_K4096
M512_N4096_K16384
M4096_N4096_K4096
```

每个 shape 使用同一机器状态、同一冻结 config、同一 experiment binary 做 flag-off/on 配对交错；三轮预注册为 `R1: off→on, R2: on→off, R3: off→on`，下一 shape 反转起始 variant，实际 order、时间和 GPU state 必须落盘。先从 Phase 4 frozen CSV 读取该 shape 的三个 round P50 `b1,b2,b3`，再从每个 variant/round raw samples 复算 P50：

```text
baseline_center_us = median(b1, b2, b3)
phase4_noise_pct = max(abs(bi - baseline_center_us) / baseline_center_us for bi in [b1,b2,b3]) * 100
noise_floor_pct = max(3.0, 2 * phase4_noise_pct)
off_us = median(off_round1_p50_us, off_round2_p50_us, off_round3_p50_us)
on_us = median(on_round1_p50_us, on_round2_p50_us, on_round3_p50_us)
ratio_r = on_round_r_p50_us / off_round_r_p50_us
speedup_pct_r = (off_round_r_p50_us / on_round_r_p50_us - 1) * 100
shape_speedup_pct = median(speedup_pct_r for r in three paired rounds)
off_cv_pct = statistics.pstdev(off_round_p50_us) / statistics.mean(off_round_p50_us) * 100
on_cv_pct = statistics.pstdev(on_round_p50_us) / statistics.mean(on_round_p50_us) * 100
off_drift_pct = abs(off_us / baseline_center_us - 1) * 100
off_drift_pass = off_drift_pct <= noise_floor_pct
```

`SUPPORTED` 要求 correctness 全通过、off drift gate 通过、**每个**预注册目标 shape 的 `shape_speedup_pct > noise_floor_pct` 且三轮符号稳定、profiler metric 按假设方向变化、flag-off/on ablation 可回溯。`abs(shape_speedup_pct) <= noise_floor_pct` 或轮次符号不稳定为 `INCONCLUSIVE`；显著反向或 metric 反证为 `REJECTED`。单 shape 结论逐 shape 判，目标子集结论要求子集每个 shape 均通过，全六 shape 结论要求六个全部通过；Phase 5 不用几何平均下全局结论，跨 shape 聚合留到 Phase 6。

### 强制脚本接口与输出隔离

所有 Phase 5 结果只能写入 `results/experiments/expNN/{off,on,paired,profiles}`，不得写入或覆盖 `results/baseline/`。以下模板中的 `expNN`、`TRITON_EXP_FLAG`、shape 顺序和 round 顺序必须在实验 README 预注册后逐项展开；不能使用未定义的 provider 名或隐式全-shape 捷径。

```bash
EXP_ROOT=$CAPSTONE/results/experiments/expNN
mkdir -p "$EXP_ROOT/paired" "$EXP_ROOT/off/correctness" "$EXP_ROOT/on/correctness" \
  "$EXP_ROOT/off/benchmark/raw-samples" "$EXP_ROOT/on/benchmark/raw-samples" \
  "$EXP_ROOT/profiles/nsys/off" "$EXP_ROOT/profiles/nsys/on" \
  "$EXP_ROOT/profiles/ncu/off" "$EXP_ROOT/profiles/ncu/on"
for VARIANT in off on; do
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_correctness.sh \
      --output-dir "$EXP_ROOT/$VARIANT/correctness" \
      --output-json "$EXP_ROOT/$VARIANT/correctness/results.json"
done

# 六 shape × 三轮 paired benchmark。脚本遍历 config-dir，并按预注册 schedule 切换 variant/cache。
bash $CAPSTONE/scripts/run_benchmark.sh \
  --config-dir "$CAPSTONE/results/baseline/configs" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$EXP_ROOT/cache" --schedule-file "$EXP_ROOT/paired/schedule.csv" \
  --rounds 3 --output-dir "$EXP_ROOT/paired" \
  --output-csv "$EXP_ROOT/paired/results.csv" \
  --raw-samples-dir "$EXP_ROOT/paired/raw-samples" \
  --gpu-state-output "$EXP_ROOT/paired/gpu-state.csv"

# nsys/ncu 只需覆盖 README 预注册的机制代表 shape；这里显式列三类代表 shape。
NCU_SECTIONS=SpeedOfLight,Occupancy,MemoryWorkloadAnalysis,SchedulerStats,InstructionStats
for SHAPE_TAG in M32_N4096_K4096 M512_N4096_K16384 M4096_N4096_K4096; do
  CONFIG_JSON=$CAPSTONE/results/baseline/configs/$SHAPE_TAG.json
  KERNEL_NAME=$($TRITON_VENV/bin/python -c \
    "import json,sys; print(json.load(open(sys.argv[1]))['kernel_name'])" "$CONFIG_JSON")
  for VARIANT in off on; do
    TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
      nsys profile --trace=cuda,nvtx --sample=none \
      --capture-range=cudaProfilerApi --force-overwrite=true \
      --output="$EXP_ROOT/profiles/nsys/$VARIANT/${SHAPE_TAG}_${KERNEL_NAME}" \
      $TRITON_VENV/bin/python $CAPSTONE/benchmark/benchmark.py \
        --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" --disable-autotune \
        --warmup-ms 100 --profile-iterations 1 --capture-profiler-api \
        --profile-metadata "$EXP_ROOT/profiles/nsys/$VARIANT/${SHAPE_TAG}_${KERNEL_NAME}.metadata.json"
    TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
      bash $CAPSTONE/scripts/run_ncu.sh \
      --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" \
      --disable-autotune --warmup-ms 100 --profile-iterations 1 \
      --sections "$NCU_SECTIONS" --kernel-name "$KERNEL_NAME" \
      --output "$EXP_ROOT/profiles/ncu/$VARIANT/${SHAPE_TAG}_${KERNEL_NAME}"
  done
done
```

项目阅读不是通用书单，而是用于填写 `docs/optimization_design.md` 中的三列表：`compiler hypothesis`、`observable metric/artifact`、`target source symbol`。

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：以 `SpeedOfLight` 初筛资源上限，以 `MemoryWorkloadAnalysis`、`Occupancy`、`SchedulerStats`、`InstructionStats` 验证归因；metric 必须保存本机 `ncu --query-metrics` 返回的全名和单位。
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：固定 kernel filter、sections、replay/capture 设置并导出 `.ncu-rep`/CSV；禁止比较不同采集设置的数字。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：只在假设涉及 global coalescing、shared bank conflict、register pressure 或 occupancy 时读取对应章节，并把建议映射到实际 TTGIR/PTX/SASS 变化。
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)：核对 `ld/st` 向量形式、predicate 与 `bar/barrier` 的语义；PTX 变化只是 compiler artifact，最终仍以 SASS 和 profiler 证明硬件行为。
- [TritonGPU `ttg.convert_layout` 文档](https://triton-lang.org/main/dialects/TritonGPUOps.html#ttg-convert-layout-triton-gpu-convertlayoutop)、官方 [`TritonGPUOps.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)、[`TritonGPUAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)、[`RemoveLayoutConversions.cpp`](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp) 与 [`test/TritonGPU`](https://github.com/triton-lang/triton/tree/main/test/TritonGPU)：滚动链接只用于导航；真正必读和修改对象必须来自 Phase 4 冻结的 `$TRITON_ROOT` commit。
- [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#rooflines)：仅用于扩展“compute-bound / memory-bound”归因框架，不能单独决定 transformation，更不能替代具体 instruction、scheduler、memory 和 occupancy 证据。

每轮实验开始前执行固定 commit 核验；把命中的真实路径、symbol 与行号写入该实验 README。路径不存在时用 `rg` 找该 commit 的等价实现，不得直接照抄 `main` 的文件名。

```bash
BASELINE_COMMIT=$(tr -d "[:space:]" < "$CAPSTONE/compiler/baseline_commit.txt")
test "$BASELINE_COMMIT" = "$(git -C "$TRITON_ROOT" rev-parse HEAD)"
test -d "$TRITON_ROOT/test/TritonGPU"
rg -n 'ConvertLayoutOp|BlockedEncoding|LinearLayout|RemoveLayoutConversions' \
  "$TRITON_ROOT/include" "$TRITON_ROOT/lib" "$TRITON_ROOT/test/TritonGPU"
ncu --list-sections
ncu --query-metrics-mode all --query-metrics
```

## Week 17：选择唯一主瓶颈并实现 Experiment 01

**Files:**

- Create: `docs/optimization_design.md`
- Create: `results/experiments/exp01/README.md`
- Create: `compiler/tests/exp01.mlir`
- Create: `compiler/reproducers/exp01.mlir`
- Create: `compiler/patches/exp01.patch`
- Create: `scripts/hash_compiler_tree.sh`
- Create: `scripts/hash_loaded_build.sh`
- Update: `scripts/run_benchmark.sh`

### 工作日

- [ ] **周一：按证据分类瓶颈**

使用以下决策：

```text
小 M 且 CTA/waves 不足 -> tile/mapping/K-parallelism 候选
layout conversion 占显著 IR/PTX -> layout propagation/elimination 候选
register/thread 高且 occupancy 低 -> tile/stages/lifetime 候选
shared bank conflict 高 -> shared layout/swizzle 候选
DRAM 高、L2 低 -> ordering/reuse 候选
```

  - **项目化必读：** 阅读 [Nsight Compute 各 section 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 和 [CUDA Best Practices 的 memory/occupancy 章节](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。为五类候选各填一行：对应的 `LaunchStats/MemoryWorkloadAnalysis/Occupancy/SchedulerStats/InstructionStats` 观测、TTGIR/PTX 证据和可能修改的 pass/symbol；没有对应证据的类别删除。

- [ ] **周二：选一个主问题**

只允许一个主指标和一个主要 compiler transformation。把未选问题写入 future work。

  - **项目化必读：** 阅读 [SpeedOfLight 与 Roofline 展示](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#rooflines)，只用它给候选排序；随后回到固定 commit 的目标 pass implementation 与最近邻 `test/TritonGPU` 测试，写下唯一 target symbol。主指标必须是本机可查询的全名，Roofline 不能作为唯一指标。

- [ ] **周三：写可证伪假设**

格式：

```text
If <compiler change>, then <IR/PTX change>,
therefore <ncu metric> should change from X toward Y,
and latency should improve on <shape set>,
while <known cost> may regress <other set>.
```

  - **项目化必读：** 阅读 [Nsight Compute metrics structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)；用 `ncu --query-metrics-mode all --query-metrics` 核对假设中的 metric 全名、suffix 和单位，并把 X、Y、受益 shape、潜在回退 shape、TTGIR/PTX 预期逐项写入 hypothesis。

- [ ] **周四：设计 feature flag**

compiler change 必须能够显式开关；flag-off 和 flag-on 必须来自同一 experiment identity、同一 binary/commit，只由明确 feature flag 区分。Phase 5 不接受用两个 commit/build 冒充 ablation。

  - **项目化必读：** 阅读固定 commit 的目标 pass registration、pipeline 构造与 option 定义，使用 `rg -n 'PassPipeline|PassOptions|add_stages|make_ttgir' "$TRITON_ROOT/include" "$TRITON_ROOT/lib" "$TRITON_ROOT/third_party/nvidia"` 定位实际 symbol。flag 值必须进入运行 metadata，但 off/on 的 patch/tree/build hash 完全相同；关闭时 IR 语义应恢复 baseline，experiment tree hash 本身不要求等于 baseline tree hash。
  - **身份核验：** 在本步实现并测试 `hash_compiler_tree.sh` 与 `hash_loaded_build.sh`；临时加入一个 tracked modification 和一个 untracked fixture 时 tree hash 必须变化，清理 fixture 后 hash 必须恢复。生成 exact patch（含 untracked additions）、experiment tree/build manifests 和 off/on→同 binary 映射后才能进入 compiler red/green test。
  - **benchmark 接口设计：** 扩展 `run_benchmark.sh`，唯一 Phase 5 接口为 Phase 4 已有的 `--config-dir/--rounds/--output-dir/--output-csv/--raw-samples-dir/--gpu-state-output`，加上本阶段明确定义的 `--variant`、`--cache-dir`、`--precompile-only` 或 paired 模式的 `--variant-env/--cache-env/--cache-root/--schedule-file`。脚本必须遍历 config-dir 中恰好六个已通过 `configs.sha256` 的 JSON；schedule 必须是六 shape × 三 round × off/on 共 36 行，拒绝缺失、重复或事后顺序变化，并把实际开始时间与执行顺序写到 output-dir 下的 `order.csv`。

- [ ] **周五：写并运行失败的 compiler test**

先定义输入 reproducer 和稳定 IR 检查，不使用 latency 作为 compiler 单测。Expected：baseline compiler 下新增 FileCheck 失败。

  - **项目化必读：** 阅读 [FileCheck 官方文档](https://llvm.org/docs/CommandGuide/FileCheck.html) 的 `CHECK-LABEL`、变量和 `CHECK-NOT`，再读固定 commit 目标 pass 的最近邻 in-tree test 与 `RUN:` 行；检查必须直接约束 hypothesis 中要改变的 op/encoding/symbol，而不是易变的整段打印。

### 周末

- [ ] **周六：实现最小 compiler change 并运行 test**

只实现足以让 reproducer 发生预期变化的最小逻辑；运行目标 test 和相关 Triton test 子集。Expected：新增 test PASS。

同时实现上一步定义的 benchmark/cache 接口，并在任何正式 compile 前验证：

```bash
bash $CAPSTONE/scripts/run_benchmark.sh --help
bash $CAPSTONE/scripts/run_benchmark.sh \
  --config-dir "$CAPSTONE/results/baseline/configs" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$CAPSTONE/results/experiments/exp01/cache" \
  --schedule-file "$CAPSTONE/results/experiments/exp01/paired/schedule.csv" \
  --rounds 3 --dry-run \
  --output-dir "$CAPSTONE/results/experiments/exp01/paired" \
  --output-csv "$CAPSTONE/results/experiments/exp01/paired/results.csv" \
  --raw-samples-dir "$CAPSTONE/results/experiments/exp01/paired/raw-samples" \
  --gpu-state-output "$CAPSTONE/results/experiments/exp01/paired/gpu-state.csv"
```

Expected：`--help` 列出上述全部参数；dry-run 验证六个 config、36 个 schedule row、off/on cache root 不同，只打印执行计划，不 compile、不写性能 CSV。固定 commit 若不使用 `TRITON_CACHE_DIR`，先把核验出的等价 env 写入 `--cache-env`，后续所有实验复用。

  - **项目化必读：** 若目标是 layout，阅读固定 commit 的 `ConvertLayoutOp`、`BlockedEncoding`/`LinearLayout` 定义、目标 transform 和相邻测试；若是其他瓶颈，仍按“op/interface → pass implementation → test”三段链定位。用 [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) 核对 rewrite 合法性，并记录实际修改 symbol。

- [ ] **周日：运行 correctness 并导出 artifact**

六组 shape 和固定 seed 全部通过后，保存 compiler diff、test、TTIR/TTGIR/LLVM/PTX before/after 和 patch。

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp01
for VARIANT in off on; do
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_correctness.sh \
    --output-dir "$EXP_ROOT/$VARIANT/correctness" \
    --output-json "$EXP_ROOT/$VARIANT/correctness/results.json"
done
```

  - **项目化必读：** 阅读 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 中本次 diff 涉及的 `ld/st` vector、predicate、`bar/barrier` 条目，把 PTX 与最终 SASS 对齐；然后严格复用 Phase 4 correctness contract。Expected：同一 experiment identity 的 flag-off/on 六 shape 与固定 seed 全量 PASS，off 的 artifact 语义回到冻结 baseline；correctness 失败时不得进入 benchmark。

### Week 17 Exit Gate

```text
唯一主瓶颈已由 profiler 证据确定
假设包含预期 IR 和硬件指标变化
compiler change 可开关
新增 compiler test 经 red/green 验证
相关 test 子集和完整 correctness 通过
IR diff 与设计预测一致
```

---

## Week 18：Experiment 01 Benchmark、Profiler 与 Ablation

**Files:**

- Create: `results/experiments/exp01/results.csv`
- Create: `results/experiments/exp01/ncu-summary.csv`
- Create: `scripts/analyze_paired_results.py`
- Update: `results/experiments/exp01/README.md`

### 工作日

- [ ] **周一：运行 baseline sanity**

先选择三组代表 shape 快速确认机器状态，再对全部六个预注册 shape 执行正式 flag-off drift gate；只有逐 shape 偏差都不超过冻结阈值，实验才有效。

  - **项目化必读：** 重读 [Phase 4 Performance 必读资料与测量协议](./Phase4_Baseline_Freeze.md#performance-必读资料)，用其中 `nvidia-smi` 状态字段和 raw-sample/P50 规则检查 drift；先覆盖 hypothesis 的受益、已知代价和中性对照，再完成六 shape。逐 shape 从 Phase 4 三轮重算 `baseline_center_us`、`phase4_noise_pct`、`noise_floor_pct`，再检查 `abs(off_us/baseline_center_us-1)*100 <= noise_floor_pct`；超阈值时保留记录并判整个 experiment identity 无效。

- [ ] **周二：运行 experiment 三轮**

使用和 baseline 完全相同的 warmup、measurement、shape 和 autotune policy。

按“强制脚本接口与输出隔离”模板设置 `EXP_ROOT=$CAPSTONE/results/experiments/exp01`；先确认 off/on 独立 cache 已分别完成首次 compile/warmup，再用已通过 `--help`/dry-run 的 `--config-dir + --cache-root + --schedule-file + output` 接口遍历六 shape × 三轮 `off/on`，正式 latency 不含 compile。

  - **项目化必读：** 阅读 [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 并对照 Phase 4 固定的 `warmup=100 ms`、`measurement=500 ms`、raw samples 和三轮协议；只改变 feature flag，禁止重新 autotune 或混入 compile time。

- [ ] **周三：计算 delta**

报告每个 shape 的逐轮 pair 和三轮聚合：

```text
baseline_center_us = median(b1, b2, b3)
phase4_noise_pct = max(abs(bi - baseline_center_us) / baseline_center_us) * 100
noise_floor_pct = max(3.0, 2 * phase4_noise_pct)
off_us = median(off_round1_p50_us, off_round2_p50_us, off_round3_p50_us)
on_us = median(on_round1_p50_us, on_round2_p50_us, on_round3_p50_us)
ratio_r = on_round_r_p50_us / off_round_r_p50_us
speedup_pct_r = (off_round_r_p50_us / on_round_r_p50_us - 1) * 100
shape_speedup_pct = median(speedup_pct_r)
off_cv_pct = statistics.pstdev(off_round_p50_us) / statistics.mean(off_round_p50_us) * 100
on_cv_pct = statistics.pstdev(on_round_p50_us) / statistics.mean(on_round_p50_us) * 100
off_drift_pct = abs(off_us / baseline_center_us - 1) * 100
```

实现并运行统一分析脚本：

```bash
python $CAPSTONE/scripts/analyze_paired_results.py --self-test
python $CAPSTONE/scripts/analyze_paired_results.py \
  --phase4-csv "$CAPSTONE/results/baseline/raw.csv" \
  --paired-csv "$CAPSTONE/results/experiments/exp01/paired/results.csv" \
  --order-csv "$CAPSTONE/results/experiments/exp01/paired/order.csv" \
  --output "$CAPSTONE/results/experiments/exp01/paired/analysis.csv"
```

CSV 每 shape 至少保存 `baseline_b1_us,b2_us,b3_us,baseline_center_us,phase4_noise_pct,noise_floor_pct,off_r1_p50_us,off_r2_p50_us,off_r3_p50_us,on_r1_p50_us,on_r2_p50_us,on_r3_p50_us,off_us,on_us,ratio_r1,ratio_r2,ratio_r3,speedup_pct_r1,speedup_pct_r2,speedup_pct_r3,shape_speedup_pct,off_cv_pct,on_cv_pct,off_drift_pct,off_drift_pass,order_path`。`--self-test` 使用 `baseline=[98,100,102]` 验证 center=100、noise=2%、floor=4%，使用 off/on 三轮 fixture 验证 median、`statistics.pstdev`、paired speedup 和 drift；任一误差超 `1e-9` 非零退出。

  - **项目化必读：** 重读 Phase 4 的三轮 P20/P50/P80 schema；从 frozen raw CSV 取三个 baseline round P50，每个 experiment round 从 raw samples 复算 off/on P50，再用上述唯一公式计算 pair。off/on CV 分开计算，保留 raw samples、order、GPU state 和无效轮原因；不得用 on/frozen baseline 混入 ablation ratio，也不得用跨 shape 汇总掩盖回退。

- [ ] **周四：采集相同 ncu sections**

只比较同一 shape、同一 kernel 语义和相同 profiler 设置。

按全局 profiler 模板设置 `EXP_ROOT=$CAPSTONE/results/experiments/exp01`，对 README 预注册的机制代表 shape 逐个执行 off/on `nsys` 与 `run_ncu.sh`；两侧必须显式传同一 `--shape-tag --config-json --disable-autotune --warmup-ms 100 --profile-iterations 1 --sections --kernel-name --output`。

  - **项目化必读：** 阅读 [Nsight Compute CLI 的 profile options/metrics](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)；复用 Phase 4 kernel filter、capture/replay 和 sections，保存 exact command、tool version、metric full name/unit 与 `.ncu-rep`。不得因为预期结果不明显而临时换 section。

- [ ] **周五：检查预期指标**

逐项标记 predicted/observed/mismatch；不能只看 latency。

  - **项目化必读：** 按 hypothesis 选择 [MemoryWorkloadAnalysis、Occupancy、SchedulerStats、InstructionStats](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 中对应项；对每个 TTGIR/PTX/SASS 预测写一个 metric 观测。scheduler 无法每周期 issue 时才分析 stall reason，不能把任意 stall 百分比直接叫瓶颈。

### 周末

- [ ] **周六：做 ablation**

关闭 feature flag；用同一 experiment identity/binary 对全部六个预注册 shape 重新执行 correctness 和三轮 paired benchmark。Profiler 可只对机制代表 shape 做 off/on。Expected：IR/指标/性能向 flag-off 恢复，且 flag-off 已通过 frozen baseline drift gate。

  - **项目化必读：** 重读固定 commit 的 flag/pipeline symbol 与 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)；验证同一 patch/tree/build hash、同一 config 下只有 `TRITON_EXP_FLAG` 变化。保存两套 identity 映射、on/off IR/PTX/SASS、六 shape correctness/paired benchmark 和机制 shape 的同 section profile，不能只关闭 Python launcher 层的标签。

- [ ] **周日：结论分类**

只允许三类结论：

```text
SUPPORTED：证据支持假设
REJECTED：证据否定假设
INCONCLUSIVE：噪声或指标不足
```

  - **项目化必读：** 用 [Nsight Compute metrics structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure) 区分 counter、ratio、throughput 和 peak；严格应用本页冻结判据：correctness、off drift、目标 shape 中位 speedup 超 noise floor且三轮符号稳定、预测 metric 和 ablation 同向才是 `SUPPORTED`；效应落入 floor 或符号不稳为 `INCONCLUSIVE`，显著反向/metric 反证为 `REJECTED`。单 shape、目标子集、全六 shape 分开陈述。

### Week 18 Exit Gate

```text
三轮结果和 ncu 对照完整
完成 ablation
结论不依赖单次最快值
```

---

## Week 19：Experiment 02 针对首轮结论迭代

**Files:**

- Create: `results/experiments/exp02/README.md`
- Create: `compiler/tests/exp02.mlir`
- Create: `compiler/patches/exp02.patch`
- Create: `results/experiments/exp02/results.csv`

### 工作日

- [ ] **周一：从 Exp01 结论定义单变量变化**

若 Exp01 rejected，改变假设而不是调参掩盖；若 supported，只改变一个参数/策略扩大适用区间。

  - **项目化必读：** 回读 Exp01 的 predicted/observed/mismatch 表，再按 [Nsight Compute section 定义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 选一个尚能区分解释的 metric；在固定 commit 目标 pass 中锁定唯一新 symbol/condition。README 明列 Exp01→Exp02 唯一变量和保持不变项。

- [ ] **周二：写 Exp02 compiler test**

  - **项目化必读：** 阅读固定 commit 的目标 pass 最近邻 `test/TritonGPU` 用例和 [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)；test 必须覆盖 Exp02 新 condition，同时保留一个负例证明未扩大到非目标 shape/layout。先在未实现状态观察预期 FAIL。

- [ ] **周三：实现最小变化并运行 test**

  - **项目化必读：** 阅读固定 commit 中 Exp02 目标 op/interface、pass implementation 和 registration；若涉及 layout，重点核对 `ConvertLayoutOp` 与 encoding verifier。用 [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/) 检查 match failure/replace 语义；只修改假设声明的 symbol，Expected：red→green。

- [ ] **周四：运行相关 test 子集和 correctness**

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp02
for VARIANT in off on; do
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_correctness.sh \
    --output-dir "$EXP_ROOT/$VARIANT/correctness" \
    --output-json "$EXP_ROOT/$VARIANT/correctness/results.json"
done
```

Expected：六组冻结 shape 和固定 seed 全部 PASS。

  - **项目化必读：** 重读 [Phase 4 Correctness 必读资料](./Phase4_Baseline_Freeze.md#correctness-必读资料)，复用 FP32 reference、容差、finite 和 source identity；compiler unit test PASS 不能替代六 shape/固定 seed correctness。

- [ ] **周五：生成四级 IR/PTX diff**

  - **项目化必读：** 阅读 [TritonGPU `ttg.convert_layout`](https://triton-lang.org/main/dialects/TritonGPUOps.html#ttg-convert-layout-triton-gpu-convertlayoutop) 和 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) 中 diff 命中的 load/store、predicate、barrier 条目；将 TTIR→TTGIR→LLVM→PTX 的首次差异回指到 Exp02 target symbol，并同时保存 SASS 以防 PTX 被后端重写。

### 周末

- [ ] **周六：三轮 benchmark 和 ncu**

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp02
# 与 Exp01 同一已验证接口遍历六个 frozen config，并按预注册 schedule 切换独立 cache。
# 先执行全局 `--precompile-only` off/on 两次并核对 variant/cache metadata，再运行下列 paired 命令。
bash $CAPSTONE/scripts/run_benchmark.sh \
  --config-dir "$CAPSTONE/results/baseline/configs" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$EXP_ROOT/cache" --schedule-file "$EXP_ROOT/paired/schedule.csv" \
  --rounds 3 --output-dir "$EXP_ROOT/paired" \
  --output-csv "$EXP_ROOT/paired/results.csv" \
  --raw-samples-dir "$EXP_ROOT/paired/raw-samples" \
  --gpu-state-output "$EXP_ROOT/paired/gpu-state.csv"

# 对 README 预注册的机制代表 shape，VARIANT=off/on 分别执行。
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_ncu.sh \
  --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" \
  --disable-autotune --warmup-ms 100 --profile-iterations 1 \
  --sections "$NCU_SECTIONS" --kernel-name "$KERNEL_NAME" \
  --output "$EXP_ROOT/profiles/ncu/$VARIANT/${SHAPE_TAG}_${KERNEL_NAME}"
```

  - **项目化必读：** 阅读 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 并复用 Exp01/Phase 4 exact sections；benchmark 复用冻结 config 和三轮 raw-sample 协议。先完成六 shape flag-off drift gate，再按预注册 order 做同 binary paired 测量；机制 profiler 可限代表 shape。只对预注册目标 shape 下主结论，额外发现进入 future work。

- [ ] **周日：ablation 和结论**

  - **项目化必读：** 重读 Exp02 的 feature-flag symbol 与 [Nsight Compute metrics structure](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-structure)；执行同 experiment identity 的 flag-on/off 六 shape correctness、drift、paired benchmark 和全 shape ablation，机制 profile 对照可限代表 shape，再按冻结公式判 `SUPPORTED/REJECTED/INCONCLUSIVE`。失败实验必须追加到 `docs/failed_experiments.md`，不能删除 patch/artifact。

### Week 19 Exit Gate

```text
Exp02 与 Exp01 只存在一个主要变量差异
compiler test/correctness/benchmark/profiler/ablation 完整
```

---

## Week 20：Experiment 03 与优化候选冻结

**Files:**

- Create: `results/experiments/exp03/README.md`
- Create: `compiler/tests/exp03.mlir`
- Create: `compiler/patches/exp03.patch`
- Create: `results/experiments/summary.csv`
- Update: `docs/failed_experiments.md`

### 工作日

- [ ] **周一：定义最后一个高信息量实验**

目标是区分仍竞争的两个解释，不是盲目继续搜索配置。

  - **项目化必读：** 用 [Nsight Compute 的 section/metric 语义](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules) 为两个解释各写一个互斥预测；若涉及 memory/occupancy，再读取 [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 对应章节。Exp03 只能修改一个固定 commit symbol，并必须产生能区分两解释的 metric/artifact。

- [ ] **周二：写 test 和实现**

  - **项目化必读：** 阅读固定 commit 的目标 op/interface、pass、registration 和最近邻 in-tree test；用 [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) 写正负例。先保存 red 输出，再实现最小 rewrite；layout 实验必须显式核对 `ConvertLayoutOp`/encoding verifier。

- [ ] **周三：运行 compiler tests**

  - **项目化必读：** 阅读固定 commit 的测试 harness 与目标 `RUN:` 行，运行新增 test、目标 pass 子集和 `test/TritonGPU` 相关 suite；记录命令、测试数、失败数和 source identity。Expected：新增与相关测试全部 PASS，不能用 `$CAPSTONE/compiler/tests` 镜像代替 in-tree test。

- [ ] **周四：运行 correctness**

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp03
for VARIANT in off on; do
  TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
    bash $CAPSTONE/scripts/run_correctness.sh \
    --output-dir "$EXP_ROOT/$VARIANT/correctness" \
    --output-json "$EXP_ROOT/$VARIANT/correctness/results.json"
done
```

  - **项目化必读：** 重读 [Phase 4 Correctness 必读资料](./Phase4_Baseline_Freeze.md#correctness-必读资料)；确认六组 shape、固定 seed、finite、容差均满足冻结契约，experiment identity 完整且 baseline identity 只读不变；不要求 experiment tree 等于 baseline tree。Expected：全部 PASS；失败则 Exp03 不得进入性能候选。

- [ ] **周五：运行代表 shape benchmark**

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp03
# 仍遍历六个 frozen config；本日先读取预注册代表 shape 的 paired 行作快速机制判别。
# 先分别完成 off/on `--precompile-only` 和 artifact/cache-key 断言。
bash $CAPSTONE/scripts/run_benchmark.sh \
  --config-dir "$CAPSTONE/results/baseline/configs" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$EXP_ROOT/cache" --schedule-file "$EXP_ROOT/paired/schedule.csv" \
  --rounds 3 --output-dir "$EXP_ROOT/paired/screen" \
  --output-csv "$EXP_ROOT/paired/screen/results.csv" \
  --raw-samples-dir "$EXP_ROOT/paired/screen/raw-samples" \
  --gpu-state-output "$EXP_ROOT/paired/screen/gpu-state.csv"
```

  - **项目化必读：** 重读 [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 和 Phase 4 测量协议；代表 shape 必须分别放大两个竞争解释，并包含中性对照。只使用冻结 config、100/500 ms、三轮 raw samples/P50。

### 周末

- [ ] **周六：完整六 shape benchmark/profiler**

```bash
EXP_ROOT=$CAPSTONE/results/experiments/exp03
# 严格执行全局模板列出的六 shape × 三 paired rounds，不使用隐式全-shape 捷径。
bash $CAPSTONE/scripts/run_benchmark.sh \
  --config-dir "$CAPSTONE/results/baseline/configs" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$EXP_ROOT/cache" --schedule-file "$EXP_ROOT/paired/schedule.csv" \
  --rounds 3 --output-dir "$EXP_ROOT/paired/final" \
  --output-csv "$EXP_ROOT/paired/final/results.csv" \
  --raw-samples-dir "$EXP_ROOT/paired/final/raw-samples" \
  --gpu-state-output "$EXP_ROOT/paired/final/gpu-state.csv"

# profiler 只覆盖 README 预注册的机制代表 shape；VARIANT=off/on 使用同一配置。
TRITON_EXP_FLAG=$VARIANT TRITON_CACHE_DIR="$EXP_ROOT/cache/$VARIANT" \
  bash $CAPSTONE/scripts/run_ncu.sh \
  --shape-tag "$SHAPE_TAG" --config-json "$CONFIG_JSON" \
  --disable-autotune --warmup-ms 100 --profile-iterations 1 \
  --sections "$NCU_SECTIONS" --kernel-name "$KERNEL_NAME" \
  --output "$EXP_ROOT/profiles/ncu/$VARIANT/${SHAPE_TAG}_${KERNEL_NAME}"
```

  - **项目化必读：** 阅读 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 和假设对应的 [memory/occupancy/scheduler/instruction sections](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#sections-and-rules)；六 shape 全部完成 correctness、off drift 和 paired benchmark，机制代表 shape 复用冻结 profiler 设置并保存 metric full name/unit、`.ncu-rep`、CSV。全六 shape 用同一 binary flag-off/on 做 ablation；profiler 代表 shape 只负责解释机制。

- [ ] **周日：冻结最佳候选**

Normal 模式比较 Exp01–03，选出一个进入泛化阶段；其余实验无论成功失败都写入记录。合法 Reduced 模式只比较预声明后实际完成的 experiment 集合，同时在 `summary.csv` 和 `docs/optimization_design.md` 明列未执行的 Exp02/Exp03、不能满足 Normal Gate 及结论范围缩小，不能伪造三实验比较。

  - **项目化必读：** 重读 [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#rooflines) 仅作为跨实验归因视图，并以具体 section metrics 为准。`summary.csv` 必须逐实验列出两层 identity、hypothesis、target symbol、预注册 shape、correctness、off drift、paired order、三轮 off/on P50、ratio/speedup/CV/noise floor、预测/观测 metric、全 shape ablation、适用/回退 shape 和单 shape/子集/全局结论；只冻结满足本页 `SUPPORTED` 判据的一个候选，其余写入 `failed_experiments.md`。Phase 5 不用几何平均下全局结论。

### Phase 5 Exit Gate

#### Normal Gate：`PASS`

```text
Exp01、Exp02、Exp03 三次实验分别在自身结果前完成预注册
三次实验均满足 identity、test、correctness、off-drift、paired benchmark、profiler、ablation 的完整证据契约
失败、REJECTED 和 INCONCLUSIVE 实验均保留
最佳候选具有 compiler test、correctness、ablation 和 profiler 证据
至少一个硬件指标按预测方向变化
```

满足 Normal Gate 时在 `progress.md` 标记 `PASS`。只有存在满足本页 `SUPPORTED` 判据的唯一冻结候选，才能进入 Phase 6；三次实验完成但没有有效候选时仍为 `FAIL`。

#### Reduced Gate：`PASS-REDUCED`

```text
在看到任何 Phase 5 experiment 结果前已按总索引记录启用 Reduced 模式
至少一次预注册的完整实验满足与 Normal Gate 相同的单次证据契约
失败、REJECTED 和 INCONCLUSIVE 实验均保留
唯一冻结候选满足本页 SUPPORTED 判据
summary 和最终报告披露未执行实验、未满足 Normal Gate 及结论范围
```

满足 Reduced Gate 时只能在 `progress.md` 标记 `PASS-REDUCED`。该状态只有在存在满足同等 correctness、identity、off-drift、paired benchmark、profiler 和 ablation 要求的唯一冻结候选时，才允许进入 Phase 6；否则为 `FAIL`。禁止在实验失败、收益不足或看过结果后切换到 Reduced Gate。

[进入 Phase 6：Generalization & Regression →](./Phase6_Generalization_Regression.md)
