# Phase 6：Generalization & Regression

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 21–22（2026-12-02 至 2026-12-15）
- 计划工时：22h

## 前置条件

[Phase 5 Exit Gate](./Phase5_Performance_Optimization.md#phase-5-exit-gate) 已通过，且只有 Phase 5 冻结的唯一最佳候选能进入本阶段。

## 可验证阶段目标

- 验证优化候选在邻近 shape、边界 shape 与目标 dtype 上的 correctness。
- 建立覆盖 compiler 行为和性能退化的 regression harness，并用真实 red/green 运行证明 gate 有效。
- 同时报告 geometric mean 与 worst-case regression；任何跨 shape 算术平均都不能替代逐 shape 硬门槛或掩盖退化。
- 冻结最终 patch、reproducer、已知限制与可由 compiler 静态判断的支持范围。

## 本阶段产出

- 邻近/边界 shape correctness 与性能矩阵。
- compiler regression tests 和性能 regression harness。
- `best`、geometric mean、worst regression、variance、compile-time/code-size delta 摘要。
- 最终 patch、reproducer、支持范围与限制说明。

## 证据契约与资料边界

本页外链已于 **2026-07-14** 在线核验。滚动版 Triton 文档用于解释 API 语义；实际 helper 签名、source path 和 compiler test 必须以 `compiler/baseline_commit.txt` 固定的 Triton commit 为准，若不一致，用 `rg` 定位该 commit 的等价接口并在结果 metadata 中记录。

Phase 6 **继承而不重写** [Phase 5 的 baseline/experiment identity、cache 隔离、paired 顺序与冻结统计](./Phase5_Performance_Optimization.md#证据契约与资料边界)：

- `results/baseline/` 是 Phase 4 immutable baseline，Phase 6 只读；不得覆盖 CSV、config、manifest、hash 或 cache。
- candidate identity **直接等于** Phase 5 已选最佳候选冻结的 `experiment_identity`，不得在 Phase 6 再拼接 patch hash、build hash 或生成另一个“等价”身份。进入 Week 21 后把该 identity 值和它已冻结的 baseline/patch/tree/build component manifests 复制到 `candidate-identity.json`；component manifests 只用于逐项核验，不改变 identity 定义。源码、build、flag、加载路径或任一 component hash 变化都意味着原候选失效：回到 Phase 5 重建完整 experiment identity，并重跑 compiler red/green、correctness、off drift、paired benchmark 与 profiler 后才能重新进入 Phase 6。
- flag-off/flag-on 使用同一 candidate binary，只由已登记 feature flag 区分；各自使用独立 cache，先 precompile/warmup，再进入 paired timing，compile time 不得混入 latency。
- 逐 shape 沿用 `R1: off→on, R2: on→off, R3: off→on`，下一 shape 反转起始 variant；保存 raw samples、round order、GPU state 和过滤原因。不能先跑完全部 off 再跑全部 on。
- 每个新增 shape 先跑 correctness；有 frozen baseline center 的原六 shape还必须通过 off drift gate。新增 shape 的 off 是 paired denominator，不得伪装成 Phase 4 frozen baseline。预注册 `supported` 与 `fallback` 后不得依据性能结果改标。
- Phase 5 的 P50、`pstdev`/CV、noise floor、符号稳定性和 `SUPPORTED/INCONCLUSIVE/REJECTED` 规则继续有效。Phase 6 新增跨 shape 聚合，但不改变逐 shape 判定。

```text
ratio_shape = optimized_p50_us / paired_off_p50_us
performance_ratio_shape = paired_off_p50_us / optimized_p50_us
geomean_ratio = statistics.geometric_mean(ratio_shape for every preregistered supported shape)
geomean_speedup_pct = (1 / geomean_ratio - 1) * 100
worst_ratio = max(ratio_shape for every preregistered supported shape)
worst_regression_pct = max(0, (worst_ratio - 1) * 100)
```

`week21-shapes.json` 在运行前把每个 canonical、neighbor、boundary shape 预注册为 `supported` 或 `fallback`。所有 `supported` shape 必须 correctness PASS、拥有完整 paired 数据且 `performance_ratio_shape >= 0.80`；canonical six 还必须满足至少四组 `>= 0.90`。静态 predicate 判定为 `fallback` 的 shape 不进入 geomean 或任何优化性能宣称，但必须验证 flag-on 正确走未优化路径并通过 fallback correctness；若错误启用优化则 gate 失败。

`statistics.geometric_mean()` 只接受正值；supported 集为空、零、负值、NaN/Inf、漏 supported shape、correctness 失败或机器状态过滤后未补跑的样本必须使汇总失败并非零退出，不能静默删除后重算。速度比是乘法量，因此用 ratio 的几何平均；**不得对各 shape 的 speedup 百分比取算术平均**。无论 geometric mean 多好，`worst_ratio` 和所有 supported shape 的逐 shape 80% 下限都独立判定，不能以平均收益抵消单点退化。

### 官方资料到证据的映射

- [Python `statistics` 文档](https://docs.python.org/3/library/statistics.html)：核对 `geometric_mean` 的正数/非空输入约束、`median`、`mean` 与 `pstdev` 的当前 API。用于重算 paired P50、CV、geomean ratio；算术平均仅可用于 Phase 5 已定义的 CV 分母，不能作为跨 shape 性能结论。
- [Triton `triton.testing`](https://triton-lang.org/main/python-api/triton.testing.html) 与 [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)：核对 warmup、rep、quantiles/`return_mode="all"` 和 `assert_close`；harness 必须保存原始 timing 或明确 quantile，不能把 helper 默认 mean 当作冻结 P50。
- [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html) 与 [skip/xfail](https://docs.pytest.org/en/stable/how-to/skipping.html)：若 harness 实际使用 pytest，shape/dtype 必须用稳定 id 参数化；已知未支持项只能用带 issue/reason 的 strict xfail，XPASS 必须失败，不能用 xfail 隐藏新 correctness/performance regression。
- [Triton Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 与 [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html)：核对 `cdiv` 后尾块、M/N/K 非整除点、load/store mask、masked load 的 `other` 和 block-pointer `boundary_check`；边界 shape 必须验证尾块输出与越界保护，不能只验证整除 shape。
- [CUDA Programming Guide 的 grid/thread indexing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-and-grid-index-intrinsics)：核对 grid 覆盖与 `threadIdx/blockIdx/blockDim/gridDim` 映射；ceil-grid 多出来的线程必须被边界条件保护。CUDA 说明用于交叉检查边界模型，Triton kernel 的实际正确性仍由固定 commit 的 mask/lowering 与 reference output 证明。

## Week 21：邻近 Shape 与边界泛化

**Files:**

- Create: `benchmark/generalization_shapes.py`
- Create: `benchmark/tests/test_generalization_shapes.py`
- Create: `benchmark/tests/test_run_benchmark_generalization.py`
- Create: `scripts/verify_week21_preregistration.sh`
- Create: `results/regression/week21-shapes.json`
- Create: `results/regression/week21-configs/`
- Create: `results/regression/week21-configs.sha256`
- Create: `results/regression/week21-schedule.csv`
- Create: `results/regression/week21-preregistration.sha256`
- Create: `results/regression/candidate-identity.json`
- Create: `results/regression/candidate-components.sha256`
- Create: `results/regression/thresholds.json`
- Create: `results/regression/progress.log`（只追加 `PREREG_COMMIT` 与结果 commit）
- Create: `results/regression/prereg-remote.json`（受保护外部 registry 记录的只读本地副本）
- Create: `results/regression/benchmark-interface-contract.json`
- Create: `results/regression/benchmark-interface-red.json`
- Create: `results/regression/benchmark-interface-red-green.txt`
- Create: `results/regression/week21-correctness.json`
- Create: `results/regression/week21-paired/`
- Create: `results/regression/week21-generalization.csv`
- Create: `results/regression/week21-summary.json`
- Create: `docs/optimization_scope.md`
- Update: `scripts/run_benchmark.sh`（保持 Phase 5 默认六 shape/36 rows；增加显式 generalization manifest 与 protected prereg anchor 模式）

### 工作日

- [ ] **周一：定义邻近 shape**

shape 集合必须在任何 exploratory run 前由以下固定算法一次性生成，**不得围绕已知获益区间选点**：

```text
canonical（固定输入顺序）:
0: M32_N4096_K4096
1: M128_N4096_K4096
2: M512_N4096_K4096
3: M512_N16384_K4096
4: M512_N4096_K16384
5: M4096_N4096_K4096

每个 canonical i 恰生成一个 neighbor:
axis = [M, N, K][i mod 3]
delta = -1 if i is even else +1
只修改该 axis，其余维度不变

boundary（固定书写顺序）:
M1_N4096_K4096
M33_N4096_K4096
M128_N4095_K4096
M512_N4096_K4097
M511_N16384_K4096
M512_N4097_K16384
```

输出顺序固定为 canonical 原顺序、六个 neighbor 按 canonical index 顺序、boundary 书写顺序；按完整 `(M,N,K,dtype)` key 保留首次出现项并拒绝重复 tag。本矩阵已知无重复，expected cardinality 必须严格为 `6 + 6 + 6 = 18`。`test_generalization_shapes.py` 必须断言完整 18 项、顺序、neighbor 公式、boundary 常量与去重规则。`supported|fallback` 只能由预先写死的 compiler 静态 predicate 分类，不能读取 timing、profiler 或已有结果。

运行前把精确 shape/classification、dtype、seed、reference 写入 `week21-shapes.json` 与 `docs/optimization_scope.md`；从 Phase 5 已选最佳候选复制冻结 `experiment_identity` 和 exact patch/tree/build/import-resolved/cache/artifact component manifests，不能重新计算拼接 identity。`thresholds.json` 同时在 Week 21 创建并固定断言：`correctness_required=true`、所有 supported `off/on >= 0.80`、canonical six 至少 4 组 `off/on >= 0.90`、geomean 只使用全部 supported 的正 `on/off` ratio、fallback 不进入性能聚合、禁止算术平均覆盖单点 gate。

```bash
set -euo pipefail
test -n "$SELECTED_EXP_ROOT"
test -f "$SELECTED_EXP_ROOT/README.md"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode preregister \
  --selected-experiment-root "$SELECTED_EXP_ROOT" \
  --candidate-identity-output "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifest-output "$CAPSTONE/results/regression/candidate-components.sha256" \
  --shape-manifest-output "$CAPSTONE/results/regression/week21-shapes.json" \
  --config-dir-output "$CAPSTONE/results/regression/week21-configs" \
  --config-manifest-output "$CAPSTONE/results/regression/week21-configs.sha256" \
  --schedule-output "$CAPSTONE/results/regression/week21-schedule.csv" \
  --thresholds-output "$CAPSTONE/results/regression/thresholds.json"

$TRITON_VENV/bin/python -m pytest -q \
  $CAPSTONE/benchmark/tests/test_generalization_shapes.py
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode assert-preregistration \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --thresholds "$CAPSTONE/results/regression/thresholds.json" \
  --expected-cardinality 18

# week21-configs.sha256 中的路径均相对 $CAPSTONE，且 config 文件名不含空白。
(
  cd "$CAPSTONE"
  sha256sum -c results/regression/week21-configs.sha256
  {
    sha256sum \
      results/regression/candidate-identity.json \
      results/regression/candidate-components.sha256 \
      results/regression/week21-shapes.json \
      results/regression/week21-configs.sha256 \
      results/regression/week21-schedule.csv \
      results/regression/thresholds.json
    while read -r HASH CONFIG; do
      test -n "$HASH" && sha256sum "$CONFIG"
    done < results/regression/week21-configs.sha256
  } > results/regression/week21-preregistration.sha256
  sha256sum -c results/regression/week21-preregistration.sha256
)

# 这是未来执行计划要求的专用 preregistration commit；当前文档编辑不执行 commit。
git -C "$CAPSTONE" add -- \
  results/regression/candidate-identity.json \
  results/regression/candidate-components.sha256 \
  results/regression/week21-shapes.json \
  results/regression/week21-configs.sha256 \
  results/regression/week21-configs/*.json \
  results/regression/week21-schedule.csv \
  results/regression/thresholds.json \
  results/regression/week21-preregistration.sha256 \
  scripts/verify_week21_preregistration.sh
git -C "$CAPSTONE" commit -m "capstone: freeze week21 preregistration"
PREREG_COMMIT=$(git -C "$CAPSTONE" rev-parse HEAD)
IDENTITY_DIGEST=$($TRITON_VENV/bin/python \
  $CAPSTONE/benchmark/generalization_shapes.py \
  --mode print-protected-identity-digest \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json")
[[ "$IDENTITY_DIGEST" =~ ^[0-9a-f]{64}$ ]]
PREREG_TAG="prereg/week21/$IDENTITY_DIGEST"

# 远端与 registry 必须通过组织 allowlist、禁止 tag force/update 的策略检查，否则 Phase 6 到此停止。
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode assert-protected-anchor-services \
  --remote "$PROTECTED_PREREG_REMOTE" \
  --registry "$PROTECTED_PREREG_REGISTRY"
git -C "$CAPSTONE" tag -s "$PREREG_TAG" "$PREREG_COMMIT" \
  -m "freeze Week 21 preregistration for $IDENTITY_DIGEST"
git -C "$CAPSTONE" push "$PROTECTED_PREREG_REMOTE" \
  "refs/tags/$PREREG_TAG:refs/tags/$PREREG_TAG"
git -C "$CAPSTONE" ls-remote --tags "$PROTECTED_PREREG_REMOTE" \
  "refs/tags/$PREREG_TAG" "refs/tags/$PREREG_TAG^{}"
git -C "$CAPSTONE" fetch "$PROTECTED_PREREG_REMOTE" \
  "refs/tags/$PREREG_TAG:refs/tags/$PREREG_TAG"
git -C "$CAPSTONE" tag -v "$PREREG_TAG"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode publish-prereg-anchor \
  --remote "$PROTECTED_PREREG_REMOTE" \
  --registry "$PROTECTED_PREREG_REGISTRY" \
  --tag "$PREREG_TAG" --tag-object "$(git -C "$CAPSTONE" rev-parse "$PREREG_TAG")" \
  --peeled-commit "$(git -C "$CAPSTONE" rev-parse "$PREREG_TAG^{commit}")" \
  --local-copy "$CAPSTONE/results/regression/prereg-remote.json"
printf 'PREREG_COMMIT=%s\n' "$PREREG_COMMIT" >> \
  "$CAPSTONE/results/regression/progress.log"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

`verify_week21_preregistration.sh` 是只读 verifier，不得接受 commit/tag 环境覆盖，也不得重生成根清单。它从 Phase 5 protected candidate record 取得 `experiment_identity` digest，唯一派生 `prereg/week21/<identity>`；随后用 `git ls-remote` 查询受保护远端的 annotated tag object 与 `^{}` peeled commit，fetch 后执行 `git tag -v`，并与外部 CI artifact/registry 中不可变记录的 remote URL、tag name、tag object、peeled commit、签名 fingerprint 全字段比较。只有该远端 peeled commit 才是内部 `PREREG_COMMIT`。然后执行 ancestry、`git show`、`git diff --exit-code`、committed root hash 和 18-shape/threshold assertion。`run_correctness.sh`、`run_benchmark.sh` 与 `run_regression.sh` 也必须在入口/出口内部调用同一解析逻辑；`prereg-remote.json` 只作 external record locator/cache，不能作为 commit 真值。若没有可禁止 force/update 的 protected remote/registry、签名无效、远端 tag 更新或 external record 不一致，Phase 6 formal run 必须以 infrastructure error 停止；本地文件和环境变量都不能另选 commit。这同时阻断 accidental rewrite 与看到结果后的 post-hoc anchor 替换。

  - **项目化必读：** 阅读 [Triton Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 中 M/N grid、K 尾块与 mask 的写法，再读固定 commit 的实际 matmul/reproducer 和 chosen config。逐轴写出“该点跨越的 tile/wave/mask 边界”和预期触发的 compiler 条件；资料输出是预注册矩阵，不是通用摘要。
  - **Expected：** 在任何 correctness/benchmark/exploratory 结果前，专用 prereg commit 已被 Phase 5 protected identity 唯一命名的 signed annotated tag 锚定并推送至禁止 force/update 的 protected remote；外部 registry 已保存 remote URL、tag object、peeled commit 与签名，remote verifier PASS。没有该外部不可变锚点不得正式开始 Phase 6。根清单和 thresholds 后续只读；任何 post-hoc identity/classification/config/order/threshold 或 tag 更新均失败，所有结果 commit 必须是远端 peeled prereg commit 的后代。

- [ ] **周二：运行全部 correctness**

对预注册 shape × dtype × seed 全量比较 reference，检查 NaN/Inf、最大绝对/相对误差、尾块和 guard region。所有 supported shape 必须在优化路径 correctness PASS；fallback shape 必须证明静态 predicate 关闭优化并在 fallback 路径 correctness PASS。任一失败都阻止 gate 通过并保留失败记录。

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode correctness --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --output "$CAPSTONE/results/regression/week21-correctness.json"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

  - **项目化必读：** 阅读 [Triton Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) 的 `mask = offsets < n_elements`，以及 [`tl.load`](https://triton-lang.org/main/python-api/generated/triton.language.load.html) 的 `mask/other/boundary_check` 语义；把 M/N/K 每个尾块对应到实际 load/store mask，并增加至少一个 guard-region canary。若使用 pytest，再按 [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html) 为每个 shape/dtype 生成稳定 test id。
  - **Expected：** 消费输入前统一 preregistration 校验 PASS；任何 identity、shape classification、config/hash 或 paired order 事后变化均在产生 correctness 结果前失败。所有 supported 组合在优化路径 PASS；所有 fallback 组合确认未启用优化且 fallback correctness PASS。已知超出两类契约的组合要么明确 fail 并退出非零，要么以 strict xfail 绑定 issue/reason，绝不被计作支持。

- [ ] **周三：运行 baseline 三轮**

这里的 baseline 是同一 candidate identity、同一 binary 的 flag-off paired side；原六 shape 还需对 Phase 4 frozen baseline 做 drift gate。Phase 5 接口原本硬性要求冻结六 config、`configs.sha256` 和 36 行 schedule，不能直接用于新增 shape。本日先按 TDD 扩展 `run_benchmark.sh`：默认/Phase 5 模式语义完全不变；只有同时显式传 `--prereg-anchor`、`--shape-manifest`、`--config-manifest`、`--candidate-identity` 与 `--component-manifests` 才进入 generalization 模式。该模式只接受预注册 supported 集，逐 config 校验 hash、schema、shape tag 与 manifest 一一对应，要求 `expected_rows = supported_count * rounds * 2 variants`；fallback、重复/缺失 config、未知 tag、hash/schema 错误或 row 数不符必须非零退出。

入口在 precompile/warmup/timing 前以及输出关闭后，都必须核对 live Triton commit、compiler tree、loaded build/shared objects、editable import resolved path、feature flag、off/on cache root/key 和 artifact hash 与 candidate/component manifests 完全一致。每个 CSV row 必须写 `experiment_identity`、`compiler_tree_hash`、`build_hash`、`import_resolved_path_hash`、`cache_identity`、`artifact_hash`、variant、shape、round 与实际 order；缺字段或前后 identity 漂移按 infrastructure error 退出，不能产生 latency 结论。

先写 `test_run_benchmark_generalization.py` 并保存 red 结果，再实现接口并保存 green 结果。测试至少证明：旧模式仍只接受六个冻结 config + `configs.sha256` + 三轮 36 rows；generalization 模式按 supported 数动态计算 rows；fallback 混入和 config hash/schema 错误都失败。不得单独跑完全部 off；接口测试通过后再 dry-run 核验 Week 21 三轮 off/on schedule、两套 cache 和输出路径。整个过程只写 `results/regression/`，不写入 `results/baseline/`。

```bash
set -euo pipefail
set +e
$TRITON_VENV/bin/python -m pytest -q \
  $CAPSTONE/benchmark/tests/test_run_benchmark_generalization.py 2>&1 | \
  tee $CAPSTONE/results/regression/benchmark-interface-red-green.txt
RED_RC=${PIPESTATUS[0]}
set -e
test "$RED_RC" -eq 1
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode assert-failure-report \
  --input "$CAPSTONE/results/regression/benchmark-interface-red.json" \
  --failure-kind gate --gate config_membership \
  --shape M1_N4096_K4096 --reason fallback_config_forbidden \
  --fixture-id iface-fallback-in-config-v1

# 实现 run_benchmark.sh 的显式 generalization manifest 模式后执行 green：
$TRITON_VENV/bin/python -m pytest -q \
  $CAPSTONE/benchmark/tests/test_run_benchmark_generalization.py 2>&1 | \
  tee -a $CAPSTONE/results/regression/benchmark-interface-red-green.txt
```

统一 CLI 退出码契约为：`0=PASS`、`1=已识别的 gate/contract failure`、`>=2=infrastructure/schema/identity/hash error`。`benchmark-interface-contract.json` 保存 legacy/generalization 两种模式的输入、supported count、rounds、expected/actual rows、live identity/component 校验、每个 test command/exit code；red/green 两次运行必须分段保存，不能只保留最终 PASS。接口 red fixture 固定 `fixture_id=iface-fallback-in-config-v1`、唯一 `shape=M1_N4096_K4096`、`gate=config_membership`、`reason=fallback_config_forbidden`；测试必须断言 `rc==1`，并解析 `benchmark-interface-red.json` 的 `failure_kind=gate`、gate/shape/reason/fixture_id。缺文件或 identity/hash 错误必须断言 `rc>=2`，不得误当 expected red。

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
bash $CAPSTONE/scripts/run_benchmark.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --config-dir "$CAPSTONE/results/regression/week21-configs" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --config-manifest "$CAPSTONE/results/regression/week21-configs.sha256" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$CAPSTONE/results/regression/cache" \
  --schedule-file "$CAPSTONE/results/regression/week21-schedule.csv" \
  --rounds 3 --dry-run \
  --output-dir "$CAPSTONE/results/regression/week21-paired" \
  --output-csv "$CAPSTONE/results/regression/week21-generalization.csv" \
  --raw-samples-dir "$CAPSTONE/results/regression/week21-paired/raw-samples" \
  --gpu-state-output "$CAPSTONE/results/regression/week21-paired/gpu-state.csv"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

`--dry-run` 必须只读已冻结 schedule；如果实现会生成、排序或规范化 schedule，必须把该动作提前到周一冻结之前。冻结后的 dry-run/formal run 均不得改写 schedule，命令结束后再次执行同一 `sha256sum -c`。

  - **项目化必读：** 阅读 [`triton.testing.do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 的 warmup、rep、quantiles 和 raw return mode，再复核 Phase 5 的 paired schedule/statistics。明确脚本究竟保存 raw timings 还是 `[0.2, 0.5, 0.8]` quantiles；若 helper 默认为 mean，必须显式覆盖，不能把默认值标成 P50。
  - **Expected：** TDD interface red 的 rc/structured JSON 精确匹配固定 fixture，infrastructure fixture 返回 `>=2`；green 后 legacy fixture 仍为六 shape/36 rows。generalization dry-run 前后 git-anchored verifier 均 PASS，只枚举所有 supported shape，实际 rows 等于 `supported_count * 3 * 2`，fallback 不出现；live commit/tree/build/import/cache/artifact 与 candidate 完全一致且 row schema 包含全部 identity hash。任何 identity/classification/config/order/threshold 变化立即失败。

- [ ] **周四：运行 optimized 三轮**

使用同一 binary 和 Phase 5 已验证接口，按预注册 schedule 对 Week 21 **全部 supported shape** 完成三轮 off/on paired 测量，显式生成 `week21-generalization.csv`；off/on 使用独立 cache。不能重排 shape、替换失败样本、测量 fallback shape 或让 on 命中 off cache。

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
bash $CAPSTONE/scripts/run_benchmark.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --config-dir "$CAPSTONE/results/regression/week21-configs" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --config-manifest "$CAPSTONE/results/regression/week21-configs.sha256" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
  --cache-root "$CAPSTONE/results/regression/cache" \
  --schedule-file "$CAPSTONE/results/regression/week21-schedule.csv" \
  --rounds 3 \
  --output-dir "$CAPSTONE/results/regression/week21-paired" \
  --output-csv "$CAPSTONE/results/regression/week21-generalization.csv" \
  --raw-samples-dir "$CAPSTONE/results/regression/week21-paired/raw-samples" \
  --gpu-state-output "$CAPSTONE/results/regression/week21-paired/gpu-state.csv"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

  - **项目化必读：** 复读 [Triton `do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) 并检查固定 commit 的 cache manager/env；把 warmup/rep、quantile、cache key、variant、round 和 order 写入 CSV。该阅读直接用于证明两侧计时窗口一致且 compile time 未混入 latency。
  - **Expected：** `week21-generalization.csv` 覆盖每个 supported shape × 三轮 × off/on；每行 on 数据都能一一配对到同 shape/round 的 off 数据，并带完整 experiment/build/import/cache/artifact identity。入口计时前和输出后 git-anchored verifier 与 live component 校验均 PASS；缺行、重复行、fallback 混入、identity/cache 交叉或机器状态不合格且未补跑均使整轮失败。

- [ ] **周五：计算 speedup distribution**

从 raw samples 重算每轮 P50、逐 shape ratio/speedup/CV/sign stability，再只对全部预注册 supported shape 的正 ratio 计算 geometric mean 和 worst regression。输出 best、median shape、geomean、worst、分位数、每个失效原因和 fallback correctness 清单；fallback 不进入性能聚合，不得只输出一个平均 speedup。

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/generalization_shapes.py \
  --mode analyze --input "$CAPSTONE/results/regression/week21-generalization.csv" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --require-all-supported \
  --summary "$CAPSTONE/results/regression/week21-summary.json"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

  - **项目化必读：** 阅读 [Python `statistics`](https://docs.python.org/3/library/statistics.html) 的 `geometric_mean`、`median`、`mean` 和 `pstdev` 定义；在脚本单测中用一组“多数小收益 + 一个大退化”的正 ratio，证明算术平均百分比可能看似良好，而 worst gate 仍失败。geomean 输入必须是 `on/off` ratio，不是可能为负的 speedup 百分比。
  - **Expected：** analyze 消费前统一 preregistration 校验 PASS；任何 identity/classification/config/order 变化先失败。汇总可由 CSV + shape manifest 独立重算，NaN/Inf/非正 ratio、漏 supported shape 或任一 supported `performance_ratio < 0.80` 非零退出；报告显式写明“fallback 不进入 geomean；geomean 不覆盖 worst/逐 shape gate”。

### 周末

- [ ] **周六：分析边界在哪里反转**

定位从获益到退化的 shape 区域，并关联 CTA waves、register、occupancy 或 memory 指标。只允许对预注册矩阵做分段，不允许看到结果后删除反转点；边界两侧至少各保留一个 correctness/performance 点。

  - **项目化必读：** 阅读 [CUDA grid/thread indexing](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#thread-and-grid-index-intrinsics) 与 [CUDA Best Practices 的 thread/block heuristics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics)，再用 Phase 5 已冻结的 NCU section 定义核对 CTA waves、occupancy、register/memory 指标。输出“shape 边界 → grid/tail-mask 变化 → metric 变化 → latency 反转”的证据链。
  - **Expected：** 每个反转区间都含边界两侧原始数据和 profiler/artifact anchor；无法归因标为 `INCONCLUSIVE`，不能猜测成 compiler 支持条件。

- [ ] **周日：写适用条件**

条件必须能由 compiler 在编译期判断，例如静态 M/N/K、layout、dtype，而不是“某些情况更快”。同时写出回退路径、已知未支持边界和 strict xfail 的移除条件。

  - **项目化必读：** 阅读固定 commit 中目标 pass 的 match/rewrite 条件、相关 op/encoding verifier 与最近邻 compiler tests；再用 [pytest skip/xfail](https://docs.pytest.org/en/stable/how-to/skipping.html) 区分“环境性 skip”和“带 issue 的已知失败”。把每条数据边界翻译为可测试的 IR/shape predicate，并为 predicate 真/假各给一个 reproducer。
  - **Expected：** `docs/optimization_scope.md` 中每条支持条件都可由 IR 静态判断并有正/负 compiler test；性能观测本身不作为 match predicate。

### Week 21 Exit Gate

```text
shape 由固定 6 canonical + 6 deterministic neighbor + 6 boundary 生成，顺序/cardinality=18 自测通过
优化适用条件可由 IR/shape 判断
退化边界有数据
全部预注册 shape 有 supported/fallback 静态分类、correctness 结论且尾块/mask 被覆盖
全部 supported canonical/neighbor/boundary shape 的 performance ratio >= 0.80
fallback shape 正确关闭优化并通过 fallback correctness，不进入 geomean 或性能宣称
geometric mean 与 worst regression 可由所有 supported shape 的原始 paired 数据重算
identity/components、18-shape classification、config/hash、paired order 与 thresholds 在任何结果前进入专用 PREREG_COMMIT
所有 consumer 前后验证 git ancestry/show/diff/hash/spec，任何事后修改或非后代 result commit 均失败
baseline identity、candidate identity、off/on cache 与 Phase 5 统计契约未被污染
```

---

## Week 22：Regression Harness、最终 Patch 与范围冻结

**Files:**

- Create: `benchmark/regression.py`
- Create: `scripts/run_regression.sh`
- Read/Verify: `results/regression/thresholds.json`（Week 21 prereg commit 冻结，Week 22 禁止修改）
- Create: `results/regression/red-input.csv`
- Create: `results/regression/red/`
- Create: `results/regression/red/failure.json`
- Create: `results/regression/green/`
- Create: `results/regression/final-correctness/results.json`
- Create: `results/regression/final/`
- Create: `results/regression/final-summary.json`
- Create: `results/regression/final-manifest.sha256`
- Create: `compiler/patches/final.patch`
- Create: `results/regression/final.csv`
- Finalize: `docs/optimization_scope.md`
- Update: `docs/failed_experiments.md`
- Update: `scripts/run_correctness.sh`（保留 Phase 4 `--output-dir/--output-json`，增加显式只读输入 `--prereg-anchor/--shape-manifest/--candidate-identity/--component-manifests`）

### 工作日

- [ ] **周一：实现 correctness 和性能 gate**

regression 首先只读 Week 21 prereg commit 中的 `thresholds.json` 并断言规范常量完全一致；Week 22 不得创建、重写或放宽 thresholds。性能硬标准为：Phase 4 canonical six 中至少四组达到 baseline 90%；**所有预注册为 supported 的 canonical、neighbor、boundary shape** 的 `performance_ratio = off_us / on_us` 均不得低于 80%。fallback shape 必须正确关闭优化并通过 fallback correctness，不进入 geomean、4/6 或性能宣称。任何失败立即返回非零，不能拿新增点替换原六组，也不能把低于 80% 的点重新标成 fallback。

```text
supported_correctness_pass = all(supported cases PASS on optimized path)
fallback_correctness_pass = all(fallback cases disable optimization and PASS)
at_least_four_90 = count(class == supported and off_us / on_us >= 0.90
                         for canonical six) >= 4
all_supported_at_least_80 = all(off_us / on_us >= 0.80 for supported shapes)
performance_pass = at_least_four_90 and all_supported_at_least_80
exit_zero = supported_correctness_pass and fallback_correctness_pass \
            and performance_pass and identity/cache/schema_pass
```

  - **项目化必读：** 阅读 [Python `statistics`](https://docs.python.org/3/library/statistics.html) 并复核 Phase 5 ratio/noise 公式；阅读 [Triton `assert_close`/testing API](https://triton-lang.org/main/python-api/triton.testing.html) 确认 correctness helper 的容差接口。逐条把 `thresholds.json` 的 supported/fallback、canonical 4/6、all-supported 80% 字段映射到 shape manifest、原始 CSV 和非零 exit path；geomean 只做汇总，不能替代 4/6 与 all-supported worst 下限。
  - **Expected：** `thresholds.json` 与 `git show <remote-verified-peeled-commit>:results/regression/thresholds.json` 一致，且 correctness/geomean/0.80/4-of-6@0.90 常量 assertion PASS；supported 与 fallback correctness 在 performance 之前执行。任一 required case、fallback 路由、schema、identity、cache、canonical 4/6 或任一 supported 80% 条件失败都返回非零并指出具体 shape/字段。

- [ ] **周二：实现 CSV/JSON 输出并做预期失败**

临时使用明显不利配置，确认 regression harness 返回非零；恢复候选后确认通过，保存 red/green 结果。这个 expected failure 是对 harness 的负向自测，不等同于把真实性能退化标为 pytest xfail。

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/regression.py \
  --mode make-red-input \
  --input "$CAPSTONE/results/regression/week21-generalization.csv" \
  --output "$CAPSTONE/results/regression/red-input.csv" \
  --fixture-id perf-supported-min-v1 \
  --shape M32_N4096_K4096 --forced-performance-ratio 0.79 \
  --reason forced_supported_ratio_below_0_80
set +e
bash $CAPSTONE/scripts/run_regression.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --input-csv "$CAPSTONE/results/regression/red-input.csv" \
  --thresholds "$CAPSTONE/results/regression/thresholds.json" \
  --failure-json "$CAPSTONE/results/regression/red/failure.json" \
  --output-dir "$CAPSTONE/results/regression/red"
RED_RC=$?
set -e
test "$RED_RC" -eq 1
$TRITON_VENV/bin/python $CAPSTONE/benchmark/regression.py \
  --mode assert-failure-report \
  --input "$CAPSTONE/results/regression/red/failure.json" \
  --failure-kind gate --gate performance_ratio \
  --shape M32_N4096_K4096 \
  --reason forced_supported_ratio_below_0_80 \
  --fixture-id perf-supported-min-v1
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
bash $CAPSTONE/scripts/run_regression.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --input-csv "$CAPSTONE/results/regression/week21-generalization.csv" \
  --thresholds "$CAPSTONE/results/regression/thresholds.json" \
  --output-dir "$CAPSTONE/results/regression/green"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

  - **项目化必读：** 若测试用 pytest，阅读 [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html) 和 [skip/xfail](https://docs.pytest.org/en/stable/how-to/skipping.html) 的 strict/XPASS 语义；用稳定 shape id 参数化 parser/gate 单测，并让故意坏 fixture 走普通失败。xfail 仅记录尚未修复且有 issue 的 correctness case，不能吞掉 red harness 的非零退出。
  - **Expected：** red/green 都读取完全相同路径与 hash 的 prereg 生产 `thresholds.json`；red 只在独立 synthetic CSV 中把唯一 fixture shape 的合法 identity row ratio 改为 0.79，其余字段/row 不变。red 必须 `rc==1` 且 structured JSON 精确报告 `performance_ratio` gate/shape/reason/fixture；缺文件、hash/identity、schema 或 remote verifier 错误必须 `>=2`。green 使用原始 input 返回 0；前后 remote-anchor verifier PASS。

- [ ] **周三：清理 compiler diff 并运行 test 子集**

移除 debug print、无关格式变化和未使用 flag；不做无关重构。运行目标 compiler tests 和相关 Triton 子集。清理后核验 exact patch/tree/build component manifests 是否仍与 Phase 5 已选最佳 `experiment_identity` 的冻结组件完全一致；若任一组件变化，当前 candidate identity 失效，必须回到 Phase 5 重建完整 experiment identity 并重跑全部 Phase 5 验证，再从 Week 21 开始，不能在 Phase 6 局部重算 hash 后复用旧性能数据。

  - **项目化必读：** 阅读固定 commit 目标 pass 最近邻 `test/TritonGPU` 的 `RUN:`/FileCheck 习惯，以及 [pytest parameterization](https://docs.pytest.org/en/stable/how-to/parametrize.html) 中稳定 case id。建立“支持 predicate 真/假、非整除 mask、回退路径”的最小 test matrix；测试只约束 compiler 行为/correctness，不写 latency FileCheck。
  - **Expected：** final diff 只含候选逻辑、feature flag/支持 predicate、必要 tests 与 harness 接口；目标 compiler tests 和相关 Triton 子集 PASS，identity 与随后数据完全一致。

- [ ] **周四：运行完整 correctness 与正式 regression**

```bash
set -euo pipefail
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
$TRITON_VENV/bin/python $CAPSTONE/benchmark/regression.py \
  --mode assert-threshold-contract \
  --thresholds "$CAPSTONE/results/regression/thresholds.json"
bash $CAPSTONE/scripts/run_correctness.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --output-dir "$CAPSTONE/results/regression/final-correctness" \
  --output-json "$CAPSTONE/results/regression/final-correctness/results.json"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
bash $CAPSTONE/scripts/run_regression.sh \
  --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
  --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
  --thresholds "$CAPSTONE/results/regression/thresholds.json" \
  --input-csv "$CAPSTONE/results/regression/week21-generalization.csv" \
  --output-dir "$CAPSTONE/results/regression/final" \
  --output-csv "$CAPSTONE/results/regression/final.csv" \
  --summary-json "$CAPSTONE/results/regression/final-summary.json"
bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
  --capstone "$CAPSTONE" \
  --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
  --protected-remote "$PROTECTED_PREREG_REMOTE" \
  --external-registry "$PROTECTED_PREREG_REGISTRY"
```

`run_correctness.sh` 沿用 Phase 4 已冻结的 `--output-dir/--output-json` 语义，并按 Files 声明只增加显式只读输入 `--prereg-anchor/--candidate-identity/--component-manifests/--shape-manifest`；它必须自行通过 protected remote tag/external registry 解析 peeled commit，不接受 commit 环境覆盖。上述原始入口必须保留，正式运行不得读取 red fixture 或写入 baseline。

  - **项目化必读：** 阅读 [Triton tutorials 的尾块 mask](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 与 [`tl.load` boundary semantics](https://triton-lang.org/main/python-api/generated/triton.language.load.html)，逐项核对正式 correctness 是否覆盖非整除 M/N/K、dtype、seed 和 guard region；再读 `triton.testing.do_bench` 核对正式 timing 参数与 Week 21 完全一致。
  - **Expected：** 两条命令都返回 0；所有 supported correctness 与 80% gate、fallback 路由/correctness、canonical six 4/6 gate、identity/cache/schema 检查均通过，final CSV/summary 可独立重算。

- [ ] **周五：导出 final patch 和 artifact hash**

同时记录 compile-time、PTX/SASS size 和 cache artifact delta。patch 从冻结 baseline commit 到最终 candidate 生成，包含必要新增文件；结果 commit 必须经 `git merge-base --is-ancestor <remote-verified-peeled-commit> "$RESULT_COMMIT"` 验证为远端 prereg tag 后代，并把 tag object、peeled commit 与 result commit 只追加到 progress。`final-manifest.sha256` 在结果完成后生成，保存远端 URL/tag object/peeled commit、threshold hash、candidate identity/components、TTIR/TTGIR/LLVM/PTX/cubin/SASS、测试、harness、shape/config/schedule、final CSV/summary 与文档 hash，不包含可变 cache 时间戳。

  - **项目化必读：** 阅读固定 commit 的 cache manager/key 与实际 artifact layout，再复核 Phase 5 `hash_compiler_tree.sh`、`hash_loaded_build.sh` 和 exact patch 算法。逐项写出 compile-time、code-size/cache delta 的同 identity off/on 配对来源；不能拿 cache hit/miss 的 wall time冒充 compiler compile-time delta。
  - **Expected：** `compiler/patches/final.patch` 可反向检查且 hash 固定；result commit 是 `PREREG_COMMIT` 后代，final manifest 明确引用 prereg commit/threshold hash；每个路径存在、hash 可重算，并能从 candidate identity 追到正式 correctness/regression 输出。

### 周末

- [ ] **周六：计算最终摘要**

必须报告 best、geometric mean、worst regression、variance、compile-time delta 和 code-size delta。geomean、best、worst 和 variance 只覆盖完整的 supported 集；另附 canonical six 4/6、所有 supported 80% gate、fallback 路由/correctness 表、失败/过滤数量和置信状态，不能只展示图或单一平均数。

```python
from statistics import geometric_mean, median, pstdev, mean

ratios = [row.optimized_p50_us / row.off_p50_us for row in supported_rows]
geomean_ratio = geometric_mean(ratios)
worst_ratio = max(ratios)
```

  - **项目化必读：** 阅读 [Python `statistics`](https://docs.python.org/3/library/statistics.html) 中 `geometric_mean` 的输入限制和 `pstdev` 的 population 语义；用手算 fixture 验证 ratio 方向、best/worst 极值和 CV。摘要必须把 geomean 与 worst 并列，且明确禁止以算术平均 speedup 抵消某个 shape 的退化。
  - **Expected：** final summary 与 shape manifest、`final.csv`/raw samples 数值一致；漏 supported shape、非正 ratio、NaN/Inf、任一 supported 低于 80%、fallback 错误启用优化或 fallback correctness 失败都会阻止生成“PASS”结论。

- [ ] **周日：完成适用范围文档**

明确支持条件、回退条件、已知限制和未解决问题。每个 supported 项链接 compiler predicate/test 与泛化数据；每个 fallback 项链接静态 predicate、未优化路径 test 和 correctness；每个限制链接 failing/xfail case、issue 和移除条件；列出最终 patch 明确不包含的 future work。观察到退化后不得事后把原 supported shape 改标为 fallback。

  - **项目化必读：** 复读固定 commit 的最终 match predicate/verifier/test，并用 [pytest skip/xfail](https://docs.pytest.org/en/stable/how-to/skipping.html) 审计所有 skip、xfail 与 XPASS。只有环境不可用可 skip；已知 bug 使用 strict xfail；性能未达标必须是 regression FAIL 或明确回退，不能写成“支持但较慢”。
  - **Expected：** `docs/optimization_scope.md` 独立说明静态支持条件、动态回退、数据边界、known failures 与 issue；读者无需猜测即可判断任一 M/N/K/layout/dtype 是否启用优化。

### Phase 6 Exit Gate

```text
final patch 干净且有 compiler test
完整 correctness/regression 通过硬门槛
泛化、最坏退化和适用范围明确
regression harness 经真实 red/green 验证且失败返回非零
geometric mean、worst regression、variance 与 compile/code-size delta 可从完整 supported 集重算
原六 shape 4/6 >= 90%，且所有 supported canonical/neighbor/boundary shape 均 >= 80%
所有 fallback shape 正确关闭优化并通过 fallback correctness，不进入 geomean 或性能宣称
平均值不能覆盖任何 supported shape 单点失败，supported/fallback 分类不得事后修改
PREREG_COMMIT 覆盖 identity/components、固定 18-shape classification、全部 config/hash、最终 paired order 与 thresholds
correctness、formal benchmark、analyze 和 regression 每次消费前后验证 ancestry/show/diff/hash/spec，事后修改必失败
所有 result commit 均为 PREREG_COMMIT 后代，final manifest 引用 prereg commit 与 threshold hash
Phase 5 identity/cache/paired/statistics 契约完整继承，results/baseline 未被覆盖
已知失败均有 strict 标记、issue 与回退，不被误报为支持
```

[进入 Phase 7：Engineering Delivery →](./Phase7_Engineering_Delivery.md)
