# Phase 7：Engineering Delivery

[← Phase 6：Generalization & Regression](./Phase6_Generalization_Regression.md) · [24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 23–24（2026-12-16 至 2026-12-29）
- 计划工时：22h

## 前置条件

[Phase 6 Exit Gate](./Phase6_Generalization_Regression.md#phase-6-exit-gate) 已通过。Phase 7 只交付已经冻结并通过 gate 的候选，不再调参、改变支持范围或重算候选身份。

## 可验证阶段目标

- 在冷环境中仅依据仓库文档独立复现安装、构建、compiler test、correctness、benchmark、profile 与关键结论。
- 完成面向评审的设计、实验、失败记录和最终报告闭环。
- 能用 manifest 中的 artifact 回答 transformation 位置、收益归因、适用范围与工程取舍。
- 让接手者不依赖作者记忆：README 中每条交付路径都有工作目录、输入、命令、预期结果和 artifact reference。

## 本阶段产出

- 冷环境复现日志与修订后的运行手册。
- 完整 patch、测试、benchmark/profile artifact 与文档索引。
- 最终报告、答辩问题清单和项目交付包。

## 证据继承与冷环境写入边界

本页外链已于 **2026-07-14** 在线核验。滚动版文档用于解释工具语义；实际 Triton 路径、依赖和测试入口必须以 `compiler/baseline_commit.txt` 固定的 commit 为准，并把差异写入 `environment/triton-build.md`。

Phase 7 **原样继承** [Phase 6 的 protected preregistration、candidate identity 与 final manifest](./Phase6_Generalization_Regression.md#证据契约与资料边界)：

- `results/baseline/`、`results/regression/`、`results/experiments/` 中已冻结的 CSV、raw samples、profile、cache、manifest 和 hash 全部只读。冷环境不得用默认输出路径覆盖它们。
- `experiment_identity` 仍是 Phase 5 冻结、Phase 6 protected remote tag 唯一命名的同一个值；不得为“交付版”生成另一个等价 identity。源码、patch、build、flag、import resolved path、逻辑 cache key 或 artifact hash 任一变化都使候选失效，必须退回 Phase 5/6 重跑，而不是在 Phase 7 局部更新 hash。cold replay 的物理 cache root 必须是新 delivery 子目录，但 off/on 逻辑 key、隔离规则和预注册 schedule 不变；manifest 分开记录逻辑 identity 与物理 replay path。
- 每次 cold run 都先令 `REPRO_ID` 唯一，并只写新目录 `results/delivery/$REPRO_ID/`。创建前执行 `test ! -e "$DELIVERY_ROOT"`；失败即停止，不能 `rm -rf`、`--force-overwrite` 或复用旧目录。
- `verify_week21_preregistration.sh` 必须在 build/test/correctness/benchmark/profile 前后解析 protected remote 的 signed annotated tag、external registry 记录和 peeled commit；本地 `prereg-remote.json` 只是 locator/cache，环境变量不能替换 commit 真值。
- 冷环境结果是冻结证据的 **replay**，不是新 baseline。报告必须同时链接 frozen artifact 与 cold replay artifact，并说明二者身份、输入、统计口径及允许的环境差异。

### 可搬迁的 normalized component identity

Protected tag 只证明预注册记录未被替换，不能代替 live source/build 核验。Phase 6 的 `candidate-components.sha256` 必须逐项引用以下三个冻结清单及其 hash：

- `candidate-source-files.sha256`：compiler source/input 的 **repo-relative path、type、content hash**；
- `candidate-build-inputs.sha256`：构建脚本、CMake/Python requirements、compiler/SDK 配置等输入的 relative path/type/hash；
- `candidate-loaded-artifacts.sha256`：editable import 实际加载的 Python extension/shared object/生成器产物的 **相对 live root path、type、content hash**。

`hash_compiler_tree.sh --root "$LIVE_TRITON" --output live-source.sha256` 和 `hash_loaded_build.sh --root "$LIVE_TRITON" --python "$LIVE_VENV/bin/python" --build-inputs-output live-build-inputs.sha256 --loaded-artifacts-output live-loaded-artifacts.sha256` 必须先用 `realpath` 证明每个文件位于 live root 内，再去掉该 root 前缀，以 `LC_ALL=C` 对 repo-relative path 排序后 hash；输出不得包含 checkout/venv 的绝对物理根、mtime 或 inode。`triton.__file__` 必须位于 live root，但只比较相对路径和内容 hash；cold checkout 的绝对路径无需等于 Phase 6 机器。source、build inputs、loaded artifacts 的 live 清单分别与 `results/regression/candidate-{source-files,build-inputs,loaded-artifacts}.sha256` 执行 `cmp`；任一内容不可复现立即失败，不能用 protected verifier 的 PASS 覆盖 component mismatch。

### `results/MANIFEST.md` 字段契约与来源

manifest 每行至少包含：`artifact_id`、`artifact_role`、`relative_path`、`sha256`、`bytes`、`producer_command_id`、`source_commit`、`experiment_identity`、`prereg_tag_object`、`prereg_peeled_commit`、`tool_versions_ref`、`created_utc`、`depends_on`。字段来源必须可机械核对：

`results/regression/final-manifest.sha256` 的所有文件名统一相对 `$CAPSTONE` 根目录；因此任何 `sha256sum -c` 都必须在 `(cd "$CAPSTONE" && ...)` 中运行。`results/delivery/` 是 Phase 7 新 replay namespace，不在 Phase 6 frozen manifest 的覆盖路径内。

- `source_commit` 来自 `git rev-parse --verify HEAD^{commit}`；commit 内容与显示字段按 [Git `rev-parse`](https://git-scm.com/docs/git-rev-parse) 和 [`show`](https://git-scm.com/docs/git-show) 取得。
- patch 的 base/head、可应用性和传输方式分别服从 [Git `diff`](https://git-scm.com/docs/git-diff)、[`apply --check`](https://git-scm.com/docs/git-apply) 与 [`bundle`](https://git-scm.com/docs/git-bundle)；如果改用提交序列，则额外记录 [`format-patch`](https://git-scm.com/docs/git-format-patch) 的 base commit。
- `experiment_identity`、tag object、peeled commit 和依赖边来自 Phase 6 只读 `candidate-identity.json`、external registry 与 `final-manifest.sha256`，不得从文件名猜测或重新拼接。
- GPU UUID/name/driver/state 来自 [NVIDIA SMI query 规范](https://docs.nvidia.com/deploy/nvidia-smi/index.html)；CUDA toolkit/host toolchain 来自 [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 的 system requirements 和 post-install verification；`.ncu-rep` 的生成/导入语义来自 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)。
- Python/Triton/pytest/ncu/nsys/git 等版本写入 `tool_versions_ref` 指向的日志；文件 hash、字节数和 UTC 时间由 manifest 生成命令实测。`artifact_id`、`artifact_role`、`producer_command_id` 是本项目 schema 字段，定义在 `results/MANIFEST.md` 顶部，必须唯一且引用存在，不能声称来自上游工具。

## Week 23：冷环境复现与文档闭环

**Files:**

- Finalize: `README.md`
- Finalize: `environment/*.md`
- Finalize: `environment/cold-toolchain.env`（仅含审核过的绝对工具路径、版本和 ABI/search path）
- Finalize: `environment/install-python-deps.sh`（固化 Phase 3 已选择的 Python/PyTorch index 与精确版本）
- Create: `compiler/triton-source.bundle`（至少包含冻结 baseline commit；供断网 cold clone）
- Create: `compiler/final_candidate_commit.txt`（从 Phase 6 protected result record 复制精确 commit）
- Create: `scripts/run_cold_reproduction.sh`（只接受 env allowlist 与显式路径；实现本周六顺序）
- Create: `scripts/register_delivery_reproduction.sh`（仅周日最终成功后登记 accepted record，并原子 create-only final pointer）
- Create: `scripts/verify_delivery_reproduction.sh`（Week 24 只读解析 protected final pointer）
- Create: `scripts/accept_delivery_report.sh`（只读核验 sealed correctness/benchmark/profile/report）
- Update: `scripts/run_ncu.sh`（接受 candidate components、variant/cache 和 metadata 输出）
- Update: `benchmark/regression.py`（验证 profile metadata 与 on artifact mapping）
- Create: `docs/reproduction.md`
- Create: `results/MANIFEST.md`
- Create: `results/delivery/week23-delivery-root.txt`（单行五字段 TSV：`directory`、canonical root、owner、mode、initial hash；位于非冻结 delivery namespace）
- Create: `results/delivery/.final-launch-$FINAL_RUN_TOKEN/final-candidate.json`（周日 fresh rerun 的独立 candidate state；不覆盖周三 state）
- Create: `results/delivery/$REPRO_ID/commands/`
- Create: `results/delivery/$REPRO_ID/environment/`
- Create: `results/delivery/$REPRO_ID/correctness/`
- Create: `results/delivery/$REPRO_ID/benchmark/`
- Create: `results/delivery/$REPRO_ID/profiles/ncu/`
- Create: `results/delivery/$REPRO_ID/reproduction-report.md`
- External Create-once: protected registry `delivery/final/<candidate_identity>`（只由周日最终 fresh rerun 成功后原子创建）

从周四起，每个独立 Bash 入口都必须定义并调用同一 `load_week23_delivery_root`：验证 state 是非 symlink 普通文件、恰好一行且 TSV `NF==5`，按 `kind/root/owner/mode/initial_hash/extra` 读取并要求 `extra` 为空、`kind=directory`；随后现场验证 canonical root、delivery namespace、owner/mode，并按创建时完全相同的 `directory\troot\towner\tmode\n` 字节公式重算 SHA-256。只有重算值等于 `initial_hash` 才可使用 `DELIVERY_ROOT`，任意额外字段或任意十六进制字符串都不能通过。周六 exploratory 与周日 final fresh rerun 生成各自 content-addressed root，不覆盖周三 state；周日另写 `final-candidate.json`，最终权威 root 只由 protected final pointer 决定。

### 工作日

- [ ] **周一：从 README 审计所有命令**

  每条命令必须包含工作目录、环境变量和预期输出。

  README 必须按 `Prerequisites → Install → Build → Test → Correctness → Benchmark → Profile → Artifact lookup → Troubleshooting` 排列。每条命令都写出工作目录、所需变量、只读输入、全量命令、成功退出码、关键预期输出和产物路径；安装与构建必须从空 venv/干净 worktree 开始，benchmark 必须注明 warmup、round/order、cache、统计口径和 frozen comparison，artifact lookup 必须用 `artifact_id` 从 `results/MANIFEST.md` 追到 patch、IR、PTX/SASS、CSV、raw samples 和 `.ncu-rep`。用一个没有参与开发的人可逐字执行的视角逐条审计，任何“同前”“运行相关测试”“按需设置”都视为缺口。

  ```bash
  cd "$CAPSTONE" || exit 1
  rg -n '^## (Prerequisites|Install|Build|Test|Correctness|Benchmark|Profile|Artifact lookup|Troubleshooting)$' README.md
  rg -n 'cwd:|Expected:|Artifacts:|Read-only inputs:' README.md
  ```

  - **项目化必读（目的：把 README 变成无口头上下文的安装/构建/验证入口）：** 阅读 [Python Packaging User Guide 的 venv/pip 指南](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)、固定 Triton commit 的 `README.md`/`python/requirements.txt`/`python/test-requirements.txt`，并对照 [Triton 官方 source build 入口](https://github.com/triton-lang/triton/blob/main/README.md#install-from-source)。逐条标出本项目命令比上游多出的 commit、CUDA、cache 和 artifact 约束。
  - **Expected：** 九个章节都存在；每条安装、构建、test、benchmark 命令均可从声明的 cwd 执行并指向可检查 artifact，不依赖作者记忆或 shell history。

- [ ] **周二：建立 manifest**

  列出 source commit、patch、脚本、CSV、ncu/nsys、IR/PTX/SASS 和 hash。

  按“字段契约与来源”建立 `results/MANIFEST.md`，列出 source commit、baseline commit、protected prereg tag object/peeled commit、candidate identity、final patch、脚本、compiler tests、correctness JSON、CSV/raw samples、ncu/nsys、TTIR/TTGIR/LLVM IR/PTX/cubin/SASS、报告和每个 hash。另建 command registry：每个 `producer_command_id` 指向 `docs/reproduction.md` 的完整命令；每个 `depends_on` 只能引用已存在的 `artifact_id`。从仓库根目录生成相对路径并拒绝绝对路径、缺文件、重复 id、悬空依赖或 hash 不一致。

  ```bash
  cd "$CAPSTONE" || exit 1
  git rev-parse --verify 'HEAD^{commit}'
  sha256sum -c results/regression/final-manifest.sha256
  git -C "$TRITON_ROOT" apply --reverse --check \
    "$CAPSTONE/compiler/patches/final.patch"
  BASELINE_COMMIT=$(cat "$CAPSTONE/compiler/baseline_commit.txt")
  CANDIDATE_COMMIT=$(cat "$CAPSTONE/compiler/final_candidate_commit.txt")
  test "$(git -C "$TRITON_ROOT" rev-parse "$BASELINE_COMMIT^{commit}")" = "$BASELINE_COMMIT"
  test "$(git -C "$TRITON_ROOT" rev-parse "$CANDIDATE_COMMIT^{commit}")" = "$CANDIDATE_COMMIT"
  cmp <(git -C "$TRITON_ROOT" diff --binary "$BASELINE_COMMIT" "$CANDIDATE_COMMIT") \
    "$CAPSTONE/compiler/patches/final.patch"
  git -C "$TRITON_ROOT" update-ref refs/delivery/baseline "$BASELINE_COMMIT"
  git -C "$TRITON_ROOT" update-ref refs/delivery/candidate "$CANDIDATE_COMMIT"
  cleanup_delivery_refs() {
    git -C "$TRITON_ROOT" update-ref -d refs/delivery/baseline
    git -C "$TRITON_ROOT" update-ref -d refs/delivery/candidate
  }
  trap cleanup_delivery_refs EXIT
  git -C "$TRITON_ROOT" bundle create \
    "$CAPSTONE/compiler/triton-source.bundle" \
    refs/delivery/baseline refs/delivery/candidate
  git -C "$TRITON_ROOT" bundle verify \
    "$CAPSTONE/compiler/triton-source.bundle"
  test "$(git -C "$TRITON_ROOT" bundle list-heads \
    "$CAPSTONE/compiler/triton-source.bundle" refs/delivery/baseline \
    | awk '{print $1}')" = "$BASELINE_COMMIT"
  test "$(git -C "$TRITON_ROOT" bundle list-heads \
    "$CAPSTONE/compiler/triton-source.bundle" refs/delivery/candidate \
    | awk '{print $1}')" = "$CANDIDATE_COMMIT"
  cleanup_delivery_refs
  trap - EXIT
  test -z "$(git -C "$TRITON_ROOT" for-each-ref --format='%(refname)' refs/delivery/)"
  ```

  - **项目化必读（目的：让 manifest 的 commit、patch、对象和文件字段各有权威来源）：** 阅读 [Git revision selection](https://git-scm.com/docs/gitrevisions)、[`git show`](https://git-scm.com/docs/git-show)、[`git apply`](https://git-scm.com/docs/git-apply) 和 [`git bundle`](https://git-scm.com/docs/git-bundle)。把每个字段旁写明采集命令或 Phase 6 只读来源；本项目自定义字段必须明确标成 local schema。
  - **Expected：** manifest 覆盖所有最终证据；每个路径存在、hash 可重算、依赖闭合，并能从任一性能结论反向追到 command、identity、patch、输入和原始数据。

- [ ] **周三：在干净 Python venv 重建 kernel 环境**

  不复制旧 venv。固定 Python 版本并保存解释器路径/版本、pip 版本、requirements 文件 hash、安装日志和最终 `pip freeze --all`。依赖安装只读取仓库中已归档的 lock/requirements 与文档指定 index；如果不能由固定输入重建，立即修订 packaging 输入，不能从旧环境“看看装了什么”后手工补包。

  ```bash
  set -euo pipefail
  ENV_RUN_ID="week23-venv-$(date -u +%Y%m%dT%H%M%SZ)"
  export DELIVERY_ROOT="$CAPSTONE/results/delivery/$ENV_RUN_ID"
  test ! -e "$DELIVERY_ROOT"
  mkdir -p "$DELIVERY_ROOT/environment" "$DELIVERY_ROOT/commands"
  DELIVERY_ROOT=$(realpath -e "$DELIVERY_ROOT")
  DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
  case "$DELIVERY_ROOT/" in
    "$DELIVERY_PARENT"/*) ;;
    *) echo "delivery root escapes delivery namespace" >&2; exit 1 ;;
  esac
  test ! -L "$DELIVERY_ROOT"
  ROOT_OWNER=$(stat -c '%u:%g' "$DELIVERY_ROOT")
  ROOT_MODE=$(stat -c '%a' "$DELIVERY_ROOT")
  INITIAL_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
    "$DELIVERY_ROOT" "$ROOT_OWNER" "$ROOT_MODE" | sha256sum | awk '{print $1}')
  STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
  test ! -e "$STATE_FILE"
  umask 077
  STATE_TMP=$(mktemp "$CAPSTONE/results/delivery/.week23-delivery-root.XXXXXX")
  cleanup_state_tmp() { rm -f "$STATE_TMP"; }
  trap cleanup_state_tmp EXIT
  printf 'directory\t%s\t%s\t%s\t%s\n' \
    "$DELIVERY_ROOT" "$ROOT_OWNER" "$ROOT_MODE" "$INITIAL_HASH" \
    > "$STATE_TMP"
  test "$(wc -l < "$STATE_TMP")" -eq 1
  mv "$STATE_TMP" "$STATE_FILE"
  trap - EXIT
  chmod 600 "$STATE_FILE"
  python3 -m venv "$DELIVERY_ROOT/venv"
  bash -x "$CAPSTONE/environment/install-python-deps.sh" \
    "$DELIVERY_ROOT/venv/bin/python" \
    "$TRITON_ROOT/python/requirements.txt" \
    "$TRITON_ROOT/python/test-requirements.txt" \
    2>&1 | tee "$DELIVERY_ROOT/commands/python-install.log"
  "$DELIVERY_ROOT/venv/bin/python" -m pip freeze --all \
    > "$DELIVERY_ROOT/environment/pip-freeze.txt"
  "$DELIVERY_ROOT/venv/bin/python" -m pip check \
    > "$DELIVERY_ROOT/environment/pip-check.txt"
  ```

  - **项目化必读（目的：区分隔离环境、安装输入和可重建版本清单）：** 阅读 [Python `venv`](https://docs.python.org/3/library/venv.html) 与 [PyPA venv/pip 指南](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)，再读取固定 commit 的两个 requirements 文件。说明 venv 本身不可搬运，交付的是创建命令、固定输入、hash 与验证日志。
  `install-python-deps.sh` 必须把 Phase 3 已核验的精确 PyTorch wheel/index 命令与两个 requirements 安装命令写在脚本内，拒绝从调用者环境读取未登记 index/version；脚本路径、内容 hash 和参数都进入 manifest。

  - **Expected：** 新 venv 的 Python/pip 路径均位于 `$DELIVERY_ROOT/venv`，精确 PyTorch 与两个 requirements 可从零安装，`pip check` 返回 0。`week23-delivery-root.txt` 以原子 rename 创建且只含一行五字段，依次记录 `directory`、delivery namespace 内非 symlink canonical root、owner、mode 与按固定字节公式生成的 initial hash；后续步骤无需继承本 shell 的 export。

- [ ] **周四：在干净 Triton worktree 应用 final patch 并构建**

  保留原计划的 worktree/apply 命令，并在 apply 前后验证 base、patch 与 candidate tree。禁止从开发 worktree 复制 build 目录。editable install 必须绑定周三的新 Python；保存 build log、resolved import path、compiler tree/build hash，并与 Phase 6 component manifests 逐项相等。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  verify_protected_identity() {
    bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
      --capstone "$CAPSTONE" \
      --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
      --protected-remote "$PROTECTED_PREREG_REMOTE" \
      --external-registry "$PROTECTED_PREREG_REGISTRY"
  }
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-build-pre.log"
  git -C "$TRITON_ROOT" worktree add "$DELIVERY_ROOT/src/triton-repro" \
    "$(cat "$CAPSTONE/compiler/baseline_commit.txt")"
  git -C "$DELIVERY_ROOT/src/triton-repro" apply \
    "$CAPSTONE/compiler/patches/final.patch"

  export LIVE_TRITON="$DELIVERY_ROOT/src/triton-repro"
  git -C "$LIVE_TRITON" diff --check
  mkdir -p "$DELIVERY_ROOT/evidence/identity"
  bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
    --root "$LIVE_TRITON" \
    --output "$DELIVERY_ROOT/evidence/identity/build-pre-source.sha256"
  cmp "$DELIVERY_ROOT/evidence/identity/build-pre-source.sha256" \
    "$CAPSTONE/results/regression/candidate-source-files.sha256"
  cd "$LIVE_TRITON"
  "$DELIVERY_ROOT/venv/bin/python" -m pip install -e . --no-build-isolation \
    2>&1 | tee "$DELIVERY_ROOT/commands/triton-build.log"
  "$DELIVERY_ROOT/venv/bin/python" -c \
    'import pathlib,triton; print(pathlib.Path(triton.__file__).resolve())' \
    | tee "$DELIVERY_ROOT/environment/triton-import.txt"
  bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
    --root "$LIVE_TRITON" \
    --output "$DELIVERY_ROOT/evidence/identity/build-post-source.sha256"
  bash "$CAPSTONE/scripts/hash_loaded_build.sh" \
    --root "$LIVE_TRITON" --python "$DELIVERY_ROOT/venv/bin/python" \
    --build-inputs-output "$DELIVERY_ROOT/evidence/identity/build-post-inputs.sha256" \
    --loaded-artifacts-output "$DELIVERY_ROOT/evidence/identity/build-post-loaded.sha256"
  cmp "$DELIVERY_ROOT/evidence/identity/build-post-source.sha256" \
    "$CAPSTONE/results/regression/candidate-source-files.sha256"
  cmp "$DELIVERY_ROOT/evidence/identity/build-post-inputs.sha256" \
    "$CAPSTONE/results/regression/candidate-build-inputs.sha256"
  cmp "$DELIVERY_ROOT/evidence/identity/build-post-loaded.sha256" \
    "$CAPSTONE/results/regression/candidate-loaded-artifacts.sha256"
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-build-post.log"
  ```

  - **项目化必读（目的：理解 detached baseline worktree、patch 预检和 Triton editable source build）：** 阅读 [Git worktree](https://git-scm.com/docs/git-worktree)、[`git apply --check`](https://git-scm.com/docs/git-apply) 与固定 commit 的 Triton `README.md` source-build 段；同时检查该 commit 的 `CMakeLists.txt`、`setup.py`/`pyproject.toml` 与 requirements，记录实际 LLVM 获取/复用方式。
  - **Expected：** 本步骤在新 shell 中从单行 state 恢复并验证 canonical `DELIVERY_ROOT`，不依赖周三 export。final patch 可应用到冻结 baseline commit，构建返回 0，`triton.__file__` 解析到干净 worktree，tree/build/import/component hash 与冻结 candidate 完全一致；否则当前交付候选失效并停止。

- [ ] **周五：运行 compiler tests 和一个 correctness smoke**

  先运行 final patch 直接关联的 lit/FileCheck/pytest test，再运行固定 commit 的 Triton 无 GPU compiler test 入口和一个代表 shape correctness smoke。目标 test 的精确路径与 `RUN:` 行必须写入 README；若固定 commit 的 test target 不同，用 `rg` 找到真实 target 并记录，不能静默跳过。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  verify_protected_identity() {
    bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
      --capstone "$CAPSTONE" \
      --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
      --protected-remote "$PROTECTED_PREREG_REMOTE" \
      --external-registry "$PROTECTED_PREREG_REGISTRY"
  }
  export LIVE_TRITON="$DELIVERY_ROOT/src/triton-repro"
  verify_live_components() {
    STAGE="$1"
    bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
      --root "$LIVE_TRITON" \
      --output "$DELIVERY_ROOT/evidence/identity/${STAGE}-source.sha256"
    bash "$CAPSTONE/scripts/hash_loaded_build.sh" \
      --root "$LIVE_TRITON" --python "$DELIVERY_ROOT/venv/bin/python" \
      --build-inputs-output "$DELIVERY_ROOT/evidence/identity/${STAGE}-inputs.sha256" \
      --loaded-artifacts-output "$DELIVERY_ROOT/evidence/identity/${STAGE}-loaded.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-source.sha256" \
      "$CAPSTONE/results/regression/candidate-source-files.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-inputs.sha256" \
      "$CAPSTONE/results/regression/candidate-build-inputs.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-loaded.sha256" \
      "$CAPSTONE/results/regression/candidate-loaded-artifacts.sha256"
  }
  cd "$LIVE_TRITON"
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-compiler-tests-pre.log"
  verify_live_components compiler-tests-pre
  PYTHON="$DELIVERY_ROOT/venv/bin/python" make test-lit \
    2>&1 | tee "$DELIVERY_ROOT/commands/compiler-tests.log"
  verify_live_components compiler-tests-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-compiler-tests-post-correctness-pre.log"
  bash "$CAPSTONE/scripts/run_correctness.sh" \
    --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
    --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
    --output-dir "$DELIVERY_ROOT/correctness" \
    --output-json "$DELIVERY_ROOT/correctness/smoke.json"
  verify_live_components correctness-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-correctness-post.log"
  ```

  - **项目化必读（目的：用项目实际 harness 证明 compiler 改动和端到端语义都可执行）：** 阅读 [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/) 的 lit/FileCheck 分类与单文件运行方式、固定 Triton commit 的 `Makefile`/test CMake 配置及目标 test 的 `RUN:` 行，再核对 `run_correctness.sh --help` 的只读 identity 输入与输出参数。
  - **Expected：** 本步骤重新从 state 恢复并验证同一 canonical root；compiler tests 返回 0。smoke 在 protected identity 校验后返回 0，JSON 位于该 delivery 目录且包含 shape/dtype/seed/reference/identity，不写 `results/baseline/` 或 `results/regression/`。

### 周末

- [ ] **周六：租赁 RTX 4090 做冷环境完整复现**

  仅依据 `docs/reproduction.md` 完成环境检查、correctness、benchmark 和一个 ncu report。

  仅依据 `docs/reproduction.md`，在从未使用过本项目的 RTX 4090 主机完成环境检查、安装、build/test、完整 correctness、正式 regression benchmark 和一个 ncu report。先记录 GPU/driver/CUDA/toolchain，再验证 protected identity；所有输出进入唯一 delivery 目录。benchmark 继续使用 Phase 6 预注册 shape/config/order/threshold 和同 candidate off/on 契约，不能因租赁时长修改样本或只挑获益 shape。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  RUN_TOKEN="week23-rtx4090-$(date -u +%Y%m%dT%H%M%SZ)"
  export DELIVERY_ROOT="$CAPSTONE/results/delivery/.unsealed-$RUN_TOKEN"
  test ! -e "$DELIVERY_ROOT"
  mkdir -p "$DELIVERY_ROOT"/{src,home,tmp,environment,commands,correctness,benchmark,profiles/ncu,cache/pip,cache/triton,cache/torch,cache/xdg,evidence/identity}
  set -a
  # shellcheck source=/dev/null
  source "$CAPSTONE/environment/cold-toolchain.env"
  set +a
  for REQUIRED in FROZEN_BASH FROZEN_PYTHON3 FROZEN_PYTHON_VERSION FROZEN_PATH \
    FROZEN_CUDA_HOME FROZEN_CC FROZEN_CXX FROZEN_LD_LIBRARY_PATH \
    FROZEN_CMAKE_PREFIX_PATH FROZEN_LLVM_SYSPATH; do
    test -n "${!REQUIRED:-}"
  done
  for TOOL in "$FROZEN_BASH" "$FROZEN_PYTHON3" "$FROZEN_CC" "$FROZEN_CXX"; do
    test "${TOOL#/}" != "$TOOL"
    test -x "$TOOL"
  done

  cold_main() {
  # env -i 已清空调用者环境；先证明常见污染变量和遗留 PIP_/TRITON_ 变量均不存在。
  for FORBIDDEN in PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV \
    PIP_CONFIG_FILE PIP_INDEX_URL PIP_EXTRA_INDEX_URL TRITON_ROOT TRITON_VENV \
    TRITON_CACHE_DIR TRITON_HOME TRITON_EXP_FLAG; do
    test -z "${!FORBIDDEN+x}"
  done
  test -z "$(env | awk -F= '$1 ~ /^PIP_/ || $1 ~ /^TRITON_/ {print $1}')"
  test "$("$FROZEN_PYTHON3" --version 2>&1)" = "$FROZEN_PYTHON_VERSION"
  export HOME="$DELIVERY_ROOT/home"
  export XDG_CACHE_HOME="$DELIVERY_ROOT/cache/xdg"
  export TORCH_HOME="$DELIVERY_ROOT/cache/torch"
  export PIP_CACHE_DIR="$DELIVERY_ROOT/cache/pip"
  export TRITON_HOME="$DELIVERY_ROOT/cache/triton-home"

  verify_protected_identity() {
    bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
      --capstone "$CAPSTONE" \
      --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
      --protected-remote "$PROTECTED_PREREG_REMOTE" \
      --external-registry "$PROTECTED_PREREG_REGISTRY"
  }
  verify_frozen_manifest() {
    (cd "$CAPSTONE" && \
      sha256sum -c results/regression/final-manifest.sha256)
  }
  snapshot_frozen_namespaces() {
    local OUTPUT="$1"
    (
      cd "$CAPSTONE"
      find results/baseline results/experiments results/regression -print0 \
        | LC_ALL=C sort -z \
        | while IFS= read -r -d '' PATHNAME; do
            if test -L "$PATHNAME"; then
              TARGET_HASH=$(printf '%s' "$(readlink "$PATHNAME")" | sha256sum | awk '{print $1}')
              printf 'symlink\t%s\t%s\n' "$PATHNAME" "$TARGET_HASH"
            elif test -f "$PATHNAME"; then
              FILE_HASH=$(sha256sum "$PATHNAME" | awk '{print $1}')
              printf 'file\t%s\t%s\n' "$PATHNAME" "$FILE_HASH"
            elif test -d "$PATHNAME"; then
              printf 'directory\t%s\t-\n' "$PATHNAME"
            else
              printf 'unsupported\t%s\t-\n' "$PATHNAME" >&2
              exit 1
            fi
          done
    ) > "$OUTPUT"
  }

  # 冷结果目录尚为空时先验证 Phase 6 frozen evidence；从 $CAPSTONE 根目录解析相对路径。
  snapshot_frozen_namespaces \
    "$DELIVERY_ROOT/evidence/frozen-namespaces-pre.tsv"
  verify_frozen_manifest \
    | tee "$DELIVERY_ROOT/commands/frozen-manifest-pre.log"
  nvidia-smi --query-gpu=timestamp,uuid,name,driver_version,pstate,temperature.gpu,power.draw,clocks.sm,clocks.mem \
    --format=csv > "$DELIVERY_ROOT/environment/nvidia-smi.csv"
  nvcc --version > "$DELIVERY_ROOT/environment/nvcc.txt"
  ncu --version > "$DELIVERY_ROOT/environment/ncu.txt"
  nsys --version > "$DELIVERY_ROOT/environment/nsys.txt"
  cp "$CAPSTONE/environment/cold-toolchain.env" \
    "$DELIVERY_ROOT/environment/cold-toolchain.env"
  {
    "$FROZEN_PYTHON3" --version
    git --version
    cmake --version | head -n 1
    ninja --version
    "$FROZEN_CC" --version | head -n 1
    "$FROZEN_CXX" --version | head -n 1
    nvcc --version
    nvidia-smi --query-gpu=uuid,name,driver_version --format=csv,noheader
    printf 'PATH=%s\nCUDA_HOME=%s\nCC=%s\nCXX=%s\n' \
      "$PATH" "$CUDA_HOME" "$CC" "$CXX"
    printf 'LD_LIBRARY_PATH=%s\nCMAKE_PREFIX_PATH=%s\nLLVM_SYSPATH=%s\n' \
      "$LD_LIBRARY_PATH" "$CMAKE_PREFIX_PATH" "$LLVM_SYSPATH"
    printf 'TMPDIR=%s\nXDG_CACHE_HOME=%s\nTORCH_HOME=%s\nPIP_CACHE_DIR=%s\n' \
      "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"
  } > "$DELIVERY_ROOT/environment/toolchain.txt"

  # 不读取旧 TRITON_ROOT/TRITON_VENV：从已归档 bundle clone，再定位冻结 baseline。
  git -C "$CAPSTONE" bundle verify \
    "$CAPSTONE/compiler/triton-source.bundle" \
    2>&1 | tee "$DELIVERY_ROOT/commands/triton-bundle-verify.log"
  git clone --no-checkout "$CAPSTONE/compiler/triton-source.bundle" \
    "$DELIVERY_ROOT/src/triton" \
    2>&1 | tee "$DELIVERY_ROOT/commands/triton-clone.log"
  export TRITON_ROOT="$DELIVERY_ROOT/src/triton"
  BASELINE_COMMIT=$(cat "$CAPSTONE/compiler/baseline_commit.txt")
  CANDIDATE_COMMIT=$(cat "$CAPSTONE/compiler/final_candidate_commit.txt")
  test "$(git -C "$TRITON_ROOT" bundle list-heads \
    "$CAPSTONE/compiler/triton-source.bundle" refs/delivery/baseline \
    | awk '{print $1}')" = "$BASELINE_COMMIT"
  test "$(git -C "$TRITON_ROOT" bundle list-heads \
    "$CAPSTONE/compiler/triton-source.bundle" refs/delivery/candidate \
    | awk '{print $1}')" = "$CANDIDATE_COMMIT"
  git -C "$TRITON_ROOT" checkout --detach "$BASELINE_COMMIT"
  test "$(git -C "$TRITON_ROOT" rev-parse HEAD)" = "$BASELINE_COMMIT"
  test -z "$(git -C "$TRITON_ROOT" status --porcelain)"
  git -C "$TRITON_ROOT" apply --check \
    "$CAPSTONE/compiler/patches/final.patch"
  git -C "$TRITON_ROOT" apply --index \
    "$CAPSTONE/compiler/patches/final.patch"
  git -C "$TRITON_ROOT" apply --reverse --check \
    "$CAPSTONE/compiler/patches/final.patch"
  cmp <(git -C "$TRITON_ROOT" diff --cached --binary "$BASELINE_COMMIT") \
    "$CAPSTONE/compiler/patches/final.patch"
  git -C "$TRITON_ROOT" diff --cached --quiet "$CANDIDATE_COMMIT" --
  bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
    --root "$TRITON_ROOT" \
    --output "$DELIVERY_ROOT/evidence/identity/source-post-patch.sha256"
  cmp "$DELIVERY_ROOT/evidence/identity/source-post-patch.sha256" \
    "$CAPSTONE/results/regression/candidate-source-files.sha256"
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-source-post-patch.log"

  # 新 venv、固定 PyTorch/index、固定 build/test requirements；全部安装日志留在 cold root。
  "$FROZEN_PYTHON3" -m venv "$DELIVERY_ROOT/venv"
  export TRITON_VENV="$DELIVERY_ROOT/venv"
  bash -x "$CAPSTONE/environment/install-python-deps.sh" \
    "$TRITON_VENV/bin/python" \
    "$TRITON_ROOT/python/requirements.txt" \
    "$TRITON_ROOT/python/test-requirements.txt" \
    2>&1 | tee "$DELIVERY_ROOT/commands/python-install.log"
  "$TRITON_VENV/bin/python" -m pip check \
    | tee "$DELIVERY_ROOT/environment/pip-check.txt"
  "$TRITON_VENV/bin/python" -m pip freeze --all \
    > "$DELIVERY_ROOT/environment/pip-freeze.txt"
  {
    "$TRITON_VENV/bin/python" -m pip --version
    "$TRITON_VENV/bin/python" -m pip config debug
    rg -n -- '--index-url|--extra-index-url|--no-index|--find-links' \
      "$CAPSTONE/environment/install-python-deps.sh"
  } > "$DELIVERY_ROOT/environment/pip-config-and-index-source.txt"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-build-pre.log"
  (cd "$TRITON_ROOT" && \
    "$TRITON_VENV/bin/python" -m pip install -e . --no-build-isolation) \
    2>&1 | tee "$DELIVERY_ROOT/commands/triton-build.log"
  TRITON_IMPORT=$("$TRITON_VENV/bin/python" -c \
    'import pathlib,triton; print(pathlib.Path(triton.__file__).resolve())')
  case "$TRITON_IMPORT" in
    "$TRITON_ROOT"/*) ;;
    *) echo "cold venv imported Triton outside delivery root: $TRITON_IMPORT" >&2; exit 1 ;;
  esac
  printf '%s\n' "$TRITON_IMPORT" \
    > "$DELIVERY_ROOT/environment/triton-import.txt"
  verify_live_components() {
    local STAGE="$1"
    bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
      --root "$TRITON_ROOT" \
      --output "$DELIVERY_ROOT/evidence/identity/${STAGE}-source.sha256"
    bash "$CAPSTONE/scripts/hash_loaded_build.sh" \
      --root "$TRITON_ROOT" --python "$TRITON_VENV/bin/python" \
      --build-inputs-output "$DELIVERY_ROOT/evidence/identity/${STAGE}-inputs.sha256" \
      --loaded-artifacts-output "$DELIVERY_ROOT/evidence/identity/${STAGE}-loaded.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-source.sha256" \
      "$CAPSTONE/results/regression/candidate-source-files.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-inputs.sha256" \
      "$CAPSTONE/results/regression/candidate-build-inputs.sha256"
    cmp "$DELIVERY_ROOT/evidence/identity/${STAGE}-loaded.sha256" \
      "$CAPSTONE/results/regression/candidate-loaded-artifacts.sha256"
  }
  verify_live_components build-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-build-post.log"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-compiler-tests-pre.log"
  verify_live_components compiler-tests-pre
  (cd "$TRITON_ROOT" && \
    PYTHON="$TRITON_VENV/bin/python" make test-lit) \
    2>&1 | tee "$DELIVERY_ROOT/commands/compiler-tests.log"
  verify_live_components compiler-tests-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-compiler-tests-post.log"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-correctness-pre.log"
  verify_live_components correctness-pre
  bash "$CAPSTONE/scripts/run_correctness.sh" \
    --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
    --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
    --output-dir "$DELIVERY_ROOT/correctness" \
    --output-json "$DELIVERY_ROOT/correctness/results.json"
  verify_live_components correctness-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-correctness-post.log"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-benchmark-pre.log"
  verify_live_components benchmark-pre
  bash "$CAPSTONE/scripts/run_benchmark.sh" \
    --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
    --config-dir "$CAPSTONE/results/regression/week21-configs" \
    --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
    --config-manifest "$CAPSTONE/results/regression/week21-configs.sha256" \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
    --variant-env TRITON_EXP_FLAG --cache-env TRITON_CACHE_DIR \
    --cache-root "$DELIVERY_ROOT/cache" \
    --schedule-file "$CAPSTONE/results/regression/week21-schedule.csv" \
    --rounds 3 \
    --output-dir "$DELIVERY_ROOT/benchmark/paired" \
    --output-csv "$DELIVERY_ROOT/benchmark/paired.csv" \
    --raw-samples-dir "$DELIVERY_ROOT/benchmark/raw-samples" \
    --gpu-state-output "$DELIVERY_ROOT/benchmark/gpu-state.csv"
  verify_live_components benchmark-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-benchmark-post.log"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-regression-pre.log"
  verify_live_components regression-pre
  bash "$CAPSTONE/scripts/run_regression.sh" \
    --prereg-anchor "$CAPSTONE/results/regression/prereg-remote.json" \
    --shape-manifest "$CAPSTONE/results/regression/week21-shapes.json" \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
    --thresholds "$CAPSTONE/results/regression/thresholds.json" \
    --input-csv "$DELIVERY_ROOT/benchmark/paired.csv" \
    --output-dir "$DELIVERY_ROOT/benchmark" \
    --output-csv "$DELIVERY_ROOT/benchmark/final.csv" \
    --summary-json "$DELIVERY_ROOT/benchmark/final-summary.json"
  verify_live_components regression-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-regression-post.log"

  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-profile-pre.log"
  verify_live_components profile-pre
  PROFILE_CONFIG="$CAPSTONE/results/baseline/configs/M512_N4096_K4096.json"
  PROFILE_CACHE="$DELIVERY_ROOT/cache/profile-on"
  PROFILE_METADATA="$DELIVERY_ROOT/profiles/ncu/M512_N4096_K4096.metadata.json"
  mkdir -p "$PROFILE_CACHE"
  KERNEL_NAME=$("$TRITON_VENV/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["kernel_name"])' \
    "$PROFILE_CONFIG")
  TRITON_EXP_FLAG=on TRITON_CACHE_DIR="$PROFILE_CACHE" \
  bash "$CAPSTONE/scripts/run_ncu.sh" \
    --shape-tag M512_N4096_K4096 \
    --config-json "$PROFILE_CONFIG" --kernel-name "$KERNEL_NAME" \
    --disable-autotune --warmup-ms 100 --profile-iterations 1 \
    --sections SpeedOfLight,Occupancy \
    --variant on --variant-env TRITON_EXP_FLAG \
    --cache-root "$PROFILE_CACHE" --cache-env TRITON_CACHE_DIR \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256" \
    --loaded-artifacts-manifest "$CAPSTONE/results/regression/candidate-loaded-artifacts.sha256" \
    --metadata-output "$PROFILE_METADATA" \
    --output "$DELIVERY_ROOT/profiles/ncu/M512_N4096_K4096"
  "$TRITON_VENV/bin/python" "$CAPSTONE/benchmark/regression.py" \
    --mode verify-profile-metadata \
    --metadata "$PROFILE_METADATA" \
    --report "$DELIVERY_ROOT/profiles/ncu/M512_N4096_K4096.ncu-rep" \
    --expected-variant on --expected-cache-root "$PROFILE_CACHE" \
    --expected-kernel "$KERNEL_NAME" --expected-shape M512_N4096_K4096 \
    --expected-config "$PROFILE_CONFIG" \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
    --component-manifests "$CAPSTONE/results/regression/candidate-components.sha256"
  verify_live_components profile-post
  verify_protected_identity \
    | tee "$DELIVERY_ROOT/commands/identity-profile-post.log"

  # cold output 不属于 Phase 6 manifest 覆盖范围；再次从 $CAPSTONE 根目录检查 frozen evidence。
  snapshot_frozen_namespaces \
    "$DELIVERY_ROOT/evidence/frozen-namespaces-post.tsv"
  cmp "$DELIVERY_ROOT/evidence/frozen-namespaces-pre.tsv" \
    "$DELIVERY_ROOT/evidence/frozen-namespaces-post.tsv"
  verify_frozen_manifest \
    | tee "$DELIVERY_ROOT/commands/frozen-manifest-post.log"
  test -s "$DELIVERY_ROOT/commands/frozen-manifest-pre.log"
  test -s "$DELIVERY_ROOT/commands/frozen-manifest-post.log"

  # 清单都使用 delivery-relative path；先聚合 toolchain、identity、results，再计算内容寻址 REPRO_ID。
  (
    cd "$DELIVERY_ROOT"
    find environment -type f -print0 | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) > "$DELIVERY_ROOT/evidence/toolchain-manifest.sha256"
  (
    cd "$DELIVERY_ROOT"
    find evidence/identity -type f -print0 | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) > "$DELIVERY_ROOT/evidence/identity-manifest.sha256"
  (
    cd "$DELIVERY_ROOT"
    find correctness benchmark profiles -type f -print0 | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) > "$DELIVERY_ROOT/evidence/results-manifest.sha256"
  BUNDLE_HASH=$(sha256sum "$CAPSTONE/compiler/triton-source.bundle" | awk '{print $1}')
  TOOLCHAIN_HASH=$(sha256sum "$DELIVERY_ROOT/evidence/toolchain-manifest.sha256" | awk '{print $1}')
  IDENTITY_HASH=$(sha256sum "$DELIVERY_ROOT/evidence/identity-manifest.sha256" | awk '{print $1}')
  RESULTS_HASH=$(sha256sum "$DELIVERY_ROOT/evidence/results-manifest.sha256" | awk '{print $1}')
  CANDIDATE_COMPONENTS_HASH=$(sha256sum \
    "$CAPSTONE/results/regression/candidate-components.sha256" | awk '{print $1}')
  {
    printf 'bundle_sha256=%s\n' "$BUNDLE_HASH"
    printf 'toolchain_manifest_sha256=%s\n' "$TOOLCHAIN_HASH"
    printf 'identity_manifest_sha256=%s\n' "$IDENTITY_HASH"
    printf 'results_manifest_sha256=%s\n' "$RESULTS_HASH"
    printf 'candidate_components_sha256=%s\n' "$CANDIDATE_COMPONENTS_HASH"
  } > "$DELIVERY_ROOT/evidence/cold-manifest.txt"
  COLD_MANIFEST_HASH=$(sha256sum \
    "$DELIVERY_ROOT/evidence/cold-manifest.txt" | awk '{print $1}')
  (
    cd "$DELIVERY_ROOT"
    find . -mindepth 1 ! -path './evidence/sealed-tree.tsv' -print0 \
      | LC_ALL=C sort -z \
      | while IFS= read -r -d '' PATHNAME; do
          if test -L "$PATHNAME"; then
            TARGET_HASH=$(printf '%s' "$(readlink "$PATHNAME")" | sha256sum | awk '{print $1}')
            printf 'symlink\t%s\t%s\n' "$PATHNAME" "$TARGET_HASH"
          elif test -f "$PATHNAME"; then
            FILE_HASH=$(sha256sum "$PATHNAME" | awk '{print $1}')
            printf 'file\t%s\t%s\n' "$PATHNAME" "$FILE_HASH"
          elif test -d "$PATHNAME"; then
            printf 'directory\t%s\t-\n' "$PATHNAME"
          else
            exit 1
          fi
        done
  ) > "$DELIVERY_ROOT/evidence/sealed-tree.tsv"
  SEALED_TREE_HASH=$(sha256sum \
    "$DELIVERY_ROOT/evidence/sealed-tree.tsv" | awk '{print $1}')
  printf 'candidate_tree_sha256=%s\n' "$SEALED_TREE_HASH" \
    > "$DELIVERY_ROOT/evidence/local-candidate-record.txt"
  REPRO_ID="$COLD_MANIFEST_HASH"
  CANDIDATE_ROOT="$CAPSTONE/results/delivery/$REPRO_ID"
  test ! -e "$CANDIDATE_ROOT"
  mv "$DELIVERY_ROOT" "$CANDIDATE_ROOT"
  DELIVERY_ROOT="$CANDIDATE_ROOT"
  printf 'local_candidate_repro_id=%s\nlocal_candidate_root=%s\n' \
    "$REPRO_ID" "$DELIVERY_ROOT"
  }

  declare -f cold_main > "$DELIVERY_ROOT/commands/cold-main.sh"
  # shellcheck disable=SC2016
  env -i \
    LANG=C.UTF-8 LC_ALL=C \
    CAPSTONE="$CAPSTONE" DELIVERY_ROOT="$DELIVERY_ROOT" \
    TMPDIR="$DELIVERY_ROOT/tmp" \
    PROTECTED_PREREG_REMOTE="$PROTECTED_PREREG_REMOTE" \
    PROTECTED_PREREG_REGISTRY="$PROTECTED_PREREG_REGISTRY" \
    PROTECTED_DELIVERY_REGISTRY="$PROTECTED_DELIVERY_REGISTRY" \
    FROZEN_BASH="$FROZEN_BASH" FROZEN_PYTHON3="$FROZEN_PYTHON3" \
    FROZEN_PYTHON_VERSION="$FROZEN_PYTHON_VERSION" \
    PATH="$FROZEN_PATH" CUDA_HOME="$FROZEN_CUDA_HOME" \
    CC="$FROZEN_CC" CXX="$FROZEN_CXX" \
    LD_LIBRARY_PATH="$FROZEN_LD_LIBRARY_PATH" \
    CMAKE_PREFIX_PATH="$FROZEN_CMAKE_PREFIX_PATH" \
    LLVM_SYSPATH="$FROZEN_LLVM_SYSPATH" \
    "$FROZEN_BASH" --noprofile --norc -c \
      'source "$1"; cold_main' _ "$DELIVERY_ROOT/commands/cold-main.sh"
  ```

  - **项目化必读（目的：在新主机正确识别 CUDA 兼容性、GPU 状态和可归档 profiler report）：** 阅读 [CUDA Installation Guide 的 requirements/verification](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)、[NVIDIA SMI query 字段](https://docs.nvidia.com/deploy/nvidia-smi/index.html) 与 [Nsight Compute CLI 的 launch/export/import](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)；再逐行复核 Phase 6 thresholds、schedule 和 component manifests，保证 cold replay 没有改实验定义。
  - **Expected：** cold driver 只在 `env -i` allowlist 中运行，未批准 Python/Conda/PIP/Triton 环境均为空；绝对 Python/CC/CXX/CUDA/search path 与版本全部匹配冻结 toolchain manifest。bundle 两个 delivery ref 精确指向 baseline/candidate，clone 后从 baseline 应用 patch，staged tree 等于 candidate；新 venv、cache、source/build 和全部结果只位于 provisional delivery root。build/test/correctness/benchmark/regression/profile 每个边界同时通过 protected verifier 与 normalized source/build-input/loaded-artifact `cmp`；profile 明确为 on variant、独立 cache、固定 kernel/config，并由 metadata 反验 live artifact。frozen manifest pre/post 均 PASS，三大 frozen namespace snapshot 相同。最终只生成内容寻址的本地 candidate `REPRO_ID`；**不 chmod seal、不写 registry、不创建 final pointer**，失败或过时 candidate 可保留本地供诊断但没有交付权威性。

- [ ] **周日：修正文档缺口**

  把 cold run 中出现的每个隐含前提写入 `docs/reproduction.md` 和 README：包名/版本来源、权限、cwd、环境变量、网络下载与离线替代、build 输出、test selector、benchmark 预期行数、profile kernel selector、artifact lookup 和失败诊断。修订后在新 shell 重新执行受影响的完整命令，不允许只改文字便关闭问题；在 `reproduction-report.md` 为每个缺口记录 symptom、root cause、doc change、rerun command、exit code 和 artifact id。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  cd "$CAPSTONE" || exit 1
  rg -n '同上|按需|相关测试|手工选择|自行设置' \
    README.md docs/reproduction.md environment
  sha256sum -c results/regression/final-manifest.sha256
  ```

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  FINAL_RUN_TOKEN="week23-final-$(date -u +%Y%m%dT%H%M%SZ)"
  FINAL_LAUNCH_ROOT="$CAPSTONE/results/delivery/.final-launch-$FINAL_RUN_TOKEN"
  test ! -e "$FINAL_LAUNCH_ROOT"
  mkdir -p "$FINAL_LAUNCH_ROOT"
  set -a
  # shellcheck source=/dev/null
  source "$CAPSTONE/environment/cold-toolchain.env"
  set +a
  finalize_delivery() {
    local FINAL_RECORD="$FINAL_LAUNCH_ROOT/final-candidate.json"
    bash "$CAPSTONE/scripts/run_cold_reproduction.sh" \
      --mode fresh-local-candidate --capstone "$CAPSTONE" \
      --output-record "$FINAL_RECORD"
    REPRO_ID=$("$FROZEN_PYTHON3" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["repro_id"])' \
      "$FINAL_RECORD")
    DELIVERY_ROOT=$("$FROZEN_PYTHON3" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["delivery_root"])' \
      "$FINAL_RECORD")
    test "$DELIVERY_ROOT" = "$CAPSTONE/results/delivery/$REPRO_ID"
    chmod -R a-w "$DELIVERY_ROOT"
    test -z "$(find "$DELIVERY_ROOT" -perm /222 -print -quit)"
    PYTHONDONTWRITEBYTECODE=1 \
    bash "$CAPSTONE/scripts/verify_delivery_reproduction.sh" \
      --mode verify-readonly-candidate --candidate-record "$FINAL_RECORD" \
      --delivery-root "$DELIVERY_ROOT" \
      --require-tests-pass --require-hashes-pass \
      --require-frozen-namespaces-unchanged
    CANDIDATE_ID=$("$FROZEN_PYTHON3" "$CAPSTONE/benchmark/regression.py" \
      --mode print-protected-identity-digest \
      --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json")
    FINAL_POINTER="delivery/final/$CANDIDATE_ID"
    # 最后一步：accepted record 与 final pointer 原子提交；pointer 已存在或更新请求都失败。
    bash "$CAPSTONE/scripts/register_delivery_reproduction.sh" \
      --mode accept-and-create-final-pointer \
      --registry "$PROTECTED_DELIVERY_REGISTRY" \
      --candidate-record "$FINAL_RECORD" \
      --final-pointer "$FINAL_POINTER" --create-only --atomic
  }
  declare -f finalize_delivery > "$FINAL_LAUNCH_ROOT/finalize-delivery.sh"
  # shellcheck disable=SC2016
  env -i LANG=C.UTF-8 LC_ALL=C \
    CAPSTONE="$CAPSTONE" FINAL_LAUNCH_ROOT="$FINAL_LAUNCH_ROOT" \
    PROTECTED_PREREG_REMOTE="$PROTECTED_PREREG_REMOTE" \
    PROTECTED_PREREG_REGISTRY="$PROTECTED_PREREG_REGISTRY" \
    PROTECTED_DELIVERY_REGISTRY="$PROTECTED_DELIVERY_REGISTRY" \
    FROZEN_BASH="$FROZEN_BASH" FROZEN_PYTHON3="$FROZEN_PYTHON3" \
    FROZEN_PYTHON_VERSION="$FROZEN_PYTHON_VERSION" \
    PATH="$FROZEN_PATH" CUDA_HOME="$FROZEN_CUDA_HOME" \
    CC="$FROZEN_CC" CXX="$FROZEN_CXX" \
    LD_LIBRARY_PATH="$FROZEN_LD_LIBRARY_PATH" \
    CMAKE_PREFIX_PATH="$FROZEN_CMAKE_PREFIX_PATH" \
    LLVM_SYSPATH="$FROZEN_LLVM_SYSPATH" \
    "$FROZEN_BASH" --noprofile --norc -c \
      'source "$1"; finalize_delivery' _ \
      "$FINAL_LAUNCH_ROOT/finalize-delivery.sh"
  ```

  - **项目化必读（目的：把复现失败转化为可验证文档修复，而不是新的口头知识）：** 阅读 [Git `Documentation/SubmittingPatches`](https://github.com/git/git/blob/master/Documentation/SubmittingPatches) 中自包含变更说明的原则、[MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/) 的 self-documenting tests 与固定 Triton commit 的 build/test 文档；用它们审计命令是否说明“为什么、怎么跑、如何判定”。
  - **Expected：** 所有文档缺口修订后，在全新目录按与周六相同的 frozen `env -i` allowlist 完成 fresh install/build/test/correctness/benchmark/profile 与全部 identity/hash/namespace gate。只有该次 fresh candidate 被 chmod seal、在只读状态再次完整复验通过后，脚本才以最后一个原子操作登记 accepted record 并 create-only `delivery/final/<candidate_identity>`；任何更早失败不写 registry，pointer 已存在时拒绝覆盖或更新。

### Week 23 Exit Gate

```text
final patch 可应用到冻结 baseline commit
干净环境可完成 build/test/run
干净环境可完成 install/build/test/correctness/benchmark/profile
README 每条交付路径含 cwd、输入、命令、Expected 和 artifact reference
week23-delivery-root.txt 单行五字段原子创建；7 个 consumer 均拒绝 extra field 并按同字节公式重算 initial hash
结果 artifact 可由 manifest 定位且依赖/hash 闭合
protected prereg/final identity 前后核验一致
cold replay 只写 content-addressed `results/delivery/$REPRO_ID`；历史本地 candidate 可共存且冻结证据未被覆盖
frozen final-manifest 在 cold run 前后均从 CAPSTONE 根目录校验 PASS，两个日志已归档
baseline/experiments/regression 完整 path/type/hash snapshot 前后 cmp 相同
live normalized source/build-input/loaded-artifact 清单在 build/test/profile 前后均等于冻结 candidate components
env -i toolchain allowlist、版本、pip index 来源和 profile on mapping 已归档并通过
周六 exploratory candidate 未写 registry；周日 fresh rerun 的唯一 REPRO_ID 在只读复验后才登记 accepted record
protected delivery/final/<candidate_identity> pointer 原子 create-only 成功且不可更新
```

---

## Week 24：最终报告与自我答辩

**Files:**

- Finalize: `docs/triton_compiler_pipeline.md`
- Finalize: `docs/baseline_report.md`
- Finalize: `docs/optimization_design.md`
- Finalize: `docs/tuning_report.md`
- Finalize: `docs/failed_experiments.md`
- Finalize: `progress.md`
- Create: `docs/final_report.md`
- Create: `docs/defense_outline.md`
- Create: `results/delivery/final-acceptance/`

### 工作日

- [ ] **周一：完成 compiler pipeline 章节**

  用一个 fused Linear 跟踪 TTIR → TTGIR → LLVM IR → PTX/SASS。每一层写 producing command、pass/pipeline 位置、关键 op/layout/addressing 变化、对应 artifact id 和 hash；明确本优化在哪一层 match/rewrite、为何不在更早或更晚层做、fallback 如何保持语义。图示只作导航，任何结论必须链接可 diff 的文本 IR/PTX/SASS 和 compiler test。

  - **项目化必读（目的：用官方 IR/test 语义解释 transformation 位置而不是只画流程图）：** 阅读 [MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)、[MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/) 和固定 Triton commit 中实际 pass registration、pipeline builder、match/rewrite 源码及相邻 `test/TritonGPU` tests；逐层核对命令与 artifact 确实来自同一 candidate identity。
  - **Expected：** 读者能从 fused Linear 输入沿 artifact id 重放到 SASS，并精确定位修改点、触发 predicate、test 与 fallback；不存在只有截图或无法重生的 IR。

- [ ] **周二：完成 baseline 与实验表格**

  不只展示最佳值，必须展示六组 shape、几何平均和最坏退化。

  不只展示最佳值，必须展示六组 canonical shape、全部预注册 supported/fallback、几何平均、best、variance、最坏退化、canonical 4/6@90% 与 all-supported@80% gate。每个数字链接 frozen CSV/raw samples、shape/config/schedule manifest、统计命令和 cold replay comparison；fallback 与过滤样本分表，不能从 geomean 分母消失而不解释。

  - **项目化必读（目的：保证汇总统计可由原始样本独立重算且不掩盖单点失败）：** 阅读 [Python `statistics`](https://docs.python.org/3/library/statistics.html) 的 `median`、`geometric_mean`、`pstdev`，并复读 Phase 5/6 的 P50、paired order、noise 与 gate 契约。用 manifest 定位 raw samples 后手算至少一组 shape 和总 geomean。
  - **Expected：** 表格覆盖六组 shape 和完整预注册集合；geomean、best/worst、variance、4/6 与 80% gate 都可独立重算，任何失败/过滤/fallback 均有数量、原因和 artifact reference。

- [ ] **周三：完成 profiler 归因**

  将每条性能结论链接到 IR/PTX diff 和 ncu 指标。

  将每条性能结论链接到同 identity 的 IR/PTX diff、kernel name、ncu report、section/metric 和 paired latency。明确 observation 与 inference：例如寄存器、occupancy、waves、DRAM/L2、Tensor Core、stall 的变化只支持相应机制，不把相关性写成因果；至少一个 ablation 排除 autotune/cache/order 噪声。

  - **项目化必读（目的：正确解释 report/metric 并保存可复查的 profiler 输入）：** 阅读 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 的 report export/import、kernel filtering、pages/sections 和 metric units，以及 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 的 performance guidelines；再核对固定 ncu 版本的 section 文件和 manifest 中 `.ncu-rep` hash。
  - **Expected：** 每个性能 claim 都有“shape → paired latency → IR/PTX diff → ncu metric → inference/alternative”链；没有跨 identity、跨 kernel 或只凭截图的归因。

- [ ] **周四：整理失败实验**

  至少包含 observation、hypothesis、change、result 和 rejected reason，并补上 experiment identity、base/candidate commit、exact command、shape/config/order、correctness、raw data/profile artifact、判废 gate 与后续影响。失败实验不能被 final manifest 丢弃；若缺 artifact，明确标为 evidence gap，不能事后重造看似同一身份的数据。

  - **项目化必读（目的：让失败实验与成功实验使用同一可追溯标准）：** 阅读 [Git `git show`](https://git-scm.com/docs/git-show) 与 [`git diff`](https://git-scm.com/docs/git-diff) 的对象/patch 表示、[Nsight Compute CLI import](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 和本计划统一实验模板；逐项核对 failure record 能回到 immutable commit、command 与 raw artifact。
  - **Expected：** 每个失败都有五字段、判废规则和 artifact 链；读者能区分语义失败、性能失败、噪声/inconclusive 与 infrastructure failure，并知道为何没有进入 final patch。

- [ ] **周五：准备 20 分钟答辩提纲**

  结构：问题 2 分钟、baseline 3 分钟、compiler pipeline 4 分钟、优化设计 4 分钟、结果 4 分钟、限制与下一步 3 分钟。每页只引用 manifest `artifact_id`，在 appendix 建立 question → claim → artifact → command 索引；准备离线 Git bundle、final patch、README 和必要报告，网络不可用时仍可验证 commit 与 hash。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  cd "$CAPSTONE" || exit 1
  git bundle create results/delivery/capstone-evidence.bundle --all
  git bundle verify results/delivery/capstone-evidence.bundle
  git -C "$TRITON_ROOT" apply --check "$CAPSTONE/compiler/patches/final.patch"
  ```

  - **项目化必读（目的：准备可离线验证而非依赖现场仓库状态的答辩包）：** 阅读 [Git bundle](https://git-scm.com/docs/git-bundle)、[`git apply`](https://git-scm.com/docs/git-apply) 与 [`git format-patch`](https://git-scm.com/docs/git-format-patch)，明确 bundle 保存 refs/objects、patch 保存 tree delta、manifest 保存 artifact 依赖，三者不可相互冒充。
  - **Expected：** 提纲严格 20 分钟且每个关键 claim 有 artifact id；bundle verify 与 patch apply check 返回 0，离线接手者能按 README 找到 build/test/benchmark/profile 证据。

### 周末

- [ ] **周六：进行一次无稿自我答辩**

  录音或逐字记录问题，必须能回答：

  ```text
  为什么选择这个 IR 层修改？
  为什么收益不是 autotune 或噪声？
  为什么某些 shape 退化？
  如何保证语义正确？
  如果支持 dynamic shape，设计会怎样变化？
  ```

  回答不得只复述结论：每题在 90 秒内给出 claim、边界/反例和至少两个 artifact id，并现场执行一个 lookup 或验证命令。把答不上、超时、链接失效、命令失败记录到 `docs/defense_outline.md`，修正文档后重新抽答，保留首次与复答记录。

  - **项目化必读（目的：用 verifier、test 与 profiler 证据约束口头回答）：** 复读 [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)、[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 和固定 commit 的目标 pass/test；为五问分别选定 correctness/compiler-test/performance/profile/scope artifact，避免把预期或设计意图说成已测事实。
  - **Expected：** 五问都有时限内、带边界和 artifact 的回答；首次遗漏被记录且复答命令成功，任何尚无证据的 future work 明确标成 hypothesis。

- [ ] **周日：最终验收**

  Week 24 **不任选 cold 目录，也不重新测量**。它从 `delivery/final/<candidate_identity>` protected create-only pointer 只读解析 Week 23 周日 accepted sealed record，取得内容寻址 `REPRO_ID` 与 delivery path；周六未登记的本地 candidate 永远不能被选中。整个 acceptance 在与 Week 23 相同的 frozen `env -i` allowlist 中执行，先验证 registry 中的 bundle/cold manifest/toolchain/identity/results/sealed-tree hash，再核对目录不可写且完整树未变化。correctness、benchmark/regression 和 profile 只消费 sealed artifact；patch applicability 只在 `$ACCEPT_ROOT/src/` 的临时 clone 中检查。

  ```bash
  set -euo pipefail
  load_week23_delivery_root() {
    local STATE_FILE="$CAPSTONE/results/delivery/week23-delivery-root.txt"
    local STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE STATE_INITIAL_HASH EXTRA
    local DELIVERY_PARENT RECOMPUTED_HASH
    test -f "$STATE_FILE" && test ! -L "$STATE_FILE"
    test "$(wc -l < "$STATE_FILE")" -eq 1
    awk -F '\t' 'NR == 1 && NF == 5 {ok=1} END {exit !(NR == 1 && ok)}' "$STATE_FILE"
    IFS=$'\t' read -r STATE_KIND RECORDED_ROOT STATE_OWNER STATE_MODE \
      STATE_INITIAL_HASH EXTRA < "$STATE_FILE"
    test -z "$EXTRA"
    test "$STATE_KIND" = directory
    DELIVERY_ROOT=$(realpath -e "$RECORDED_ROOT")
    DELIVERY_PARENT=$(realpath -e "$CAPSTONE/results/delivery")
    test "$DELIVERY_ROOT" = "$RECORDED_ROOT"
    case "$DELIVERY_ROOT/" in "$DELIVERY_PARENT"/*) ;; *) return 1 ;; esac
    test -d "$DELIVERY_ROOT" && test ! -L "$DELIVERY_ROOT"
    test "$(stat -c '%u:%g' "$DELIVERY_ROOT")" = "$STATE_OWNER"
    test "$(stat -c '%a' "$DELIVERY_ROOT")" = "$STATE_MODE"
    RECOMPUTED_HASH=$(printf 'directory\t%s\t%s\t%s\n' \
      "$RECORDED_ROOT" "$STATE_OWNER" "$STATE_MODE" | sha256sum | awk '{print $1}')
    test "$RECOMPUTED_HASH" = "$STATE_INITIAL_HASH"
    export DELIVERY_ROOT
  }
  load_week23_delivery_root
  export ACCEPT_ROOT="$CAPSTONE/results/delivery/final-acceptance"
  test ! -e "$ACCEPT_ROOT"
  mkdir -p "$ACCEPT_ROOT"/{commands,evidence,src,report,tmp}
  set -a
  # shellcheck source=/dev/null
  source "$CAPSTONE/environment/cold-toolchain.env"
  set +a
  acceptance_main() {
  for FORBIDDEN in PYTHONPATH VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV \
    PIP_CONFIG_FILE PIP_INDEX_URL PIP_EXTRA_INDEX_URL TRITON_ROOT TRITON_VENV \
    TRITON_CACHE_DIR TRITON_HOME TRITON_EXP_FLAG; do
    test -z "${!FORBIDDEN+x}"
  done
  test -z "$(env | awk -F= '$1 ~ /^PIP_/ || $1 ~ /^TRITON_/ {print $1}')"
  test "$("$FROZEN_PYTHON3" --version 2>&1)" = "$FROZEN_PYTHON_VERSION"
  verify_protected_identity() {
    bash "$CAPSTONE/scripts/verify_week21_preregistration.sh" \
      --capstone "$CAPSTONE" \
      --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json" \
      --protected-remote "$PROTECTED_PREREG_REMOTE" \
      --external-registry "$PROTECTED_PREREG_REGISTRY"
  }
  verify_frozen_manifest() {
    (cd "$CAPSTONE" && \
      sha256sum -c results/regression/final-manifest.sha256)
  }

  verify_frozen_manifest \
    | tee "$ACCEPT_ROOT/commands/frozen-manifest-pre.log"

  REGISTRY_RECORD="$ACCEPT_ROOT/evidence/reproduction-registry.json"
  CANDIDATE_ID=$("$FROZEN_PYTHON3" "$CAPSTONE/benchmark/regression.py" \
    --mode print-protected-identity-digest \
    --candidate-identity "$CAPSTONE/results/regression/candidate-identity.json")
  FINAL_POINTER="delivery/final/$CANDIDATE_ID"
  bash "$CAPSTONE/scripts/verify_delivery_reproduction.sh" \
    --mode resolve-final-pointer --registry "$PROTECTED_DELIVERY_REGISTRY" \
    --final-pointer "$FINAL_POINTER" --require-create-only \
    --output "$REGISTRY_RECORD"
  REPRO_ID=$("$FROZEN_PYTHON3" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["repro_id"])' \
    "$REGISTRY_RECORD")
  DELIVERY_ROOT=$("$FROZEN_PYTHON3" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["delivery_root"])' \
    "$REGISTRY_RECORD")
  test "$DELIVERY_ROOT" = "$CAPSTONE/results/delivery/$REPRO_ID"
  test -d "$DELIVERY_ROOT"
  test -z "$(find "$DELIVERY_ROOT" -perm /222 -print -quit)"
  cmp "$DELIVERY_ROOT/environment/cold-toolchain.env" \
    "$CAPSTONE/environment/cold-toolchain.env"

  bash "$CAPSTONE/scripts/verify_delivery_reproduction.sh" \
    --mode verify-sealed --registry-record "$REGISTRY_RECORD" \
    --delivery-root "$DELIVERY_ROOT" \
    --bundle "$CAPSTONE/compiler/triton-source.bundle" \
    --candidate-components "$CAPSTONE/results/regression/candidate-components.sha256" \
    --frozen-namespace-pre "$DELIVERY_ROOT/evidence/frozen-namespaces-pre.tsv" \
    --frozen-namespace-post "$DELIVERY_ROOT/evidence/frozen-namespaces-post.tsv"
  (
    cd "$DELIVERY_ROOT"
    sha256sum -c evidence/toolchain-manifest.sha256
    sha256sum -c evidence/identity-manifest.sha256
    sha256sum -c evidence/results-manifest.sha256
  ) | tee "$ACCEPT_ROOT/commands/sealed-manifests-pre.log"

  export LIVE_TRITON="$DELIVERY_ROOT/src/triton"
  export LIVE_VENV="$DELIVERY_ROOT/venv"
  export PYTHONDONTWRITEBYTECODE=1
  bash "$CAPSTONE/scripts/hash_compiler_tree.sh" \
    --root "$LIVE_TRITON" \
    --output "$ACCEPT_ROOT/evidence/accept-source.sha256"
  bash "$CAPSTONE/scripts/hash_loaded_build.sh" \
    --root "$LIVE_TRITON" --python "$LIVE_VENV/bin/python" \
    --build-inputs-output "$ACCEPT_ROOT/evidence/accept-build-inputs.sha256" \
    --loaded-artifacts-output "$ACCEPT_ROOT/evidence/accept-loaded-artifacts.sha256"
  cmp "$ACCEPT_ROOT/evidence/accept-source.sha256" \
    "$CAPSTONE/results/regression/candidate-source-files.sha256"
  cmp "$ACCEPT_ROOT/evidence/accept-build-inputs.sha256" \
    "$CAPSTONE/results/regression/candidate-build-inputs.sha256"
  cmp "$ACCEPT_ROOT/evidence/accept-loaded-artifacts.sha256" \
    "$CAPSTONE/results/regression/candidate-loaded-artifacts.sha256"
  verify_protected_identity \
    | tee "$ACCEPT_ROOT/commands/protected-identity.log"

  git clone --no-checkout "$CAPSTONE/compiler/triton-source.bundle" \
    "$ACCEPT_ROOT/src/triton-verify"
  BASELINE_COMMIT=$(cat "$CAPSTONE/compiler/baseline_commit.txt")
  git -C "$ACCEPT_ROOT/src/triton-verify" checkout --detach "$BASELINE_COMMIT"
  git -C "$ACCEPT_ROOT/src/triton-verify" apply --check \
    "$CAPSTONE/compiler/patches/final.patch"

  bash "$CAPSTONE/scripts/accept_delivery_report.sh" \
    --registry-record "$REGISTRY_RECORD" --repro-id "$REPRO_ID" \
    --delivery-root "$DELIVERY_ROOT" \
    --correctness "$DELIVERY_ROOT/correctness/results.json" \
    --paired-csv "$DELIVERY_ROOT/benchmark/paired.csv" \
    --regression-summary "$DELIVERY_ROOT/benchmark/final-summary.json" \
    --profile-report "$DELIVERY_ROOT/profiles/ncu/M512_N4096_K4096.ncu-rep" \
    --profile-metadata "$DELIVERY_ROOT/profiles/ncu/M512_N4096_K4096.metadata.json" \
    --thresholds "$CAPSTONE/results/regression/thresholds.json" \
    --output "$ACCEPT_ROOT/report/final-acceptance.json"

  bash "$CAPSTONE/scripts/verify_delivery_reproduction.sh" \
    --mode verify-sealed --registry-record "$REGISTRY_RECORD" \
    --delivery-root "$DELIVERY_ROOT" \
    --bundle "$CAPSTONE/compiler/triton-source.bundle" \
    --candidate-components "$CAPSTONE/results/regression/candidate-components.sha256" \
    --frozen-namespace-pre "$DELIVERY_ROOT/evidence/frozen-namespaces-pre.tsv" \
    --frozen-namespace-post "$DELIVERY_ROOT/evidence/frozen-namespaces-post.tsv"
  verify_frozen_manifest \
    | tee "$ACCEPT_ROOT/commands/frozen-manifest-post.log"
  test -s "$ACCEPT_ROOT/commands/frozen-manifest-pre.log"
  test -s "$ACCEPT_ROOT/commands/frozen-manifest-post.log"
  }
  declare -f acceptance_main > "$ACCEPT_ROOT/commands/acceptance-main.sh"
  # shellcheck disable=SC2016
  env -i LANG=C.UTF-8 LC_ALL=C \
    CAPSTONE="$CAPSTONE" ACCEPT_ROOT="$ACCEPT_ROOT" \
    TMPDIR="$ACCEPT_ROOT/tmp" \
    PROTECTED_PREREG_REMOTE="$PROTECTED_PREREG_REMOTE" \
    PROTECTED_PREREG_REGISTRY="$PROTECTED_PREREG_REGISTRY" \
    PROTECTED_DELIVERY_REGISTRY="$PROTECTED_DELIVERY_REGISTRY" \
    FROZEN_BASH="$FROZEN_BASH" FROZEN_PYTHON3="$FROZEN_PYTHON3" \
    FROZEN_PYTHON_VERSION="$FROZEN_PYTHON_VERSION" \
    PATH="$FROZEN_PATH" CUDA_HOME="$FROZEN_CUDA_HOME" \
    CC="$FROZEN_CC" CXX="$FROZEN_CXX" \
    LD_LIBRARY_PATH="$FROZEN_LD_LIBRARY_PATH" \
    CMAKE_PREFIX_PATH="$FROZEN_CMAKE_PREFIX_PATH" \
    LLVM_SYSPATH="$FROZEN_LLVM_SYSPATH" \
    "$FROZEN_BASH" --noprofile --norc -c \
      'source "$1"; acceptance_main' _ \
      "$ACCEPT_ROOT/commands/acceptance-main.sh"
  ```

  Expected：无论调用者预设何种 Python/Conda/PIP/Triton 污染变量，`env -i` 内都观测不到它们；PATH、Python、CUDA、CC/CXX 与 search paths 来自 sealed toolchain 对应的冻结 allowlist。protected final pointer 只返回一个 immutable `REPRO_ID`；其值等于 sealed cold manifest hash，路径精确为 `results/delivery/$REPRO_ID`，目录无 writable entry。sealed tree、bundle、toolchain、normalized components、correctness/paired/regression/profile 与 frozen namespace 全部通过；patch 在 `$ACCEPT_ROOT/src/triton-verify` 通过 `apply --check`。只读 acceptance 返回 0 且 sealed directory 前后不变，新输出只写 `final-acceptance`。

  - **项目化必读（目的：以干净 patch、真实 test 与不可变 identity 完成最终验收）：** 阅读 [`git apply --check`](https://git-scm.com/docs/git-apply)、[Triton 官方 source build](https://github.com/triton-lang/triton/blob/main/README.md#install-from-source) 与 [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/)，并逐项复核 Phase 6 Final Exit Gate、protected verifier 和 manifest 字段契约。
  - **Expected：** 所有只读验收命令返回 0；六组及完整 supported/fallback gate、patch/test/reproducer、ablation/profile、成功/失败文档和 manifest lookup 都来自 registry 固定的同一 `REPRO_ID`。验收前后 sealed tree 与 frozen namespaces 未变化，只有 `final-acceptance` 新增报告。

### Final Exit Gate

```text
硬验收全部满足
六组 shape correctness 全部通过
至少四组达到 Triton baseline 90%
任何一组不低于 baseline 80%
compiler patch/test/reproducer 可复现
性能结论有 ablation 和 profiler 证据
成功与失败实验均已文档化
冷环境可按 README 独立完成 install/build/test/correctness/benchmark/profile
README 的每条交付路径都有 cwd、只读输入、完整命令、Expected 与 artifact reference
manifest 字段来源明确，artifact/hash/command/dependency/identity 全部闭合
Phase 6 protected prereg tag、peeled commit、candidate identity 与 final manifest 前后相同
所有 cold/final replay 只写新 delivery 目录，Phase 4/5/6 冻结证据未被覆盖
frozen final-manifest 在 cold/final replay 前后各校验一次且两次均 PASS，pre/post 日志可定位
Week 24 只消费 protected registry 唯一 REPRO_ID，sealed tree、normalized components 与结果 hash 前后不变
最终报告和五问自我答辩均能从 claim 定位到原始 artifact 与重放命令
```

[返回 24 周总索引进行最终验收 →](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)
