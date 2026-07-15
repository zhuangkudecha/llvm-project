# Phase 1：CUDA 与 Triton Kernel 基础

[← 返回 24 周总索引](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md)

- 周期：Week 1–4（2026-07-15 至 2026-08-11）
- 计划工时：44h

## 前置条件

无（本阶段是 24 周计划的起点）。

## 可验证阶段目标

- 建立可复现的本地与 RTX 4090 环境记录。
- 能实现并验证 CUDA 基础 kernel 与 Triton fused kernel。
- 形成 correctness 优先、同步后计时的 GPU 测量纪律。

## 本阶段产出

- `GPUCodeGenCapstone` 固定目录、README 与进度记录。
- CUDA vector add、reduction、naive/tiled matmul。
- Triton vector add、softmax、matmul 与 fused linear 初版。
- 环境验收、correctness、测量结果与阶段复盘材料。

> 外部链接核验日期：2026-07-14

## Week 1：项目初始化与 GPU 执行模型

**Files:**

- Create: `LearningSteps/GPUCodeGenCapstone/README.md`
- Create: `LearningSteps/GPUCodeGenCapstone/progress.md`
- Create: `LearningSteps/GPUCodeGenCapstone/environment/local.md`
- Create: `LearningSteps/GPUCodeGenCapstone/environment/rtx4090.md`
- Create: `LearningSteps/GPUCodeGenCapstone/cuda/vector_add.cu`
- Create: `LearningSteps/GPUCodeGenCapstone/cuda/CMakeLists.txt`
- Create: `LearningSteps/GPUCodeGenCapstone/docs/cuda_kernel_notes.md`
- Create: remaining directories from [总索引 §2.2 最终目录](../RTX4090_GPU_CodeGen_24Week_Execution_Plan.md#22-最终目录)

### 工作日

- [ ] **周一：记录本地环境**

运行：

```bash
uname -a
cmake --version
ninja --version
$MLIR_BIN/mlir-opt --version
$MLIR_BIN/llvm-lit --version
```

将输出和当前 Git commit 写入 `environment/local.md`。

**参考资料：**

- **必读文档：** [CMake `--version` 与命令行用法](https://cmake.org/cmake/help/latest/manual/cmake.1.html)——确认如何记录可复现实验所需的 CMake 版本与生成器入口。
- **必读文档：** [Git `git-rev-parse` 文档](https://git-scm.com/docs/git-rev-parse)——明确如何取得并保存可唯一定位源码状态的 commit。

- [ ] **周二：画出 CUDA 执行层级**

在 `docs/cuda_kernel_notes.md` 写出：

```text
kernel -> grid -> block -> warp -> thread
SM -> warp scheduler -> CUDA core/Tensor Core
```

为每个层级写明它负责的并行粒度和可见资源。

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programming-model)——建立 grid、block、thread 与 memory hierarchy 的准确对应关系。
- **必读文档：** [CUDA C++ Programming Guide：Hardware Implementation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#hardware-implementation)——把 warp、warp scheduler、SM 与 CUDA/Tensor Core 放到真实硬件执行关系中。

- [ ] **周三：编写 CUDA vector add**

实现接口：

```cpp
__global__ void vectorAdd(const float *a, const float *b, float *c, int64_t n);
```

必须包含 `idx < n` 边界判断和 CPU reference。

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#kernels)——掌握 `__global__`、内建索引和 execution configuration，完成带越界保护的 vector add。
- **实践参考：** [CUDA Samples：vectorAdd](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/vectorAdd)——对照 NVIDIA 官方样例的 host/device 数据流与结果检查方式，但自行实现本计划的 CPU reference 和三组边界 case。

- [ ] **周四：加入错误检查**

为 `cudaMalloc`、`cudaMemcpy`、kernel launch 和 `cudaDeviceSynchronize` 加统一错误检查宏；添加 `n=1`、`n=1025`、`n=1<<20` 三组 case。

**参考资料：**

- **必读文档：** [CUDA Runtime API：Error Handling](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html)——区分 API 返回错误与异步 kernel 错误，设计统一检查宏。
- **必读文档：** [CUDA C++ Programming Guide：Error Checking](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#error-checking)——理解同步点为何能暴露先前异步 launch 的失败，并覆盖非整 block 的 `n=1025`。

- [ ] **周五：准备租赁环境验收清单**

在 `environment/rtx4090.md` 预先写入：

```text
GPU exact name
VRAM
driver version
CUDA version
power limit
ncu version
nsys version
performance-counter permission
persistent storage path
```

**参考资料：**

- **必读文档：** [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/)——确认 GPU、显存、驱动、时钟、温度与 power limit 的可查询字段。
- **必读文档：** [NVIDIA：ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)——提前识别性能计数器权限要求，避免租到无法运行 `ncu` 的实例。

### 周末

- [ ] **周六：建立 CUDA 构建文件和项目 README**

`cuda/CMakeLists.txt` 至少定义 `vector_add` target，并启用 C++17/CUDA17。README 写清当前目标、目录和第一条构建命令。

**参考资料：**

- **必读文档：** [CMake：`CUDA_STANDARD`](https://cmake.org/cmake/help/latest/prop_tgt/CUDA_STANDARD.html)——确认 target 请求 CUDA17 标准的声明方式及 `CUDA_STANDARD_REQUIRED` 的作用。
- **必读文档：** [CMake `add_executable`](https://cmake.org/cmake/help/latest/command/add_executable.html)——用最小且可复现的 target 定义连接 `vector_add.cu`。

- [ ] **周日：本地结构验证与复盘**

运行：

```bash
test -f $CAPSTONE/cuda/vector_add.cu
test -f $CAPSTONE/environment/local.md
test -f $CAPSTONE/progress.md
find $CAPSTONE -maxdepth 2 -type d | sort
```

Expected：所有固定目录和 Week 1 文件存在。将验证命令写入 `progress.md`。

**参考资料：**

- **必读文档：** [POSIX `test` utility](https://pubs.opengroup.org/onlinepubs/9799919799/utilities/test.html)——准确理解 `-f` 的成功条件，让结构验收命令可作为明确的 pass/fail 证据。
- **必读文档：** [GNU Findutils：Directories](https://www.gnu.org/software/findutils/manual/html_node/find_html/Directories.html)——理解 `-maxdepth` 的遍历边界，保证目录清单稳定且不过度展开构建产物。

### Week 1 Exit Gate

```text
能准确解释 grid/block/warp/thread
vector_add.cu 包含边界处理、CPU reference 和 CUDA error checking
租赁环境验收表已经准备好
```

---

## Week 2：首次 RTX 4090 实验与测量纪律

**Files:**

- Update: `environment/rtx4090.md`
- Create: `scripts/check_environment.sh`
- Update: `cuda/vector_add.cu`
- Create: `results/experiments/week02-vector-add.csv`

### 工作日

- [ ] **周一：租用并验收 RTX 4090 实例**

运行：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv
nvcc --version
ncu --version
nsys --version
cmake -S $CAPSTONE/cuda -B $CAPSTONE/cuda/build -G Ninja
cmake --build $CAPSTONE/cuda/build --target vector_add
mkdir -p $CAPSTONE/profiles/ncu
ncu --set basic --force-overwrite \
  --export $CAPSTONE/profiles/ncu/week02-permission-smoke \
  $CAPSTONE/cuda/build/vector_add \
  >$CAPSTONE/profiles/ncu/week02-permission-smoke.log 2>&1
ncu_rc=$?
cat $CAPSTONE/profiles/ncu/week02-permission-smoke.log
test $ncu_rc -eq 0
test -s $CAPSTONE/profiles/ncu/week02-permission-smoke.ncu-rep
! grep -q ERR_NVGPUCTRPERM $CAPSTONE/profiles/ncu/week02-permission-smoke.log
ncu --import $CAPSTONE/profiles/ncu/week02-permission-smoke.ncu-rep \
  --page raw --csv \
  >$CAPSTONE/profiles/ncu/week02-permission-smoke.csv
ncu_import_rc=$?
test $ncu_import_rc -eq 0
test -s $CAPSTONE/profiles/ncu/week02-permission-smoke.csv
grep -Eq 'sm__|gpu__|dram__|launch__' \
  $CAPSTONE/profiles/ncu/week02-permission-smoke.csv
```

Expected：GPU 名称包含 RTX 4090；真实 `vector_add` profiling 的 `ncu_rc=0`，日志中无 `ERR_NVGPUCTRPERM`，非空 `.ncu-rep` 可成功导入且 CSV 中至少出现一项 metric。任一检查失败则停止 GPU 计划并更换实例，不接受无 profiler 权限的服务器。

**参考资料：**

- **必读文档：** [Nsight Compute CLI：Quickstart](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#quickstart)——按真实 kernel profiling、report export 与 report import 流程验收 metric 采集，而不把 metric 名称查询误当作计数器权限证明。
- **必读文档：** [NVIDIA：ERR_NVGPUCTRPERM](https://developer.nvidia.com/ERR_NVGPUCTRPERM)——识别真实 profiling 时的性能计数器权限失败，并将其作为立即更换实例的条件。
- **实践参考：** [Nsight Systems User Guide：Command Line Options](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#command-line-options)——核对 `nsys` CLI 安装和版本查询入口，为后续系统级时间线分析留证。

- [ ] **周二：编译运行 vector add**

```bash
cmake -S $CAPSTONE/cuda -B $CAPSTONE/cuda/build -G Ninja
cmake --build $CAPSTONE/cuda/build --target vector_add
$CAPSTONE/cuda/build/vector_add
```

Expected：三组 size 全部与 CPU reference 一致。

**参考资料：**

- **必读文档：** [CMake buildsystem 手册](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html)——理解 configure、生成 Ninja buildsystem 与指定 target 构建三步的边界。
- **实践参考：** [CUDA Samples：vectorAdd](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/0_Introduction/vectorAdd)——核对官方样例的 device-to-host 拷贝后再比较结果这一 correctness 顺序。

- [ ] **周三：加入 CUDA Event timing**

只测 kernel execution；初始化、allocation、H2D/D2H 分开记录。至少 warmup 20 次、测量 100 次。

**参考资料：**

- **必读文档：** [CUDA Best Practices Guide：Timing](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#timing)——理解 GPU 异步执行、CUDA Event 计时及同步要求，避免把 host enqueue 时间误当 kernel 时间。
- **必读文档：** [CUDA Runtime API：Event Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)——确认 event record、synchronize 与 elapsed time 的合法调用顺序。

- [ ] **周四：运行 compute-sanitizer**

```bash
compute-sanitizer --tool memcheck $CAPSTONE/cuda/build/vector_add
```

Expected：0 errors。

**参考资料：**

- **必读文档：** [Compute Sanitizer：Memcheck Tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool)——理解 memcheck 能发现的越界、misaligned 与硬件异常，并以 0 errors 作为本日门槛。

- [ ] **周五：记录第一个 ncu report**

```bash
ncu --set basic --target-processes all \
  --export $CAPSTONE/profiles/ncu/week02-vector-add \
  $CAPSTONE/cuda/build/vector_add
```

**参考资料：**

- **必读文档：** [Nsight Compute CLI：Quickstart](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#quickstart)——掌握 section set、目标进程与 `--export` report 的基本采集流程。
- **必读文档：** [Nsight Compute：Profiler Report](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#profiler-report)——知道导出的 `.ncu-rep` 中 Details/Raw/Source 页面分别承载什么证据。

### 周末

- [ ] **周六：做 block-size ablation**

测量 block size `64/128/256/512`，每项至少三轮；CSV 列：

```text
timestamp,gpu,driver,cuda,n,block_size,median_us,p20_us,p80_us,bandwidth_gbps
```

**参考资料：**

- **必读文档：** [CUDA Best Practices Guide：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#occupancy)——解释 block size 如何经 active warps、register 与 shared-memory 约束影响 occupancy，而非预设单调关系。
- **必读文档：** [CUDA Best Practices Guide：Bandwidth](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#bandwidth)——按读写字节数和耗时计算 `bandwidth_gbps`，并区分理论与有效带宽。

- [ ] **周日：解释结果**

在 `docs/cuda_kernel_notes.md` 回答：

```text
为什么极小输入主要受 launch latency 影响？
为什么 block size 改变不一定线性改变性能？
有效带宽如何计算？
```

**参考资料：**

- **必读文档：** [CUDA Best Practices Guide：Performance Metrics](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#performance-metrics)——用执行时间和有效带宽组织复盘，避免只比较裸延迟。
- **必读文档：** [CUDA C++ Programming Guide：Maximize Utilization](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#maximize-utilization)——从 launch 数量、block/warp 调度与 multiprocessor utilization 解释小输入和 block-size 结果。

### Week 2 Exit Gate

```text
RTX 4090、ncu、nsys 权限确认
vector add correctness 和 memcheck 通过
有第一份可复现 CSV 和 ncu report
```

---

## Week 3：Reduction 与 CUDA Matmul

**Files:**

- Create: `cuda/reduction.cu`
- Create: `cuda/matmul_naive.cu`
- Create: `cuda/matmul_tiled.cu`
- Create: `results/experiments/week03-cuda-matmul.csv`

### 工作日

- [ ] **周一：实现 reduction baseline**

实现一个 block 内 shared-memory tree reduction；使用 FP32 输入，和 CPU double accumulation reference 比较。

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory)——理解 block 内共享数据的生命周期与可见范围，正确放置 reduction 中间值。
- **必读源码：** [CUDA Samples：reduction](https://github.com/NVIDIA/cuda-samples/tree/master/cpp/2_Concepts_and_Techniques/reduction)——阅读 NVIDIA 官方 reduction 各阶段实现，重点辨认 tree reduction、同步位置与 reference 检查；本日只实现计划要求的 block 内 baseline。

- [ ] **周二：实现 naive matmul**

接口固定为 row-major：

```text
A[M,K] × B[K,N] -> C[M,N]
```

每个 thread 计算一个 `C[m,n]`，先只覆盖 `M=N=K=512`。

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Matrix Multiplication without Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory)——把 row-major 下的 thread-to-output 映射和 A/B 索引公式落实到 naive kernel。

- [ ] **周三：实现 shared-memory tiled matmul**

先固定 `TILE=16`，包含 K 维循环、两次 `__syncthreads()` 和边界 mask。

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Matrix Multiplication with Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-shared-memory)——逐步理解 cooperative tile load、K-tile 累加与两个 barrier 的必要性。
- **必读文档：** [CUDA C++ Programming Guide：Synchronization Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)——确认 `__syncthreads()` 是 block barrier，并避免让部分 thread 因边界分支跳过 barrier。

- [ ] **周四：加入非整除 case**

至少加入：

```text
(M,N,K)=(127,251,509)
(M,N,K)=(513,769,1025)
```

**参考资料：**

- **必读文档：** [CUDA C++ Programming Guide：Thread Hierarchy](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-hierarchy)——根据 grid/block 向上取整后的多余 thread，分别推导 M/N 输出 mask 和 K-tail 输入 mask。
- **实践参考：** [Compute Sanitizer：Memcheck Tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#memcheck-tool)——用非整除 case 检查 tail tile 的非法 global/shared memory 访问。

- [ ] **周五：检查访存模式**

在笔记中逐式写出 A/B global address，并说明哪个维度连续、哪些 load 能 coalesce。

**参考资料：**

- **必读文档：** [CUDA Best Practices Guide：Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#coalesced-access-to-global-memory)——把相邻 thread 的 A/B 地址代入 cache-line/transaction 规则，判断 load 是否 coalesce。

### 周末

- [ ] **周六：比较 naive 与 tiled**

```bash
cmake --build $CAPSTONE/cuda/build
compute-sanitizer --tool memcheck $CAPSTONE/cuda/build/matmul_tiled
$CAPSTONE/cuda/build/matmul_naive --m 1024 --n 1024 --k 1024
$CAPSTONE/cuda/build/matmul_tiled --m 1024 --n 1024 --k 1024
```

记录 latency、effective TFLOPS 和 correctness。

**参考资料：**

- **必读文档：** [CUDA Best Practices Guide：Effective Bandwidth Calculation](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#effective-bandwidth-calculation)——建立统一的 bytes/time 口径，与 TFLOPS 一起判断 kernel 是计算还是带宽受限。
- **必读文档：** [CUDA Best Practices Guide：Shared Memory and Memory Banks](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks)——检查 tiled 版本是否以减少 global load 的代价引入 shared-memory bank conflict。

- [ ] **周日：采集 ncu 并复盘**

```bash
ncu --set full --kernel-name regex:matmul \
  --export $CAPSTONE/profiles/ncu/week03-matmul \
  $CAPSTONE/cuda/build/matmul_tiled --m 1024 --n 1024 --k 1024
```

重点记录 memory throughput、occupancy、register、shared memory 和 stall reason。

**参考资料：**

- **必读文档：** [Nsight Compute Profiling Guide：Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide)——正确解读 throughput、occupancy 与 warp stall 指标，避免把单个百分比直接当根因。
- **必读文档：** [Nsight Compute CLI：Profile Options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html#command-line-options-profile)——理解 kernel regex、`--set full` 和 report export 的采集范围与 replay 成本。

### Week 3 Exit Gate

```text
naive/tiled matmul 均通过整除和非整除 correctness
能解释 tiled 版本减少了哪些 global load
能指出当前版本仍未接近 cuBLAS/Triton 的至少三个原因
```

---

## Week 4：Triton Kernel 与 Fused Linear

**Files:**

- Create: `triton_kernels/vector_add.py`
- Create: `triton_kernels/fused_softmax.py`
- Create: `triton_kernels/matmul.py`
- Create: `triton_kernels/fused_linear.py`
- Create: `results/experiments/week04-triton.csv`

### 工作日

- [ ] **周一：运行并重写 Triton vector add**

先运行官方 tutorial，再关闭 tutorial，从空文件实现 `tl.program_id/tl.arange/tl.load/tl.store` 版本。

**参考资料：**

- **必读文档：** [Triton 官方教程：Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)——理解 Triton program instance、block offset、mask 和 load/store；关闭教程后用这些概念独立重写。

- [ ] **周二：运行 fused softmax 并解释 row mapping**

记录一个 Triton program instance 处理哪一行、mask 如何处理边界、为什么 fusion 减少 global traffic。

**参考资料：**

- **必读文档：** [Triton 官方教程：Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)——追踪 program-to-row mapping、power-of-two padding mask 和单 kernel fusion 的读写次数。

- [ ] **周三：运行 matmul tutorial**

固定当前环境的 Triton package/version，完成官方 correctness 和 benchmark。

**参考资料：**

- **必读文档：** [Triton 官方教程：Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)——复现官方 block matmul、grouped ordering、correctness 和 benchmark，并记录可复现版本。
- **实践参考：** [Triton 安装文档](https://triton-lang.org/main/getting-started/installation.html)——确认 package 安装来源与版本记录方式，避免环境漂移使 benchmark 不可比较。

- [ ] **周四：实现 fused Linear + Bias + ReLU**

固定语义：

```text
FP16 A/B/bias
FP32 accumulator
acc += bias in FP32
relu in FP32
FP16 output
```

**参考资料：**

- **必读文档：** [Triton 官方教程：Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)——沿官方 matmul epilogue 与 fused activation 示例扩展 bias 和 ReLU，同时保持 accumulator/epilogue 的 FP32 语义。
- **必读文档：** [Triton `tl.dot` API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)——核对输入、accumulator 与输出精度约束，防止隐式降精度破坏固定语义。

- [ ] **周五：加入三组最小 correctness**

```text
(32,4096,4096)
(512,4096,4096)
(127,251,509)
```

reference 使用 PyTorch FP32 matmul + bias + ReLU 后 cast FP16。

**参考资料：**

- **必读文档：** [Triton 官方教程：Vector Addition（Compute Kernel）](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html#compute-kernel)——从当前有效的 compute-kernel 章节复核 mask 对非整除边界的处理，再沿教程页面中的 PyTorch reference 比较方式设计三组 correctness。
- **必读文档：** [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)——理解浮点 matmul 的非结合性与误差来源，为 FP32 reference 到 FP16 输出选择合理容差。

### 周末

- [ ] **周六：增加 autotune**

至少搜索：

```text
BLOCK_M: 32,64,128
BLOCK_N: 32,64,128
BLOCK_K: 32,64
num_warps: 4,8
num_stages: 2,3,4
```

过滤明显超资源或不合法组合，key 固定为 `M,N,K`。

**参考资料：**

- **必读文档：** [Triton `triton.autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html)——理解 config、key 与 benchmark cache 如何按 `M,N,K` 选择配置。
- **必读文档：** [Triton 官方 Matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)——参考其中的 autotuning configurations 组织 `BLOCK_*`、`num_warps`、`num_stages`，并识别需预过滤的资源组合。

- [ ] **周日：做第一轮 Triton 结果分析**

使用 `triton.testing.do_bench` 输出 median/P20/P80；写清 CUDA thread-level 与 Triton blocked programming model 的差异。

**参考资料：**

- **必读文档：** [Triton `triton.testing.do_bench` API](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)——核对 quantiles 参数与返回顺序，确保 median/P20/P80 CSV 列不被写反。
- **必读文档：** [Triton Programming Guide：Introduction](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)——用 block-based program instance 与 CUDA SIMT thread hierarchy 的差异组织本周复盘。

### Phase 1 Exit Gate

```text
独立实现 CUDA tiled matmul
独立实现 Triton fused Linear
能够解释 coalescing/shared memory/occupancy/tile trade-off
RTX 4090 benchmark 和 ncu 已真实运行
```

未通过时重复 Week 3-4，不进入 Phase 2。

---

[进入 Phase 2：MLIR Transformation Bridge →](./Phase2_MLIR_Transformation_Bridge.md)
