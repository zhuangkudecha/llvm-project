// =====================================================================
// transformer.cu — 单 Transformer Block 前向（FP32, CUDA C++）
//
// 对应 24 周 GPU CodeGen 计划 §16.1 扩展路线里的 attention/softmax/
// layernorm 练习。所有 kernel 从零手写，复用 Phase 1 学到的技能：
//   - tiled matmul（16x16 tile + 两次 __syncthreads + 边界 mask）
//   - block 内 tree reduction（double 累加，reference 也是 double）
//   - 统一 CUDA 错误检查宏
//   - correctness 优先、同步后计时（warmup 20 / 测量 100 次）
//
// 计算图（GPT-2 风格 decoder block，无 causal mask，见 README）：
//
//   X ──┬─ LN1 ── QKV ── Attention ── O ── (+) ── H1 ── LN2 ── FF1(+b) ── GELU ── FF2(+b) ── (+) ── Out
//       │                                   ↑                          ↑
//       └───────────────────────────────────┘                          └──────────────┘
//                              (residual)                              (residual)
//
// 权重布局（全部 row-major）：
//   Wqkv [D, 3D]   Wo [D, D]   W1 [D, FF]  b1 [FF]   W2 [FF, D]  b2 [D]
//   g1/b1/g2/b2 [D]（LayerNorm 的 scale/shift）
//
// 正确性纪律：CPU double 累加 reference 逐元素对比；所有 CUDA 调用走
// CHECK_CUDA；kernel launch 后立即 cudaGetLastError 暴露配置错误。
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

// 统一错误检查宏：API 返回错误与 kernel 异步错误都要能定位到文件和行
#define CHECK_CUDA(call)                                                    \
  do {                                                                      \
    cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,      \
              __LINE__, cudaGetErrorString(err_));                          \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

namespace {

constexpr int TILE = 16;           // matmul tile 大小（与 Phase 1 一致）
constexpr int ATTEN_THREADS = 256; // attention/layernorm block 线程数
constexpr float LN_EPS = 1e-5f;

// ---------------------------------------------------------------------
// 1) tiled matmul：C[M,N] = A[M,K] * B[K,N]（可选 bias 沿列广播）
//    block 16x16 线程算一个 16x16 输出 tile；A/B 的越界元素以 0 填充，
//    保证非整除尺寸下不读非法地址。
//    访存：As 加载在 warp 内沿 threadIdx.y 连续，Bs 沿 threadIdx.x 连续，
//    两者都构成 coalesced/广播访问；每轮 tile 两次 __syncthreads()。
// ---------------------------------------------------------------------
template <bool HasBias>
__global__ void matmulTiledKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  const float* __restrict__ bias,
                                  float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int row = blockIdx.y * TILE + threadIdx.y;
  const int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int ka = k0 + threadIdx.y; // A[row, ka]
    const int kb = k0 + threadIdx.x; // B[kb, col]
    As[threadIdx.y][threadIdx.x] =
        (row < M && ka < K) ? A[(size_t)row * K + ka] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] =
        (col < N && kb < K) ? B[(size_t)kb * N + col] : 0.0f;
    __syncthreads(); // 全部 shared tile 加载完成后才能读
#pragma unroll
    for (int k = 0; k < TILE; ++k)
      acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads(); // 下一轮覆盖 As/Bs 前，确保无人再读
  }
  if (row < M && col < N) {
    if (HasBias) acc += bias[col];
    C[(size_t)row * N + col] = acc;
  }
}

// ---------------------------------------------------------------------
// 2) LayerNorm：一个 block 负责一行（N 个 block）。
//    一行两遍扫描：第一遍同时累加 sum 与 sumsq（double），tree 归约后
//    得到 mean/var；第二遍归一化。var 用 E[x^2]-E[x]^2 可能因浮点误差
//    为负，需要 clamp。
// ---------------------------------------------------------------------
__global__ void layernormKernel(const float* __restrict__ x,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ y, int N, int D,
                                float eps) {
  const int i = blockIdx.x;
  const float* row = x + (size_t)i * D;
  float* yrow = y + (size_t)i * D;
  const int tid = threadIdx.x;

  __shared__ double s1[ATTEN_THREADS];
  __shared__ double s2[ATTEN_THREADS];
  __shared__ double mean_s;
  __shared__ double var_s;

  double sum = 0.0, sumsq = 0.0;
  for (int j = tid; j < D; j += blockDim.x) {
    const double v = row[j];
    sum += v;
    sumsq += v * v;
  }
  s1[tid] = sum;
  s2[tid] = sumsq;
  __syncthreads();
  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      s1[tid] += s1[tid + off];
      s2[tid] += s2[tid + off];
    }
    __syncthreads();
  }
  if (tid == 0) {
    mean_s = s1[0] / D;
    var_s = s2[0] / D - mean_s * mean_s;
    if (var_s < 0.0) var_s = 0.0;
  }
  __syncthreads();
  const double inv_std = 1.0 / sqrt(var_s + eps);
  for (int j = tid; j < D; j += blockDim.x)
    yrow[j] = (float)((row[j] - mean_s) * inv_std * gamma[j] + beta[j]);
}

// ---------------------------------------------------------------------
// 3) fused attention：一个 block 负责 (query 行 i, head h) 的输出行
//        O[i, :] = softmax( Q[i,:] * K^T / sqrt(d_head) ) * V
//    在线 softmax（flash attention 的 rescale 技巧）：只扫一遍 K/V，
//    用增量 max 纠正指数缩放，避免把 N*N 的 scores 矩阵写进 global
//    memory 再读回来 —— 这正是 fused attention 省 global traffic 的
//    关键。每个 key 要两次 block 同步（s 归约 + m/l 发布）。
//
//    约束：d_head <= ATTEN_THREADS（默认 64，足够；大 d_head 时线程
//    stride 循环覆盖）。
// ---------------------------------------------------------------------
__global__ void attentionKernel(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O, int N, int inStride,
                                int outStride, int d_head, float scale) {
  const int i = blockIdx.x; // query 行
  const int h = blockIdx.y; // head
  const int tid = threadIdx.x;
  const int col0 = h * d_head; // 该 head 在行内的列起点

  __shared__ float qs[ATTEN_THREADS]; // Q 行拷贝（只读一次）
  __shared__ float o[ATTEN_THREADS];  // 输出行累加器（每线程独占自己列）
  __shared__ float s_part[ATTEN_THREADS];
  __shared__ float s_scaled_s, m_s, l_s, alpha_s, p_s;

  if (tid == 0) {
    m_s = -INFINITY;
    l_s = 0.0f;
  }
  for (int c = tid; c < d_head; c += ATTEN_THREADS) {
    qs[c] = Q[(size_t)i * inStride + col0 + c];
    o[c] = 0.0f;
  }
  __syncthreads(); // 确保 qs/m_s/l_s 对所有线程可见

  for (int j = 0; j < N; ++j) {
    // (a) 部分和 -> tree 归约，得到 s = q_i . k_j
    float s = 0.0f;
    for (int c = tid; c < d_head; c += ATTEN_THREADS)
      s += qs[c] * K[(size_t)j * inStride + col0 + c];
    s_part[tid] = s;
    __syncthreads();
    for (int off = ATTEN_THREADS / 2; off > 0; off >>= 1) {
      if (tid < off) s_part[tid] += s_part[tid + off];
      __syncthreads();
    }

    // (b) 在线 softmax 更新：m_new = max(m, s)，
    //     alpha = exp(m - m_new) 重缩放历史累加，p = exp(s - m_new)。
    //     m/l/alpha/p 都要广播，所以 thread 0 写完必须再同步一次。
    if (tid == 0) {
      s_scaled_s = s_part[0] * scale;
      const float m_new = fmaxf(m_s, s_scaled_s);
      alpha_s = expf(m_s - m_new);
      p_s = expf(s_scaled_s - m_new);
      m_s = m_new;
      l_s = l_s * alpha_s + p_s;
    }
    __syncthreads();

    // (c) 重缩放累加器并吸收 V 行。每线程写自己的 o[c]，无竞争；
    //     V 的列地址沿 tid 连续 -> coalesced。
    for (int c = tid; c < d_head; c += ATTEN_THREADS)
      o[c] = o[c] * alpha_s + p_s * V[(size_t)j * inStride + col0 + c];
  }

  __syncthreads(); // 防御性同步（实际无竞争：每线程只读写自己的 o[c]）
  for (int c = tid; c < d_head; c += ATTEN_THREADS)
    O[(size_t)i * outStride + col0 + c] = o[c] / l_s;
}

// ---------------------------------------------------------------------
// 4) GELU（tanh 近似，GPT-2 风格）：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
//    与 CPU reference 使用同一公式，比较才有意义。
// ---------------------------------------------------------------------
__global__ void geluKernel(const float* __restrict__ x,
                           float* __restrict__ y, size_t n) {
  constexpr float c = 0.7978845608f; // sqrt(2/pi)
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n;
       idx += (size_t)gridDim.x * blockDim.x)
    y[idx] =
        0.5f * x[idx] * (1.0f + tanhf(c * (x[idx] + 0.044715f * x[idx] *
                                                       x[idx] * x[idx])));
}

// ---------------------------------------------------------------------
// 5) 残差连接：out = a + b（逐元素，grid-stride loop）
// ---------------------------------------------------------------------
__global__ void residualAddKernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ out, size_t n) {
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n;
       idx += (size_t)gridDim.x * blockDim.x)
    out[idx] = a[idx] + b[idx];
}

struct BlockParams {
  int N;      // sequence length
  int D;      // hidden dim
  int H;      // number of heads
  int d_head; // D / H
  int FF;     // feed-forward hidden dim
};

} // namespace

// ---------------------------------------------------------------------
// host 前向：按计算图顺序 launch 全部 kernel。所有 kernel 在同一默认
// stream 上串行执行，每次 launch 后立即检查，异步错误不拖到后面才暴露。
// ---------------------------------------------------------------------
void transformerBlockForward(const float* dX, const float* dWqkv,
                             const float* dWo, const float* dW1,
                             const float* db1, const float* dW2,
                             const float* db2, const float* dg1,
                             const float* dbeta1, const float* dg2,
                             const float* dbeta2, float* dOut,
                             const BlockParams& p, float* dLN1, float* dQKV,
                             float* dAttn, float* dOatt, float* dH1,
                             float* dLN2, float* dFF1, float* dFF2) {
  const int N = p.N, D = p.D, FF = p.FF;
  const size_t nTok = (size_t)N * D;
  const size_t nFF = (size_t)N * FF;

  // 1) X -> LN1
  layernormKernel<<<N, ATTEN_THREADS>>>(dX, dg1, dbeta1, dLN1, N, D, LN_EPS);
  CHECK_CUDA(cudaGetLastError());

  // 2) QKV = LN1 @ Wqkv： [N,D] x [D,3D] -> [N,3D]
  {
    dim3 block(TILE, TILE);
    dim3 grid((3 * D + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiledKernel<false><<<grid, block>>>(dLN1, dWqkv, nullptr, dQKV, N,
                                              3 * D, D);
    CHECK_CUDA(cudaGetLastError());
  }

  // 3) Attention：Q/K/V 是 QKV 缓冲的列切片（偏移 0/D/2D），零拷贝
  {
    const float scale = 1.0f / sqrtf((float)p.d_head);
    dim3 grid(N, p.H);
    attentionKernel<<<grid, ATTEN_THREADS>>>(dQKV, dQKV + D, dQKV + 2 * D,
                                             dAttn, N, 3 * D, D, p.d_head,
                                             scale);
    CHECK_CUDA(cudaGetLastError());
  }

  // 4) Oatt = Attn @ Wo： [N,D] x [D,D] -> [N,D]
  {
    dim3 block(TILE, TILE);
    dim3 grid((D + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiledKernel<false><<<grid, block>>>(dAttn, dWo, nullptr, dOatt, N, D,
                                              D);
    CHECK_CUDA(cudaGetLastError());
  }

  // 5) H1 = X + Oatt（第一个残差）
  residualAddKernel<<<(unsigned)((nTok + 255) / 256), 256>>>(dX, dOatt, dH1,
                                                             nTok);
  CHECK_CUDA(cudaGetLastError());

  // 6) H1 -> LN2
  layernormKernel<<<N, ATTEN_THREADS>>>(dH1, dg2, dbeta2, dLN2, N, D, LN_EPS);
  CHECK_CUDA(cudaGetLastError());

  // 7) FF1 = LN2 @ W1 + b1： [N,D] x [D,FF] -> [N,FF]（带 bias）
  {
    dim3 block(TILE, TILE);
    dim3 grid((FF + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiledKernel<true><<<grid, block>>>(dLN2, dW1, db1, dFF1, N, FF, D);
    CHECK_CUDA(cudaGetLastError());
  }

  // 8) GELU 激活
  geluKernel<<<(unsigned)((nFF + 255) / 256), 256>>>(dFF1, dFF1, nFF);
  CHECK_CUDA(cudaGetLastError());

  // 9) FF2 = FF1 @ W2 + b2： [N,FF] x [FF,D] -> [N,D]（带 bias）
  {
    dim3 block(TILE, TILE);
    dim3 grid((D + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiledKernel<true><<<grid, block>>>(dFF1, dW2, db2, dFF2, N, D, FF);
    CHECK_CUDA(cudaGetLastError());
  }

  // 10) Out = H1 + FF2（第二个残差）
  residualAddKernel<<<(unsigned)((nTok + 255) / 256), 256>>>(dH1, dFF2, dOut,
                                                             nTok);
  CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------
// CPU reference：与 kernel 完全相同的计算图，double 累加。
// 公式必须与 device 端逐一对应（GELU 用同一个 tanh 近似），否则比较
// 没有意义。attention 用朴素两遍 softmax 即可——reference 不需要
// 在线 rescale 技巧。
// ---------------------------------------------------------------------

static void matmulRef(const float* A, const float* B, const float* bias,
                      float* C, int M, int N, int K) {
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k)
        acc += (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
      C[(size_t)i * N + j] = (float)(acc + (bias ? (double)bias[j] : 0.0));
    }
}

static void layernormRef(const float* x, const float* gamma,
                         const float* beta, float* y, int N, int D,
                         float eps) {
  for (int i = 0; i < N; ++i) {
    double sum = 0.0, sumsq = 0.0;
    for (int j = 0; j < D; ++j) {
      const double v = x[(size_t)i * D + j];
      sum += v;
      sumsq += v * v;
    }
    const double mean = sum / D;
    double var = sumsq / D - mean * mean;
    if (var < 0.0) var = 0.0;
    const double inv = 1.0 / sqrt(var + eps);
    for (int j = 0; j < D; ++j)
      y[(size_t)i * D + j] =
          (float)((x[(size_t)i * D + j] - mean) * inv * gamma[j] + beta[j]);
  }
}

// Q/K/V 共享同一行宽 inStride（指向 QKV 缓冲的列切片），输出写向
// outStride 宽的缓冲。head h 的列起点是 h*d_head。
static void attentionRef(const float* Q, const float* K, const float* V,
                         float* O, int N, int H, int inStride, int outStride,
                         int d_head, float scale) {
  std::vector<double> score(N);
  for (int i = 0; i < N; ++i)
    for (int h = 0; h < H; ++h) {
      const int col0 = h * d_head;
      for (int j = 0; j < N; ++j) {
        double s = 0.0;
        for (int c = 0; c < d_head; ++c)
          s += (double)Q[(size_t)i * inStride + col0 + c] *
               K[(size_t)j * inStride + col0 + c];
        score[j] = s * scale;
      }
      double maxs = -1e300;
      for (int j = 0; j < N; ++j) maxs = fmax(maxs, score[j]);
      double sum = 0.0;
      for (int j = 0; j < N; ++j) sum += exp(score[j] - maxs);
      for (int c = 0; c < d_head; ++c) {
        double acc = 0.0;
        for (int j = 0; j < N; ++j)
          acc += exp(score[j] - maxs) * V[(size_t)j * inStride + col0 + c];
        O[(size_t)i * outStride + col0 + c] = (float)(acc / sum);
      }
    }
}

static float geluRef(double x) {
  constexpr double c = 0.7978845608028654; // sqrt(2/pi)
  return (float)(0.5 * x * (1.0 + tanh(c * (x + 0.044715 * x * x * x))));
}

void transformerBlockReference(const float* X, const float* Wqkv,
                               const float* Wo, const float* W1,
                               const float* b1, const float* W2,
                               const float* b2, const float* g1,
                               const float* beta1, const float* g2,
                               const float* beta2, float* Out,
                               const BlockParams& p) {
  const int N = p.N, D = p.D, FF = p.FF;
  std::vector<float> LN1((size_t)N * D), QKV((size_t)N * 3 * D),
      Attn((size_t)N * D), Oatt((size_t)N * D), H1((size_t)N * D),
      LN2((size_t)N * D), FF1((size_t)N * FF), FF2((size_t)N * D);

  layernormRef(X, g1, beta1, LN1.data(), N, D, LN_EPS);
  matmulRef(LN1.data(), Wqkv, nullptr, QKV.data(), N, 3 * D, D);
  const float scale = 1.0f / sqrtf((float)p.d_head);
  attentionRef(QKV.data(), QKV.data() + D, QKV.data() + 2 * D, Attn.data(), N,
               p.H, 3 * D, D, p.d_head, scale);
  matmulRef(Attn.data(), Wo, nullptr, Oatt.data(), N, D, D);
  for (size_t t = 0; t < (size_t)N * D; ++t) H1[t] = X[t] + Oatt[t];
  layernormRef(H1.data(), g2, beta2, LN2.data(), N, D, LN_EPS);
  matmulRef(LN2.data(), W1, b1, FF1.data(), N, FF, D);
  for (size_t t = 0; t < (size_t)N * FF; ++t) FF1[t] = geluRef(FF1[t]);
  matmulRef(FF1.data(), W2, b2, FF2.data(), N, D, FF);
  for (size_t t = 0; t < (size_t)N * D; ++t) Out[t] = H1[t] + FF2[t];
}

// ---------------------------------------------------------------------

static void usage(const char* prog) {
  fprintf(stderr,
          "Usage: %s [--seq N] [--d D] [--heads H] [--ffn F] [--seed S]\n"
          "  --seq    sequence length (default 64)\n"
          "  --d      hidden dim (default 256, 需能被 heads 整除)\n"
          "  --heads  number of heads (default 4, d_head = D/H <= 256)\n"
          "  --ffn    FFN hidden dim (default 4*D)\n"
          "  --seed   RNG seed (default 42)\n",
          prog);
}

int main(int argc, char** argv) {
  int N = 64, D = 256, H = 4, FF = 0;
  unsigned seed = 42;
  for (int a = 1; a < argc; ++a) {
    if (!strcmp(argv[a], "--seq") && a + 1 < argc)
      N = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--d") && a + 1 < argc)
      D = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--heads") && a + 1 < argc)
      H = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--ffn") && a + 1 < argc)
      FF = atoi(argv[++a]);
    else if (!strcmp(argv[a], "--seed") && a + 1 < argc)
      seed = (unsigned)atoi(argv[++a]);
    else {
      usage(argv[0]);
      return 1;
    }
  }
  if (N <= 0 || D <= 0 || H <= 0 || D % H != 0) {
    fprintf(stderr, "参数非法：需要 N>0, D>0, H>0 且 D %% H == 0\n");
    usage(argv[0]);
    return 1;
  }
  if (FF <= 0) FF = 4 * D;
  const int d_head = D / H;
  if (d_head > ATTEN_THREADS) {
    fprintf(stderr, "d_head=%d > %d：超出 attention kernel 的 block 线程数"
                    "约束，请增大 heads 或减小 d\n",
            d_head, ATTEN_THREADS);
    return 1;
  }
  const BlockParams p{N, D, H, d_head, FF};
  const size_t nTok = (size_t)N * D;
  const size_t nFF = (size_t)N * FF;

  // ---- host 数据（固定种子，结果可复现）----
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> hX(nTok), hWqkv((size_t)D * 3 * D), hWo((size_t)D * D),
      hW1((size_t)D * FF), hb1(FF), hW2((size_t)FF * D), hb2(D), hg1(D, 1.0f),
      hbeta1(D, 0.0f), hg2(D, 1.0f), hbeta2(D, 0.0f);
  const auto fill = [&](std::vector<float>& v) {
    for (float& e : v) e = dist(rng);
  };
  fill(hX);
  fill(hWqkv);
  fill(hWo);
  fill(hW1);
  fill(hb1);
  fill(hW2);
  fill(hb2);
  // LayerNorm 的 gamma/beta 保持 1/0

  // ---- device 分配与 H2D ----
  float *dX, *dWqkv, *dWo, *dW1, *db1, *dW2, *db2, *dg1, *dbeta1, *dg2, *dbeta2;
  float *dLN1, *dQKV, *dAttn, *dOatt, *dH1, *dLN2, *dFF1, *dFF2, *dOut;
  CHECK_CUDA(cudaMalloc(&dX, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dWqkv, (size_t)D * 3 * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dWo, (size_t)D * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW1, (size_t)D * FF * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db1, FF * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW2, (size_t)FF * D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&db2, D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dg1, D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dbeta1, D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dg2, D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dbeta2, D * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dLN1, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dQKV, nTok * 3 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dAttn, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOatt, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dH1, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dLN2, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dFF1, nFF * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dFF2, nTok * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dOut, nTok * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dX, hX.data(), nTok * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWqkv, hWqkv.data(), (size_t)D * 3 * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dWo, hWo.data(), (size_t)D * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW1, hW1.data(), (size_t)D * FF * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db1, hb1.data(), FF * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dW2, hW2.data(), (size_t)FF * D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(db2, hb2.data(), D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dg1, hg1.data(), D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dbeta1, hbeta1.data(), D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dg2, hg2.data(), D * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dbeta2, hbeta2.data(), D * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ---- CPU reference ----
  std::vector<float> refOut(nTok);
  transformerBlockReference(hX.data(), hWqkv.data(), hWo.data(), hW1.data(),
                            hb1.data(), hW2.data(), hb2.data(), hg1.data(),
                            hbeta1.data(), hg2.data(), hbeta2.data(),
                            refOut.data(), p);

  // ---- correctness 优先：先验证再谈性能 ----
  transformerBlockForward(dX, dWqkv, dWo, dW1, db1, dW2, db2, dg1, dbeta1,
                          dg2, dbeta2, dOut, p, dLN1, dQKV, dAttn, dOatt, dH1,
                          dLN2, dFF1, dFF2);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> gpuOut(nTok);
  CHECK_CUDA(cudaMemcpy(gpuOut.data(), dOut, nTok * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float maxAbs = 0.0f, maxRel = 0.0f;
  for (size_t t = 0; t < nTok; ++t) {
    const float d = std::fabs(gpuOut[t] - refOut[t]);
    maxAbs = std::max(maxAbs, d);
    maxRel = std::max(maxRel, d / (1.0f + std::fabs(refOut[t])));
  }
  printf("[correctness] max_abs_err=%.3e  max_rel_err=%.3e  -> %s\n", maxAbs,
         maxRel, maxRel <= 1e-3f ? "PASS" : "FAIL");
  if (!(maxRel <= 1e-3f)) return 1;

  // ---- 测量纪律：warmup 20 次，正式 100 次，CUDA Event 只量 kernel ----
  constexpr int kWarmup = 20;
  constexpr int kRuns = 100;
  for (int i = 0; i < kWarmup; ++i)
    transformerBlockForward(dX, dWqkv, dWo, dW1, db1, dW2, db2, dg1, dbeta1,
                            dg2, dbeta2, dOut, p, dLN1, dQKV, dAttn, dOatt,
                            dH1, dLN2, dFF1, dFF2);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t evStart, evStop;
  CHECK_CUDA(cudaEventCreate(&evStart));
  CHECK_CUDA(cudaEventCreate(&evStop));
  std::vector<float> us(kRuns);
  for (int i = 0; i < kRuns; ++i) {
    CHECK_CUDA(cudaEventRecord(evStart));
    transformerBlockForward(dX, dWqkv, dWo, dW1, db1, dW2, db2, dg1, dbeta1,
                            dg2, dbeta2, dOut, p, dLN1, dQKV, dAttn, dOatt,
                            dH1, dLN2, dFF1, dFF2);
    CHECK_CUDA(cudaEventRecord(evStop));
    CHECK_CUDA(cudaEventSynchronize(evStop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, evStart, evStop));
    us[i] = ms * 1000.0f;
  }
  std::sort(us.begin(), us.end());
  const auto pct = [&](double q) { return us[(int)(q * (kRuns - 1))]; };
  printf("[timing] block forward: median=%.1f us  p20=%.1f  p80=%.1f"
         "  (warmup=%d, runs=%d)\n",
         pct(0.5), pct(0.2), pct(0.8), kWarmup, kRuns);
  printf("[dims]   N=%d D=%d H=%d d_head=%d FF=%d\n", p.N, p.D, p.H, p.d_head,
         p.FF);
  printf("[sample] out[0:5] =");
  for (int t = 0; t < 5; ++t) printf(" %.4f", gpuOut[t]);
  printf("\n");

  CHECK_CUDA(cudaEventDestroy(evStart));
  CHECK_CUDA(cudaEventDestroy(evStop));
  CHECK_CUDA(cudaFree(dX));
  CHECK_CUDA(cudaFree(dWqkv));
  CHECK_CUDA(cudaFree(dWo));
  CHECK_CUDA(cudaFree(dW1));
  CHECK_CUDA(cudaFree(db1));
  CHECK_CUDA(cudaFree(dW2));
  CHECK_CUDA(cudaFree(db2));
  CHECK_CUDA(cudaFree(dg1));
  CHECK_CUDA(cudaFree(dbeta1));
  CHECK_CUDA(cudaFree(dg2));
  CHECK_CUDA(cudaFree(dbeta2));
  CHECK_CUDA(cudaFree(dLN1));
  CHECK_CUDA(cudaFree(dQKV));
  CHECK_CUDA(cudaFree(dAttn));
  CHECK_CUDA(cudaFree(dOatt));
  CHECK_CUDA(cudaFree(dH1));
  CHECK_CUDA(cudaFree(dLN2));
  CHECK_CUDA(cudaFree(dFF1));
  CHECK_CUDA(cudaFree(dFF2));
  CHECK_CUDA(cudaFree(dOut));
  return 0;
}
