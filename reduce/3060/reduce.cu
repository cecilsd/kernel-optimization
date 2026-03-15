/**
 * CUDA Reduce 算子优化 —— RTX 3060 专版
 *
 * 硬件：NVIDIA GeForce RTX 3060 (GA106, SM 8.6, Ampere)
 *   - SM 数量：28
 *   - 每 SM 最大线程数：1536
 *   - 每 SM Shared Memory：100 KB（每 Block 最大 48 KB）
 *   - 理论内存带宽：360 GB/s（GDDR6, 192-bit, 7501 MHz）
 *   - L2 Cache：2304 KB
 *   - 寄存器/SM：65536
 *
 * 版本列表：
 *   v0: Naive 交错寻址（基线）
 *   v1: 顺序寻址（消除 warp divergence）
 *   v2: 加载时第一次加法（First Add During Load）
 *   v3: 展开最后一个 Warp（Unroll Last Warp）
 *   v4: 完整模板展开（Complete Unrolling via Template）
 *   v5: 每线程多元素（Grid-Stride Loop, 28×4=112 blocks）
 *   v6: Warp Shuffle Reduction（__shfl_down_sync）
 *   v7: Float4 向量化加载 + Warp Shuffle（128-bit LD.E.128）
 *
 * 数据规模：N = 32M（128 MB），充分利用 360 GB/s 带宽
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(_e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// RTX 3060 硬件常量
static const int RTX3060_SM_COUNT = 28;

// ============================================================
// v0: Naive — 交错寻址（Interleaved Addressing）
//     tid % (2*s) == 0 导致 warp divergence
// ============================================================
__global__ void reduce_v0(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v1: 顺序寻址（Sequential Addressing）
//     stride 从 blockDim.x/2 向下折半，消除 warp divergence
// ============================================================
__global__ void reduce_v1(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v2: 加载时第一次加法（First Add During Load）
//     每 block 处理 2×blockDim.x 个元素，block 数减半
// ============================================================
__global__ void reduce_v2(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float val = 0.f;
    if (i < n)               val  = g_idata[i];
    if (i + blockDim.x < n)  val += g_idata[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v3: 展开最后一个 Warp（Unroll Last Warp）
//     s <= 32 时 warp 内 SIMT 隐式同步，省掉 5 次 __syncthreads()
// ============================================================
__device__ void warpReduce_v3(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8];
    sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2];
    sdata[tid] += sdata[tid +  1];
}

__global__ void reduce_v3(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float val = 0.f;
    if (i < n)               val  = g_idata[i];
    if (i + blockDim.x < n)  val += g_idata[i + blockDim.x];
    sdata[tid] = val;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) warpReduce_v3(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v4: 完整模板展开（Complete Unrolling via Template）
//     blockSize 编译期已知，所有 if 分支静态消除
// ============================================================
template <unsigned int blockSize>
__device__ void warpReduce_v4(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void reduce_v4(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockSize * 2) + threadIdx.x;
    float val = 0.f;
    if (i < n)              val  = g_idata[i];
    if (i + blockSize < n)  val += g_idata[i + blockSize];
    sdata[tid] = val;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) warpReduce_v4<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v5: 每线程多元素（Grid-Stride Loop）
//     RTX 3060: 28 SM × 4 = 112 blocks，每线程处理 ~2368 元素
// ============================================================
template <unsigned int blockSize>
__global__ void reduce_v5(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    float val = 0.f;
    while (i < n) {
        val += g_idata[i];
        if (i + blockSize < n) val += g_idata[i + blockSize];
        i += gridSize;
    }
    sdata[tid] = val;
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }
    if (tid < 32) warpReduce_v4<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v6: Warp Shuffle Reduction（__shfl_down_sync）
//     仅 blockSize/32 个 float 的 shared memory（32 bytes for BS=256）
// ============================================================
template <unsigned int blockSize>
__global__ void reduce_v6(const float *g_idata, float *g_odata, int n) {
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    unsigned int lane     = tid & 31u;
    unsigned int wid      = tid >> 5u;

    float val = 0.f;
    while (i < n) {
        val += g_idata[i];
        if (i + blockSize < n) val += g_idata[i + blockSize];
        i += gridSize;
    }

    // Warp 内 shuffle reduce（寄存器直接交换，~1 cycle latency）
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    __shared__ float shared[blockSize / 32];
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    constexpr unsigned int numWarps = blockSize / 32;
    val = (tid < numWarps) ? shared[tid] : 0.f;
    if (wid == 0) {
        val += __shfl_down_sync(0xffffffff, val, numWarps >> 1);
        if (numWarps >= 4) val += __shfl_down_sync(0xffffffff, val, numWarps >> 2);
        if (numWarps >= 8) val += __shfl_down_sync(0xffffffff, val, numWarps >> 3);
    }
    if (tid == 0) g_odata[blockIdx.x] = val;
}

// ============================================================
// v7: Float4 向量化加载 + Warp Shuffle（RTX 3060 专属优化）
//
// 原理：
//   SM 8.6 的全局内存加载单元支持 128-bit 宽度事务（LDG.E.128）
//   float4 = 128 bit，单次指令加载 4 个 float，相比 4 次 float 加载：
//     - 指令数减少 4×：降低 instruction issue overhead
//     - 内存事务数不变（均为 128-bit），但 warp 内指令发射压力降低
//     - L1/L2 命中率提升（连续地址访问，cache line 复用更好）
//
//   同时使用 __ldg()（只读缓存，通过纹理缓存路径加载）避免 L1 污染
//
// Grid 配置：
//   RTX 3060: 28 SM × 4 = 112 blocks（确保所有 SM 满载）
//   每线程处理 n/(112×blockSize×4) 组 float4 ≈ ~930 元素（N=32M）
// ============================================================
template <unsigned int blockSize>
__launch_bounds__(256, 4)          // 告知编译器: 每块 256 线程，每 SM 至少 4 块
__global__ void reduce_v7(const float *__restrict__ g_idata,
                           float       *__restrict__ g_odata, int n) {
    unsigned int tid      = threadIdx.x;
    unsigned int lane     = tid & 31u;
    unsigned int wid      = tid >> 5u;
    constexpr unsigned int numWarps = blockSize / 32;

    // float4 视图（n 必须是 4 的倍数，余数单独处理）
    const float4 *g4      = reinterpret_cast<const float4*>(g_idata);
    int           n4      = n / 4;                      // float4 元素数
    // grid-stride: 每线程处理多个 float4，步长 = blockSize*2*gridDim（float4 视图）
    int           i4      = blockIdx.x * (blockSize * 2) + tid; // float4 索引
    int           stride4 = blockSize * 2 * gridDim.x;

    float val = 0.f;

    // Phase 1: float4 grid-stride loop —— 每次加载 128-bit
    while (i4 < n4) {
        float4 a = __ldg(&g4[i4]);               // 只读缓存（L1 texture path）
        val += a.x + a.y + a.z + a.w;
        if (i4 + blockSize < n4) {
            float4 b = __ldg(&g4[i4 + blockSize]);
            val += b.x + b.y + b.z + b.w;
        }
        i4 += stride4;
    }

    // Phase 2: 处理尾部不足 4 个的元素
    int tail_start = n4 * 4 + tid;
    int tail_stride = blockSize;
    while (tail_start < n) {
        val += __ldg(&g_idata[tail_start]);
        tail_start += tail_stride;
    }

    // Phase 3: Warp shuffle reduce（与 v6 相同）
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    __shared__ float shared[numWarps];
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (tid < numWarps) ? shared[tid] : 0.f;
    if (wid == 0) {
        val += __shfl_down_sync(0xffffffff, val, numWarps >> 1);
        if (numWarps >= 4) val += __shfl_down_sync(0xffffffff, val, numWarps >> 2);
        if (numWarps >= 8) val += __shfl_down_sync(0xffffffff, val, numWarps >> 3);
    }
    if (tid == 0) g_odata[blockIdx.x] = val;
}

// ============================================================
// 计时工具
// ============================================================
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start() { cudaEventRecord(start_); }
    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// CPU 双精度参考
static double cpu_reduce(const float *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data[i];
    return sum;
}

// ============================================================
// Main
// ============================================================
int main() {
    // ---------- 硬件信息 ----------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  (SM %d.%d, %d SMs, %.0f GB/s BW)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    const int N       = 1 << 25;   // 32M elements = 128 MB
    const int THREADS = 256;
    const int BLOCKS_STRIDE = RTX3060_SM_COUNT * 4;  // 112 blocks for v5-v7
    const int WARMUP  = 10;
    const int ITERS   = 100;

    // ---------- 主机数据 ----------
    float *h_in = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 1000) / 1000.f;

    double cpu_ans = cpu_reduce(h_in, N);
    printf("CPU reference sum = %.2f  (N=%d, %.1f MB)\n\n",
           cpu_ans, N, N * 4.0 / 1024 / 1024);

    // ---------- GPU 内存 ----------
    // v0/v1 最多需要 N/THREADS = 131072 个 partial 结果
    const int MAX_BLOCKS = (N + THREADS - 1) / THREADS;
    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in,      (long long)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, MAX_BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, (long long)N * sizeof(float),
                          cudaMemcpyHostToDevice));
    float *h_partial = new float[MAX_BLOCKS];

    GpuTimer timer;
    float peak_bw = 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6f;

    // 打印表头
    printf("%-14s %7s %7s %14s %10s %10s %10s\n",
           "Version", "Blocks", "Threads", "Result",
           "Time(us)", "BW(GB/s)", "BW(%peak)");
    printf("%-14s %7s %7s %14s %10s %10s %10s\n",
           "--------------", "------", "-------",
           "------------", "--------", "--------", "--------");

    auto bench = [&](const char *name, int blocks, auto launch_fn) {
        // Warmup
        for (int it = 0; it < WARMUP; it++) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timing
        timer.start();
        for (int it = 0; it < ITERS; it++) launch_fn();
        float ms = timer.stop_ms() / ITERS;

        // 读取结果
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double gpu_ans = 0.0;
        for (int i = 0; i < blocks; i++) gpu_ans += (double)h_partial[i];

        double err = fabs(gpu_ans - cpu_ans) / (cpu_ans + 1e-6);
        // 有效带宽：读 N floats（v7 效率相同，但指令更少）
        float bw_gbs = (N * 4.0f / 1e9f) / (ms / 1000.f);
        printf("%-14s %7d %7d %14.2f %10.2f %10.1f %10.1f%%\n",
               name, blocks, THREADS, gpu_ans, ms * 1000.f,
               bw_gbs, bw_gbs / peak_bw * 100.f);
        return ms;
    };

    // ---- v0 ----
    {
        int b = (N + THREADS - 1) / THREADS;
        bench("v0_naive", b, [&](){
            reduce_v0<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v1 ----
    {
        int b = (N + THREADS - 1) / THREADS;
        bench("v1_seqaddr", b, [&](){
            reduce_v1<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v2 ----
    {
        int b = (N + THREADS * 2 - 1) / (THREADS * 2);
        bench("v2_1stadd", b, [&](){
            reduce_v2<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v3 ----
    {
        int b = (N + THREADS * 2 - 1) / (THREADS * 2);
        bench("v3_unwarp", b, [&](){
            reduce_v3<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v4 ----
    {
        const unsigned int BS = THREADS;
        int b = (N + BS * 2 - 1) / (BS * 2);
        bench("v4_unroll", b, [&](){
            reduce_v4<BS><<<b, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v5 ----  112 blocks（28 SM × 4）
    {
        const unsigned int BS = THREADS;
        bench("v5_gridstride", BLOCKS_STRIDE, [&](){
            reduce_v5<BS><<<BLOCKS_STRIDE, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v6 ----
    {
        const unsigned int BS = THREADS;
        bench("v6_shuffle", BLOCKS_STRIDE, [&](){
            reduce_v6<BS><<<BLOCKS_STRIDE, BS, (BS/32) * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v7 ---- float4 向量化（N 必须是 4 的倍数，32M = 2^25 满足）
    {
        const unsigned int BS = THREADS;
        // v7 的 float4 索引步长：每个 block 处理 BS*2 个 float4 = BS*8 floats
        // 需要确保 partial result buffer 足够
        bench("v7_float4", BLOCKS_STRIDE, [&](){
            reduce_v7<BS><<<BLOCKS_STRIDE, BS, (BS/32) * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    printf("\nPeak Memory BW: %.0f GB/s\n", peak_bw);
    printf("Data: %d elements = %.1f MB\n", N, N * 4.0 / 1024 / 1024);

    delete[] h_in;
    delete[] h_partial;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return 0;
}
