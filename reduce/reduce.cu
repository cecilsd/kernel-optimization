/**
 * CUDA Reduce 算子优化
 *
 * 针对长度为 100000 的 float 数组进行 reduce（求和）
 * 参考 NVIDIA 官方文档 "Optimizing Parallel Reduction in CUDA"
 *
 * 版本列表：
 *   v0: Naive（交错寻址，存在大量 bank conflict + warp divergence）
 *   v1: 顺序寻址（Sequential Addressing，消除 bank conflict）
 *   v2: 加载时第一次加法（First Add During Load，减少 idle 线程）
 *   v3: 展开最后一个 Warp（Unroll Last Warp）
 *   v4: 完整循环展开（Complete Unrolling via Template）
 *   v5: 每线程处理多个元素（Multiple Elements Per Thread，grid-stride）
 *   v6: Warp Shuffle Reduction（__shfl_down_sync，极少 shared mem 访问）
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 工具宏
// ============================================================
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// ============================================================
// v0: Naive — 交错寻址（Interleaved Addressing）
//     stride 以 2 的幂次增长
//     问题: tid%2 判断导致 warp divergence + shared memory bank conflict
// ============================================================
__global__ void reduce_v0(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v1: 顺序寻址（Sequential Addressing）
//     stride 从 blockDim.x/2 向下折半，消除 bank conflict
//     但后半 warp 的线程仍然 idle
// ============================================================
__global__ void reduce_v1(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ============================================================
// v2: 加载时完成第一次加法（First Add During Load）
//     每 block 处理 2*blockDim.x 个元素，block 数量减半
//     全局加载阶段即完成一次合并，减少空闲线程占比
// ============================================================
__global__ void reduce_v2(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.f;
    if (i < n)                val  = g_idata[i];
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
//     当 s <= 32 时，warp 内部 SIMT 已隐式同步，无需 __syncthreads()
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
    if (i < n)                val  = g_idata[i];
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
// v4: 完整循环展开（Complete Unrolling via Template）
//     blockSize 在编译期已知，编译器静态展开所有 if 分支
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
// v5: 每线程处理多个元素（Multiple Elements Per Thread）
//     使用 grid-stride loop 在全局加载阶段连续累加多段数据
//     固定较少 block 数量，让每个线程负责更多全局内存读取
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
// v6: Warp Shuffle Reduction
//     用 __shfl_down_sync 完成 warp 内归约，
//     只用极少量 shared memory 存储各 warp 结果
// ============================================================
template <unsigned int blockSize>
__global__ void reduce_v6(const float *g_idata, float *g_odata, int n) {
    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    unsigned int lane     = tid & 31u;
    unsigned int wid      = tid >> 5u;

    // Grid-stride 累加
    float val = 0.f;
    while (i < n) {
        val += g_idata[i];
        if (i + blockSize < n) val += g_idata[i + blockSize];
        i += gridSize;
    }

    // Warp 内 shuffle reduce
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);

    // 每个 warp 的 lane0 写入 shared memory
    __shared__ float shared[blockSize / 32];
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // 第一个 warp 归约所有 warp 结果
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
// 计时工具
// ============================================================
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start()        { cudaEventRecord(start_); }
    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// CPU 参考实现
static float cpu_reduce(const float *data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data[i];
    return (float)sum;
}

// ============================================================
// Main
// ============================================================
int main() {
    const int N       = 100000;
    const int THREADS = 256;
    const int WARMUP  = 5;
    const int ITERS   = 200;

    // 初始化主机数据
    float *h_in = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 100) / 100.f;

    float cpu_ans = cpu_reduce(h_in, N);
    printf("CPU reference sum = %.4f\n\n", cpu_ans);

    // GPU 内存
    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    float *h_partial = new float[1024];

    GpuTimer timer;

    // 打印表头
    printf("%-14s %8s %8s %12s %10s %10s\n",
           "Version", "Blocks", "Threads", "Result", "Time(us)", "Error");
    printf("%-14s %8s %8s %12s %10s %10s\n",
           "--------------", "------", "-------",
           "----------", "--------", "--------");

    auto bench = [&](const char *name, int blocks, int threads, int smem,
                     auto launch_fn) {
        // Warmup
        for (int it = 0; it < WARMUP; it++) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timing
        timer.start();
        for (int it = 0; it < ITERS; it++) launch_fn();
        float ms = timer.stop_ms() / ITERS;

        // 读取结果
        int nb = blocks;
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, nb * sizeof(float),
                              cudaMemcpyDeviceToHost));
        float gpu_ans = 0.f;
        for (int i = 0; i < nb; i++) gpu_ans += h_partial[i];

        float err = fabsf(gpu_ans - cpu_ans) / (cpu_ans + 1e-6f);
        printf("%-14s %8d %8d %12.4f %10.4f %10.2e\n",
               name, blocks, threads, gpu_ans, ms * 1000.f, err);
        return ms;
    };

    // ---- v0 ----
    {
        int blocks = (N + THREADS - 1) / THREADS;
        bench("v0_naive", blocks, THREADS, THREADS * sizeof(float), [&](){
            reduce_v0<<<blocks, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v1 ----
    {
        int blocks = (N + THREADS - 1) / THREADS;
        bench("v1_seqaddr", blocks, THREADS, THREADS * sizeof(float), [&](){
            reduce_v1<<<blocks, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v2 ----
    {
        int blocks = (N + THREADS * 2 - 1) / (THREADS * 2);
        bench("v2_1stadd", blocks, THREADS, THREADS * sizeof(float), [&](){
            reduce_v2<<<blocks, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v3 ----
    {
        int blocks = (N + THREADS * 2 - 1) / (THREADS * 2);
        bench("v3_unwarp", blocks, THREADS, THREADS * sizeof(float), [&](){
            reduce_v3<<<blocks, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v4 ----
    {
        const unsigned int BS = 256;
        int blocks = (N + BS * 2 - 1) / (BS * 2);
        bench("v4_unroll", blocks, BS, BS * sizeof(float), [&](){
            reduce_v4<BS><<<blocks, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v5 ----
    {
        const unsigned int BS = 256;
        int blocks = 64; // 固定 block 数，每线程处理更多元素
        bench("v5_multielem", blocks, BS, BS * sizeof(float), [&](){
            reduce_v5<BS><<<blocks, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    // ---- v6 ----
    {
        const unsigned int BS = 256;
        int blocks = 64;
        bench("v6_shuffle", blocks, BS, (BS/32) * sizeof(float), [&](){
            reduce_v6<BS><<<blocks, BS, (BS/32) * sizeof(float)>>>(d_in, d_partial, N);
        });
    }

    printf("\nData size: %zu bytes (%.2f KB), %d elements\n",
           N * sizeof(float), N * sizeof(float) / 1024.f, N);

    delete[] h_in;
    delete[] h_partial;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return 0;
}
