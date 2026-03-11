/**
 * CUDA Matrix Transpose 算子优化
 *
 * 针对 4096×4096 的 float 矩阵进行转置
 * 参考 NVIDIA CUDA SDK 示例 "transpose"
 *
 * 版本列表：
 *   v0: Naive（非合并写，global memory stride 访问）
 *   v1: Shared Memory Tiled（合并读写，但存在 shared memory bank conflict）
 *   v2: Padded Shared Memory（+1 padding 消除 bank conflict）
 *   v3: Wide Tile（32×8 block，每线程处理 4 行，提升 ILP 和 Occupancy）
 *   v4: __ldg 只读缓存（v2 基础上使用 __ldg）
 *   v5: 对角线 Block 映射（Diagonal，消除 L2 partition camping）
 *   ref: 内存拷贝（bandwidth ceiling 参考）
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

static const int TILE = 32;

// ============================================================
// ref: 内存拷贝（理论带宽上限参考）
// ============================================================
__global__ void transpose_ref(float *out, const float *in, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N)
        out[row * N + col] = in[row * N + col];
}

// ============================================================
// v0: Naive
//     读：out[col*N+row]，连续 threadIdx.x → 连续 col（合并）
//     写：out[col*N+row]，col 固定，row 连续 → stride N（非合并！）
// ============================================================
__global__ void transpose_v0(float *out, const float *in, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N)
        out[col * N + row] = in[row * N + col];  // 写非合并
}

// ============================================================
// v1: Shared Memory Tiled（32×32 tile，存在 bank conflict）
//     读 global：合并  写 global：合并
//     读 sdata[threadIdx.x][threadIdx.y]：threadIdx.x 为行 → stride=32 → 32-way bank conflict
// ============================================================
__global__ void transpose_v1(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (y < N && x < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];   // 合并读
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (y < N && x < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // bank conflict！
}

// ============================================================
// v2: Padded Shared Memory（+1 padding 消除 bank conflict）
//     sdata[TILE][TILE+1]：列宽从 32 → 33
//     tile[tx][ty] 访问：地址步长 33*4=132 bytes → 33 mod 32 = 1 → 各访不同 bank
// ============================================================
__global__ void transpose_v2(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];   // +1 padding

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (y < N && x < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (y < N && x < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // 无 bank conflict
}

// ============================================================
// v3: Wide Tile（blockDim=32×8，每线程处理 4 行）
//     256 线程/block vs 1024 → occupancy 从 1 block/SM → 6 blocks/SM
//     #pragma unroll 展开 4 次循环，提升 ILP
// ============================================================
__global__ void transpose_v3(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x  = blockIdx.x * TILE + threadIdx.x;
    int y0 = blockIdx.y * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    x  = blockIdx.y * TILE + threadIdx.x;
    y0 = blockIdx.x * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================
// v4: __ldg 只读缓存（Read-Only Cache）
//     __ldg 通过 texture/read-only cache 路径加载，绕过 L1 data cache
//     对于非重复读取的场景可提升吞吐，基于 v3
// ============================================================
__global__ void transpose_v4(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x  = blockIdx.x * TILE + threadIdx.x;
    int y0 = blockIdx.y * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[y * N + x]);  // __ldg
    }
    __syncthreads();

    x  = blockIdx.y * TILE + threadIdx.x;
    y0 = blockIdx.x * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================
// v5: 对角线 Block 映射（Diagonal Block Ordering）
//     解决 L2/DRAM partition camping：
//     默认 block 行优先排列时，同一时刻的 block 读写相同内存 partition
//     对角线映射让同时活跃的 block 分散到不同内存区域
// ============================================================
__global__ void transpose_v5(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

    // 对角线坐标映射
    int grid_w = gridDim.x;
    int bx = (blockIdx.x + blockIdx.y) % grid_w;
    int by = blockIdx.x;

    int x  = bx * TILE + threadIdx.x;
    int y0 = by * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    x  = by * TILE + threadIdx.x;
    y0 = bx * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

// ============================================================
// 计时工具
// ============================================================
struct GpuTimer {
    cudaEvent_t start_, stop_;
    GpuTimer()  { cudaEventCreate(&start_); cudaEventCreate(&stop_); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start()    { cudaEventRecord(start_); }
    float stop_ms() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// CPU 参考转置（验证正确性）
static void cpu_transpose(float *out, const float *in, int N) {
    for (int r = 0; r < N; r++)
        for (int c = 0; c < N; c++)
            out[c * N + r] = in[r * N + c];
}

// ============================================================
// Main
// ============================================================
int main() {
    const int N      = 2048;
    const int WARMUP = 5;
    const int ITERS  = 100;

    size_t bytes = (size_t)N * N * sizeof(float);

    // 主机内存
    float *h_in  = new float[N * N];
    float *h_out = new float[N * N];
    float *h_ref = new float[N * N];

    srand(42);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)(rand() % 1000) / 1000.f;
    cpu_transpose(h_ref, h_in, N);

    // GPU 内存
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;

    // grid/block 配置
    dim3 block32x32(TILE, TILE);                     // 1024 threads/block
    dim3 block32x8 (TILE, TILE / 4);                 // 256  threads/block
    dim3 grid(N / TILE, N / TILE);

    double bw_peak = 0.0;  // 记录最佳带宽

    printf("Matrix: %d×%d float (%.1f MB in + %.1f MB out = %.1f MB total)\n",
           N, N, bytes/1e6, bytes/1e6, 2*bytes/1e6);
    printf("TILE=%d, WARMUP=%d, ITERS=%d\n\n", TILE, WARMUP, ITERS);
    printf("%-16s %10s %10s %12s %12s\n",
           "Version", "Time(us)", "BW(GB/s)", "BW%Peak", "MaxErr");
    printf("%-16s %10s %10s %12s %12s\n",
           "----------------", "--------", "--------", "--------", "--------");

    // 理论带宽（Orin SM 8.7，LPDDR5 ~68 GB/s）
    // 实测 copy 带宽作为参考
    double theory_bw = 68.0;

    auto bench = [&](const char *name, dim3 blk, dim3 grd, auto launch_fn, bool check) {
        // Warmup
        for (int it = 0; it < WARMUP; it++) launch_fn(grd, blk);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timing
        timer.start();
        for (int it = 0; it < ITERS; it++) launch_fn(grd, blk);
        float ms = timer.stop_ms() / ITERS;

        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;  // GB/s

        // 正确性验证
        float max_err = 0.f;
        if (check) {
            CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
            for (int i = 0; i < N * N; i++)
                max_err = fmaxf(max_err, fabsf(h_out[i] - h_ref[i]));
        }

        if (bw > bw_peak) bw_peak = bw;
        printf("%-16s %10.2f %10.2f %11.1f%% %12.2e\n",
               name, ms * 1000.f, bw, bw / theory_bw * 100.0, (double)max_err);
        return ms;
    };

    // ---- ref: copy ----
    bench("ref_copy", block32x32, grid, [&](dim3 g, dim3 b) {
        transpose_ref<<<g, b>>>(d_out, d_in, N);
    }, false);

    // ---- v0: naive ----
    bench("v0_naive", block32x32, grid, [&](dim3 g, dim3 b) {
        transpose_v0<<<g, b>>>(d_out, d_in, N);
    }, true);

    // ---- v1: smem tile (bank conflict) ----
    bench("v1_smem", block32x32, grid, [&](dim3 g, dim3 b) {
        transpose_v1<<<g, b>>>(d_out, d_in, N);
    }, true);

    // ---- v2: padded smem ----
    bench("v2_padded", block32x32, grid, [&](dim3 g, dim3 b) {
        transpose_v2<<<g, b>>>(d_out, d_in, N);
    }, true);

    // ---- v3: wide tile 32x8 ----
    bench("v3_wide", block32x8, grid, [&](dim3 g, dim3 b) {
        transpose_v3<<<g, b>>>(d_out, d_in, N);
    }, true);

    // ---- v4: __ldg ----
    bench("v4_ldg", block32x8, grid, [&](dim3 g, dim3 b) {
        transpose_v4<<<g, b>>>(d_out, d_in, N);
    }, true);

    // ---- v5: diagonal ----
    bench("v5_diagonal", block32x8, grid, [&](dim3 g, dim3 b) {
        transpose_v5<<<g, b>>>(d_out, d_in, N);
    }, true);

    printf("\nTheoretical peak BW (LPDDR5): %.1f GB/s\n", theory_bw);
    printf("Peak measured BW: %.2f GB/s (%.1f%%)\n",
           bw_peak, bw_peak / theory_bw * 100.0);

    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
