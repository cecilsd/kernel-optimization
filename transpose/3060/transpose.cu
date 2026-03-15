/**
 * CUDA Matrix Transpose 算子优化 —— RTX 3060 (SM 8.6)
 *
 * 针对 4096×4096 的 float 矩阵进行转置
 * 参考 NVIDIA CUDA SDK 示例 "transpose"
 *
 * 版本列表：
 *   ref : 内存拷贝（bandwidth ceiling 参考）
 *   v0  : Naive（非合并写）
 *   v1  : Shared Memory Tiled（合并读写，有 bank conflict）
 *   v2  : Padded Shared Memory（+1 padding 消除 bank conflict）
 *   v3  : Wide Tile（32×8 block，每线程处理 4 行，提升 Occupancy）
 *   v4  : __ldg 只读缓存（v3 基础上使用 __ldg）
 *   v5  : 对角线 Block 映射（消除 L2 partition camping）
 *   v6  : float4 向量化 + shared memory 转置（正确转置，减少指令数）
 *
 * 编译：nvcc -O3 -arch=sm_86 -std=c++17 -o transpose transpose.cu
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
//     读：in[row*N+col]，threadIdx.x 连续 → col 连续 → 合并
//     写：out[col*N+row]，col 连续，row 固定 → stride N（非合并！）
// ============================================================
__global__ void transpose_v0(float *out, const float *in, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N)
        out[col * N + row] = in[row * N + col];   // 写非合并
}

// ============================================================
// v1: Shared Memory Tiled（32×32 tile，存在 bank conflict）
//     读 global：合并   写 global：合并
//     读 smem tile[tx][ty]：tx 连续，同一 bank → 32-way conflict
// ============================================================
__global__ void transpose_v1(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (y < N && x < N)
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];    // 合并读
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (y < N && x < N)
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];   // bank conflict！
}

// ============================================================
// v2: Padded Shared Memory（+1 padding 消除 bank conflict）
//     tile[TILE][TILE+1]：列步长 33 → bank 地址 = (k*33+ty)%32，k=0~31 各不同
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
        out[y * N + x] = tile[threadIdx.x][threadIdx.y];   // 无 bank conflict
}

// ============================================================
// v3: Wide Tile（blockDim=32×8，每线程处理 4 行）
//     256 线程/block → 1536/256 = 6 blocks/SM → Occupancy 接近 100%
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
// v4: __ldg 只读缓存（v3 基础上，读使用 Read-Only Cache 路径）
// ============================================================
__global__ void transpose_v4(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x  = blockIdx.x * TILE + threadIdx.x;
    int y0 = blockIdx.y * TILE + threadIdx.y;

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[y * N + x]);
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
//     同时调度的 block 沿对角线分布 → 读写分散到不同 L2/DRAM partition
//     RTX 3060 有 28 SM，多 partition 场景下 camping 效应更明显
// ============================================================
__global__ void transpose_v5(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

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
// v6: float4 向量化 + shared memory 转置
//
//     思路：
//       - 输入按 float4 读取（每线程 16B），减少全局读指令
//       - shared memory 仍以 float 存储，保持完整正确的转置
//       - 输出按 float4 写回（每线程 16B），减少全局写指令
//
//     blockDim = (TILE/4, TILE/4) = (8, 8)
//       - 每个 block 覆盖 32×32 float
//       - 每线程处理 4 行 × 4 列 = 16 元素
// ============================================================
__global__ void transpose_v6(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x * 4;  // float 列起点
    int y0 = blockIdx.y * TILE + threadIdx.y;     // float 行起点

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && (x + 3) < N) {
            float4 v = *reinterpret_cast<const float4 *>(&in[y * N + x]);
            tile[threadIdx.y + j][threadIdx.x * 4 + 0] = v.x;
            tile[threadIdx.y + j][threadIdx.x * 4 + 1] = v.y;
            tile[threadIdx.y + j][threadIdx.x * 4 + 2] = v.z;
            tile[threadIdx.y + j][threadIdx.x * 4 + 3] = v.w;
        }
    }
    __syncthreads();

    int ox = blockIdx.y * TILE + threadIdx.x * 4;  // 输出列起点
    int oy0 = blockIdx.x * TILE + threadIdx.y;     // 输出行起点

#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int oy = oy0 + j;
        if (oy < N && (ox + 3) < N) {
            float4 v;
            v.x = tile[threadIdx.x * 4 + 0][threadIdx.y + j];
            v.y = tile[threadIdx.x * 4 + 1][threadIdx.y + j];
            v.z = tile[threadIdx.x * 4 + 2][threadIdx.y + j];
            v.w = tile[threadIdx.x * 4 + 3][threadIdx.y + j];
            *reinterpret_cast<float4 *>(&out[oy * N + ox]) = v;
        }
    }
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
    const int N      = 4096;
    const int WARMUP = 10;
    const int ITERS  = 200;

    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_in  = new float[N * N];
    float *h_out = new float[N * N];
    float *h_ref = new float[N * N];

    srand(42);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)(rand() % 1000) / 1000.f;
    cpu_transpose(h_ref, h_in, N);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;

    dim3 block32x32(TILE, TILE);
    dim3 block32x8 (TILE, TILE / 4);   // 256 threads/block
    dim3 grid(N / TILE, N / TILE);

    // RTX 3060 理论内存带宽：192-bit * 15 Gbps / 8 = 360 GB/s
    double theory_bw = 360.0;
    double bw_peak   = 0.0;

    printf("Matrix: %d×%d float (%.1f MB in + %.1f MB out = %.1f MB total)\n",
           N, N, bytes/1e6, bytes/1e6, 2*bytes/1e6);
    printf("GPU: RTX 3060 (SM 8.6), Theory BW: %.0f GB/s\n", theory_bw);
    printf("TILE=%d, WARMUP=%d, ITERS=%d\n\n", TILE, WARMUP, ITERS);
    printf("%-16s %10s %10s %12s %12s\n",
           "Version", "Time(us)", "BW(GB/s)", "BW%Peak", "MaxErr");
    printf("%-16s %10s %10s %12s %12s\n",
           "----------------", "--------", "--------", "--------", "--------");

    auto bench = [&](const char *name, auto launch_fn, bool check) {
        for (int it = 0; it < WARMUP; it++) launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        for (int it = 0; it < ITERS; it++) launch_fn();
        float ms = timer.stop_ms() / ITERS;

        double bw = 2.0 * bytes / (ms * 1e-3) / 1e9;

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
    bench("ref_copy", [&]() {
        transpose_ref<<<grid, block32x32>>>(d_out, d_in, N);
    }, false);

    // ---- v0: naive ----
    bench("v0_naive", [&]() {
        transpose_v0<<<grid, block32x32>>>(d_out, d_in, N);
    }, true);

    // ---- v1: smem tile (bank conflict) ----
    bench("v1_smem", [&]() {
        transpose_v1<<<grid, block32x32>>>(d_out, d_in, N);
    }, true);

    // ---- v2: padded smem ----
    bench("v2_padded", [&]() {
        transpose_v2<<<grid, block32x32>>>(d_out, d_in, N);
    }, true);

    // ---- v3: wide tile 32x8 ----
    bench("v3_wide", [&]() {
        transpose_v3<<<grid, block32x8>>>(d_out, d_in, N);
    }, true);

    // ---- v4: __ldg ----
    bench("v4_ldg", [&]() {
        transpose_v4<<<grid, block32x8>>>(d_out, d_in, N);
    }, true);

    // ---- v5: diagonal ----
    bench("v5_diagonal", [&]() {
        transpose_v5<<<grid, block32x8>>>(d_out, d_in, N);
    }, true);

    // ---- v6: float4 vectorized + smem ----
    // blockDim = (TILE/4, TILE/4) = (8, 8)
    {
        dim3 blk6(TILE / 4, TILE / 4);   // (8, 8) = 64 threads/block
        dim3 grd6(N / TILE, N / TILE);   // 每 block 覆盖 TILE cols × TILE rows
        bench("v6_float4", [&]() {
            transpose_v6<<<grd6, blk6>>>(d_out, d_in, N);
        }, true);
    }

    printf("\nTheoretical peak BW (GDDR6): %.1f GB/s\n", theory_bw);
    printf("Peak measured BW: %.2f GB/s (%.1f%%)\n",
           bw_peak, bw_peak / theory_bw * 100.0);

    delete[] h_in;
    delete[] h_out;
    delete[] h_ref;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
