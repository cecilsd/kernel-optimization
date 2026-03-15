/**
 * NCU Profiling 程序 —— 矩阵转置 RTX 3060
 *
 * 用法：./transpose_profile <version>
 *   0 = ref_copy
 *   1 = v0_naive
 *   2 = v1_smem
 *   3 = v2_padded
 *   4 = v3_wide
 *   5 = v4_ldg
 *   6 = v5_diagonal
 *   7 = v6_float4
 *
 * 编译：nvcc -O3 -arch=sm_86 -std=c++17 -o transpose_profile transpose_profile.cu
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

__global__ void transpose_ref(float *out, const float *in, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N) out[row * N + col] = in[row * N + col];
}

__global__ void transpose_v0(float *out, const float *in, int N) {
    int col = blockIdx.x * TILE + threadIdx.x;
    int row = blockIdx.y * TILE + threadIdx.y;
    if (row < N && col < N) out[col * N + row] = in[row * N + col];
}

__global__ void transpose_v1(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (y < N && x < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_v2(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (y < N && x < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_v3(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x  = blockIdx.x * TILE + threadIdx.x;
    int y0 = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N) tile[threadIdx.y + j][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();
    x  = blockIdx.y * TILE + threadIdx.x;
    y0 = blockIdx.x * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transpose_v4(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x  = blockIdx.x * TILE + threadIdx.x;
    int y0 = blockIdx.y * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N) tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[y * N + x]);
    }
    __syncthreads();
    x  = blockIdx.y * TILE + threadIdx.x;
    y0 = blockIdx.x * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

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
        if (y < N && x < N) tile[threadIdx.y + j][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();
    x  = by * TILE + threadIdx.x;
    y0 = bx * TILE + threadIdx.y;
#pragma unroll
    for (int j = 0; j < TILE; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__ void transpose_v6(float *out, const float *in, int N) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x * 4;
    int y0  = blockIdx.y * TILE + threadIdx.y;
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
    int ox = blockIdx.y * TILE + threadIdx.x * 4;
    int oy0  = blockIdx.x * TILE + threadIdx.y;
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

int main(int argc, char **argv) {
    int version = (argc > 1) ? atoi(argv[1]) : 1;
    const int N = 4096;

    size_t bytes = (size_t)N * N * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // 初始化数据
    {
        float *h = new float[N * N];
        for (int i = 0; i < N * N; i++) h[i] = (float)i;
        CUDA_CHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));
        delete[] h;
    }

    dim3 block32x32(TILE, TILE);
    dim3 block32x8(TILE, TILE / 4);
    dim3 grid(N / TILE, N / TILE);

    const char *names[] = {"ref_copy","v0_naive","v1_smem","v2_padded",
                           "v3_wide","v4_ldg","v5_diagonal","v6_float4"};
    printf("Profiling: %s (N=%d)\n", names[version], N);

    switch (version) {
        case 0: transpose_ref    <<<grid, block32x32>>>(d_out, d_in, N); break;
        case 1: transpose_v0     <<<grid, block32x32>>>(d_out, d_in, N); break;
        case 2: transpose_v1     <<<grid, block32x32>>>(d_out, d_in, N); break;
        case 3: transpose_v2     <<<grid, block32x32>>>(d_out, d_in, N); break;
        case 4: transpose_v3     <<<grid, block32x8 >>>(d_out, d_in, N); break;
        case 5: transpose_v4     <<<grid, block32x8 >>>(d_out, d_in, N); break;
        case 6: transpose_v5     <<<grid, block32x8 >>>(d_out, d_in, N); break;
        case 7: {
            dim3 blk6(TILE / 4, TILE / 4);
            dim3 grd6(N / TILE, N / TILE);
            transpose_v6<<<grd6, blk6>>>(d_out, d_in, N);
            break;
        }
        default:
            fprintf(stderr, "Unknown version %d\n", version);
            return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
