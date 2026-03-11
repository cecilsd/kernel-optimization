/**
 * NCU 专用 profiling 程序 — Matrix Transpose
 * 每个版本单独运行一次，供 ncu 逐一分析
 * 用法: ./transpose_profile <version>
 *   0 = ref_copy, 1 = v0_naive, 2 = v1_smem, 3 = v2_padded,
 *   4 = v3_wide,  5 = v4_ldg,   6 = v5_diagonal
 */

#include <cstdio>
#include <cstdlib>
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <version 0-6>\n", argv[0]);
        return 1;
    }
    int ver = atoi(argv[1]);

    const int N = 2048;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_in = new float[N * N];
    srand(42);
    for (int i = 0; i < N * N; i++) h_in[i] = (float)(rand() % 1000) / 1000.f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block32x32(TILE, TILE);
    dim3 block32x8 (TILE, TILE / 4);
    dim3 grid(N / TILE, N / TILE);

    switch (ver) {
    case 0: transpose_ref    <<<grid, block32x32>>>(d_out, d_in, N); break;
    case 1: transpose_v0     <<<grid, block32x32>>>(d_out, d_in, N); break;
    case 2: transpose_v1     <<<grid, block32x32>>>(d_out, d_in, N); break;
    case 3: transpose_v2     <<<grid, block32x32>>>(d_out, d_in, N); break;
    case 4: transpose_v3     <<<grid, block32x8 >>>(d_out, d_in, N); break;
    case 5: transpose_v4     <<<grid, block32x8 >>>(d_out, d_in, N); break;
    case 6: transpose_v5     <<<grid, block32x8 >>>(d_out, d_in, N); break;
    default: fprintf(stderr, "Unknown version %d\n", ver); return 1;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    delete[] h_in;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
