/**
 * NCU 专用 profiling 程序
 * 每个版本只运行一次，供 ncu 单独分析每个 kernel
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

// -------- v0 --------
__global__ void reduce_v0(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// -------- v1 --------
__global__ void reduce_v1(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < n) ? g_idata[idx] : 0.f;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// -------- v2 --------
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

// -------- v3 --------
__device__ void warpReduce_v3(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32]; sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid +  8]; sdata[tid] += sdata[tid +  4];
    sdata[tid] += sdata[tid +  2]; sdata[tid] += sdata[tid +  1];
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

// -------- v4 --------
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

// -------- v5 --------
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

// -------- v6 --------
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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <version 0-6>\n", argv[0]);
        return 1;
    }
    int ver = atoi(argv[1]);

    const int N       = 100000;
    const int THREADS = 256;

    float *h_in = new float[N];
    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = (float)(rand() % 100) / 100.f;

    float *d_in, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_in,      N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, 1024 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    switch (ver) {
    case 0: {
        int b = (N + THREADS - 1) / THREADS;
        reduce_v0<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 1: {
        int b = (N + THREADS - 1) / THREADS;
        reduce_v1<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 2: {
        int b = (N + THREADS * 2 - 1) / (THREADS * 2);
        reduce_v2<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 3: {
        int b = (N + THREADS * 2 - 1) / (THREADS * 2);
        reduce_v3<<<b, THREADS, THREADS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 4: {
        const unsigned int BS = 256;
        int b = (N + BS * 2 - 1) / (BS * 2);
        reduce_v4<BS><<<b, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 5: {
        const unsigned int BS = 256;
        reduce_v5<BS><<<64, BS, BS * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    case 6: {
        const unsigned int BS = 256;
        reduce_v6<BS><<<64, BS, (BS/32) * sizeof(float)>>>(d_in, d_partial, N);
        break;
    }
    default:
        fprintf(stderr, "Unknown version %d\n", ver);
        return 1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    delete[] h_in;
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_partial));
    return 0;
}
