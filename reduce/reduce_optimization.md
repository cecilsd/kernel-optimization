# CUDA Reduce 算子优化报告

## 环境信息

| 项目 | 值 |
|------|-----|
| 硬件平台 | NVIDIA Jetson Orin (nvgpu) |
| Compute Capability | SM 8.7 (Ampere架构) |
| SM 数量 | 8 |
| SM 频率 | ~305 MHz |
| CUDA 版本 | 12.6 |
| 编译器 | nvcc 12.6 `-O3 -arch=sm_87` |
| 数组规模 | N = 100,000，float32 |
| 数据大小 | 400,000 bytes（390.6 KB） |

---

## 问题描述

对长度为 **100,000** 的 `float` 数组进行**规约求和（Reduce Sum）**，从 Naive 实现出发，逐步引入优化手段，最终获得接近硬件极限的实现。

参考 NVIDIA 经典白皮书：*Optimizing Parallel Reduction in CUDA*（Mark Harris, 2007）。

---

## 性能测试结果汇总

基准测试参数：预热 5 次，计时 200 次迭代，取平均值（单位 μs）。

| 版本 | 描述 | Blocks | Threads | 结果 | 时间(μs) | 相对v0加速比 |
|------|------|--------|---------|------|----------|-------------|
| v0 | Naive 交错寻址 | 391 | 256 | 49493.54 | 151.16 | 1.0× |
| v1 | 顺序寻址 | 391 | 256 | 49493.54 | 26.63 | 5.7× |
| v2 | 加载时第一次加法 | 196 | 256 | 49493.52 | 24.39 | 6.2× |
| v3 | 展开最后一个 Warp | 196 | 256 | 49493.52 | 18.55 | 8.2× |
| v4 | 完整模板展开 | 196 | 256 | 49493.52 | 13.14 | 11.5× |
| v5 | 每线程多元素 | 64 | 256 | 49493.52 | 10.47 | 14.4× |
| v6 | Warp Shuffle | 64 | 256 | 49493.52 | 9.98 | 15.1× |

CPU 参考结果：49493.5195（双精度累加），所有 GPU 版本相对误差 < 1e-6。

---

## NCU Profiling 数据

使用 `ncu --set basic` 和自定义 metrics 采集。

### 基础性能指标（ncu --set basic）

| 版本 | Duration(μs) | Elapsed Cycles | Mem BW(%) | Compute(%) | L1 BW(%) | L2 BW(%) | Occupancy(%) | Waves/SM |
|------|-------------|----------------|-----------|------------|----------|----------|--------------|----------|
| v0 | 127.90 | 39,103 | 31.97 | 65.47 | 34.04 | 10.41 | 91.44 | 8.15 |
| v1 | 70.98 | 21,662 | 59.02 | 52.32 | 66.81 | 18.63 | 87.13 | 8.15 |
| v2 | 47.17 | 14,387 | 48.22 | 43.64 | 60.04 | 27.51 | 85.93 | 4.08 |
| v3 | 39.17 | 11,915 | 33.21 | 28.80 | 34.64 | 33.21 | 75.48 | 4.08 |
| v4 | 37.47 | 11,402 | 34.68 | 24.79 | 36.96 | 34.68 | 74.20 | 4.08 |
| v5 | 37.60 | 11,446 | 34.44 | 16.42 | 16.97 | 34.44 | 76.85 | 1.33 |
| v6 | 36.03 | 10,958 | 35.86 | 17.66 | 15.75 | 35.86 | 81.11 | 1.33 |

> **注**：NCU 的 Duration 是单次 kernel 调用时间（profiling overhead 较大），基准测试时间更具参考价值。

### 详细指标（自定义 metrics）

| 版本 | Global读(KB) | Shared Bank冲突(LD) | Shared Bank冲突(ST) | Warp活跃率(%) | 平均指令延迟(cycle) | FADD指令数 | Stall-Branch(%) | Stall-Scoreboard(%) | Stall-Wait(%) |
|------|------------|---------------------|---------------------|--------------|--------------------|-----------|--------------|--------------------|--------------|
| v0 | 400 | 0 | 0 | 91.69 | 15.94 | 99,705 | 1.14 | 13.39 | 25.93 |
| v1 | 400 | 0 | 0 | 87.91 | 18.86 | 99,705 | 1.77 | 13.73 | 16.18 |
| v2 | 400 | 0 | 0 | 87.41 | 20.68 | 99,900 | 1.47 | 11.17 | 13.76 |
| v3 | 400 | 0 | 0 | 78.32 | 27.04 | 125,184 | 1.61 | 5.42 | 10.41 |
| v4 | 400 | 0 | 0 | 78.28 | 34.09 | 125,184 | 1.49 | 7.27 | 7.21 |
| v5 | 400 | 0 | 0 | 83.01 | 51.39 | 124,576 | 0.79 | 2.57 | 5.81 |
| v6 | 400 | 0 | 0 | 81.91 | 49.23 | 188,064 | 0.81 | 3.40 | 6.25 |

---

## 各版本优化分析

### v0 → v1：顺序寻址消除 Warp Divergence

**v0 核心代码：**
```cuda
for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {   // 问题所在
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**v1 核心代码：**
```cuda
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {               // 顺序寻址
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

**分析：**

v0 的 `tid % (2*s) == 0` 条件在同一个 warp 内制造了 **warp divergence**：
- 第一轮（s=1）：warp 内 16 个线程执行加法，16 个线程 idle → 实际吞吐折半
- 第二轮（s=2）：只有 8 个线程工作 → 7/8 的 SM 时间浪费

**NCU 观测：**
- v0：`Stall-Wait` 高达 **25.93%**（线程等待 barrier 时的 idle 浪费）
- v0：`Compute(SM) Throughput` = 65.47%，但实际 FADD 数量与 v1 相同（99,705），说明 SM 在"空转"
- v1：Duration 从 127.90 → **70.98 μs**，加速 **1.80×**

**关于 Shared Memory Bank Conflict（重要发现）：**
v0 和 v1 的 `bank conflicts = 0`。这是因为：
- v0 的访问模式是 `sdata[0], sdata[2], sdata[4]...`（步长为 2 的偶数下标），各访问不同 bank，**不产生冲突**
- Ampere 架构的 bank conflict 检测机制也有所不同
- v0 的真正性能瓶颈是 **warp divergence，而非 bank conflict**

---

### v1 → v2：加载时第一次加法（First Add During Load）

**v2 核心变化：**
```cuda
// 每 block 处理 2×blockDim.x 个元素，加载时即完成第一次加法
unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
float val = 0.f;
if (i < n)               val  = g_idata[i];
if (i + blockDim.x < n)  val += g_idata[i + blockDim.x];
sdata[tid] = val;
```

**分析：**

v1 存在一个根本性问题：**一半的线程在第一次 shared memory reduce 时就立即 idle**。
通过在全局内存加载阶段合并两个元素，同样的工作量用一半的 block 完成：
- Blocks 从 391 → **196**，减少了 kernel launch 开销和同步开销
- 所有 256 个线程在进入 shared memory 归约前都"充分工作"了
- Waves per SM 从 8.15 → **4.08**（每次 SM 调度的 block 数减半）

**NCU 观测：**
- `Stall-Wait` 从 16.18% → **13.76%**（同步等待减少）
- Duration 从 70.98 → **47.17 μs**，加速 **1.50×**

---

### v2 → v3：展开最后一个 Warp

**v3 核心变化：**
```cuda
// __syncthreads() 只在 s > 32 时需要
for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
// 最后一个 warp（32个线程）直接展开，无需 __syncthreads
if (tid < 32) warpReduce_v3(sdata, tid);
```

```cuda
__device__ void warpReduce_v3(volatile float *sdata, unsigned int tid) {
    sdata[tid] += sdata[tid + 32];  // 无 __syncthreads，SIMT 隐式同步
    sdata[tid] += sdata[tid + 16];
    // ...
}
```

**分析：**

当 `s ≤ 32` 时，只有一个 warp（32线程）在活动。同一 warp 内的所有线程在 SIMT 下隐式同步，不需要 `__syncthreads()`。因此可以把最后 5 次 `__syncthreads()` 全部消除，减少 barrier 同步开销。

**NCU 观测：**
- `Stall-Scoreboard`（等待 shared memory 结果）从 11.17% → **5.42%**（展开后流水线更流畅）
- `FADD` 数从 99,900 → **125,184**（展开 warpReduce 多生成了 fadd 指令，是正常的展开效果）
- Duration 从 47.17 → **39.17 μs**，加速 **1.21×**

---

### v3 → v4：完整模板展开（Complete Unrolling）

**v4 核心变化：**
```cuda
template <unsigned int blockSize>
__global__ void reduce_v4(...)
// blockSize 编译期已知，编译器可以静态展开所有 if 分支
if (blockSize >= 512) { if (tid < 256) sdata[tid] += ...; __syncthreads(); }
if (blockSize >= 256) { if (tid < 128) sdata[tid] += ...; __syncthreads(); }
if (blockSize >= 128) { if (tid <  64) sdata[tid] += ...; __syncthreads(); }
```

**分析：**

由于 `blockSize=256` 在编译期确定，编译器会静态求值所有 `if (blockSize >= X)` 分支，完全消除运行时的条件判断，生成直线代码（straight-line code）。这减少了分支预测失误和条件跳转的开销。

**NCU 观测：**
- `Stall-Wait` 从 10.41% → **7.21%**（更少的 barrier 等待）
- Duration 从 39.17 → **37.47 μs**，加速 **1.05×**（效果有限，因为 blockSize=256 时 v3 已经只有 128→64 两个 __syncthreads 被展开）

---

### v4 → v5：每线程处理多个元素（Grid-Stride Loop）

**v5 核心变化：**
```cuda
// 固定 64 个 block，使用 grid-stride 循环覆盖所有数据
unsigned int gridSize = blockSize * 2 * gridDim.x;
float val = 0.f;
while (i < n) {
    val += g_idata[i];
    if (i + blockSize < n) val += g_idata[i + blockSize];
    i += gridSize;
}
```

**分析：**

v4 使用 196 个 block，每个 block 处理 512 个元素，只需 1 次 grid-stride 循环。
v5 使用 **64 个 block**（远少于数据），每个 block 平均处理 ~1563 个元素，循环 ~3 次。

好处：
- Waves per SM 从 4.08 → **1.33**（极大减少 block 调度开销）
- 全局内存访问更**连续合并**（coalesced），L1 cache 利用率更高
- 更好的**内存流水线**：线程在 while 循环中持续从全局内存加载，hiding memory latency

**NCU 观测：**
- Waves/SM：4.08 → **1.33**
- `Stall-Branch`：1.49% → **0.79%**（循环展开减少分支）
- `Stall-Scoreboard`：7.27% → **2.57%**（流水线更顺畅）
- Duration 从 37.47 → **37.60 μs**（NCU overhead 下 v5 略慢，但基准测试中 v5=10.47μs < v4=13.14μs）

> **说明**：NCU profiling 时单次调用开销较大，而基准测试 200 次平均更准确。基准测试显示 v5 比 v4 快 **1.26×**。

---

### v5 → v6：Warp Shuffle 替代 Shared Memory

**v6 核心变化：**
```cuda
// 用 __shfl_down_sync 替代 shared memory 的 warp 内归约
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val,  8);
val += __shfl_down_sync(0xffffffff, val,  4);
val += __shfl_down_sync(0xffffffff, val,  2);
val += __shfl_down_sync(0xffffffff, val,  1);

// 每个 warp 的 lane0 写入 shared memory（只需存 numWarps=8 个值）
__shared__ float shared[blockSize / 32];  // 仅 8 个 float = 32 bytes
if (lane == 0) shared[wid] = val;
```

**分析：**

`__shfl_down_sync` 是 warp 内线程间直接通过寄存器交换数据的指令，不经过 shared memory，延迟更低（寄存器访问 < 1 cycle，shared memory ~4 cycles）。

v6 只用 32 bytes 的 shared memory（8 个 warp 的结果），相比 v5 的 1024 bytes 大幅减少。这意味着：
- 每个 SM 可以同时容纳更多 block（shared memory 是 occupancy 的限制因素之一）
- 消除了 warp 内 6 次 shared memory 读写

**NCU 观测：**
- `Dynamic Shared Memory Per Block`：v5 = 1.02 KB → v6 = **32 bytes**（减少 97%）
- `FADD` 数从 124,576 → **188,064**（shuffle 指令需要更多 fadd 来完成等价操作）
- Achieved Occupancy：76.85% → **81.11%**（shared memory 减少，更多 block 可以并发）
- Duration（NCU）：37.60 → **36.03 μs**，基准测试 10.47 → **9.98 μs**，加速 **1.05×**

---

## 优化效果总结

```
时间（μs）对比（基准测试，200次平均）：

  v0_naive    ████████████████████████████████████████ 151.16 μs
  v1_seqaddr  ███████ 26.63 μs  (5.7× vs v0)
  v2_1stadd   ██████ 24.39 μs  (6.2× vs v0)
  v3_unwarp   █████ 18.55 μs   (8.2× vs v0)
  v4_unroll   ████ 13.14 μs    (11.5× vs v0)
  v5_multielem ███ 10.47 μs    (14.4× vs v0)
  v6_shuffle  ███ 9.98 μs      (15.1× vs v0)
```

### 各优化手段的收益排序

| 优化手段 | 加速比（相对上一版本）| 主要原理 |
|----------|---------------------|---------|
| 顺序寻址（v0→v1） | **5.7×** | 消除 warp divergence，减少 SM 空转 |
| 多元素/线程（v4→v5） | **1.26×** | 减少 block 调度开销，提升内存流水线 |
| First Add（v1→v2） | **1.09×** | 减少 idle 线程，节省 block 数 |
| 展开最后 Warp（v2→v3）| **1.21×** | 消除不必要的 __syncthreads |
| 模板展开（v3→v4） | **1.41×** | 编译期静态展开所有分支 |
| Warp Shuffle（v5→v6）| **1.05×** | 寄存器直接交换，减少 shared mem 访问 |

---

## 关键洞察

### 1. Orin（SM 8.7）上 Bank Conflict = 0
所有版本的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum = 0`。

这与经典教材描述不同——v0 理论上应有 bank conflict，但在 Orin 上：
- v0 的交错访问模式（`sdata[0], sdata[2], sdata[4]...`）每个线程访问不同 bank，确实不产生冲突
- **v0 的真正瓶颈是 warp divergence（`tid % 2s == 0` 导致同 warp 内分支分叉）**，而非 bank conflict

### 2. 顺序寻址（v0→v1）是收益最大的单步优化
消除 `Stall-Wait`（25.93% → 16.18%）带来 **5.7×** 提速，是所有版本中单步提升最大的。

### 3. 内存带宽瓶颈
理论上 reduce 操作需要读取 400 KB 全局内存，L2 Cache Throughput 在 v4-v6 均约 34-36%，说明仍有带宽提升空间，但对于如此小的数据集（400 KB），主要瓶颈已是 **kernel 调度开销** 而非内存带宽。

### 4. Compute (SM) Throughput 下降是正常现象
从 v0(65.47%) 到 v6(17.66%) 的 SM 计算利用率持续下降，但性能却持续提升。这说明 v0 的"高计算利用率"是假象——大量时钟周期浪费在 warp 内的串行化和 barrier 等待上，而优化版本让更多时钟周期用于真正有效的计算。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `reduce.cu` | 全部7个版本的 kernel + 性能基准测试框架 |
| `reduce_profile.cu` | NCU 专用 profiling 程序（接受版本号命令行参数）|
| `reduce` | 编译后的基准测试可执行文件 |
| `reduce_profile` | 编译后的 profiling 可执行文件 |
| `reduce_optimization.md` | 本文档 |

### 编译命令
```bash
nvcc -O3 -arch=sm_87 -std=c++17 -o reduce reduce.cu
nvcc -O3 -arch=sm_87 -std=c++17 -o reduce_profile reduce_profile.cu
```

### 运行命令
```bash
# 性能基准测试
./reduce

# NCU profiling（需要 root 权限）
sudo /usr/local/cuda/bin/ncu --set basic -o /tmp/reduce_v0 ./reduce_profile 0
sudo /usr/local/cuda/bin/ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,... ./reduce_profile 0
```
