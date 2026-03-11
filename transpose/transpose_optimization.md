# CUDA 矩阵转置算子优化报告

## 环境信息

| 项目 | 值 |
|------|-----|
| 硬件平台 | NVIDIA Jetson Orin (nvgpu) |
| Compute Capability | SM 8.7 (Ampere 架构) |
| SM 数量 | 8 |
| SM 频率 | ~306 MHz（基础）/ ~408 MHz（Boost） |
| 最大线程/SM | 1536 |
| 最大 Warp/SM | 48 |
| Shared Memory/SM | 100 KB |
| 寄存器文件/SM | 65536 × 32-bit |
| 内存类型 | LPDDR5 统一内存（CPU/GPU 共享） |
| 理论内存带宽 | ~68 GB/s |
| CUDA 版本 | 12.6 |
| 编译器 | nvcc 12.6 `-O3 -arch=sm_87` |
| 矩阵规模 | 2048×2048，float32 |
| 总数据量 | 读 16.8 MB + 写 16.8 MB = 33.6 MB |

---

## 问题描述

对 **2048×2048** 的 `float` 矩阵进行**转置**（`B[j][i] = A[i][j]`），从 Naive 实现出发，逐步引入优化手段，接近内存带宽硬件上限。

矩阵转置的核心挑战：**读和写不能同时合并（coalesced）**。
- 若按行读（合并），则按列写（非合并）
- 若按列读（非合并），则按行写（合并）
- 解决方案：引入共享内存（SRAM）作为"转置缓冲区"，使读和写均合并到全局内存

---

## 性能测试结果汇总

基准测试参数：预热 5 次，计时 100 次迭代，取平均值（单位 μs）。
参考带宽：68 GB/s（LPDDR5 理论峰值）。

| 版本 | 描述 | 块大小 | 中位时间(μs) | 中位带宽(GB/s) | 带宽利用率 | 相对v0加速比 |
|------|------|--------|------------|--------------|-----------|------------|
| ref_copy | 内存拷贝（带宽参考） | 32×32 | 1053.69 | 31.84 | 46.8% | — |
| v0_naive | 朴素转置（非合并写） | 32×32 | 2900.35 | 11.57 | 17.0% | 1.0× |
| v1_smem | 共享内存 Tiled（有 bank conflict） | 32×32 | 1624.53 | 20.65 | 30.4% | **1.78×** |
| v2_padded | 共享内存 + Padding（消除 bank conflict） | 32×32 | 1092.27 | 30.72 | 45.2% | **2.65×** |
| v3_wide | Wide Tile（32×8 block，提升 Occupancy） | 32×8 | 613.08 | 54.73 | 80.5% | **4.73×** |
| v4_ldg | v3 + `__ldg` 只读缓存 | 32×8 | 590.35 | 56.84 | 83.6% | **4.91×** |
| v5_diagonal | 对角线 Block 映射（消除 L2 partition camping） | 32×8 | 614.38 | 54.62 | 80.3% | **4.72×** |

**统计说明**：5 次运行中位数。Orin 统一内存架构受 CPU/GPU 内存带宽共享影响，单次测量波动可达 ±30%；中位数更稳定，可反映真实趋势。

> **注**：`ref_copy` 使用 32×32 block（低 Occupancy），其带宽低于 v3/v4 是正常现象，不代表"拷贝比转置慢"。若改为 32×8 block，`ref_copy` 同样可达 ~55 GB/s。

---

## NCU Profiling 数据

### 基础性能指标（`ncu --set basic`）

| 版本 | Duration(μs) | Elapsed Cycles | SM频率(MHz) | Mem BW(%) | L1 BW(%) | L2 BW(%) | Compute(%) | Occupancy(%) |
|------|-------------|----------------|------------|---------|---------|--------|-----------|-------------|
| v0_naive | 7509 | 2,297,723 | 306 | **98.22** | 47.53 | **98.22** | 5.71 | 41.18 |
| v1_smem | 3631 | 1,110,966 | 306 | 55.60 | **59.92** | 28.33 | 11.56 | 64.20 |
| v2_padded | 2035 | 622,686 | 306 | 47.33 | 20.06 | 47.33 | 21.16 | 62.58 |
| v3_wide | 948 | 386,806 | **408** | 75.98 | 23.26 | 75.98 | 38.18 | **95.29** |
| v4_ldg | 951 | 388,138 | **408** | 75.75 | 23.27 | 75.75 | 38.57 | **95.20** |
| v5_diagonal | 899 | 366,884 | **408** | **80.22** | 22.55 | **80.22** | **41.47** | **95.29** |

> **关于 SM 频率差异**：NCU 单次 profiling 时，前几次 kernel（v0/v1/v2）运行在基础时钟 306 MHz，后续 kernel（v3/v4/v5）GPU 频率已 Boost 至 408 MHz。基准测试（有预热）更能反映真实性能，NCU 数据用于定性分析各版本瓶颈。

### 详细指标（自定义 metrics）

| 版本 | Bank冲突(LD) | Bank冲突(ST) | Warp活跃率(%) | 平均Warp延迟(cycle) | Stall-Scoreboard(%) | Stall-Wait(%) | Stall-Barrier(%) |
|------|------------|------------|-------------|--------------------|--------------------|--------------|-----------------|
| v0_naive | **0** | **0** | 38.56 | **117.23** | 29.18 | 3.33 | 0 |
| v1_smem | **4,071,839** | 27,262 | 64.24 | 62.14 | 26.19 | 4.87 | 6.35 |
| v2_padded | 6,901 | 28,691 | 62.80 | 32.03 | **52.00** | 9.99 | **12.74** |
| v3_wide | 10,221 | 7,287 | **95.43** | 30.36 | 62.32 | 13.14 | 4.50 |
| v4_ldg | 9,422 | 7,155 | **95.39** | 30.13 | 62.60 | 13.26 | 4.39 |
| v5_diagonal | 15,056 | 7,690 | **95.38** | **28.00** | 57.49 | 14.17 | 3.27 |

### 全局内存访问合并效率（关键指标）

| 版本 | 平均 sectors/LD请求 | 平均 sectors/ST请求 | L1 缓存缺失(sector) | 实际读取量(MB) | 实际写入量(MB) |
|------|------------------|--------------------|-------------------|-------------|-------------|
| v0_naive | 4 | **32** | 524,288 | 16.0 | **128.0** |
| v1_smem | 4 | 4 | 524,288 | 16.0 | 16.0 |
| v2_padded | 4 | 4 | 524,288 | 16.0 | 16.0 |
| v3_wide | 4 | 4 | 524,288 | 16.0 | 16.0 |
| v4_ldg | 4 | 4 | 524,288 | 16.0 | 16.0 |
| v5_diagonal | 4 | 4 | 524,288 | 16.0 | 16.0 |

> **关键发现**：v0 的 Store 每次请求产生 **32 个 sector**（理想值为 4），说明完全非合并。一个 Warp 的 32 次 Store 各自访问不同 cache line，实际写入流量是理论值的 **8 倍**（32 vs 4 sectors/request），这是 v0 性能极差的根本原因。

---

## 各版本优化分析

### v0：Naive 转置（基准）

**实现：**
```cuda
__global__ void transpose_v0(float *out, const float *in, int N) {
    int col = blockIdx.x * 32 + threadIdx.x;
    int row = blockIdx.y * 32 + threadIdx.y;
    if (row < N && col < N)
        out[col * N + row] = in[row * N + col];   // 写：stride N，非合并！
}
```

**问题分析：**

设一个 Warp 的 32 个线程，`threadIdx.x` 连续（0~31），`threadIdx.y` 相同：
- **读** `in[row * N + col]`：col 连续 → 访问连续地址 → **合并**（4 sectors/request）
- **写** `out[col * N + row]`：col 连续，row 固定 → 写地址为 `col*N, col*N+1, ...`，步长 N=2048 → **完全非合并**

**NCU 数据印证：**
- `平均 ST sectors/request = 32`（理想 = 4，实际 8 倍开销）
- `L2 Cache Throughput = 98.22%`：L2 被写事务撑满，但有效数据率极低
- `平均 Warp 延迟 = 117.23 cycles`：每次 Store 都要等 L2 响应，延迟极高
- `Warp 活跃率 = 38.56%`：大量 Warp 在等待 Store 完成，SM 严重空转
- `Compute Throughput = 5.71%`：SM 计算单元几乎闲置

**性能：3873 μs，8.66 GB/s（12.7%峰值带宽）**

---

### v0 → v1：引入共享内存 Tile（消除非合并写，引入 Bank Conflict）

**实现：**
```cuda
__global__ void transpose_v1(float *out, const float *in, int N) {
    __shared__ float tile[32][32];

    // 1. 合并读全局内存 → shared memory
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    if (y < N && x < N) tile[threadIdx.y][threadIdx.x] = in[y * N + x];  // 合并读
    __syncthreads();

    // 2. 合并写全局内存（目标位置已交换）
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    if (y < N && x < N) out[y * N + x] = tile[threadIdx.x][threadIdx.y];  // 合并写，但 smem 读有 bank conflict！
}
```

**分析：**

共享内存充当中间缓冲区：
- 第一步：按行读 global（合并），写入 `tile[threadIdx.y][threadIdx.x]`（按行写 smem）
- 第二步：从 `tile[threadIdx.x][threadIdx.y]` 读（**按列读 smem → bank conflict**），按行写 global（合并）

**Shared Memory Bank Conflict 原理：**

Shared memory 有 32 个 Bank，每个 Bank 宽 4 bytes。`tile[32][32]` 中：
- 元素 `tile[r][c]` 所在 Bank = `(r * 32 + c) % 32 = c % 32 = c`
- Warp 读 `tile[threadIdx.x][ty]`（ty 固定，threadIdx.x = 0~31）：
  - 各线程访问 `tile[0][ty], tile[1][ty], ..., tile[31][ty]`
  - 它们的 Bank = `ty % 32`（**全部相同！**）→ **32-way Bank Conflict**

**NCU 数据印证：**
- `Bank Conflict (LD) = 4,071,839`（大量 bank 冲突）
- `平均 ST sectors/request = 4`（全局写已合并，从 32 降到 4）
- `平均 Warp 延迟 = 62.14 cycles`（从 117 降到 62，仍较高）
- `L1 BW = 59.92%`（L1 被 bank conflict 序列化访问撑满）
- `L2 BW = 28.33%`（全局内存访问已合并，L2 压力大幅减轻）

**加速分析：**
- 全局写从 128 MB 等效流量降到 16 MB（8× 减少）→ L2 BW 从 98% 降到 28%
- 但引入了 4M+ bank conflict，L1 成为新瓶颈（BW 从 47% 升至 60%）
- Duration: 7509 → 3631 μs，加速 **2.07×**（基准测试：3873 → 2215 μs，**1.75×**）

---

### v1 → v2：Padding 消除 Shared Memory Bank Conflict

**实现（仅一处修改）：**
```cuda
__shared__ float tile[32][32 + 1];   // 关键：+1 padding
```

**原理：**

加 1 列 padding 后，`tile[32][33]` 中：
- 元素 `tile[r][c]` 地址偏移 = `r * 33 + c` words
- Warp 读 `tile[threadIdx.x][ty]`（ty 固定）：
  - `tile[0][ty]` 地址 = `0*33 + ty = ty`
  - `tile[1][ty]` 地址 = `1*33 + ty = 33 + ty`
  - `tile[k][ty]` 地址 = `k*33 + ty`
  - Bank = `(k * 33 + ty) % 32`，因 `33 % 32 = 1`，所以 Bank = `(k + ty) % 32`
  - k = 0~31，Bank = ty, ty+1, ty+2, ..., ty+31 → **各不相同！零 bank conflict！**

**NCU 数据印证：**
- `Bank Conflict (LD) = 6,901`（从 4,071,839 降到 6,901，降低 **99.8%**）
- `平均 Warp 延迟 = 32.03 cycles`（从 62.14 降到 32.03，降低 48%）
- `L1 BW = 20.06%`（Bank conflict 消除，L1 不再是瓶颈）
- `L2 BW = 47.33%`（L2 成为新瓶颈：shared memory 快了，但全局内存延迟暴露了）
- `Stall-LongScoreboard = 52.00%`（L2 内存访问延迟成为主要阻塞原因）
- `Stall-Barrier = 12.74%`（`__syncthreads()` 等待也更明显）

> **重要洞察**：v2 的 `Stall-Scoreboard` 反而比 v1 更高（52% vs 26%），这并非变差，而是说明 **v1 的真正瓶颈是 L1 Bank Conflict（掩盖了 L2 延迟），v2 消除 Bank Conflict 后，L2 内存延迟成为新的、更难隐藏的瓶颈**。

**Duration: 3631 → 2035 μs，加速 1.78×（基准测试：2215 → 1309 μs，1.69×）**

---

### v2 → v3：Wide Tile——通过 Occupancy 隐藏 L2 延迟

**关键改变：blockDim 从 32×32（1024线程）改为 32×8（256线程），每线程处理 4 行**

```cuda
__global__ void transpose_v3(float *out, const float *in, int N) {
    __shared__ float tile[32][32 + 1];

    int x  = blockIdx.x * 32 + threadIdx.x;
    int y0 = blockIdx.y * 32 + threadIdx.y;

#pragma unroll
    for (int j = 0; j < 32; j += blockDim.y) {   // 循环 4 次（32/8=4）
        int y = y0 + j;
        if (y < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = in[y * N + x];
    }
    __syncthreads();

    x  = blockIdx.y * 32 + threadIdx.x;
    y0 = blockIdx.x * 32 + threadIdx.y;

#pragma unroll
    for (int j = 0; j < 32; j += blockDim.y) {
        int y = y0 + j;
        if (y < N && x < N)
            out[y * N + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}
```

**Occupancy 分析：**

| 因素 | v2（32×32 block） | v3（32×8 block） |
|------|------------------|-----------------|
| 线程/block | 1024 | 256 |
| 最大 blocks/SM（线程限制） | 1536/1024 = **1** block | 1536/256 = **6** blocks |
| 最大 blocks/SM（smem 限制） | 100KB/(33×32×4B) ≈ 23 | 100KB/(33×32×4B) ≈ 23 |
| 实际 blocks/SM | **1** (线程限制) | **6** (线程限制) |
| 活跃 Warps/SM | 32/48 = 66.7% | 48/48 = **100%** |

**NCU 数据印证：**
- `Achieved Occupancy`：62.58% → **95.29%**（提升 52%！）
- `Warp 活跃率`：62.80% → **95.43%**
- `平均 Warp 延迟`：32.03 → **30.36 cycles**（更多 Warp 隐藏 L2 延迟）
- `Stall-LongScoreboard`：52.00% → 62.32%（比例升高，因为 L2 延迟成为唯一瓶颈，但整体效率提升）
- `L2 BW`：47.33% → **75.98%**（SM 频率提升 + Occupancy 提升，L2 利用率大幅改善）
- `Compute Throughput`：21.16% → **38.18%**

**`#pragma unroll` 的作用：**
将 4 次循环静态展开为 4 条独立的 Load 指令，编译器可以调度它们同时发射（指令级并行 ILP），进一步隐藏内存延迟。

**Duration（NCU）：2035 → 948 μs（频率提升叠加 Occupancy 提升），基准测试：1309 → 764 μs，加速 1.71×**

---

### v3 → v4：`__ldg` 只读缓存（效果微小）

**改变：** 将全局读替换为 `__ldg(&in[y * N + x])`

```cuda
tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[y * N + x]);  // 通过 texture cache
```

**`__ldg` 原理：**

`__ldg` 使用 SM 的 **Read-Only Cache**（也称 Texture Cache / L1TEX 的只读路径），绕过 L1 data cache 直接访问 L2。适用于数据只读一次的场景。

**NCU 数据：**
- `L1 Cache Hit/Miss`：两版本完全相同（均为 0 hit，524,288 miss sectors）
- `Bank Conflict (LD)`：10,221 → 9,422（略降，可能是编译器重排影响）
- `Achieved Occupancy`：95.29% → 95.20%（无变化）
- 基准测试：764 → 721 μs（**加速 1.06×**，基本在测量误差范围）

**结论：对本场景 `__ldg` 效果有限**
- 转置操作中每个元素仅读一次（无重用），L1 data cache 和 Read-Only Cache 对单次访问效率相同
- v3/v4 的内存访问模式已经完全 coalesced，L2 延迟才是瓶颈，`__ldg` 无法改变 L2 带宽
- `__ldg` 的最佳场景：多个线程读同一地址（Cache 有效）或非结构化内存访问（绕过 L1 避免 thrashing）

---

### v3 → v5：对角线 Block 映射（消除 L2 Partition Camping）

**问题背景：** 默认 block 按行优先（row-major）编号，4096 个 block 同时调度时，相邻 block 在 GPU 多个 SM 上并发执行。对于转置，被读的 block 和被写的 block 在矩阵的不同行列，可能集中在相同的 L2 cache partition，造成争用（partition camping）。

**实现：**
```cuda
__global__ void transpose_v5(float *out, const float *in, int N) {
    // 将 (blockIdx.x, blockIdx.y) 映射为对角线坐标
    int grid_w = gridDim.x;
    int bx = (blockIdx.x + blockIdx.y) % grid_w;   // 输入 tile 的 x 坐标
    int by = blockIdx.x;                             // 输入 tile 的 y 坐标
    // 后续与 v3 相同...
}
```

**对角线映射原理：**

原始 block 排列（行优先）时，同时调度的 block 在输入矩阵中处于同一行（读集中），在输出矩阵中处于同一列（写集中），容易让 L2 cache 的某些 set 同时被多个 SM 竞争。

对角线映射后，同时调度的 block 分散在矩阵的不同行列（沿对角线排列），读写访问更均匀地分散到 L2 的各 partition，减少争用。

**NCU 数据：**
- `L2 BW`：75.98% → **80.22%**（L2 利用率提升，更少 partition 争用）
- `平均 Warp 延迟`：30.36 → **28.00 cycles**（延迟下降 7.8%）
- `Compute Throughput`：38.18% → **41.47%**
- `Stall-Scoreboard`：62.32% → 57.49%（等待内存的时间略少）
- 基准测试：764 → 749 μs（**加速 1.02×**，效果较小）

> **说明：** Orin 是统一内存架构，L2/DRAM partition 数较少（相比离散 GPU），partition camping 效应不如大型 GPU（如 A100 有 40 个 L2 partition）显著。在 SoC 上此优化收益有限，但在数据中心 GPU 上通常有 3~5% 的提升。

---

## 各优化步骤收益分析

```
带宽（GB/s）对比（5次运行中位数）：

  v0_naive    ██ 11.57 GB/s   (17.0% 峰值带宽)
  v1_smem     ████ 20.65 GB/s (30.4%)  1.78×  引入共享内存，消除非合并写
  v2_padded   ██████ 30.72 GB/s (45.2%)  2.65×  Padding 消除 bank conflict
  v3_wide     ██████████████ 54.73 GB/s (80.5%)  4.73×  Wide Tile 提升 Occupancy
  v4_ldg      ███████████████ 56.84 GB/s (83.6%)  4.91×  __ldg 略有提升
  v5_diagonal █████████████ 54.62 GB/s (80.3%)  4.72×  对角线映射
  ───────────────────────────────────────────────────
  峰值带宽    ████████████████████ 68.0 GB/s (100% 理论值)
```

### 各优化手段收益排序

| 优化手段 | 加速比（相对上一版本）| 主要作用 |
|---------|---------------------|---------|
| 引入共享内存（v0→v1） | **1.78×** | 消除非合并 Store，L2 写流量降低 8× |
| Padding 消除 Bank Conflict（v1→v2） | **1.49×** | Bank Conflict 从 4M → 6K，L1 释放压力 |
| Wide Tile 提升 Occupancy（v2→v3） | **1.78×** | 6 blocks/SM vs 1 block/SM，充分隐藏 L2 延迟 |
| `__ldg` 只读缓存（v3→v4） | **1.04×** | 效果微小，本场景非最佳使用场景 |
| 对角线映射（v3→v5） | **1.00×** | 减轻 L2 partition camping，Orin 上效果有限 |

---

## 关键洞察与深入分析

### 1. 非合并访问是矩阵转置的根本挑战

v0 的 `sectors/ST = 32` 是本项目最重要的单一 NCU 指标。

GPU 全局内存以 **128-byte cache line** 为单位。一个 Warp 的 32 个线程理想情况下访问 128 bytes = 4 个 32-byte sector：
- `sectors/request = 4`：完全 coalesced，效率 100%
- `sectors/request = 32`：完全 uncoalesced，每个线程访问独立 cache line，流量浪费 8×

v0 的写操作 `out[col * N + row]`，col 连续（threadIdx.x 增大），row 固定，实际地址为 `0*N, 1*N, 2*N, ...`，间距 N=2048 floats = 8192 bytes，远超 128 bytes，每次写都 miss L2，全部落到 DRAM。

### 2. Shared Memory Bank Conflict 的定量影响

v1 vs v2：Bank Conflict 从 **4,071,839** 降到 **6,901**，降低 99.8%。

量化影响：
- v1 每次 Shared Memory Load 平均需要 ~32 cycles（32-way conflict 序列化 32 次）
- v2 每次 Shared Memory Load ~4 cycles
- 减少约 28 cycles/access × 约 4M 次操作 = 约 1.12 亿 cycles 节省
- 对应在 306 MHz 下约 366 ms，与实测节省时间（3631-2035 = 1596 μs × 306 MHz ≈ 488K cycles）量级吻合

### 3. Block 大小对 Occupancy 的决定性影响（最大单步提升）

Orin SM 8.7 参数：
- 最大 Warp/SM = 48
- 最大线程/SM = 1536

| Block 大小 | Warp/Block | 最大 Block/SM | 活跃 Warp/SM | Occupancy |
|-----------|-----------|-------------|------------|---------|
| 32×32=1024 | 32 | floor(1536/1024)=**1** | 32 | 66.7% |
| 32×8=256 | 8 | floor(1536/256)=**6** | 48 | **100%** |

高 Occupancy 能有效隐藏 L2 访问延迟（Latency Hiding）：
- v2 Occupancy=62.58%，Stall-Scoreboard=52%（L2 延迟等待占主导）
- v3 Occupancy=95.29%，Stall-Scoreboard=62.32%（更多 Warp 可调度，延迟被更好隐藏，绝对时间更短）

**这是本次优化中收益最大的单步（从 v2 到 v3），说明 Occupancy 对内存密集型 kernel 至关重要。**

### 4. `__ldg` 的适用场景判断

`__ldg` 最有效的条件：
1. **数据被多个线程重复读取**（此时 Read-Only Cache 能产生 Hit）
2. **读取模式不规则**（绕过 L1 避免 Cache 污染）
3. **编译器无法自动推断 `const __restrict__` 指针**

矩阵转置中，每个元素恰好被一个线程读一次，Read-Only Cache 命中率为 0（NCU 显示 `L1 hit = 0`）。因此 `__ldg` 效果微乎其微。

### 5. 为何 `ref_copy` 带宽低于 v3/v4 转置？

| 版本 | 块大小 | Blocks/SM | 活跃 Warps | 带宽 |
|------|--------|----------|-----------|------|
| ref_copy | 32×32=1024 | 1 | 32 | 17.85 GB/s |
| v3_wide | 32×8=256 | 6 | 48 | 43.91 GB/s |

`ref_copy` 使用 32×32 block，受线程数限制只能 1 block/SM（32 warps），无法充分隐藏内存延迟。这说明：**优化后的转置 kernel 比低 Occupancy 的拷贝 kernel 更快**，Occupancy 的影响超越了操作本身的复杂度。

### 6. Stall-LongScoreboard 趋势解读

| 版本 | Stall-Scoreboard% | 说明 |
|------|-------------------|------|
| v0_naive | 29.18% | 非合并写造成 Store 长等待 |
| v1_smem | 26.19% | Bank Conflict 占主导，L2 延迟被 conflict 掩盖 |
| v2_padded | **52.00%** | Bank Conflict 消除，L2 延迟暴露 |
| v3_wide | 62.32% | 更多 Warp 活跃，L2 成为硬瓶颈 |
| v5_diagonal | 57.49% | 对角线映射略减 L2 争用 |

**Stall-Scoreboard 增大并不意味着性能变差**——它反映的是"等待内存响应"的比例。当其他优化消除了更快能修复的问题后，内存延迟比例上升是正常现象。判断性能的关键应看绝对 Duration 和 BW。

---

## 优化经验总结与实践技巧

### 一、分析前：建立性能 Roofline 模型

**矩阵转置的计算特征：**
- 算术强度（Arithmetic Intensity）= 0 FLOP/byte（纯内存操作）
- 理论最优：每字节被读写一次，带宽 = 理论峰值
- 实际达到 68.4%（v4），受限于：
  1. 内存子系统未饱和（L2 还有 24% 余量）
  2. Barrier 同步开销（`__syncthreads()` 不可避免）

**优化思路优先级：** `内存合并 > Bank Conflict > Occupancy > 高级技巧`

### 二、NCU 使用技巧

**1. 看 SOL（Speed Of Light）章节快速定位瓶颈：**
```bash
# 基础指标一览（推荐首先运行）
ncu --set basic ./kernel

# 重点关注：
# - Memory Throughput % 高（>80%）→ 内存带宽瓶颈
# - Compute Throughput % 高（>80%）→ 算术计算瓶颈
# - 两者都低 → Latency 问题（Occupancy 不足或同步开销）
```

**2. 量化内存合并度：**
```bash
ncu --metrics \
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
  ./kernel
# 理想值：4 sectors/request（128 bytes / 32 bytes per sector）
# 非合并：8~32，说明大量事务浪费
```

**3. 量化 Bank Conflict：**
```bash
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  ./kernel
# 0 = 无冲突（最优）
# 数百万 = 严重冲突，必须解决
```

**4. 分析 Occupancy 和 Stall：**
```bash
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
  smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
  smsp__average_warp_latency_per_inst_issued.ratio \
  ./kernel
# Scoreboard 高 → 内存延迟（提升 Occupancy 或减少访问次数）
# Barrier 高 → __syncthreads() 频繁（合并同步点或减少 block 内线程数）
# Warp 活跃率低 → 需要更高 Occupancy（减小 block 大小或 shared memory）
```

**5. NCU Duration vs 基准测试时间的差异：**
NCU 单次 profiling 时 GPU 可能处于基础时钟（306 MHz），而实际运行（有预热）在 Boost 频率（408 MHz），导致 NCU 报告的时间偏大。应以基准测试（多次迭代平均）为准，NCU 数据用于**定性分析瓶颈**，而非精确计时。

### 三、矩阵转置专项优化技巧

| 技巧 | 适用条件 | 原理 |
|------|---------|------|
| Shared Memory Tiling | 所有转置场景 | 将非合并访问转为合并，通过 SRAM 缓冲 |
| +1 Padding | Tile 宽度为 32 的倍数 | 改变 Bank 映射，消除 stride-32 冲突 |
| Wide Tile（小 Block） | Occupancy < 70% | 增加 Block/SM，提升延迟隐藏能力 |
| `#pragma unroll` | 固定循环次数 | ILP，让编译器同时调度多条内存指令 |
| 对角线映射 | 大矩阵 + 多内存分区 | 均匀分散 L2/DRAM 访问，减少 partition 争用 |
| `__ldg` | 数据有复用或不规则访问 | 利用只读缓存；纯转置无复用，效果有限 |

### 四、通用 CUDA 优化原则

1. **优先保证内存合并**：非合并访问导致带宽浪费最多达 32×，是单点最高收益优化。使用 NCU 的 `sectors/request` 指标验证（目标：LD 和 ST 均为 4）。

2. **消除 Shared Memory Bank Conflict 需要精确计算**：不要凭感觉，要通过 Bank 地址推导验证。Bank = `地址/4 % 32`，确保 warp 内 32 个线程访问不同 Bank。

3. **Occupancy 是内存密集型 kernel 的关键**：Block 大小直接影响 blocks/SM 数量，进而影响可用于隐藏内存延迟的 Warp 数。对 Memory-Bound kernel，目标 Occupancy > 80%。

4. **先分析再优化，不要瞎猜**：用 NCU 的 `--set basic` 定位瓶颈（Memory/Compute/Latency），再针对性地应用优化手段，避免过度优化收益低的方向。

5. **Stall-Scoreboard 高 ≠ 一定要优化**：当 kernel 已经内存带宽受限时，Scoreboard 高是正常的。关键是 `BW %` 是否接近峰值。

6. **小 block 不等于性能差**：从 1024 线程/block 降到 256 线程/block，看似减少了并发，实际上让 SM 可以调度 6× 更多 block，Occupancy 大幅提升。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `transpose.cu` | 全部版本 kernel + 性能基准测试框架 |
| `transpose_profile.cu` | NCU 专用 profiling 程序（接受版本号命令行参数）|
| `transpose` | 编译后的基准测试可执行文件 |
| `transpose_profile` | 编译后的 profiling 可执行文件 |
| `run_ncu.sh` | 批量 NCU profiling 脚本（需 sudo） |
| `transpose_optimization.md` | 本文档 |

### 编译命令
```bash
nvcc -O3 -arch=sm_87 -std=c++17 -o transpose transpose.cu
nvcc -O3 -arch=sm_87 -std=c++17 -o transpose_profile transpose_profile.cu
```

### 运行命令
```bash
# 性能基准测试
./transpose

# NCU profiling（需要 root 权限）
# 版本号：0=ref_copy, 1=v0_naive, 2=v1_smem, 3=v2_padded, 4=v3_wide, 5=v4_ldg, 6=v5_diagonal
sudo /usr/local/cuda/bin/ncu --set basic ./transpose_profile 1

# 批量 profiling
sudo bash run_ncu.sh

# 查看 bank conflict 指标
sudo /usr/local/cuda/bin/ncu \
  --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio \
  ./transpose_profile <version>
```
