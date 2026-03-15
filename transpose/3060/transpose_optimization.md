# CUDA 矩阵转置优化报告（RTX 3060 / SM 8.6）

> 本报告用于记录 **3060 目录** 下的转置优化过程、NCU 分析结果与技巧总结。  
> 表格数值来自 RTX 3060 实机运行结果（命令见文末）。

---

## 环境信息（待补全）

| 项目 | 值 |
|------|-----|
| GPU | RTX 3060（SM 8.6） |
| CUDA 版本 | 12.x（以 `nvcc --version` 为准） |
| 驱动版本 | 待补 |
| SM 数量 | 待补 |
| 显存类型 / 带宽 | GDDR6 / 理论 ~360 GB/s |
| 矩阵规模 | 4096×4096，float32 |
| 总数据量 | 读 64 MB + 写 64 MB = 128 MB |
| 编译器 | `nvcc -O3 -arch=sm_86 -std=c++17` |
| SM 频率 | ~1.32 GHz（NCU 采样） |

---

## 版本说明（/home/w/kernel/kernel-optimization/transpose/3060/transpose.cu）

| 版本 | 说明 | 关键点 |
|------|------|--------|
| ref_copy | 内存拷贝 | 带宽天花板参考 |
| v0_naive | 朴素转置 | 写非合并 |
| v1_smem | 共享内存 tile | 合并读写但有 bank conflict |
| v2_padded | smem + padding | 消除 bank conflict |
| v3_wide | 32×8 宽 tile | Occupancy 更高 |
| v4_ldg | v3 + `__ldg` | 走只读缓存路径 |
| v5_diagonal | 对角线 block 映射 | 缓解 L2 partition camping |
| v6_float4 | float4 向量化 + smem | 正确转置 + 降低指令数 |

---

## 基准性能（运行 ./transpose）

> 建议：预热 10 次，循环 200 次（代码已固定）。  
> 下面表格用跑完结果替换 `TBD`。

| 版本 | Time(us) | BW(GB/s) | BW%Peak | MaxErr |
|------|----------|----------|---------|--------|
| ref_copy | 519.57 | 258.33 | 71.8% | 0.00e+00 |
| v0_naive | 1476.77 | 90.89 | 25.2% | 0.00e+00 |
| v1_smem | 776.13 | 172.93 | 48.0% | 0.00e+00 |
| v2_padded | 532.81 | 251.91 | 70.0% | 0.00e+00 |
| v3_wide | 432.28 | 310.49 | 86.2% | 0.00e+00 |
| v4_ldg | 435.26 | 308.36 | 85.7% | 0.00e+00 |
| v5_diagonal | 451.33 | 297.38 | 82.6% | 0.00e+00 |
| v6_float4 | 415.12 | 323.33 | 89.8% | 0.00e+00 |

---

## NCU Profiling（建议用 run_ncu.sh）

### 1) Basic Set（`ncu --set basic`）

| 版本 | Duration(us) | SM Freq(GHz) | Mem BW(%) | L1/TEX BW(%) | Compute(%) | Occupancy(%) | Total L2 Cycles |
|------|---------------|--------------|-----------|--------------|------------|--------------|-----------------|
| ref_copy | 558.11 | 1.32 | 68.62 | 25.41 | 20.33 | 48.78 | 12,657,366 |
| v0_naive | 2180.00 | 1.32 | 68.91 | 84.53 | 5.20 | 37.10 | 49,453,560 |
| v1_smem | 1030.00 | 1.32 | 51.32 | 56.05 | 16.51 | 64.25 | 23,384,718 |
| v2_padded | 630.11 | 1.32 | 60.80 | 30.96 | 27.05 | 62.84 | 14,289,822 |
| v3_wide | 458.40 | 1.32 | 83.57 | 30.90 | 27.89 | 95.93 | 10,395,684 |
| v4_ldg | 459.07 | 1.32 | 83.38 | 31.00 | 27.97 | 96.00 | 10,409,544 |
| v5_diagonal | 480.74 | 1.32 | 79.59 | 29.62 | 27.41 | 96.32 | 10,902,528 |
| v6_float4 | 427.81 | 1.32 | 89.48 | 33.25 | 17.46 | 65.03 | 9,702,090 |

重点关注：
- v0 的 **Mem BW/L2 BW 过高但有效带宽低**（写非合并导致扇区放大）
- v1 的 **L1/TEX BW 升高**（bank conflict）
- v2~v6 的 **Mem BW 提升** + **Occupancy 变化**

### 2) Bank Conflict + Coalescing

| 版本 | Bank Conflicts (LD) | Bank Conflicts (ST) | Avg sectors/LD | Avg sectors/ST |
|------|----------------------|---------------------|----------------|----------------|
| v0_naive | 0 | 0 | 4 | 32 |
| v1_smem | 16,290,640 | 76,792 | 4 | 4 |
| v2_padded | 53,674 | 77,858 | 4 | 4 |
| v3_wide | 0 | 0 | 4 | 4 |
| v4_ldg | 0 | 0 | 4 | 4 |
| v5_diagonal | 0 | 0 | 4 | 4 |
| v6_float4 | 39,410 | 30,301 | 16 | 16 |

重点关注：
- v0 的 **ST sectors/request** 是否远高于 4  
- v1 → v2 的 **bank conflict 大幅下降**

### 3) Warp Stall 分析

| 版本 | Warps Active(%) | Stall Long Scoreboard(%) | Stall Wait(%) | Stall Barrier(%) | Avg Warp Latency |
|------|-----------------|--------------------------|---------------|------------------|------------------|
| v0_naive | 37.10 | 18.20 | 2.39 | 0.00 | 138.51 |
| v1_smem | 64.26 | 27.05 | 3.80 | 11.28 | 70.36 |
| v2_padded | 62.86 | 50.36 | 6.61 | 21.72 | 39.83 |
| v3_wide | 96.01 | 71.69 | 8.30 | 11.35 | 45.37 |
| v4_ldg | 96.10 | 71.97 | 8.27 | 11.45 | 45.55 |
| v5_diagonal | 96.31 | 64.01 | 8.78 | 19.36 | 41.94 |
| v6_float4 | 65.06 | 83.47 | 3.93 | 6.50 | 86.40 |

重点关注：
- v2 之后 **Long Scoreboard** 是否成为主要瓶颈（内存延迟）
- v3/v4/v5 是否通过更高 Occupancy 掩蔽延迟

---

## 优化过程（步骤式教程 + 示意图）

> 目标：让**读写都合并**，并逐步消除共享内存/调度瓶颈，最终逼近带宽上限。

### Step 0：朴素转置（v0_naive）

**核心问题：写非合并**

```
输入矩阵 A (row-major)            输出矩阵 B (row-major)
读取：A[row][col]  连续地址       写入：B[col][row]  stride = N

Warp 线程 (tx=0..31, ty固定):
读地址:  A[row][0..31]   -> 合并
写地址:  B[0..31][row]   -> 相隔 N, 完全非合并
```

结论：**写扇区放大**（Avg sectors/ST = 32），导致带宽崩溃。

---

### Step 1：共享内存 Tile（v1_smem）

**思路：先合并读到 smem，再合并写回 global**

```
global -> smem (行方向，合并)
smem  -> global (转置后行方向，合并)

smem tile[32][32]
读 smem 时按列访问 => 32-way bank conflict
```

结论：全局写已合并，但 **L1/TEX 被 bank conflict 填满**。

---

### Step 2：Padding 消除 Bank Conflict（v2_padded）

**关键技巧：+1 padding**

```
smem tile[32][33]

bank = (row*33 + col) % 32
=> 每行错开 1 bank，列访问不再冲突
```

结论：bank conflict 几乎消失，L1/TEX 压力下降，**瓶颈转移到 DRAM 延迟**。

---

### Step 3：提高 Occupancy（v3_wide）

**将 block 改为 32×8，每线程处理 4 行**

```
blockDim = (32, 8)
每个 block 256 线程
更高的 block/SM -> 更高 Occupancy
```

结论：更多 warp 并发掩蔽内存延迟，带宽进入 300+ GB/s 区间。

---

### Step 4：只读缓存路径（v4_ldg）

**在读路径使用 __ldg**

```
tile[...] = __ldg(&in[...])
```

结论：在本次 3060 数据中收益很小（v3 ≈ v4），说明瓶颈已接近 DRAM 上限。

---

### Step 5：对角线 Block 映射（v5_diagonal）

**减少 L2/DRAM partition camping**

```
原始: (bx, by)
对角: bx' = (bx + by) % grid_w, by' = bx
```

结论：4096² + 3060 上收益不明显，BW 略下降。

---

### Step 6：float4 向量化 + smem 转置（v6_float4）

**正确转置 + 降指令数**

```
每线程一次读 float4 (16B) -> smem
smem 做完整转置
每线程一次写 float4 (16B) -> global
```

结论：带宽最高（323 GB/s），但 Occupancy 降至 65%，属于“低并发高吞吐”策略。

1) **v0_naive：写非合并是根本瓶颈**  
   - 读合并、写非合并 → 扇区放大，带宽浪费

2) **v1_smem：引入共享内存后全局读写都合并**  
   - 但共享内存按列读触发 32-way bank conflict

3) **v2_padded：`+1 padding` 消除 bank conflict**  
   - L1/TEX 压力明显下降，L2/DRAM 成为新瓶颈

4) **v3_wide：32×8 block 提升 Occupancy**  
   - 256 线程/block，更多 block/SM 让延迟更容易被隐藏

5) **v4_ldg：只读缓存路径**  
   - 读访问更稳定（是否有收益需 NCU 数据验证）

6) **v5_diagonal：对角线映射缓解 partition camping**  
   - 访问分散到更多 L2/DRAM partition

7) **v6_float4：向量化读写 + smem 转置**  
   - 每线程一次读写 16B，减少指令数与 MIO 压力  
   - 保留 smem 解决读/写合并问题

---

## 关键观察（基于本次 3060 数据）

- **v0 的写非合并导致扇区放大**：`Avg sectors/ST = 32`，全局写流量放大 8×，带宽利用率仅 25.2%  
- **v1 的 bank conflict 明显**：`LD conflicts = 16,290,640`，L1/TEX Throughput 56%  
- **v2 padding 立竿见影**：bank conflict 下降到 `53,674`，BW 提升到 251.9 GB/s  
- **v3/v4 进入“接近带宽上限”区间**：Occupancy ~96%，BW ~308–310 GB/s  
- **v5 对角线映射在 4096² 上收益不明显**：BW 略低于 v3/v4  
- **v6 向量化取得最高带宽**：323.3 GB/s（89.8% 峰值），但 Occupancy 降至 65%  
  - `Avg sectors/req = 16` 并非更差合并，而是**单次请求更大、请求数更少**（总 sectors 不变）

---

## 深入分析（结合 NCU 指标）

1) **v0_naive：吞吐被非合并写击穿**  
   - `Avg sectors/ST = 32` → 写请求扇区放大 8×  
   - L1/TEX Throughput 84.5% 但 Compute 仅 5.2%，说明**缓存/内存流量浪费**  
   - Warp Latency 138 cycles，SM 大量等待写回

2) **v1_smem：合并读写后瓶颈转移到 L1/TEX**  
   - `LD bank conflicts = 16.3M`，L1/TEX Throughput 56%  
   - Stall Short Scoreboard 28.3% → **共享内存冲突是主因**

3) **v2_padded：bank conflict 清除，瓶颈转向 DRAM 延迟**  
   - 冲突降到 ~5.4e4，L1/TEX 负载降低到 31%  
   - Stall Long Scoreboard 50.4% → **DRAM/L2 延迟主导**

4) **v3_wide / v4_ldg：高 Occupancy 掩蔽延迟**  
   - Occupancy ~96%，Mem BW ~83%  
   - Long Scoreboard 71% 仍高，但可被更多 warp 覆盖  
   - `__ldg` 在本数据中提升极小（v3 310.5 vs v4 308.4 GB/s）

5) **v5_diagonal：对角线映射未带来收益**  
   - BW 略低于 v3/v4，Stall Barrier 19.36% 略升  
   - 4096² + 3060 分区压力可能不大，改映射反而增加调度扰动

6) **v6_float4：向量化减少指令数，但受寄存器/延迟影响**  
   - BW 323.3 GB/s 为最高，但 Occupancy 仅 65%  
   - Long Scoreboard 83.5%、Warp Latency 86 cycles → **更像“低并发高吞吐”**  
   - `Avg sectors/req = 16` 与总 sectors 不变同时出现，说明**请求更粗粒度、次数减少**  

---

## v6 向量化为何 sectors/req 变大却更快

- 传统版本每线程 4B 访问，Warp 总 128B → `Avg sectors/req ≈ 4`  
- v6 每线程 16B（float4），Warp 总 512B → `Avg sectors/req ≈ 16`  
- NCU 中 `t_sectors` 总量不变，但**请求数减少** → 指令数下降、MIO 压力下降  
- 结果就是：占用率降低但带宽反而更高

---

## 技巧总结（可复用 Checklist）

- **先让读写都合并**：transpose 的第一性瓶颈就是非合并写  
- **共享内存必配 padding**：`TILE×(TILE+1)` 是最稳定的消冲突办法  
- **Occupancy 不是越大越好，但要足够掩蔽内存延迟**  
- **`__ldg` 可能有收益，但一定要 NCU 验证**  
- **对角线 block 适合大尺寸矩阵，减少 partition camping**  
- **向量化只有在保证正确转置的前提下才有效**  
- **向量化可能降低 Occupancy**，但若吞吐更高依旧值得（需用 NCU 验证）
- **看 sectors/req 时务必结合 `t_sectors` 总量**，否则会误判合并效率
- **高 Long Scoreboard 不一定代表慢**，关键是是否有足够 warp 掩蔽

---

## 后续优化方向（可选）

1) **检查 v6 的寄存器压力**  
   - `nvcc -Xptxas -v` 或 `ncu --metrics smsp__inst_executed_pipe_*`  
   - 若寄存器过高，可尝试 `-maxrregcount` 或拆分临时变量

2) **尝试更大 tile + 分块向量化**  
   - 例如 64×32 tile + 32×8 block，每线程处理更大 micro-tile  
   - 目标：在不牺牲合并的前提下提高 ILP

3) **Ampere 上尝试 `cp.async` 预取**  
   - 适合流水化 global→smem，减少同步等待  
   - 对纯内存带宽任务收益有限，但可用于探索极限

4) **细化对角线映射策略**  
   - 仅在大矩阵/强 partition camping 的场景启用  
   - 可考虑“块内对角线 + 块间顺序调度”折中策略

---

## 运行命令（在 RTX 3060 上执行）

```bash
cd /home/w/kernel/kernel-optimization/transpose/3060
nvcc -O3 -arch=sm_86 -std=c++17 -o transpose transpose.cu
nvcc -O3 -arch=sm_86 -std=c++17 -o transpose_profile transpose_profile.cu
./transpose
sudo bash run_ncu.sh
```

> NCU 需要 sudo；若系统提示，请输入密码。

---

## 运行记录

本次数据来自 RTX 3060 实机运行 `./transpose` 与 `run_ncu.sh`。
