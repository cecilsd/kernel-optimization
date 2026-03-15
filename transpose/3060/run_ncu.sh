#!/usr/bin/env bash
# NCU Profiling 脚本 —— 矩阵转置 RTX 3060
# 用法：sudo bash run_ncu.sh
# 需要 sudo 权限

set -e
cd "$(dirname "$0")"

NCU=/usr/local/cuda/bin/ncu
BIN=./transpose_profile

VERSIONS=(0 1 2 3 4 5 6 7)
NAMES=(ref_copy v0_naive v1_smem v2_padded v3_wide v4_ldg v5_diagonal v6_float4)

echo "========================================="
echo " NCU Basic Set (SOL / Occupancy / BW)"
echo "========================================="
for i in "${!VERSIONS[@]}"; do
    v=${VERSIONS[$i]}
    n=${NAMES[$i]}
    echo ""
    echo "--- ${n} (version ${v}) ---"
    $NCU --set basic --kernel-name-base function $BIN $v 2>&1 | \
        grep -E "Duration|Memory Throughput|L1/TEX|Compute \(SM\)|Achieved Occupancy|Elapsed Cycles|SM Frequency" || true
done

echo ""
echo "========================================="
echo " NCU 自定义指标：Bank Conflict + Coalescing"
echo "========================================="
METRICS="\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"

for i in "${!VERSIONS[@]}"; do
    v=${VERSIONS[$i]}
    n=${NAMES[$i]}
    echo ""
    echo "--- ${n} (version ${v}) ---"
    $NCU --metrics "$METRICS" $BIN $v 2>&1 | \
        grep -E "bank_conflicts|sectors_per_request|sectors_pipe" || true
done

echo ""
echo "========================================="
echo " NCU 自定义指标：Warp Stall 分析"
echo "========================================="
STALL_METRICS="\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__average_warp_latency_per_inst_issued.ratio"

for i in "${!VERSIONS[@]}"; do
    v=${VERSIONS[$i]}
    n=${NAMES[$i]}
    echo ""
    echo "--- ${n} (version ${v}) ---"
    $NCU --metrics "$STALL_METRICS" $BIN $v 2>&1 | \
        grep -E "warps_active|stalled|warp_latency" || true
done

echo ""
echo "Done."
