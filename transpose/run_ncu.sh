#!/bin/bash
# Run with: sudo bash run_ncu.sh
# Profiles all transpose versions with ncu and saves results

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
NCU=/usr/local/cuda/bin/ncu
BIN=$DIR/transpose_profile

METRICS="sm__memory_throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
sm__sass_data_bytes_mem_global_op_ld.sum,\
sm__sass_data_bytes_mem_global_op_st.sum,\
l2__global_load_bytes.sum,\
l2__global_store_bytes.sum,\
smsp__sass_average_data_bytes_per_wavefront_mem_global.avg,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
achieved_occupancy"

NAMES=("ref_copy" "v0_naive" "v1_smem" "v2_padded" "v3_wide" "v4_ldg" "v5_diagonal")

for i in 0 1 2 3 4 5 6; do
  NAME=${NAMES[$i]}
  echo "===== Profiling ${NAME} (version $i) ====="
  
  # Basic metrics
  $NCU --set basic --csv $BIN $i > $DIR/ncu_${NAME}_basic.csv 2>&1
  echo "  Basic metrics saved to ncu_${NAME}_basic.csv"
  
  # Detailed custom metrics
  $NCU --metrics "$METRICS" --csv $BIN $i > $DIR/ncu_${NAME}_detail.csv 2>&1
  echo "  Detail metrics saved to ncu_${NAME}_detail.csv"
  
  # Save ncu-rep for GUI viewing
  $NCU --set full -o $DIR/ncu_${NAME} $BIN $i 2>&1 | tail -3
  echo "  Report saved to ncu_${NAME}.ncu-rep"
  echo ""
done

echo "All profiling complete!"
