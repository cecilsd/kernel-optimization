CXX      = nvcc
ARCH     = sm_86
CFLAGS   = -O3 -arch=$(ARCH) -std=c++17 -Xptxas -v

.PHONY: all clean run profile

all: reduce reduce_profile

reduce: reduce.cu
	$(CXX) $(CFLAGS) -o $@ $<

reduce_profile: reduce_profile.cu
	$(CXX) $(CFLAGS) -o $@ $<

run: reduce
	./reduce

# NCU profiling targets (需要 root 或 paranoia_level=0)
NCU = sudo /usr/local/cuda/bin/ncu

# 一键采集所有版本的 basic 指标
profile_all: reduce_profile
	@for v in 0 1 2 3 4 5 6 7; do \
		echo "=== Profiling v$$v ==="; \
		$(NCU) --set basic \
		       --kernel-name-base function \
		       -o profile_v$$v \
		       ./reduce_profile $$v; \
	done

# 采集单个版本的完整指标
profile_v%: reduce_profile
	$(NCU) --set full \
	       --import-source yes \
	       -o profile_v$* \
	       ./reduce_profile $*

# 采集关键内存指标（带宽分析）
profile_mem_%: reduce_profile
	$(NCU) \
	  --metrics \
	  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
	  lts__t_bytes.sum.per_second,\
	  dram__bytes_read.sum.per_second,\
	  dram__bytes_write.sum.per_second,\
	  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
	  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
	  smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
	  smsp__warp_issue_stalled_branch_not_taken_per_warp_active.pct,\
	  smsp__warp_issue_stalled_scoreboard_per_warp_active.pct,\
	  smsp__warp_issue_stalled_wait_per_warp_active.pct,\
	  sm__warps_active.avg.pct_of_peak_sustained_active,\
	  sm__throughput.avg.pct_of_peak_sustained_elapsed \
	  ./reduce_profile $*

clean:
	rm -f reduce reduce_profile *.ncu-rep profile_v*
