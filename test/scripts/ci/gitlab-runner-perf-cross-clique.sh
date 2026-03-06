#!/bin/bash
#
# Cross-clique P2P performance tests.
# Emulates cross-clique by assigning per-node NCCL_MNNVL_CLIQUE_ID values.
# Requires the cross-clique-wrapper.sh script in the srun command line.

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

WRAPPER="test/scripts/ci/cross-clique-wrapper.sh"

opts="-w 1 -G 0 -s 512M -M 1"
range="-b 8 -e $MAX -f 2"

# Args for run_command
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"

# ============================================================================
# Standard collectives with user buffer registration
# ============================================================================
for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
  run_command "xclique_${func}" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/$func" "$range $opts -R 1"
done

# AlltoAll with user buffer registration
run_command "xclique_alltoall_perf" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 1"

# ============================================================================
# AllReduce without NVLS (pure cross-clique P2P, no hierarchical NVLS)
# ============================================================================
run_command "xclique_all_reduce_perf_nvls_off" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE} NCCL_NVLS_ENABLE=0" "$WRAPPER $NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -R 1"

# ============================================================================
# Force TREE algorithm to test pull mode
# ============================================================================
run_command "xclique_all_reduce_perf_tree" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE} NCCL_ALGO=Tree" "$WRAPPER $NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -R 1"
run_command "xclique_all_reduce_perf_tree_nvls_off" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE} NCCL_ALGO=Tree NCCL_NVLS_ENABLE=0" "$WRAPPER $NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -R 1"

# ============================================================================
# CE (Copy Engine) collectives - zero-SM path with unicast sync
# ============================================================================
for func in all_gather_perf alltoall_perf scatter_perf gather_perf; do
  run_command "xclique_${func}_ce" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2 -w 1 -M 1"
done

# ============================================================================
# AlltoAll with LSA kernels (Device API implementations)
# ============================================================================
for impl in 1 2; do
  run_command "xclique_alltoall_perf_lsa_${impl}" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
done

# ============================================================================
# AlltoAll with GIN kernels
# ============================================================================
for impl in 3; do
  run_command "xclique_alltoall_perf_gin_proxy_${impl}" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE} NCCL_GIN_TYPE=2" "$WRAPPER $NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
  run_command "xclique_alltoall_perf_gin_gdaki_${impl}" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE} NCCL_GIN_TYPE=3" "$WRAPPER $NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
done

# ============================================================================
# AlltoAll with hybrid (LSA/GIN) kernels
# ============================================================================
for impl in 4; do
  run_command "xclique_alltoall_perf_hybrid_${impl}" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
done

# ============================================================================
# Symmetric memory tests for cross-clique
# ============================================================================
for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
  run_command "xclique_${func}_symm" $RUN_MODE $NGPUS "" "CLIQUE_SIZE=${CLIQUE_SIZE}" "$WRAPPER $NCCL_HOME/test/perf/$func" "$range $opts -R 2"
done

print_failed_commands
end_junit_file
ci_exit
