#!/bin/bash
#
# The tests in this script are specific to the NCCL profiler.
# They make sure the profiler works without any error/crashes for all communication APIs
#
set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

opts="-w 1 -G \${graph}"
range="-b 8 -e 128M -f 2"

# Args for run_command
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"

# Build the example profiler plugin
make -C $NCCL_HOME/../ext-profiler/example

# Intra-node tests: enable all the events in NCCL and dump events to a trace file (one per rank)
# Alltoall exercises the group path in the kernel profiler
# Graph capturing exercises the graph path in the kernel profiler
echo "Profiler Perf TESTs Default"
for graph in 0 2 ; do
  local_opts=$(eval "echo ${opts}")
  for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
    run_command "${func}_all_sizes" $RUN_MODE $NGPUS "" "NCCL_PROFILER_PLUGIN=$NCCL_HOME/../ext-profiler/example/libnccl-profiler.so NCCL_PROFILE_EVENT_MASK=255 NCCL_PROFILE_DUMP_FILE=${func} NCCL_PROFILE_GROUP_POOL_SIZE=300 NCCL_PROFILE_COLL_POOL_SIZE=300" "$NCCL_HOME/test/perf/$func" "$range $local_opts"
  done
done

# Inter-node tests: compared with intra-node tests, these tests also exercise the events in the network proxy.
echo "Profiler Perf TESTs Network"
for graph in 0 2 ; do
  local_opts=$(eval "echo ${opts}")
  for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
    NCCL_PROFILE_DUMP_FILE=${func}
    run_command "${func}_all_sizes" $RUN_MODE $NGPUS "" "NCCL_PROFILER_PLUGIN=$NCCL_HOME/../ext-profiler/example/libnccl-profiler.so NCCL_PROFILE_EVENT_MASK=255 NCCL_PROFILE_DUMP_FILE=${func} NCCL_PROFILE_GROUP_POOL_SIZE=300 NCCL_PROFILE_COLL_POOL_SIZE=300 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1" "$NCCL_HOME/test/perf/$func" "$range $local_opts"
  done
done

print_failed_commands
end_junit_file
ci_exit
