#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

opts="-w 1 -G $GRAPH -s 512M"
range="-b 8 -e $MAX -f 2"
enable_ft="-B 0 -F 1"
enable_split_test="-S 1 -P 1"
split_range="-b 8 -e 1G -f 2"
enable_local_register="-R 1"
enable_graph_register="-G 1"
enable_parallel_init="-p 1"

# Args for run_command
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
  run_command "${func}_all_sizes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts"
done

if [ "$NGPUS" -ge "3" ];
then
  for func in all_reduce_perf alltoall_perf; do
    run_command "${func}_multi_threaded_3gpu_per_node_all_sizes" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -t 3 -g 1 -n 1"
    if [ "$NGPUS" -ge "6" ];
    then
      run_command "${func}_multi_threaded_6gpu_per_node_all_sizes" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -t 6 -g 1 -n 1"
    fi
  done
fi

if [ "$SKIP_MULTI_GPU" != "1" ];
then
  let np=$NNODES
  for func in all_reduce_perf alltoall_perf; do
    run_command "${func}_all_gpu_all_sizes" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -t 1 -g $NGPUS -n 1"
    let nthreads=$NGPUS/2
    if [ $nthreads -gt 0 ];
    then
      run_command "${func}_2_gpu_parallel_init_all_sizes" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -t $nthreads -g2 $enable_parallel_init -n 1"
    fi
    let nthreads=$NGPUS/4
    if [ $nthreads -gt 0 ];
    then
      run_command "${func}_4_gpu_all_sizes" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -t $nthreads -g4 -n 1"
    fi
  done
fi

rangetype="-b 16M -e 16M -o all -d all -n 5"
for func in all_reduce_perf reduce_perf reduce_scatter_perf; do
  run_command "${func}_all_ops_dtypes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$rangetype $opts"
done

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf hypercube_perf; do
  run_command "${func}_split_share_all_sizes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$split_range $opts $enable_split_test -n 1"
done

if [ "$ENABLE_NVLS" == "1" ]; then
  for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
    run_command "${func}_nvls_local_registration_all_sizes" $RUN_MODE $NGPUS "" "NCCL_ALGO=NVLS" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register -n 1"
  done
fi

run_command "all_reduce_tree_local_registration_all_sizes" $RUN_MODE $NGPUS "" "NCCL_ALGO=Tree" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts $enable_local_register -n 1"

for func in all_reduce_perf all_gather_perf broadcast_perf; do
  run_command "${func}_ring_local_registration_all_sizes" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register -n 1"
done

for func in all_reduce_perf all_gather_perf broadcast_perf; do
  run_command "${func}_ring_graph_registration_all_types" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring" "$NCCL_HOME/test/perf/$func" "-b 1G -e 1G -n 5 -w 5 -d all $enable_graph_register"
done


if [ "$NO_LOOPBACK_NETWORKING" != "1" ]
then
    for func in all_reduce_perf all_gather_perf broadcast_perf; do
    run_command "${func}_ring_1rpn_graph_registration_all_types" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/perf/$func" "-b $MAX -e $MAX -n 5 -w 5 -d all $enable_graph_register"
    run_command "${func}_ring_1rpn_local_registration_all_types" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register"
    done
else
  echo "Skipping Ring 1RPN registration tests..."
fi

for func in sendrecv_perf alltoall_perf; do
  run_command "${func}_local_registration_all_sizes" $RUN_MODE $NGPUS "" "NCCL_PXN_DISABLE=1" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register"
done

NCCL_DEBUG_OLD=$NCCL_DEBUG
export NCCL_DEBUG=VERSION
run_command "all_reduce_output_file" $RUN_MODE $NGPUS "" "NCCL_PXN_DISABLE=1" "$NCCL_HOME/test/perf/all_reduce_perf" "-b8 -e8 -w0 -n1 -J test_out.json"

enable_ft="$enable_ft -L allreduce,alltoall,abort,split"
if [ "$SKIP_FT_INIT" != "1" ]
then
  enable_ft+=",init"
  # Mark as 0 for clarity of test label
  SKIP_FT_INIT=0
else
  echo "WARNING : Skipping init FT test"
fi

if [ "$SKIP_FT_FINALIZE" != "1" ]
then
  enable_ft+=",finalize"
  # Mark as 0 for clarity of test label
  SKIP_FT_FINALIZE=0
else
  echo "WARNING : Skipping finalize FT test"
fi

run_command "ft_test_skip_init_${SKIP_FT_INIT}_skip_finalize_${SKIP_FT_FINALIZE}" $RUN_MODE $NGPUS "" "NCCL_SOCKET_RETRY_SLEEP_MSEC=1" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts $enable_ft"
# https://nvbugspro.nvidia.com/bug/4934665
# run_command "ft_test_1ppn_skip_ft_init_$SKIP_FT_INIT" $RUN_MODE 1 "-x NCCL_SOCKET_RETRY_SLEEP_MSEC=1 " "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -t $NGPUS $enable_ft"

export NCCL_DEBUG=$NCCL_DEBUG_OLD

for func in all_reduce_perf alltoall_perf; do
  run_command "${func}_nic_fusion_phb"      $RUN_MODE $NGPUS "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -n 1"
  run_command "${func}_nic_fusion_phb_1ppn" $RUN_MODE 1      "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -t $NGPUS -n 1"
done

print_failed_commands
end_junit_file
ci_exit
