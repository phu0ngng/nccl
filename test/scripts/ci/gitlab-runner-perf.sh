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

if [ "$CE_COLL" == "1" ];
then
  for func in all_gather_perf alltoall_perf scatter_perf gather_perf; do
    run_command "${func}_ce_nvls_enable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
    run_command "${func}_ce_nvls_disable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=0" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
    run_command "${func}_ce_graph_nvls_enable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 2 -x 2"
    run_command "${func}_ce_graph_nvls_disable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=0" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 2 -x 2"
    if [ "$NNODES" == "1" ];
    then
      run_command "${func}_ce_single_proc_nvls_enable" $RUN_MODE 1 "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-t 1 -g $NGPUS -b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
      run_command "${func}_ce_single_proc_nvls_disable" $RUN_MODE 1 "" "NCCL_NVLS_ENABLE=0" "$NCCL_HOME/test/perf/$func" "-t 1 -g $NGPUS -b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
    fi
  done
fi

if [ "$SYMMETRIC" == "1" ];
then
  for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
    run_command "${func}_symm_memory_min_size" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-w 1 -n 1 -b 1K -e 1K -d all -R 2"
    run_command "${func}_symm_memory_max_size" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-w 1 -n 1 -b 16G -e 16G -d all -R 2"
    run_command "${func}_symm_memory_sweep" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -R 2"
    run_command "${func}_symm_memory_min_size_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-G 1 -w 1 -n 1 -b 1K -e 1K -d all -R 2"
    run_command "${func}_symm_memory_max_size_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-G 1 -w 1 -n 1 -b 16G -e 16G -d all -R 2"
    run_command "${func}_symm_memory_sweep_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -G 1 -R 2"
    run_command "${func}_group_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -R 2 -n 5 -m 10"
    run_command "${func}_large_group_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -R 2 -n 5 -m 100"
  done
fi

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

enable_ft="$enable_ft -L allreduce,alltoall,split,shrink"
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

if [ "$SKIP_FT_ABORT" != "1" ]
then
  enable_ft+=",abort"
  # Mark as 0 for clarity of test label
  SKIP_FT_ABORT=0
else
  echo "WARNING : Skipping abort FT test"
fi

run_command "ft_test_skip_init_${SKIP_FT_INIT}_skip_finalize_${SKIP_FT_FINALIZE}" $RUN_MODE $NGPUS "" "NCCL_SOCKET_RETRY_SLEEP_MSEC=1" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts $enable_ft"
# https://nvbugspro.nvidia.com/bug/4934665
# run_command "ft_test_1ppn_skip_ft_init_$SKIP_FT_INIT" $RUN_MODE 1 "-x NCCL_SOCKET_RETRY_SLEEP_MSEC=1 " "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -t $NGPUS $enable_ft"

export NCCL_DEBUG=$NCCL_DEBUG_OLD

for func in all_reduce_perf alltoall_perf; do
  run_command "${func}_nic_fusion_phb"      $RUN_MODE $NGPUS "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -n 1"
  run_command "${func}_nic_fusion_phb_1ppn" $RUN_MODE 1      "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -t $NGPUS -n 1"
done

# socket NET testing
for func in all_reduce_perf all_gather_perf broadcast_perf; do
  run_command "${func}_socket_net" $RUN_MODE $NGPUS "" "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket" "$NCCL_HOME/test/perf/$func" "-b 8 -e 16M -f2 $opts -n 1"
done

# Test tuner plugin with MNNVL data
if [ "$ENABLE_MNNVL_TUNER_PLUGIN" == "1" ]; then
  for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
    run_command "tuner_plugin_mnnvl_test_${func}" $RUN_MODE $NGPUS "" "NCCL_TUNER_PLUGIN=$NCCL_HOME/test/unit/plugins/libnccl-tuner-example.so NCCL_DEBUG=INFO" "$NCCL_HOME/test/perf/${func}" "-b 8 -e 128M -f2 $opts -n 5"
  done
fi

print_failed_commands
end_junit_file
ci_exit
