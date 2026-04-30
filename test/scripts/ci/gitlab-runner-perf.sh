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
small_msg_range="-b 8 -e 16K -f 2"
enable_ft="-B 0 -F 1"
enable_split_test="-S 1 -P 1"
split_range="-b 8 -e 1G -f 2"
enable_local_register="-R 1"
enable_graph_register="-G 1"
enable_parallel_init="-p 1"
if [ -z "$MAX_GROUP" ]; then
  range_group=$range
else
  range_group="-b 8 -e $MAX_GROUP -f 2"
fi


# Args for run_command
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"

# When running cross-clique-only jobs, skip all standard perf tests
if [ "$CROSS_CLIQUE_NATIVE" == "1" ] || [ "$CROSS_CLIQUE_EMULATED" == "1" ]; then
  SKIP_STANDARD_PERF=1
fi

if [ "$SKIP_STANDARD_PERF" != "1" ]; then

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf all_gatherv_perf; do
  run_command "${func}_all_sizes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts"
done

if [ "$RUN_DDP_TEST" == "1" ];
then
  for func in all_reduce_perf alltoall_perf all_gather_perf; do
    run_command "${func}_all_sizes_ddp_force_enable" $RUN_MODE $NGPUS "" "NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_OOO_RQ=1" "$NCCL_HOME/test/perf/$func" "$range $opts"
    run_command "${func}_all_sizes_ddp_force_enable_no_p2p" $RUN_MODE $NGPUS "" "NCCL_P2P_DISABLE=1 NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_OOO_RQ=1" "$NCCL_HOME/test/perf/$func" "$range $opts"
    run_command "${func}_all_sizes_ddp_force_enable_no_mnnvl" $RUN_MODE $NGPUS "" "NCCL_MNNVL_ENABLE=0 NCCL_IB_ADAPTIVE_ROUTING=1 NCCL_IB_OOO_RQ=1" "$NCCL_HOME/test/perf/$func" "$range $opts"
  done
fi

if [ "$RMA" == "1" ];
then
  for func in all_gather_perf alltoall_perf broadcast_perf gather_perf scatter_perf sendrecv_perf; do
    if [ "$NNODES" == "1" ];
    then
      run_command "${func}_single_rma" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 0 -R 2 -H"
      run_command "${func}_single_rma_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 1 -R 2 -H"
    fi
    if [ "$NNODES" -gt "1" ];
    then
      run_command "${func}_multi_rma" $RUN_MODE 1 "" "NCCL_NET=IB" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 0 -R 2 -H"
      run_command "${func}_multi_rma_graph" $RUN_MODE 1 "" "NCCL_NET=IB" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 1 -R 2 -H"
      if [ "$NGPUS" -gt "1" ];
      then
        run_command "${func}_multi_rma" $RUN_MODE $NGPUS "" "NCCL_NET=IB" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 0 -R 2 -H"
        run_command "${func}_multi_rma_graph" $RUN_MODE $NGPUS "" "NCCL_NET=IB" "$NCCL_HOME/test/perf/$func" "-b 128 -e 1G -f 2 -G 1 -R 2 -H"
      fi
    fi
  done
fi


if [ "$CE_COLL" == "1" ];
then
  for func in all_gather_perf alltoall_perf scatter_perf gather_perf; do
    run_command "${func}_ce_nvls_enable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
    run_command "${func}_ce_nvls_disable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=0" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2"
    run_command "${func}_ce_graph_nvls_enable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 2 -x 2"
    run_command "${func}_ce_graph_nvls_disable" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=0" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 2 -x 2"
    run_command "${func}_ce_send_reg" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 3 -x 2"
    run_command "${func}_ce_recv_reg" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 1 -R 4 -x 2"
    run_command "${func}_ce_legacy_stream" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 1G -e 1G -f 2 -G 0 -R 2 -x 2 -y 1"
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
    run_command "${func}_symm_memory_max_size" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-w 1 -n 1 -b $MAX -e $MAX -d all -R 2"
    run_command "${func}_symm_memory_sweep" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -R 2"
    run_command "${func}_symm_memory_min_size_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-G 1 -w 1 -n 1 -b 1K -e 1K -d all -R 2"
    run_command "${func}_symm_memory_max_size_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-G 1 -w 1 -n 1 -b $MAX -e $MAX -d all -R 2"
    run_command "${func}_symm_memory_sweep_graph" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -G 1 -R 2"
    run_command "${func}_group_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range_group $opts -R 2 -n 1 -m 10"
    run_command "${func}_large_group_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range_group $opts -R 2 -n 1 -m 100"
    run_command "${func}_ll_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$small_msg_range $opts -n 1"
    run_command "${func}_send_reg_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -n 1 -R 3"
    run_command "${func}_recv_reg_symm_kernel" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts -n 1 -R 4"
  done

  if [ "$TMA" == "1" ];
  then
    for func in all_reduce_perf reduce_scatter_perf; do
      run_command "${func}_tma_symm_memory_sweep" $RUN_MODE $NGPUS "" "NCCL_NVLS_ENABLE=0 NCCL_SYM_TMA_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 32K -e $MAX -f 2 -R 2"
    done

    func=all_gather_perf
    run_command "${func}_tma_symm_memory_sweep" $RUN_MODE $NGPUS "" "NCCL_SYM_CE_THRESHOLD=$((1024*1024*1024)) NCCL_NVLS_ENABLE=0 NCCL_SYM_TMA_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 32K -e $MAX -f 2 -R 2"
    run_command "${func}_tma_symm_nvls_memory_sweep" $RUN_MODE $NGPUS "" "NCCL_SYM_CE_THRESHOLD=$((1024*1024*1024)) NCCL_NVLS_ENABLE=1 NCCL_SYM_TMA_ENABLE=1" "$NCCL_HOME/test/perf/$func" "-b 32K -e $MAX -f 2 -R 2"
    unset
  fi
fi

if [ "$DEVICE_API" != "0" ]; then
  if [ "$NNODES" == "1" ]; then
    # lsa tests
    for impl in 1 2; do
      run_command "all_reduce_perf_device_lsa_${impl}" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -R 2 -D $impl"
      if [ "$NGPUS" -ge "3" ]; then
        run_command "all_reduce_perf_device_lsa_${impl}_multithread" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -t $NGPUS -g 1 -R 2 -D $impl"
        if [ "${GRAPH:-0}" == "0" ]; then
          run_command "all_reduce_perf_device_lsa_${impl}_multigpu" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -t 1 -g $NGPUS -R 2 -D $impl"
        fi
      fi
    done
    for impl in 1 2; do
      run_command "alltoall_perf_device_lsa_${impl}" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
    done
    # multimem tests
    for impl in 3 4; do
      run_command "all_reduce_perf_device_multimem_${impl}" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -R 2 -D $impl"
    done
  fi
  # gin tests
  for impl in 3; do
    # try proxy and gdaki
    run_command "alltoall_perf_device_gin_proxy_${impl}" $RUN_MODE $NGPUS "" "NCCL_GIN_TYPE=2" "$NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
    run_command "alltoall_perf_device_gin_gdaki_${impl}" $RUN_MODE $NGPUS "" "NCCL_GIN_TYPE=3" "$NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
    if [[ "${GIN_TESTS_GPI}" -eq 1 ]] ; then
      run_command "alltoall_perf_device_gin_gpi_${impl}" $RUN_MODE $NGPUS "" "NCCL_GIN_TYPE=4 NCCL_NET_PLUGIN=$SPCX_PLUGIN" "$NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
    fi
  done
  # hybrid (lsa/gin) tests
  for impl in 4; do
    run_command "alltoall_perf_device_hybrid_${impl}" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/alltoall_perf" "$range $opts -R 2 -D $impl"
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

if [ "$SKIP_MULTI_GPU" != "1" ] && [ "${GRAPH:-0}" == "0" ];
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

if [ "$NNODES" == "1" ] && [ "$NGPUS" == 1 ]; then
  rangetype="-b 2164744 -e $((8 * 2164744)) -o all -n 5 -f 2"
  run_command "all_reduce_perf_one_rank_non_blocknum_mult_all_opts_dtypes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/all_reduce_perf" "$rangetype $opts"
fi

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf hypercube_perf all_gatherv_perf; do
  run_command "${func}_split_share_all_sizes" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$split_range $opts $enable_split_test -n 1"
done

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
  run_command "${func}_dyn_mem_test" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "-b 1G -e 1G -w 5 -n 1 -Z 1"
  if [ "$NNODES" == "1" ]; then
    run_command "${func}_dyn_mem_test_single_proc" $RUN_MODE 1 "" "" "$NCCL_HOME/test/perf/$func" "-b 1G -e 1G -w 5 -n 1 -Z 1 -t 1 -g $NGPUS"
  fi
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

enable_ft="$enable_ft -L allreduce,alltoall,split,shrink,revoke,revoke_shrink,revoke_split"
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

if [ "$SKIP_FT_TEST" != "1" ]
then
  # Mark as 0 for clarity of test label
  SKIP_FT_TEST=0
  run_command "ft_test_skip_init_${SKIP_FT_INIT}_skip_finalize_${SKIP_FT_FINALIZE}" $RUN_MODE $NGPUS "" "NCCL_SOCKET_RETRY_SLEEP_MSEC=1" "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts $enable_ft"
  # https://nvbugspro.nvidia.com/bug/4934665
  # run_command "ft_test_1ppn_skip_ft_init_$SKIP_FT_INIT" $RUN_MODE 1 "-x NCCL_SOCKET_RETRY_SLEEP_MSEC=1 " "$NCCL_HOME/test/perf/all_reduce_perf" "$range $opts -t $NGPUS $enable_ft"
else
  echo "WARNING : Skipping all FT test"
fi


export NCCL_DEBUG=$NCCL_DEBUG_OLD


if [ "$SKIP_NIC_FUSION" != "1" ]
then
  # Mark as 0 for clarity of test label
  SKIP_NIC_FUSION=0
  for func in all_reduce_perf alltoall_perf; do
    run_command "${func}_nic_fusion_phb"      $RUN_MODE $NGPUS "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -n 1"
    run_command "${func}_nic_fusion_phb_1ppn" $RUN_MODE 1      "" "NCCL_NET_MERGE_LEVEL=PHB" "$NCCL_HOME/test/perf/$func" "-b 8 -e 128M -f2 $opts -t $NGPUS -n 1"
  done
else
  echo "WARNING : Skipping NIC fusion test"
fi


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

# tests w/o allgatherv enabled
export NCCL_ALLGATHERV_ENABLE=0
for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf all_gatherv_perf; do
  run_command "${func}_all_sizes_allgatherv" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/$func" "$range $opts"
done

for func in all_gatherv_perf broadcast_perf; do
  run_command "${func}_ring_local_registration_all_sizes_allgatherv" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register -n 1"
done

for func in all_gatherv_perf broadcast_perf; do
  run_command "${func}_ring_graph_registration_all_types_allgatherv" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring" "$NCCL_HOME/test/perf/$func" "-b 1G -e 1G -n 5 -w 5 -d all $enable_graph_register"
done

if [ "$NO_LOOPBACK_NETWORKING" != "1" ]
then
    for func in all_gatherv_perf broadcast_perf; do
    run_command "${func}_ring_1rpn_graph_registration_all_types_allgatherv" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/perf/$func" "-b $MAX -e $MAX -n 5 -w 5 -d all $enable_graph_register"
    run_command "${func}_ring_1rpn_local_registration_all_types_allgatherv" $RUN_MODE $NGPUS "" "NCCL_ALGO=Ring NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/perf/$func" "$range $opts $enable_local_register"
    done
else
  echo "Skipping Ring 1RPN registration tests for allgatherv..."
fi

# allgatherv socket NET testing
for func in all_gatherv_perf broadcast_perf; do
  run_command "${func}_socket_net_allgatherv" $RUN_MODE $NGPUS "" "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_NET=Socket" "$NCCL_HOME/test/perf/$func" "-b 8 -e 16M -f2 $opts -n 1"
done

# allgatherv Test tuner plugin with MNNVL data
if [ "$ENABLE_MNNVL_TUNER_PLUGIN" == "1" ]; then
  for func in all_gatherv_perf broadcast_perf; do
    run_command "tuner_plugin_mnnvl_test_${func}_allgatherv" $RUN_MODE $NGPUS "" "NCCL_TUNER_PLUGIN=$NCCL_HOME/test/unit/plugins/libnccl-tuner-example.so NCCL_DEBUG=INFO" "$NCCL_HOME/test/perf/${func}" "-b 8 -e 128M -f2 $opts -n 5"
  done
fi
unset NCCL_ALLGATHERV_ENABLE

if [ "$NNODES" == "1" ]; then
  # alltoallv_perf smoke tests (including generated pattern and matrix file mode)
  run_command "alltoallv_perf_generated_uniform" $RUN_MODE $NGPUS "" "ALLTOALLV_SPREAD=0.0" "$NCCL_HOME/test/perf/alltoallv_perf" "-w 1 -n 1 -b 1M -e 64M -f 2"
  run_command "alltoallv_perf_generated_weighted" $RUN_MODE $NGPUS "" "ALLTOALLV_SPREAD=1.0" "$NCCL_HOME/test/perf/alltoallv_perf" "-w 1 -n 1 -b 1M -e 64M -f 2"

  a2av_matrix_file="$(mktemp "$(pwd)/alltoallv_ci_traffic_matrix.XXXXXX")"
  cat >"$a2av_matrix_file" <<'EOF'
    0 8388608 0 0 8388608 0 0 0
    0 0 8388608 0 0 8388608 0 0
    0 0 0 8388608 0 0 8388608 0
    0 0 0 0 8388608 0 0 8388608
    8388608 0 0 0 0 8388608 0 0
    0 8388608 0 0 0 0 8388608 0
    0 0 8388608 0 0 0 0 8388608
    8388608 0 0 8388608 0 0 0 0
EOF
  run_command "alltoallv_perf_matrix_file" $RUN_MODE $NGPUS "" "ALLTOALLV_MATRIX_FILE=$a2av_matrix_file" "$NCCL_HOME/test/perf/alltoallv_perf" "-w 1 -n 1 -b 16M -e 16M -f 2"
  rm -f "$a2av_matrix_file"
fi

if [ "$SKIP_COMM_MGT_TESTS" != "1" ]; then
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"
  run_command "comm_ops_perf_init"       $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "init"
  run_command "comm_ops_perf_init_abort" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "init --abort"

  run_command "comm_ops_perf_split"       $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "split"
  run_command "comm_ops_perf_split_share" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "split --share"
  run_command "comm_ops_perf_split_abort" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "split --abort"

  run_command "comm_ops_perf_shrink"       $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "shrink"
  run_command "comm_ops_perf_shrink_share" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "shrink --share"
  run_command "comm_ops_perf_shrink_abort" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "shrink --abort"

  if [ "${WORKAROUND_NVBUG_5793707}" == "1" ]; then
      export NCCL_NET_MERGE_LEVEL=LOC
  fi
  if [ "${SKIP_COMM_OPS_GROW_NVBUG_6029839}" != "1" ]; then
      run_command "comm_ops_perf_grow"       $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "grow"
      run_command "comm_ops_perf_grow_abort" $RUN_MODE $NGPUS "" "" "$NCCL_HOME/test/perf/comm_ops_perf" "grow --abort"
  fi
  if [ "${WORKAROUND_NVBUG_5793707}" == "1" ]; then
    unset NCCL_NET_MERGE_LEVEL
  fi
fi

# Multi-rank GPU correctness tests using r/3 (3 ranks per GPU). We use 3
# rather than 2 to produce non-power-of-2 rank counts, which exercise
# alignment edge cases. The _mt, _mg, and _mt_mg variants each produce
# 2*NGPUS total ranks per node, using ceil(2*NGPUS/3) physical GPUs with
# uneven fill on the last GPU. The _mt_mg variant has threads sharing
# physical GPUs across thread boundaries.
if [ "$SKIP_MULTI_RANK_GPU" != "1" ]; then
  let ppn_3rpg=$NGPUS*3
  multi_rank_range="-b 8 -e 16M -f 2"
  multi_rank_opts="-w 1 -n 1 -G 0"
  multi_rank_env="NCCL_MULTI_RANK_GPU_ENABLE=1 NCCL_TESTS_DEVICE=r/3"

  for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf all_gatherv_perf; do
    # 3*NGPUS processes, 1 rank each, 3 ranks per GPU across all GPUs
    # P0=0. P1=0, P2=0, P3=1, P4=1, P5=1, P6=2, ...
    run_command "${func}_multi_rank_gpu_3rpg" $RUN_MODE $ppn_3rpg "--oversubscribe" "$multi_rank_env" "$NCCL_HOME/test/perf/$func" "$multi_rank_range $multi_rank_opts"
    # NGPUS processes, 2 threads each, 1 GPU per thread
    # P0t0=0, P0t1=0, P1t0=0, P1t1=1, P2t0=1, P2t1=1, P3t0=2, P3t1=2, ...
    run_command "${func}_multi_rank_gpu_mt" $RUN_MODE $NGPUS "--oversubscribe" "$multi_rank_env" "$NCCL_HOME/test/perf/$func" "$multi_rank_range $multi_rank_opts -t 2 -g 1"
    # NGPUS processes, 1 thread each, 2 GPUs per thread
    # P0=(0,0), P1=(0,1), P2=(1,1), P3=(2,2), P4=(2,3), ...
    run_command "${func}_multi_rank_gpu_mg" $RUN_MODE $NGPUS "--oversubscribe" "$multi_rank_env" "$NCCL_HOME/test/perf/$func" "$multi_rank_range $multi_rank_opts -t 1 -g 2"
    # NGPUS/2 processes, 2 threads each, 2 GPUs per thread
    # P0t0=(0,0), P0t1=(0,1), P1t0=(1,1), P1t1=(2,2), ...
    run_command "${func}_multi_rank_gpu_mt_mg" $RUN_MODE $((NGPUS/2)) "--oversubscribe" "$multi_rank_env" "$NCCL_HOME/test/perf/$func" "$multi_rank_range $multi_rank_opts -t 2 -g 2"
  done
fi

fi # SKIP_STANDARD_PERF

if [ "${RUN_GDRCOPY_TESTS:-0}" == "1" ]; then
  gdrcopy_funcs="all_reduce_perf alltoall_perf sendrecv_perf"
  xc_pxn_wrapper="test/scripts/ci/cross-clique-wrapper.sh"

  export NCCL_GDRCOPY_ENABLE=1

  # FIFO-only
  export NCCL_GDRCOPY_FIFO_ENABLE=1 NCCL_GDRCOPY_SYNC_ENABLE=0 NCCL_GDRCOPY_FLUSH_ENABLE=0
  for func in $gdrcopy_funcs; do
    run_command "${func}_gdrcopy_fifo" $RUN_MODE $NGPUS "" "" \
      "$NCCL_HOME/test/perf/$func" "$range $opts -t 1 -g 1"
  done

  # SYNC-only
  export NCCL_GDRCOPY_FIFO_ENABLE=0 NCCL_GDRCOPY_SYNC_ENABLE=1 NCCL_GDRCOPY_FLUSH_ENABLE=0
  for func in $gdrcopy_funcs; do
    run_command "${func}_gdrcopy_sync" $RUN_MODE $NGPUS "" "" \
      "$NCCL_HOME/test/perf/$func" "$range $opts -t 1 -g 1"
  done

  # FLUSH-only
  export NCCL_GDRCOPY_FIFO_ENABLE=0 NCCL_GDRCOPY_SYNC_ENABLE=0 NCCL_GDRCOPY_FLUSH_ENABLE=1
  for func in $gdrcopy_funcs; do
    run_command "${func}_gdrcopy_flush" $RUN_MODE $NGPUS "" "" \
      "$NCCL_HOME/test/perf/$func" "$range $opts -t 1 -g 1"
  done

  # PXN: single NIC forces cross-GPU proxying and exercises the GDRCopy
  # sameDevice guard. cross-clique-wrapper.sh (CLIQUE_SIZE=1) forces
  # inter-node traffic onto IB on MNNVL clusters; no-op elsewhere.
  export NCCL_GDRCOPY_FIFO_ENABLE=1 NCCL_GDRCOPY_SYNC_ENABLE=1 NCCL_GDRCOPY_FLUSH_ENABLE=0
  if [ "${GRAPH:-0}" == "0" ]; then
    for func in $gdrcopy_funcs; do
      run_command "${func}_gdrcopy_pxn" $RUN_MODE 1 "" "CLIQUE_SIZE=1 NCCL_IB_HCA=mlx5_0" \
        "$xc_pxn_wrapper $NCCL_HOME/test/perf/$func" "$range $opts -t 1 -g $NGPUS"
    done
  fi
fi

# Native cross-clique P2P (real rack topology, NCCL_MNNVL_CLIQUE_ID=-2)
if [ "$CROSS_CLIQUE_NATIVE" == "1" ]; then
  xc_env="NCCL_MNNVL_CLIQUE_ID=-2 NCCL_MNNVL_CROSS_CLIQUE=1 NCCL_NET_PLUGIN=none"
  xc_opts="-w 1 -G 0 -s 512M -M 1"
  xc_range="-b 8 -e ${MAX:-1G} -f 2"

  if [ "${CROSS_CLIQUE_QUICK:-0}" == "1" ]; then
    xc_range="-b 8 -e 1G -f 2"
    run_command "nvl576_all_reduce_perf" $RUN_MODE $NGPUS "" "$xc_env" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"
    run_command "nvl576_all_reduce_nvls_off" $RUN_MODE $NGPUS "" "$xc_env NCCL_NVLS_ENABLE=0" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"
    run_command "nvl576_all_gather_perf" $RUN_MODE $NGPUS "" "$xc_env" \
      "$NCCL_HOME/test/perf/all_gather_perf" "$xc_range $xc_opts -R 1 -g 1"
    run_command "nvl576_alltoall_ce" $RUN_MODE $NGPUS "" "$xc_env" \
      "$NCCL_HOME/test/perf/alltoall_perf" "-b 128 -e 2G -f 2 -G 0 -R 2 -x 2 -w 1 -M 1 -g 1"
    run_command "nvl576_all_reduce_tree" $RUN_MODE $NGPUS "" "$xc_env NCCL_ALGO=Tree" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"
  else
    for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
      run_command "nvl576_${func}" $RUN_MODE $NGPUS "" "$xc_env" \
        "$NCCL_HOME/test/perf/$func" "$xc_range $xc_opts -R 1 -g 1"
    done

    run_command "nvl576_alltoall_perf" $RUN_MODE $NGPUS "" "$xc_env" \
      "$NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 1 -g 1"

    run_command "nvl576_all_reduce_nvls_off" $RUN_MODE $NGPUS "" "$xc_env NCCL_NVLS_ENABLE=0" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"

    run_command "nvl576_all_reduce_tree" $RUN_MODE $NGPUS "" "$xc_env NCCL_ALGO=Tree" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"
    run_command "nvl576_all_reduce_tree_nvls_off" $RUN_MODE $NGPUS "" \
      "$xc_env NCCL_ALGO=Tree NCCL_NVLS_ENABLE=0" \
      "$NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1 -g 1"

    xc_ce_max="${CROSS_CLIQUE_CE_MAX:-8G}"
    for func in all_gather_perf alltoall_perf scatter_perf gather_perf; do
      run_command "nvl576_${func}_ce" $RUN_MODE $NGPUS "" "$xc_env" \
        "$NCCL_HOME/test/perf/$func" "-b 128 -e $xc_ce_max -f 2 -G 0 -R 2 -x 2 -w 1 -M 1 -g 1"
    done

    for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
      run_command "nvl576_${func}_symm" $RUN_MODE $NGPUS "" "$xc_env" \
        "$NCCL_HOME/test/perf/$func" "$xc_range $xc_opts -R 2 -g 1"
    done
  fi
fi

# Emulated cross-clique P2P (wrapper-based, optional multi-NVLD via NVLDS env var)
if [ "$CROSS_CLIQUE_EMULATED" == "1" ]; then
  xc_wrapper="test/scripts/ci/cross-clique-wrapper.sh"
  xc_env="CLIQUE_SIZE=${CLIQUE_SIZE:-1}"
  xc_opts="-w 1 -G 0 -s 512M -M 1"
  xc_range="-b 8 -e $MAX -f 2"

  for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
    run_command "xclique_${func}" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/$func" "$xc_range $xc_opts -R 1"
  done

  run_command "xclique_alltoall_perf" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 1"

  run_command "xclique_all_reduce_perf_nvls_off" $RUN_MODE $NGPUS "" "$xc_env NCCL_NVLS_ENABLE=0" "$xc_wrapper $NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1"

  run_command "xclique_all_reduce_perf_tree" $RUN_MODE $NGPUS "" "$xc_env NCCL_ALGO=Tree" "$xc_wrapper $NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1"
  run_command "xclique_all_reduce_perf_tree_nvls_off" $RUN_MODE $NGPUS "" "$xc_env NCCL_ALGO=Tree NCCL_NVLS_ENABLE=0" "$xc_wrapper $NCCL_HOME/test/perf/all_reduce_perf" "$xc_range $xc_opts -R 1"

  for func in all_gather_perf alltoall_perf scatter_perf gather_perf; do
    run_command "xclique_${func}_ce" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/$func" "-b 128 -e 8G -f 2 -G 0 -R 2 -x 2 -w 1 -M 1"
  done

  if [ "$DEVICE_API" != "0" ]; then
    for impl in 1 2; do
      run_command "xclique_alltoall_perf_lsa_${impl}" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 2 -D $impl"
    done
    for impl in 3; do
      run_command "xclique_alltoall_perf_gin_proxy_${impl}" $RUN_MODE $NGPUS "" "$xc_env NCCL_GIN_TYPE=2" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 2 -D $impl"
      run_command "xclique_alltoall_perf_gin_gdaki_${impl}" $RUN_MODE $NGPUS "" "$xc_env NCCL_GIN_TYPE=3" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 2 -D $impl"
      if [[ "${GIN_TESTS_GPI}" -eq 1 ]] ; then
        run_command "xclique_alltoall_perf_gin_gpi_${impl}" $RUN_MODE $NGPUS "" "NCCL_GIN_TYPE=4 NCCL_NET_PLUGIN=$SPCX_PLUGIN" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 2 -D $impl"
      fi
    done
    for impl in 4; do
      run_command "xclique_alltoall_perf_hybrid_${impl}" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/alltoall_perf" "$xc_range $xc_opts -R 2 -D $impl"
    done
  fi

  for func in all_reduce_perf reduce_scatter_perf all_gather_perf; do
    run_command "xclique_${func}_symm" $RUN_MODE $NGPUS "" "$xc_env" "$xc_wrapper $NCCL_HOME/test/perf/$func" "$xc_range $xc_opts -R 2"
  done
fi

print_failed_commands
end_junit_file
ci_exit
