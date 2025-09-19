#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

# Args for run_command
# run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"
export LD_LIBRARY_PATH_BACKUP=$LD_LIBRARY_PATH

function run_gin_test_suite() {
  local backend_label="$1"
  run_command "gin_test_${backend_label}_put_signal_ping_pong_gin" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/put_signal_ping_pong_gin" "-v"
  run_command "gin_test_${backend_label}_put_signal_ping_pong_gin_4GiB" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/put_signal_ping_pong_gin" "-v -b $((4 * 1024 * 1024 * 1024)) -e $((4 * 1024 * 1024 * 1024)) -w 1 -i 1"
  run_command "gin_test_${backend_label}_put_gin_alltoall" "$RUN_MODE" ${NP} "--oversubscribe" "" "$NCCL_HOME/test/unit/put_gin_alltoall" ""
  run_command "gin_test_${backend_label}_devapi_barrier" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/devapi_barrier" ""
  run_command "gin_test_${backend_label}_devapi_data_ring" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/devapi_data_ring" ""
  run_command "gin_test_${backend_label}_devapi_data_ring2" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/devapi_data_ring2" ""
  run_command "gin_test_${backend_label}_devapi_signal_ring" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/devapi_signal_ring" ""
  run_command "gin_test_${backend_label}_devapi_uts" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/devapi_uts" ""
}

if [[ ${ENQUEUE_TESTS_ARGS} -eq 1 ]] ; then
  run_command "enqueue_tests_args" "$RUN_MODE" 1 "--oversubscribe" "NCCL_WORK_FIFO_BYTES=0 NCCL_WORK_ARGS_BYTES=512" "$NCCL_HOME/test/unit/enqueue_test" ""
else
  echo -e "Disabled Enqueue TESTS Args test\n\n"
fi

if [[ ${ENQUEUE_TESTS_FIFO} -eq 1 ]] ; then
  run_command "enqueue_tests_fifo" "$RUN_MODE" 1 "--oversubscribe" "NCCL_WORK_FIFO_BYTES=1024" "$NCCL_HOME/test/unit/enqueue_test" ""
else
  echo -e "Disabled Enqueue TESTS Fifo test\n\n"
fi

if [[ ${GRAPH_TESTS_DEFAULT} -eq 1 ]] ; then
  run_command "graph_test_default" "$RUN_MODE" 1 "--oversubscribe" "TOPO_DIR=$NCCL_HOME/test/unit/" "$NCCL_HOME/test/unit/graph_test" ""
else
  echo -e "Disabled Graph TESTS Default test\n\n"
fi

if [[ ${MEMLEAK_TESTS_DEFAULT} -eq 1 ]] ; then
  run_command "single_process_mem_leak_test_default" "$RUN_MODE" 1 "--oversubscribe" "ASAN_OPTIONS=protect_shadow_gap=0" "$NCCL_HOME/test/unit/comm_leak_test" ""
else
  echo -e "Disabled Single-Process Mem Leak TESTS Default test\n\n"
fi

if [[ ${MEMLEAK_TESTS_NO_P2P} -eq 1 ]] ; then
  run_command "single_process_mem_leak_test_no_p2p" "$RUN_MODE" 1 "--oversubscribe" "ASAN_OPTIONS=protect_shadow_gap=0 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/unit/comm_leak_test" ""
else
  echo -e "Disabled Single-Process Mem Leak TESTS no_p2p test\n\n"
fi

if [[ ${MEMLEAK_TESTS_NETWORK} -eq 1 ]] ; then
  run_command "single_process_mem_leak_test_network" "$RUN_MODE" 1 "--oversubscribe" "ASAN_OPTIONS=protect_shadow_gap=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/unit/comm_leak_test" ""
else
  echo -e "Disabled Single-Process Mem Leak TESTS network test\n\n"
fi

NCCL_DEBUG_OLD=$NCCL_DEBUG
export NCCL_DEBUG=VERSION

if [[ ${FT_TESTS_DEFAULT} -eq 1 ]] ; then
  run_command "ft_test_default" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/ft_test" ""
else
  echo -e "Disabled FT TESTS Default test\n\n"
fi

if [[ "$NGPUS" -gt 1 ]] && [[ ${GIN_TESTS} -ne 1 ]]; then
  run_command "ft_abort_rank0" "$RUN_MODE" 2 "--oversubscribe" "" "$NCCL_HOME/test/unit/ft_abort_rank0" ""
fi

if [[ ${FT_TESTS_NO_P2P} -eq 1 ]] ; then
  run_command "ft_test_no_p2p" "$RUN_MODE" 1 "--oversubscribe" "NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/unit/ft_test" ""
else
  echo -e "Disabled FT TESTS no_p2p test\n\n"
fi

if [[ ${FT_TESTS_NETWORK} -eq 1 ]] ; then
  run_command "ft_test_network" "$RUN_MODE" 1 "--oversubscribe" "NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/unit/ft_test" ""
else
  echo -e "Disabled FT TESTS network test\n\n"
fi

if [[ ${DEVICE_ID_TESTS} -eq 1 ]] ; then
  run_command "device_id_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/device_id_test" ""
else
  "Disabled Device ID TESTS test\n\n"
fi

if [[ ${REGISTER_MEMCPY_TESTS} -eq 1 ]] ; then
  run_command "register_memcpy_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/register_memcpyTest" ""
else
  echo -e "Disabled Register Memcpy TESTS test\n\n"
fi

export NCCL_DEBUG=$NCCL_DEBUG_OLD
if [[ ${PLUGIN_TESTS_NET_TUNER} -eq 1 ]] ; then
  run_command "make_mixed_tuner" "CMD" 1 "" "" "make" "-C ext-mixed/example test"
else
  echo -e "Disabled Net/Tuner TESTS Mixed test\n\n"
fi

# PAT / Log Algo Tests
if [[ ${LOG_ALGO_RS_TESTS} -eq 1 ]] ; then
  run_command "log_algo_rs_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/log_algo" "rs 128"
else
  echo -e "Disabled PAT Log Algo RS TESTS test\n\n"
fi

if [[ ${LOG_ALGO_AG_TESTS} -eq 1 ]] ; then
  run_command "log_algo_ag_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/log_algo" "ag 128"
else
  echo -e "Disabled PAT Log Algo AG TESTS test\n\n"
fi

# Plugin Load Tests
export LD_LIBRARY_PATH=$NCCL_HOME/test/unit/plugins
if [[ ${PLUGIN_LOADING_TESTS} ]] ; then
  run_command "plugin_loading_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/plugin_load" ""
else
  echo -e "Disabled Plugin Loading TESTS test\n\n"
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_BACKUP

# More unit test

if [[ "${GIN_TESTS}" -eq 1 ]] ; then
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$MPI_HOME/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH"
  export DOCA_GPUNETIO_LITE_DEBUG=0

  run_gin_test_suite "auto"

  if [[ "${GIN_TESTS_GDAKI_GPU_SM}" -eq 1 ]] ; then
    export NCCL_GIN_TYPE=3
    export NCCL_GIN_GDAKI_NIC_HANDLER=2
    run_gin_test_suite "gdaki_gpusm"
  fi

  if [[ "${GIN_TESTS_GDAKI_CPU_ASSISTED}" -eq 1 ]] ; then
    export NCCL_GIN_TYPE=3
    export NCCL_GIN_GDAKI_NIC_HANDLER=1
    run_gin_test_suite "gdaki_cpuassisted"
  fi

  if [[ "${GIN_TESTS_CPU_PROXY}" -eq 1 ]] ; then
    export NCCL_GIN_TYPE=2
    run_gin_test_suite "cpuproxy"
  fi
else
  echo -e "Disabled GIN_TESTS test\n\n"
fi

print_failed_commands
end_junit_file
ci_exit
