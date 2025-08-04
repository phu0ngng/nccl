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

if [ "$NGPUS" -gt "1" ]; then
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

export LD_LIBRARY_PATH=$NCCL_HOME/test/unit/plugins

if [[ ${PLUGIN_LOADING_TESTS} ]] ; then
  run_command "plugin_loading_tests" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/unit/plugin_load" ""
else
  echo -e "Disabled Plugin Loading TESTS test\n\n"
fi

print_failed_commands
end_junit_file
ci_exit
