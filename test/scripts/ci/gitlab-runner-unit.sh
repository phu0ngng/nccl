#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

run_command "enqueue_tests_args" "$RUN_MODE" 1 "-x NCCL_WORK_FIFO_BYTES=0 -x NCCL_WORK_ARGS_BYTES=512 --oversubscribe" "$NCCL_HOME/test/unit/enqueue_test" ""
run_command "enqueue_tests_fifo" "$RUN_MODE" 1 "-x NCCL_WORK_FIFO_BYTES=1024 --oversubscribe" "$NCCL_HOME/test/unit/enqueue_test" ""
run_command "graph_test_default" "$RUN_MODE" 1 "-x TOPO_DIR=$NCCL_HOME/test/unit/ --oversubscribe " "$NCCL_HOME/test/unit/graph_test" ""

if [ "$DISABLE_MEMLEAK_DEFAULT" == "1" ]; then
  echo "Disabled Single-Process Mem Leak TESTS Default test\n\n"
else
  run_command "single_process_mem_leak_test_default" "$RUN_MODE" 1 "-x ASAN_OPTIONS=protect_shadow_gap=0 --oversubscribe" "$NCCL_HOME/test/unit/comm_leak_test" ""
fi

if [ "$DISABLE_MEMLEAK_NETWORK" == "1" ]; then
  echo "Disabled Single-Process Mem Leak TESTS no_p2p and network test\n\n"
else
  run_command "single_process_mem_leak_test_no_p2p" "$RUN_MODE" 1 "-x ASAN_OPTIONS=protect_shadow_gap=0 -x NCCL_P2P_DISABLE=1 --oversubscribe " "$NCCL_HOME/test/unit/comm_leak_test" ""
  run_command "single_process_mem_leak_test_network" "$RUN_MODE" 1 "-x ASAN_OPTIONS=protect_shadow_gap=0 -x NCCL_SHM_DISABLE=1 -x NCCL_P2P_DISABLE=1 --oversubscribe" "$NCCL_HOME/test/unit/comm_leak_test" ""
fi

NCCL_DEBUG_OLD=$NCCL_DEBUG
export NCCL_DEBUG=VERSION
run_command "ft_test_default" "$RUN_MODE" 1 "--oversubscribe" "$NCCL_HOME/test/unit/ft_test" ""

if [ "$NGPUS" -gt "1" ]; then
  run_command "ft_abort_rank0" "$RUN_MODE" 2 "--oversubscribe" "$NCCL_HOME/test/unit/ft_abort_rank0" ""
fi

run_command "ft_test_no_p2p" "$RUN_MODE" 1 "-x NCCL_P2P_DISABLE=1 --oversubscribe" "$NCCL_HOME/test/unit/ft_test" ""
run_command "ft_test_network" "$RUN_MODE" 1 "-x NCCL_SHM_DISABLE=1 -x NCCL_P2P_DISABLE=1 --oversubscribe" "$NCCL_HOME/test/unit/ft_test" ""
export NCCL_DEBUG=$NCCL_DEBUG_OLD
run_command "make_mixed_tuner" "CMD" 1 "" "make" "-C ext-mixed/example test"

# PAT / Log Algo Tests
run_command "log_algo_rs_tests" "$RUN_MODE" 1 "--oversubscribe" "$NCCL_HOME/test/unit/log_algo" "rs 128"
run_command "log_algo_ag_tests" "$RUN_MODE" 1 "--oversubscribe" "$NCCL_HOME/test/unit/log_algo" "ag 128"

print_failed_commands
end_junit_file
ci_exit
