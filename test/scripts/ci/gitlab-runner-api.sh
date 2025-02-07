#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

run_command "apitest_default" "$RUN_MODE" 1 " --oversubscribe" "$NCCL_HOME/test/apitest/apitest" ""
run_command "apitest_no_p2p" "$RUN_MODE" 1 "-x NCCL_P2P_DISABLE=1 --oversubscribe" "$NCCL_HOME/test/apitest/apitest" ""
run_command "apitest_no_p2p_no_shm" "$RUN_MODE" 1 "-x NCCL_P2P_DISABLE=1 -x NCCL_SHM_DISABLE=1 --oversubscribe" "$NCCL_HOME/test/apitest/apitest" ""

print_failed_commands
end_junit_file
ci_exit
