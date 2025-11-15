#!/bin/bash

set +e  # Disable exit on error
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables
init_junit_file
get_slurm_planned_time

export LD_LIBRARY_PATH=$NCCL_HOME/test/apitest/plugin:$LD_LIBRARY_PATH

# argument list: testPrefix, gtest_filter
run_api_test(){
    echo "RUNNING API TEST with $1 and --gtest_filter = $2"
    # Args for run_command
    # run_command "label" "run_mode" "ppn" "test_mpi_flags" "test_env_vars" "binary" "args"
    if [[ ${API_TESTS_DEFAULT} -eq 1 ]] ; then
      run_command "apitest_${1}default" "$RUN_MODE" 1 "--oversubscribe" "" "$NCCL_HOME/test/apitest/apitest --gtest_filter=${2}" ""
    else
      echo -e "Disabled Api TESTS Default test\n\n"
    fi

    # NvBug 5210770
    if [[ ${API_TESTS_NO_P2P} -eq 1 ]] ; then
      run_command "apitest_${1}no_p2p" "$RUN_MODE" 1 "--oversubscribe" "NCCL_P2P_DISABLE=1" "$NCCL_HOME/test/apitest/apitest --gtest_filter=${2}" ""
    else
      echo -e "Disabled Api TESTS no_p2p test\n\n"
    fi

    if [[ ${API_TESTS_NETWORK} -eq 1 ]] ; then
      run_command "apitest_${1}no_p2p_no_shm" "$RUN_MODE" 1 "--oversubscribe" "NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1" "$NCCL_HOME/test/apitest/apitest --gtest_filter=${2}" ""
    else
      echo -e "Disabled Api TESTS network test\n\n"
    fi
}

# list of tests with a special config
multinetTests="ncclCommInitRankConfig_test.multi_net_plugin_*"
sharedPluginTest="ncclCommInitRankConfig_test.shared_plugin_lib"

# run all tests except the ones with a special config
run_api_test "" "-${multinetTests}:${sharedPluginTest}"

# run multinet tests with special config
export NCCL_NET_PLUGIN="plugin_nodev_v6,plugin_v7,plugin_nodev_v8,plugin_nodev_v9,plugin_nodev_v10,plugin_nodev_v11"
run_api_test "multinet_" "${multinetTests}"
unset NCCL_NET_PLUGIN

export NCCL_NET_PLUGIN="libnccl-shared-plugins.so"
run_api_test "" "${sharedPluginTest}"
unset NCCL_NET_PLUGIN


# run w/o allgatherv
export NCCL_ALLGATHERV_ENABLE=0
run_api_test "" ""
unset NCCL_ALLGATHERV_ENABLE

print_failed_commands
end_junit_file
ci_exit
