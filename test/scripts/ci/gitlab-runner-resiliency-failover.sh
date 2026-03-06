#!/bin/bash

# ============================== Description ===================================

# NCCL port-failover test script.
# This script tests NCCL's port failover feature by emulating port failure.
# The port failure emulation binary is launched on the first node of the SLURM allocation.

# ==============================================================================

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gitlab-runner-resiliency-utils.sh"
source "$SCRIPT_DIR/ci-utils.sh"

# Check if CLUSTER_CONFIG is set and points to a real file
if [[ ! -z "${CLUSTER_CONFIG:-}" ]]; then
    load_test_ci_variables
    source $CLUSTER_CONFIG
    load_cluster_ci_variables
fi

# Validate environment variables
validate_environment NIC_TO_FAIL

# Setup test environment
export LOGS_DIR=.

PORT_FAILURE_EMU_NODE=$(get_port_failure_emu_node)
NCCL_PARAMS_COMMON=$(setup_nccl_common_params 0)  # 0 = no recovery

failure_count=0
failure_names=()

NCCL_CYCLES=500

# Sweep over different NCCL perf tests and AR thresholds
NCCL_PERF_TESTS=("all_reduce" "alltoall")
NCCL_AR_THRESHOLDS=(0 2147483647)

for ar_threshold in "${NCCL_AR_THRESHOLDS[@]}"; do
for func in "${NCCL_PERF_TESTS[@]}"; do

NCCL_PARAMS_EXTRA=$(setup_nccl_ar_params $ar_threshold)
NCCL_PARAMS="${NCCL_PARAMS_COMMON} ${NCCL_PARAMS_EXTRA}"
NCCL_TEST_PARAMS=$(setup_nccl_test_params $NCCL_CYCLES)

LOG_PORT_EMULATION_FILE="$LOGS_DIR/test_port_emulation_${func}_${ar_threshold}.log"
echo "Log file for port failure emulation: $LOG_PORT_EMULATION_FILE"
LOG_NCCL_TEST_FILE="$LOGS_DIR/test_failover_${func}_${ar_threshold}.log"
echo "Log file for NCCL test: $LOG_NCCL_TEST_FILE"

# Start NCCL perf test
start_nccl_perf_test "$func" "$NCCL_PARAMS" "$NCCL_TEST_PARAMS" "$LOG_NCCL_TEST_FILE" NCCL_PERFTEST_PID

# Wait for NCCL test to be underway before launching port failure emulation
wait_for_nccl_cycles "$LOG_NCCL_TEST_FILE" 5 $NCCL_CYCLES

# Start port failure emulation
start_port_failure_emulation "$PORT_FAILURE_EMU_NODE" "$NIC_TO_FAIL" "$LOG_PORT_EMULATION_FILE" PORT_FAILURE_EMU_PID

# Verify port failure emulation is running
verify_port_failure_emulation "$PORT_FAILURE_EMU_NODE" "$PORT_FAILURE_EMU_PID"

echo "Port failure emulation app started on NIC: $NIC_TO_FAIL (PID: $PORT_FAILURE_EMU_PID)"
echo "NCCL test should experience failover"

# Wait for NCCL perf test to complete
echo "Waiting for NCCL perf test process (PID=${NCCL_PERFTEST_PID}) to complete."
wait_for_nccl_finish $NCCL_PERFTEST_PID 600
NCCL_TEST_RESULT=$?

# Stop port failure emulation after NCCL test is fully complete
stop_port_failure_emulation "$PORT_FAILURE_EMU_NODE" "$PORT_FAILURE_EMU_PID" 1

# Record test result
record_test_result $NCCL_TEST_RESULT "$func" "failover" "$ar_threshold" failure_count failure_names

# Print logs
print_test_logs "$LOG_NCCL_TEST_FILE" "$LOG_PORT_EMULATION_FILE"

done;
done;

print_test_summary $failure_count "${failure_names[@]}"
exit $failure_count
