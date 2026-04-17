#!/bin/bash
# Run the reduceCopy device API test suite for all 12 types.
# Intended for weekly CI runs on GB200/Hopper/Blackwell nodes.
# Uses run_reducecopy_types.sh which stops on first type failure.
# Set NCCL_TEST_SKIP_MULTIMEM=1 to skip multimem tests (for clusters without multimem support).

set +e  # Disable exit on error; ci_exit handles final exit code
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables

BINARY="$NCCL_HOME/test/apitest/device_api/device_api_test"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: device_api_test binary not found or not executable: $BINARY"
    exit 1
fi

export NCCL_TEST_VERBOSE="${NCCL_TEST_VERBOSE:-0}"

SKIP_MULTIMEM_FLAG=""
if [ "${NCCL_TEST_SKIP_MULTIMEM:-0}" = "1" ]; then
    SKIP_MULTIMEM_FLAG="-m"
fi

# run_reducecopy_types.sh runs the binary once per type (all 12 types by default)
# and stops at first failure.  Pass "" as the extra-filter so all tests run.
# Explicitly pass LD_LIBRARY_PATH in --export to match what run_command() does;
# relying on --export=ALL alone is insufficient because load_cluster_ci_variables
# sets LD_LIBRARY_PATH without exporting it.
srun -n 1 --label --export=ALL,LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    test/apitest/device_api/reduceCopy/run_reducecopy_types.sh $SKIP_MULTIMEM_FLAG "" "$BINARY"
