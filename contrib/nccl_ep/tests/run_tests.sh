#!/usr/bin/env bash
# Run ncclEp unit tests across multiple GPUs.
#
# Uses mpirun if available (provides PMI environment for NCCL bootstrapping
# on EOS/Enroot containers); falls back to bare background processes.
#
# Usage:
#   NCCL_HOME=/path/to/nccl/build bash run_tests.sh [num_gpus]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCCL_HOME="${NCCL_HOME:-$(cd "${SCRIPT_DIR}/../../../build" && pwd)}"
NUM_GPUS="${1:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
MPI_HOME="${MPI_HOME:-/usr/local/mpi}"

export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"

GTEST_ARGS="${GTEST_FILTER:+--gtest_filter=${GTEST_FILTER}}"
OVERALL_FAIL=0

run_suite() {
    local BINARY="$1"
    local SUITE_NAME="$2"
    local MIN_GPUS="${3:-4}"
    local TEST_BIN="${NCCL_HOME}/test/nccl_ep/${BINARY}"

    if [[ ! -x "${TEST_BIN}" ]]; then
        echo "ERROR: binary not found: ${TEST_BIN}"
        echo "Build first:  make -C ${SCRIPT_DIR} NCCL_HOME=${NCCL_HOME}"
        return 1
    fi

    if (( NUM_GPUS < MIN_GPUS )); then
        echo "${SUITE_NAME}: requires at least ${MIN_GPUS} GPUs, found ${NUM_GPUS}. Skipping."
        return 0
    fi

    local TMPDIR_L="${TMPDIR:-/tmp}"
    local UID_FILE="${TMPDIR_L}/te_ep_uid_${BINARY}_$$"
    rm -f "${UID_FILE}"
    trap "rm -f '${UID_FILE}'" EXIT INT TERM

    local LOG_DIR
    LOG_DIR=$(mktemp -d)
    local FAIL=0

    echo "=== ${SUITE_NAME} ==="
    echo "  GPUs: ${NUM_GPUS}   Binary: ${TEST_BIN}"
    echo

    if [[ -x "${MPI_HOME}/bin/mpirun" ]]; then
        # mpirun gives each process a proper PMI environment, which NCCL
        # needs for socket bootstrapping inside Enroot containers on EOS.
        "${MPI_HOME}/bin/mpirun" \
            -np "${NUM_GPUS}" \
            --bind-to none \
            --allow-run-as-root \
            --oversubscribe \
            -x LD_LIBRARY_PATH \
            bash -c "\"${TEST_BIN}\" \
                --rank=\${OMPI_COMM_WORLD_RANK} \
                --nranks=${NUM_GPUS} \
                --uid-file=\"${UID_FILE}\" \
                ${GTEST_ARGS} \
                > \"${LOG_DIR}/rank_\${OMPI_COMM_WORLD_RANK}.log\" 2>&1" \
            || FAIL=1
    else
        # Fallback: bare background processes (works on single-node with
        # direct TCP access; may fail in restricted container environments).
        local PIDS=()
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            "${TEST_BIN}" \
                --rank="${i}" \
                --nranks="${NUM_GPUS}" \
                --uid-file="${UID_FILE}" \
                ${GTEST_ARGS} \
                > "${LOG_DIR}/rank_${i}.log" 2>&1 &
            PIDS+=($!)
        done
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            wait "${PIDS[$i]}" || FAIL=1
        done
    fi

    echo "--- Rank 0 output ---"
    cat "${LOG_DIR}/rank_0.log"

    if (( FAIL )); then
        for i in $(seq 1 $((NUM_GPUS - 1))); do
            echo "--- Rank ${i} output ---"
            cat "${LOG_DIR}/rank_${i}.log"
        done
        echo "=== ${SUITE_NAME}: FAILED ==="
        OVERALL_FAIL=1
    else
        echo "=== ${SUITE_NAME}: ALL PASSED ==="
    fi

    rm -rf "${LOG_DIR}"
}

run_suite "test_output_layout" "EP Output Layout Tests"
run_suite "test_handle_maps"   "EP Handle Maps Tests"

exit "${OVERALL_FAIL}"
