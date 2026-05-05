#!/bin/bash
#
# DeepEP-v2 CI: build + run sanity tests in a single SLURM allocation.
# Used identically by CI and for manual reproduction.
#
# Usage:
#   ./run.sh <cluster>                              # build + run all tests
#                                                   # (cluster: eos|theia|ptyche)
#   ./run.sh <cluster> sanity_ep                    # single test
#   ./run.sh <cluster> sanity_ep sanity_engram      # multiple
#
# Required env: NCCL_INSTALL_DIR, DEEP_EP_INSTALL_DIR
# Optional env:
#   TIMEOUT     salloc -t in minutes; default 15
#   SKIP_BUILD  =1 to skip venv + DeepEP build and reuse the existing one at
#               $DEEP_EP_INSTALL_DIR (useful for iterative manual debug)
#
# Outer entry point only — in-allocation work lives in _alloc.sh.
set -e

export CLUSTER=${1:?cluster required (eos|theia|ptyche)}
shift
TESTS=("$@")

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_pretty.sh"
source "$SCRIPT_DIR/clusters.sh"
source "$SCRIPT_DIR/tests.sh"

: "${NCCL_INSTALL_DIR:?env NCCL_INSTALL_DIR required}"
: "${DEEP_EP_INSTALL_DIR:?env DEEP_EP_INSTALL_DIR required}"
export NCCL_INSTALL_DIR DEEP_EP_INSTALL_DIR SKIP_BUILD

# Default test set = ALL_TESTS minus GLOBAL_SKIP_TESTS minus this cluster's
# SKIP_TESTS.
if [ ${#TESTS[@]} -eq 0 ]; then
    TESTS=()
    for t in "${ALL_TESTS[@]}"; do
        skip=0
        for s in "${GLOBAL_SKIP_TESTS[@]}" "${SKIP_TESTS[@]}"; do
            [ "$t" = "$s" ] && { skip=1; break; }
        done
        [ $skip -eq 0 ] && TESTS+=("$t")
    done
fi

TIMEOUT=${TIMEOUT:-15}
SLURM_JOB_NAME="coreai_libraries_nccl-DeepEP-v2.ci"

banner "DeepEP-v2 CI :: $CLUSTER"
info "NCCL build:    $NCCL_INSTALL_DIR"
info "DeepEP dir:    $DEEP_EP_INSTALL_DIR"
info "Tests:         ${TESTS[*]}"
[ ${#GLOBAL_SKIP_TESTS[@]} -gt 0 ] && info "Skipped (all): ${GLOBAL_SKIP_TESTS[*]} (per tests.sh)"
[ ${#SKIP_TESTS[@]} -gt 0 ]        && info "Skipped:       ${SKIP_TESTS[*]} (per clusters.sh)"
[ -n "${SKIP_BUILD:-}" ]    && info "Build:         SKIPPED (SKIP_BUILD=$SKIP_BUILD; reusing $DEEP_EP_INSTALL_DIR)"
info "Cluster:       $CLUSTER (partition=$SLURM_PARTITION, gpus/node=$GPUS_PER_NODE)"
[ -n "${CLUSTER_ENV_NOTES:-}" ] && info "Cluster env:   $CLUSTER_ENV_NOTES (set in clusters.sh)"
info "Toolkits:      NVSHMEM=$NVSHMEM_INSTALL_DIR"
info "               CUDA=$CUDA_HOME"
info "               UV=$UV_INSTALL_DIR"
info "Allocation:    -N 2 -A $SLURM_ACCOUNT --exclusive -t $TIMEOUT (job: $SLURM_JOB_NAME)"

# Arrays don't survive salloc/srun env forwarding — pass as space-separated.
export TESTS_STR="${TESTS[*]}"

# salloc body runs on the compute master, not the submitting host — so its
# arch matches the GPUs (matters for ARM clusters).
exec salloc -N 2 -p "$SLURM_PARTITION" -A "$SLURM_ACCOUNT" \
            -J "$SLURM_JOB_NAME" \
            --exclusive -t "$TIMEOUT" \
            bash "$SCRIPT_DIR/_alloc.sh"
