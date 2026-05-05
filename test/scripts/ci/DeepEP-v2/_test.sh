#!/bin/bash
#
# One task per compute node, invoked by _alloc.sh via
# `srun --mpi=pmix -N 2 --ntasks-per-node=1`.
#
# Required env: NCCL_INSTALL_DIR, NVSHMEM_INSTALL_DIR, DEEP_EP_INSTALL_DIR,
# TEST_MODULE, MASTER_PORT, TEST_ARGS.
# Plus any cluster-wide env exported by clusters.sh's cluster_set_env (NCCL_IB_HCA,
# OMPI_MCA_*, etc.) and per-test env from tests.sh's test_set_env.
set -e

cd "$DEEP_EP_INSTALL_DIR/DeepEP"
source "$DEEP_EP_INSTALL_DIR/venv/bin/activate"

# Preload toolkit nvshmem + our NCCL so torch's pip-installed nvidia-nvshmem-cu13
# and nvidia-nccl-cu13 (same SONAMEs) don't win symbol resolution.
export LD_PRELOAD="$NVSHMEM_INSTALL_DIR/lib/libnvshmem_host.so.3:$NCCL_INSTALL_DIR/lib/libnccl.so.2"
# DeepEP JIT-compiles kernels at runtime; without EP_NCCL_ROOT_DIR it falls back
# to torch's pip nvidia-nccl-cu13 (older, missing GIN device symbols).
export EP_NCCL_ROOT_DIR="$NCCL_INSTALL_DIR"
export EP_JIT_CACHE_DIR=/tmp/deep_ep_jit_cache
export EP_BUFFER_DEBUG=0
export EP_SUPPRESS_NCCL_CHECK=1
export NCCL_DEBUG=WARN
export MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export WORLD_SIZE=$SLURM_NNODES
export RANK=$SLURM_PROCID

[ "${SLURM_PROCID:-0}" = "0" ] && echo "[exec] python -m $TEST_MODULE $TEST_ARGS"
# TEST_ARGS intentionally unquoted so it word-splits into separate args.
exec python -m "$TEST_MODULE" $TEST_ARGS
