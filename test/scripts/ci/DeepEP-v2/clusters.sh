# Per-cluster config + per-(cluster, test) overrides for the DeepEP-v2 CI.
# Sourced by run.sh and _alloc.sh. Expects $CLUSTER to be set.
#
# Vars are organized into three groups; case branches follow the same order.
#   CI INFRA:     SLURM_ACCOUNT, SLURM_PARTITION, GPUS_PER_NODE,
#                 SKIP_TESTS, CLUSTER_ENV_NOTES
#   DEEPEP BUILD: TOOLKITS_DIR, UV_INSTALL_DIR, NVSHMEM_INSTALL_DIR (also
#                 LD_PRELOAD'd at test — dual-use), CUDA_HOME, TORCH_INDEX_URL
#   TEST RUNTIME: cluster-wide NCCL/EP env exports (NCCL_IB_HCA, ...)
#                 set via cluster_set_env (mirror of test_set_env in tests.sh).
# Functions:
#   cluster_set_env VAR=value   — export + append to CLUSTER_ENV_NOTES.
#   apply_overrides             — per-(cluster, test) test_set_env tweaks.

# --- CI INFRA (shared) ---
SLURM_ACCOUNT=coreai_libraries_nccl

# --- DEEPEP BUILD (shared) ---
TOOLKITS_DIR=/lustre/fsw/coreai_libraries_nccl/toolkits
UV_INSTALL_DIR=$TOOLKITS_DIR/uv_install
# Toolkit NVSHMEM: DeepEP's setup.py links `-l:libnvshmem_host.so` (unversioned
# symlink), which the pip wheel doesn't ship — only `libnvshmem_host.so.3`.
NVSHMEM_INSTALL_DIR=$TOOLKITS_DIR/DeepEPv2-CI/nvshmem-3.6.5

# Exports VAR=value cluster-wide and appends it to CLUSTER_ENV_NOTES so the
# banner stays in sync. Mirror of test_set_env in tests.sh.
cluster_set_env() {
    local kv="$1"
    export "$kv"
    CLUSTER_ENV_NOTES="${CLUSTER_ENV_NOTES:+$CLUSTER_ENV_NOTES }$kv"
}

# --- per-cluster config ---
# To add a cluster: copy a case branch, keep ci-infra/build/test-runtime grouping.
apply_cluster_config() {
    # build defaults
    CUDA_HOME=$TOOLKITS_DIR/cuda-13.2
    TORCH_INDEX_URL=""           # empty = PyPI default cu13 wheel
    # ci-infra defaults
    CLUSTER_ENV_NOTES=""

    case "$CLUSTER" in
        eos)
            # ci infra
            SLURM_PARTITION=batch
            GPUS_PER_NODE=8       # H100 DGX
            SKIP_TESTS=()
            # build — driver caps at CUDA 12.2; pin toolkit + wheel to cu129
            # (latest cu12 PyTorch ships). MVC handles 12.9 → 12.2.
            CUDA_HOME=$TOOLKITS_DIR/cuda-12.9
            TORCH_INDEX_URL=https://download.pytorch.org/whl/cu129
            # test runtime
            cluster_set_env OMPI_MCA_coll=^hcoll
            cluster_set_env OMPI_MCA_btl=tcp,self
            cluster_set_env PMIX_MCA_gds=hash
            ;;
        theia)
            # ci infra
            SLURM_PARTITION=gb300
            GPUS_PER_NODE=4       # GB300
            SKIP_TESTS=()
            # test runtime
            cluster_set_env NCCL_IB_DATA_DIRECT=0
            cluster_set_env NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
            cluster_set_env OMPI_MCA_coll=^hcoll
            cluster_set_env OMPI_MCA_btl=tcp,self
            cluster_set_env PMIX_MCA_gds=hash
            ;;
        ptyche)
            # ci infra
            SLURM_PARTITION=batch
            GPUS_PER_NODE=4       # GB200 (ARM)
            SKIP_TESTS=()
            # test runtime
            cluster_set_env NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
            cluster_set_env OMPI_MCA_coll=^hcoll
            cluster_set_env OMPI_MCA_btl=tcp,self
            cluster_set_env PMIX_MCA_gds=hash
            ;;
        *)
            echo "Unsupported cluster: '$CLUSTER' (expected: eos|theia|ptyche)" >&2
            exit 1
            ;;
    esac
}

apply_cluster_config

# --- TEST RUNTIME — per-(cluster, test) overrides ---
# Called from _alloc.sh after set_test_config. Use `test_set_env VAR=value`.
apply_overrides() {
    case "$CLUSTER:$TEST" in
        # Add `cluster:test)` branches here as needed.
        *) ;;
    esac
}

export SLURM_ACCOUNT SLURM_PARTITION GPUS_PER_NODE
export UV_INSTALL_DIR NVSHMEM_INSTALL_DIR CUDA_HOME TORCH_INDEX_URL
