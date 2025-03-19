# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for Pre-Tyche cluster
#
# Targeting build to run on Pre-Tyche

# Pre-Tyche GPU Type: GB200
# https://confluence.nvidia.com/display/DSE/Pre-Tyche+Hardware
PRETYCHE_GPU_ARCHS="100,120"

# NCCL CI deps
PRETYCHE_CUDA_VERSION="12.8.0"
PRETYCHE_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${PRETYCHE_CUDA_VERSION}"

PRETYCHE_OPENMPI_VERSION="5.0.6"
PRETYCHE_OPENMPI_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/openmpi-${PRETYCHE_OPENMPI_VERSION}"

# Slurm account for executing jobs
PRETYCHE_SLURM_ACCOUNT="coreai_libraries_nccl"

# Use with sacct on to get Slurm wait times
PRETYCHE_PLANNED_RESERVED="Planned"

PRETYCHE_NCCL_SOCKET_IFNAME="enP6p3s0f1np1"

# Target configs
function get_gpu_archs() {
    echo "$PRETYCHE_GPU_ARCHS"
}

function get_slurm_account() {
    echo "$PRETYCHE_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    echo "srun \
            --account=$PRETYCHE_SLURM_ACCOUNT \
            -J $PRETYCHE_SLURM_ACCOUNT-ci:nccl-build \
            -t 00:05:00 \
            -n 1 \
            -c 64 \
            -p batch \
            docker/build_nccl_bm.sh"
}

function get_cuda_home() {
    echo "$PRETYCHE_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$PRETYCHE_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$PRETYCHE_CUDA_HOME/lib64:$PRETYCHE_OPENMPI_HOME/lib"
}

function get_mpi_params() {
    # Expose MPI params as test env variables
    echo ""
}

function get_extra_path() {
    echo "$PRETYCHE_CUDA_HOME/bin:$PRETYCHE_OPENMPI_HOME/bin"
}

function get_planned_reserved() {
    echo "$PRETYCHE_PLANNED_RESERVED"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$PRETYCHE_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp

    # Note: These are the E/W IB interfaces on Pre-Tyche
    export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_4,mlx5_5"

    # MPI params
    # --oversubscribe
    export OMPI_MCA_rmaps_oversubscribe=1

    # --bind-to none
    export OMPI_MCA_rmaps_binding_policy=none

    # --mca btl tcp,self
    export OMPI_MCA_btl="tcp,self"

    # --mca btl_tcp_if_include <socket>
    export OMPI_MCA_btl_tcp_if_include=$PRETYCHE_NCCL_SOCKET_IFNAME
}
