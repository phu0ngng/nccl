# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for Pre-Nyx cluster
#
# Targeting build to run on Pre-Nyx

# Pre-Nyx GPU Type: B200
# https://confluence.nvidia.com/display/DSE/Pre-Nyx+Hardware
PRENYX_GPU_ARCHS="100,120"

# NCCL CI deps
PRENYX_CUDA_VERSION="12.8"
PRENYX_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${PRENYX_CUDA_VERSION}"

PRENYX_OPENMPI_VERSION="5.0.6"
PRENYX_OPENMPI_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/openmpi-${PRENYX_OPENMPI_VERSION}"

# Slurm account for executing jobs
PRENYX_SLURM_ACCOUNT="coreai_libraries_nccl"

# Use with sacct on to get Slurm wait times
PRENYX_PLANNED_RESERVED="Planned"

PRENYX_NCCL_SOCKET_IFNAME="ibp24s0"

# Target configs
function get_gpu_archs() {
    echo "$PRENYX_GPU_ARCHS"
}

function get_slurm_account() {
    echo "$PRENYX_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    echo "srun \
            --account=$PRENYX_SLURM_ACCOUNT \
            -J $PRENYX_SLURM_ACCOUNT-ci:nccl-build \
            -t 00:05:00 \
            -n 1 \
            -c 64 \
            -p batch \
            docker/build_nccl_bm.sh"
}

function get_cuda_home() {
    echo "$PRENYX_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$PRENYX_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$PRENYX_CUDA_HOME/lib64:$PRENYX_OPENMPI_HOME/lib"
}

function get_mpi_params() {
    # Expose MPI params as test env variables
    echo ""
}

function get_extra_path() {
    echo "$PRENYX_CUDA_HOME/bin:$PRENYX_OPENMPI_HOME/bin"
}

function get_planned_reserved() {
    echo "$PRENYX_PLANNED_RESERVED"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$PRENYX_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp

    # Note: These are the non-E/W IB interfaces on Pre-Nyx
    export NCCL_IB_HCA="^mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_11,mlx5_6,mlx5_12"

    # MPI params
    # --oversubscribe
    export OMPI_MCA_rmaps_oversubscribe=1

    # --bind-to none
    export OMPI_MCA_rmaps_binding_policy=none

    # --mca btl tcp,self
    export OMPI_MCA_btl="tcp,self"

    # --mca btl_tcp_if_include <socket>
    export OMPI_MCA_btl_tcp_if_include=$PRENYX_NCCL_SOCKET_IFNAME
}
