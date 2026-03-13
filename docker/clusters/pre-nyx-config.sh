# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for Pre-Nyx cluster
#
# Targeting build to run on Pre-Nyx

# Pre-Nyx GPU Type: B200
# https://confluence.nvidia.com/display/DSE/Pre-Nyx+Hardware
PRENYX_GPU_ARCHS="100,120"

PRENYX_TOOLKIT_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits"

PRENYX_OS_VERSION="20.04"
PRENYX_CUDA_VERSION="12.8.0"
PRENYX_BUILD_TOOLS_VERSION="2.0.0"
PRENYX_BUILD_IMAGE_VERSION="${PRENYX_BUILD_TOOLS_VERSION}-c${PRENYX_CUDA_VERSION}-u${PRENYX_OS_VERSION}"

PRENYX_DOCKER_IMAGE_DIR="$PRENYX_TOOLKIT_DIR/docker_sqsh"

PRENYX_CUDA_HOME="$PRENYX_TOOLKIT_DIR/cuda-${PRENYX_CUDA_VERSION}"

PRENYX_OPENMPI_VERSION="5.0.6"
PRENYX_OPENMPI_HOME="$PRENYX_TOOLKIT_DIR/openmpi-${PRENYX_OPENMPI_VERSION}"

# Slurm account for executing jobs
PRENYX_SLURM_ACCOUNT="coreai_libraries_nccl"

# Use with sacct on to get Slurm wait times
PRENYX_PLANNED_RESERVED="Planned"

PRENYX_NCCL_SOCKET_IFNAME="ens6f1np1"

# Target configs
function get_gpu_archs() {
    echo "$PRENYX_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$PRENYX_BUILD_IMAGE_VERSION"
}

function get_build_tools_image() {
    build_image_version="$1"
    echo "$PRENYX_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_slurm_account() {
    echo "$PRENYX_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_command_bm() {
    echo "srun \
            --account=$PRENYX_SLURM_ACCOUNT \
            -J $PRENYX_SLURM_ACCOUNT-ci:nccl-build \
            -p batch \
            -t 00:10:00 \
            -n 1 \
            -c 64 \
            docker/build_nccl_bm.sh"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"
    build_tools_image="$(get_build_tools_image $build_image_version)"

    echo "srun \
        --account=$PRENYX_SLURM_ACCOUNT \
        -J ${PRENYX_SLURM_ACCOUNT}-ci:nccl-build \
        -p batch \
        -t 00:10:00 \
        -n 1 \
	-c 64 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_cuda_home() {
    echo "$PRENYX_CUDA_HOME"
}

function get_ccache_bin() {
    echo "$PRENYX_TOOLKIT_DIR/ccache-4.12.1-linux-x86_64/ccache"
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

function get_nccl_socket_ifname() {
    echo "$PRENYX_NCCL_SOCKET_IFNAME"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$PRENYX_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp

    # Note: These are the E/W IB interfaces on Pre-Nyx
    export NCCL_IB_HCA="=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"

    # MPI params
    # --oversubscribe
    export OMPI_MCA_rmaps_oversubscribe=1

    # --bind-to none
    export OMPI_MCA_rmaps_binding_policy=none
    export OMPI_MCA_hwloc_base_binding_policy=none

    # --mca btl tcp,self
    export OMPI_MCA_btl="tcp,self"

    # --mca btl_tcp_if_include <socket>
    export OMPI_MCA_btl_tcp_if_include=$PRENYX_NCCL_SOCKET_IFNAME
}
