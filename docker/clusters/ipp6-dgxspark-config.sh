# Config parameters for ipp6 cluster
# No host-specific code executes in this module to allow it to be served over http in the future
#
# Targeting build to run in ipp6 environment (A100/A40/L40S/H100)
DGX_GPU_ARCHS="121"

DGX_OS_VERSION="24.04"
DGX_CUDA_VERSION="13.0.2"
DGX_BUILD_TOOLS_VERSION="2.1.1"
DGX_BUILD_IMAGE_VERSION="${DGX_BUILD_TOOLS_VERSION}-c${DGX_CUDA_VERSION}-u${DGX_OS_VERSION}"

DGX_TOOLKIT_DIR="/storage/toolkits-arm"
DGX_CUDA_HOME="/usr/local/cuda"
DGX_OPENMPI_HOME="/cm/shared/apps/openmpi4/gcc/4.1.8"

DGX_DOCKER_IMAGE_DIR="$DGX_TOOLKIT_DIR/docker_sqsh"

DGX_PLANNED_RESERVED="Skip"

DGX_NCCL_SOCKET_IFNAME="enP7s7"

# Target configs
function get_gpu_archs() {
    echo "$DGX_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$DGX_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$DGX_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_num_build_procs() {
    echo "nproc"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    echo "srun \
        -p dgxspark \
        -J nccl:build \
        -t 00:30:00 \
        -n 1 \
        --exclusive \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}

function get_build_command_bm() {
    echo "srun \
        -p dgxspark \
        -J nccl:build \
        -t 00:30:00 \
        -n 1 \
        --exclusive \
        docker/build_nccl_bm.sh"
}

function get_cuda_home() {
    echo "$DGX_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$DGX_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$DGX_CUDA_HOME/lib64:$DGX_OPENMPI_HOME/lib"
}

function get_extra_path() {
    echo "$DGX_CUDA_HOME/bin"
}

function get_planned_reserved() {
    echo "$DGX_PLANNED_RESERVED"
}

function get_nccl_socket_ifname() {
    echo "$DGX_NCCL_SOCKET_IFNAME"
}

function get_mpi_params() {
    echo ""
}

function configure_test_env() {
    export NCCL_DEBUG=WARN
    export UCX_NET_DEVICES=$DGX_NCCL_SOCKET_IFNAME
    export NCCL_SOCKET_IFNAME=$DGX_NCCL_SOCKET_IFNAME
    export OMPI_MCA_btl_tcp_if_include=$DGX_NCCL_SOCKET_IFNAME
}
