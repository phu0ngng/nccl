# Config parameters for ipp6 cluster
# No host-specific code executes in this module to allow it to be served over http in the future
#
# Targeting build to run in ipp6 environment (A100/A40/L40S/H100)
IPP6_GPU_ARCHS="80,86,89,90"

IPP6_OS_VERSION="20.04"
IPP6_CUDA_VERSION="12.8.0"
IPP6_BUILD_TOOLS_VERSION="1.1.0"
IPP6_BUILD_IMAGE_VERSION="${IPP6_BUILD_TOOLS_VERSION}-c${IPP6_CUDA_VERSION}-u${IPP6_OS_VERSION}"

IPP6_TOOLKIT_DIR="/storage/toolkits"
IPP6_CUDA_HOME="$IPP6_TOOLKIT_DIR/cuda-$IPP6_CUDA_VERSION"
IPP6_OPENMPI_HOME="/cm/shared/apps/openmpi4/gcc/4.1.5"

IPP6_DOCKER_IMAGE_DIR="$IPP6_TOOLKIT_DIR/docker_sqsh"

IPP6_PLANNED_RESERVED="Skip"

IPP6_NCCL_SOCKET_IFNAME=""
if [ -n "$SLURM_JOB_NODELIST" ] && [[ ! "$SLURM_JOB_NODELIST" =~ worker ]]; then
    # Dynamically get the N/S interface from Slurm allocation.
    # If the nodelist has a 'worker' node, then this is not applicable because the cpuonly partition was explicitly requested.
    # This assumes that the cluster config file is being sourced in after the allocation is granted
    nodename=$(srun hostname -s | head -n1)
    IPP6_NCCL_SOCKET_IFNAME=$(ssh $nodename 'ip route get 8.8.8.8' | sed -E 's/.*?dev (\S+) .*/\1/;t;d')
fi

# Target configs
function get_gpu_archs() {
    echo "$IPP6_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$IPP6_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$IPP6_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$IPP6_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_num_build_procs() {
    # ignore on ipp6
    hostname="$1"

    echo "nproc"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    echo "srun \
        -p cpuonly \
        -J nccl:build \
        -t 00:30:00 \
        -n 1 \
        --exclusive \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_cuda_home() {
    echo "$IPP6_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$IPP6_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$IPP6_CUDA_HOME/lib64"
}

function get_extra_path() {
    echo "$IPP6_CUDA_HOME/bin"
}

function get_planned_reserved() {
    echo "$IPP6_PLANNED_RESERVED"
}

function get_nccl_socket_ifname() {
    echo "$IPP6_NCCL_SOCKET_IFNAME"
}

function get_mpi_params() {
    echo "--oversubscribe --bind-to none --mca btl tcp,self --mca btl_tcp_if_include $IPP6_NCCL_SOCKET_IFNAME"
}

function configure_test_env() {
    export UCX_NET_DEVICES=$IPP6_NCCL_SOCKET_IFNAME
    export UCX_TLS=tcp

    # Note: These are the E/W IB interfaces on Pre-Nyx
    # export NCCL_IB_HCA="=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15"
}
