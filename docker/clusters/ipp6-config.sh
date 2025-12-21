# Config parameters for ipp6 cluster
# No host-specific code executes in this module to allow it to be served over http in the future
#
# Targeting build to run in ipp6 environment (A100/A40/L40)
IPP6_GPU_ARCHS="80,86,89"

IPP6_OS_VERSION="20.04"
IPP6_CUDA_VERSION="12.8.0"
IPP6_BUILD_TOOLS_VERSION="2.1.1"
IPP6_BUILD_IMAGE_VERSION="${IPP6_BUILD_TOOLS_VERSION}-c${IPP6_CUDA_VERSION}-u${IPP6_OS_VERSION}"

#   Parameters for building on ipp6
DOCKER_BUILD_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools"

#   Relevant paths for bare metal
IPP6_TOOLKIT_DIR="/nfs_mnt1/toolkits"
IPP6_OPENMPI_HOME="$IPP6_TOOLKIT_DIR/openmpi"
IPP6_CUDA_HOME="$IPP6_TOOLKIT_DIR/cuda-$IPP6_CUDA_VERSION"

#   Parameters for running in a container
DOCKER_RUN_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_run_tools"
IPP6_RUN_TOOLS_VERSION="1.0.1"
IPP6_RUN_IMAGE_VERSION="${IPP6_RUN_TOOLS_VERSION}-c${IPP6_CUDA_VERSION}-u${IPP6_OS_VERSION}"


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
    echo "${DOCKER_BUILD_TOOLS_REPO}:$build_image_version"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "${DOCKER_RUN_TOOLS_REPO}:$run_image_version"
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

    # pass in relevant env vars
    echo "docker run --rm \
        -e CI_BUILD \
        -e NCCL_BUILD_TRACE \
        -e NVCC_GENCODE \
        -e NUM_BUILD_PROCS \
        --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
        -v ${current_dir}:/nccl \
        $build_tools_image \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_cuda_home() {
    echo "$IPP6_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$IPP6_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$IPP6_CUDA_HOME/lib64:$IPP6_OPENMPI_HOME/lib"
}

