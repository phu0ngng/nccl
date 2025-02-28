# Config parameters for gc cluster
# No host-specific code executes in this module to allow it to be served over http in the future
#
# Targeting build to run in gc environment
GC_GPU_ARCHS="60,70,80"

GC_OS_VERSION="20.04"
GC_CUDA_VERSION="12.6.1"
GC_BUILD_TOOLS_VERSION="1.0.1"
GC_BUILD_IMAGE_VERSION="${GC_BUILD_TOOLS_VERSION}-c${GC_CUDA_VERSION}-u${GC_OS_VERSION}"

# Using gc
#   Parameters for building on gc
DOCKER_BUILD_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_build_tools"

#   Relevant paths for bare metal
GC_OPENMPI_HOME="/home/nvshmem_shared/openmpi"
GC_CUDA_HOME="/usr/local/cuda"

#   Parameters for running in a container
DOCKER_RUN_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_run_tools"
GC_RUN_TOOLS_VERSION="1.0.1"
GC_RUN_IMAGE_VERSION="${GC_RUN_TOOLS_VERSION}-c${GC_CUDA_VERSION}-u${GC_OS_VERSION}"


# Target configs
function get_gpu_archs() {
    echo "$GC_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$GC_BUILD_IMAGE_VERSION"
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

# be nice on gc00
function get_num_build_procs() {
    hostname="$1"

    if [ "$hostname" = "gc00" ]; then
        echo "echo 8"
    else
	echo "nproc"
    fi
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    # pass in relevant env vars
    echo "docker run --rm \
        -e NVCC_GENCODE \
        -e NUM_BUILD_PROCS \
        --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
        -v ${current_dir}:/nccl \
        $build_tools_image \
        /nccl/docker/build_nccl.sh"
}

function get_cuda_home() {
    echo "$GC_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$GC_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$GC_CUDA_HOME/lib64:$GC_OPENMPI_HOME/lib"
}

