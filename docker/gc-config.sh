# Config parameters for gc cluster
# no host-specific code executes in this module to allow it to live elsewhere
GC_GPU_ARCHS="60,70,80"

# Relevant paths for bare metal
GC_OPENMPI_HOME="/home/nvshmem_shared/openmpi"
GC_CUDA_HOME="/usr/local/cuda"

# Container parameters
DOCKER_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_build_tools"
OS_VERSION="20.04"
CUDA_VERSION="12.6.1"
BUILD_TOOLS_VERSION="1.0.1"
GC_BUILD_TOOLS_IMAGE="${DOCKER_TOOLS_REPO}:${BUILD_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}"

# Target configs
function get_gpu_archs() {
    echo "$GC_GPU_ARCHS"
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

# Build host configs
function get_build_tools_image() {
    echo "${GC_BUILD_TOOLS_IMAGE}"
}

# be nice on gc00
function get_docker_job_command() {
    hostname="$1"

    if [ "$hostname" = "gc00" ]; then
        echo "echo 8"
    else
	echo "nproc"
    fi
}

function get_build_command() {
    current_dir="$1"

    # pass in relevant env vars
    echo "docker run --rm \
        -e NVCC_GENCODE \
        -e DOCKER_JOB_COMMAND \
        -e DOCKER_USER_ID \
        -e DOCKER_GROUP_ID \
        -v ${current_dir}:/nccl \
        ${GC_BUILD_TOOLS_IMAGE} \
        /nccl/docker/build_nccl.sh"
}

