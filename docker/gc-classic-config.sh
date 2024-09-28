# No host-specific code executes in this module to allow it to be served via web
#
# Config parameters for gc cluster used in a classic gc00-based CI/CD flow:
# --build for the oldest supported CUDA/MPI versions
# --run on bare metal

# Targeting build to run in classic gc environment
GCCL_GPU_ARCHS="60,70,80"

GCCL_OS_VERSION="20.04"
GCCL_CUDA_VERSION="11.7.1"
GCCL_BUILD_TOOLS_VERSION="1.0.1"
GCCL_BUILD_IMAGE_VERSION="${GCCL_BUILD_TOOLS_VERSION}-c${GCCL_CUDA_VERSION}-u${GCCL_OS_VERSION}"

# Also build container uses this openmpi
GCCL_OPENMPI_VERSION="1.10.7"


# Using gc
#   Parameters for building on gc
DOCKER_BUILD_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_build_tools"

#   Relevant paths for bare metal
GCCL_OPENMPI_HOME="/opt/mpi/openmpi-${GCCL_OPENMPI_VERSION}"
GCCL_CUDA_HOME="/usr/local/cuda-${GCCL_CUDA_VERSION}"

#   Parameters for running in a container (unused)
# DOCKER_RUN_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools/nccl_run_tools"
# GCCL_RUN_IMAGE_VERSION="${GCCL_RUN_TOOLS_VERSION}-c${GCCL_CUDA_VERSION}-u${GCCL_OS_VERSION}"


# Target configs
function get_gpu_archs() {
    echo "$GCCL_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$GCCL_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "${DOCKER_BUILD_TOOLS_REPO}:$build_image_version"
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
    build_image_version="$2"

    build_tools_image="$(get_build_tools_image $build_image_version)"

    # pass in relevant env vars
    echo "docker run --rm \
        -e NVCC_GENCODE \
        -e DOCKER_JOB_COMMAND \
        -e DOCKER_USER_ID \
        -e DOCKER_GROUP_ID \
        -v ${current_dir}:/nccl \
        $build_tools_image \
        /nccl/docker/build_nccl.sh"
}

function get_cuda_home() {
    echo "$GCCL_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$GCCL_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$GCCL_CUDA_HOME/lib64:$GCCL_OPENMPI_HOME/lib"
}
