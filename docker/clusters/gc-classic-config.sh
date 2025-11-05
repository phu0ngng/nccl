# No host-specific code executes in this module to allow it to be served via web
#
# Config parameters for gc cluster used in a classic gc00-based CI/CD flow:
# --build for the oldest supported CUDA/MPI versions
# --run on bare metal
# --this is not used as config for a build cluster, only as a build target

# Targeting build to run in classic gc environment
GCCL_GPU_ARCHS="60,70,80"

GCCL_OS_VERSION="20.04"
GCCL_CUDA_VERSION="11.7.1"
GCCL_BUILD_TOOLS_VERSION="2.0.0"
GCCL_BUILD_IMAGE_VERSION="${GCCL_BUILD_TOOLS_VERSION}-c${GCCL_CUDA_VERSION}-u${GCCL_OS_VERSION}"

# Also build container uses this openmpi
GCCL_OPENMPI_VERSION="1.10.7"
GCCL_PLANNED_RESERVED="Reserved"

# Using gc
#   Parameters for building on gc
DOCKER_BUILD_TOOLS_REPO="gitlab-master.nvidia.com:5005/gpucomms/nccl_docker_tools"

#   Relevant paths for bare metal
GCCL_OPENMPI_HOME="/opt/mpi/openmpi-${GCCL_OPENMPI_VERSION}"
GCCL_ACTUAL_CUDA_VERSION="11.7"
GCCL_CUDA_HOME="/usr/local/cuda-${GCCL_ACTUAL_CUDA_VERSION}"

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

function get_cuda_home() {
    echo "$GCCL_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$GCCL_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$GCCL_CUDA_HOME/lib64:$GCCL_OPENMPI_HOME/lib"
}

function get_mpi_params() {
    echo "--mca btl ^openib --mca shmem_mmap_enable_nfs_warning 0"
}

function get_extra_path() {
    echo "$GCCL_CUDA_HOME/bin:$GCCL_OPENMPI_HOME/bin"
}

function get_planned_reserved() {
    echo "$GCCL_PLANNED_RESERVED"
}

function configure_test_env() {
    echo ""
}
