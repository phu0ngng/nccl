# config parameters for Draco-RNO cluster
# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
DO_OPENMPI_HOME="/usr/mpi/gcc/openmpi-4.1.2a1"
DO_CUDA_HOME="/lustre/fsw/portfolios/coreai/users/vkhodel/local/cuda-12.0.1"
DO_DOCKER_IMAGE_DIR="/lustre/fsw/portfolios/coreai/users/vkhodel/images"

OS="ubuntu20.04"
CUDA_VERSION="12.0.1"

DO_SLURM_ACCOUNT="coreai_libraries_nccl"

RUN_TOOLS_VERSION="1.0.5"
RUN_TOOLS_IMAGE_NAME="nccl_run_tools-${RUN_TOOLS_VERSION}-${CUDA_VERSION}-${OS}.sqsh"
DO_RUN_TOOLS_IMAGE="$DO_DOCKER_IMAGE_DIR/$RUN_TOOLS_IMAGE_NAME"

BUILD_TOOLS_VERSION="1.0.5"
BUILD_TOOLS_IMAGE_NAME="nccl_build_tools-${BUILD_TOOLS_VERSION}-${CUDA_VERSION}-${OS}.sqsh"
DO_BUILD_TOOLS_IMAGE="$DO_DOCKER_IMAGE_DIR/$BUILD_TOOLS_IMAGE_NAME"

# Target configs
# Draco-OCI only has one type of GPU: A100
function get_nvcc_gencode() {
    echo "-gencode=arch=compute_80,code=sm_80"
}

function get_cuda_home() {
    echo "$DO_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$DO_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$DO_CUDA_HOME/lib64:$DR_OPENMPI_HOME/lib"
}

function get_slurm_account() {
    echo "$DO_SLURM_ACCOUNT"
}

function get_run_tools_image() {
    echo "$DO_RUN_TOOLS_IMAGE"
}

# Build host configs
function get_build_tools_image() {
    echo "$DO_BUILD_TOOLS_IMAGE"
}

function get_docker_job_command() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    current_dir="$1"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$DO_SLURM_ACCOUNT \
        -J ${DO_SLURM_ACCOUNT}-nccl:test \
	-p batch_block1 \
        -t 00:20:00 \
        -n 1 \
	-c 48 \
        --container-image=$DO_BUILD_TOOLS_IMAGE \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}
