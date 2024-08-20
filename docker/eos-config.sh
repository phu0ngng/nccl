# config parameters for EOS cluster
# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
EOS_OPENMPI_HOME="/usr/mpi/gcc/openmpi-4.1.5rc2"
EOS_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-12.0.1"
EOS_DOCKER_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"

OS_VERSION="20.04"
CUDA_VERSION="12.0.1"

EOS_SLURM_ACCOUNT="coreai_libraries_nccl"

RUN_TOOLS_VERSION="1.0.5"
RUN_TOOLS_IMAGE_NAME="nccl_run_tools-${RUN_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}.sqsh"
EOS_RUN_TOOLS_IMAGE="$EOS_DOCKER_IMAGE_DIR/$RUN_TOOLS_IMAGE_NAME"

BUILD_TOOLS_VERSION="1.0.5"
BUILD_TOOLS_IMAGE_NAME="nccl_build_tools-${BUILD_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}.sqsh"
EOS_BUILD_TOOLS_IMAGE="$EOS_DOCKER_IMAGE_DIR/$BUILD_TOOLS_IMAGE_NAME"

# Target configs
# EOS only has one type of GPU: H100
function get_nvcc_gencode() {
    echo "-gencode=arch=compute_90,code=sm_90"
}

function get_cuda_home() {
	echo "$EOS_CUDA_HOME"
}

function get_openmpi_home() {
	echo "$EOS_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$EOS_CUDA_HOME/lib64:$EOS_OPENMPI_HOME/lib"
}

function get_slurm_account() {
    echo "$EOS_SLURM_ACCOUNT"
}

function get_run_tools_image() {
    echo "$EOS_RUN_TOOLS_IMAGE"
}

# Build host configs
function get_build_tools_image() {
    echo "$EOS_BUILD_TOOLS_IMAGE"
}

function get_docker_job_command() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    current_dir="$1"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$EOS_SLURM_ACCOUNT \
        -J ${EOS_SLURM_ACCOUNT}-nccl:test \
        -t 00:05:00 \
        -n 1 \
	-c 48 \
        --container-image=$EOS_BUILD_TOOLS_IMAGE \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}
