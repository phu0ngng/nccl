# config parameters for Draco-RNO cluster
# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
DR_OPENMPI_HOME="/usr/mpi/gcc/openmpi-4.1.0rc5"
DR_CUDA_HOME="/gpfs/fs1/projects/sw_gpucomms/users/vkhodel/cuda-12.0.1"
DR_DOCKER_IMAGE_DIR="/gpfs/fs1/projects/sw_gpucomms/users/vkhodel/images"

OS="ubuntu20.04"
CUDA_VERSION="12.0.1"

DR_SLURM_ACCOUNT="coreai_libraries_nccl"

RUN_TOOLS_VERSION="1.0.5"
RUN_TOOLS_IMAGE_NAME="nccl_run_tools-${RUN_TOOLS_VERSION}-${CUDA_VERSION}-${OS}.sqsh"
DR_RUN_TOOLS_IMAGE="$DR_DOCKER_IMAGE_DIR/$RUN_TOOLS_IMAGE_NAME"

BUILD_TOOLS_VERSION="1.0.5"
BUILD_TOOLS_IMAGE_NAME="nccl_build_tools-${BUILD_TOOLS_VERSION}-${CUDA_VERSION}-${OS}.sqsh"
DR_BUILD_TOOLS_IMAGE="$DR_DOCKER_IMAGE_DIR/$BUILD_TOOLS_IMAGE_NAME"

# Target configs
# Draco-RNO only has one type of GPU: V100
function get_nvcc_gencode() {
    echo "-gencode=arch=compute_70,code=sm_70"
}

function get_cuda_home() {
    echo "$DR_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$DR_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$DR_CUDA_HOME/lib64:$DR_OPENMPI_HOME/lib"
}

function get_slurm_account() {
    echo "$DR_SLURM_ACCOUNT"
}

function get_run_tools_image() {
    echo "$DR_RUN_TOOLS_IMAGE"
}

# Build host configs
function get_build_tools_image() {
    echo "$DR_BUILD_TOOLS_IMAGE"
}

function get_docker_job_command() {
    # ignore the arg
    # echo "nproc" - this returns 2 in srun!
    echo "echo 48"
}

function get_build_command() {
    current_dir="$1"

    echo "srun \
        --account=$DR_SLURM_ACCOUNT \
        -J ${DR_SLURM_ACCOUNT}-nccl:test \
        --nv-meta ml-model.dlss \
        --mpi=pmix \
        -t 00:20:00 \
        -p batch_dgx1_m2 \
        -n 1 \
        --container-image=$DR_BUILD_TOOLS_IMAGE \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}
