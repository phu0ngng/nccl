# Config parameters for Draco-RNO cluster
# No host-specific code executes in this module to allow it to live elsewhere

# Targeting build to run on Draco-RNO
DR_GPU_ARCHS="70"

DR_OS_VERSION="20.04"
DR_CUDA_VERSION="12.0.1"
DR_BUILD_TOOLS_VERSION="1.0.5"
DR_BUILD_IMAGE_VERSION="${DR_BUILD_TOOLS_VERSION}-c${DR_CUDA_VERSION}-u${DR_OS_VERSION}"

DR_OPENMPI_VERSION="4.1.0rc5"

# Parameters for executing tasks on Draco-RNO
DR_DOCKER_IMAGE_DIR="/gpfs/fs1/projects/sw_gpucomms/users/vkhodel/docker_sqsh"
DR_SLURM_ACCOUNT="coreai_libraries_nccl"

# Relevant paths for running test on bare metal
DR_OPENMPI_HOME="/usr/mpi/gcc/openmpi-${DR_OPENMPI_VERSION}"
DR_CUDA_HOME="/gpfs/fs1/projects/sw_gpucomms/users/vkhodel/cuda-${DR_CUDA_VERSION}"

# Running test in run container
DR_RUN_TOOLS_VERSION="1.0.5"
DR_RUN_IMAGE_VERSION="${DR_RUN_TOOLS_VERSION}-c${DR_CUDA_VERSION}-u${DR_OS_VERSION}.sqsh"



BUILD_TOOLS_IMAGE_NAME="nccl_build_tools-${BUILD_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}.sqsh"
DR_BUILD_TOOLS_IMAGE="$DR_DOCKER_IMAGE_DIR/$BUILD_TOOLS_IMAGE_NAME"

# Target configs
# Draco-RNO only has one type of GPU: V100
function get_gpu_archs() {
    echo "$DR_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$DR_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$DR_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$DR_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_slurm_account() {
    echo "$DR_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"
    build_tools_image="$(get_build_tools_image $build_image_version)"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$DR_SLURM_ACCOUNT \
        -J ${DR_SLURM_ACCOUNT}-nccl:test \
        --nv-meta ml-model.dlss \
        -t 00:05:00 \
        -p batch_dgx1_m2 \
        -n 1 \
	-c 48 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
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

