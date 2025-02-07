# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for EOS cluster
#
# Targeting build to run on EOS
EOS_GPU_ARCHS="90"

EOS_OS_VERSION="20.04"
EOS_CUDA_VERSION="12.6.1"
EOS_BUILD_TOOLS_VERSION="1.0.1"
EOS_BUILD_IMAGE_VERSION="${EOS_BUILD_TOOLS_VERSION}-c${EOS_CUDA_VERSION}-u${EOS_OS_VERSION}"

EOS_OPENMPI_VERSION="4.1.5rc2"

# Parameters for executing tasks on EOS
EOS_DOCKER_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"
EOS_SLURM_ACCOUNT="coreai_libraries_nccl"

# Relevant paths for running tests on EOS on bare metal
EOS_OPENMPI_HOME="/usr/mpi/gcc/openmpi-${EOS_OPENMPI_VERSION}"
EOS_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${EOS_CUDA_VERSION}"

# Running tests on EOS in a run container
EOS_RUN_TOOLS_VERSION="1.0.1"
EOS_RUN_IMAGE_VERSION="${EOS_RUN_TOOLS_VERSION}-c${EOS_CUDA_VERSION}-u${EOS_OS_VERSION}"

EOS_NCCL_SOCKET_IFNAME="eth3"
EOS_MPI_PARAMS="-mca btl tcp,self --mca btl_tcp_if_include $EOS_NCCL_SOCKET_IFNAME"
EOS_NCCL_IB_SL="1"
EOS_PLANNED_RESERVED="Planned"

# Target configs
# EOS only has one type of GPU: H100
function get_gpu_archs() {
    echo "$EOS_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$EOS_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$EOS_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$EOS_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_slurm_account() {
    echo "$EOS_SLURM_ACCOUNT"
}

function get_docker_job_command() {
    # ignore the arg
    echo "nproc"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"
    build_tools_image="$(get_build_tools_image $build_image_version)"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$EOS_SLURM_ACCOUNT \
        -J ${EOS_SLURM_ACCOUNT}-nccl:test \
        -t 00:05:00 \
        -n 1 \
	-c 48 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
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

function get_mpi_params() {
    echo "$EOS_MPI_PARAMS"
}

function get_extra_path() {
    echo "$EOS_CUDA_HOME/bin:$EOS_OPENMPI_HOME/bin"
}

function get_nccl_socket_ifname() {
    echo "$EOS_NCCL_SOCKET_IFNAME"
}

function get_nccl_ib_sl() {
    echo "$EOS_NCCL_IB_SL"
}

function get_planned_reserved() {
    echo "$EOS_PLANNED_RESERVED"
}
