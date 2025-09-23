# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for CW-DFW cluster
#
# Targeting build to run on CW-DFW
CW_DFW_GPU_ARCHS="90"

CW_DFW_OS_VERSION="20.04"
CW_DFW_CUDA_VERSION="12.8.0"
CW_DFW_BUILD_TOOLS_VERSION="1.0.1"
CW_DFW_BUILD_IMAGE_VERSION="${CW_DFW_BUILD_TOOLS_VERSION}-c${CW_DFW_CUDA_VERSION}-u${CW_DFW_OS_VERSION}"

CW_DFW_OPENMPI_VERSION="4.1.4"

# Parameters for executing tasks on CW-DFW
CW_DFW_DOCKER_IMAGE_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/toolkits/docker_sqsh"
CW_DFW_SLURM_ACCOUNT="coreai_libraries_nccl"

# Relevant paths for running tests on CW-DFW on bare metal
CW_DFW_OPENMPI_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/toolkits/openmpi-${CW_DFW_OPENMPI_VERSION}"
CW_DFW_CUDA_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/toolkits/cuda-${CW_DFW_CUDA_VERSION}"

# Running tests on CW-DFW in a run container
CW_DFW_RUN_TOOLS_VERSION="1.0.1"
CW_DFW_RUN_IMAGE_VERSION="${CW_DFW_RUN_TOOLS_VERSION}-c${CW_DFW_CUDA_VERSION}-u${CW_DFW_OS_VERSION}"

CW_DFW_NCCL_SOCKET_IFNAME="enp90s0f0np0"
CW_DFW_MPI_PARAMS="-mca btl tcp,self --mca btl_tcp_if_include $CW_DFW_NCCL_SOCKET_IFNAME"
CW_DFW_NCCL_IB_SL="1"
CW_DFW_PLANNED_RESERVED="Planned"

# Target configs
# CW-DFW only has one type of GPU: H100
function get_gpu_archs() {
    echo "$CW_DFW_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$CW_DFW_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$CW_DFW_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$CW_DFW_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_slurm_account() {
    echo "$CW_DFW_SLURM_ACCOUNT"
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
        --account=$CW_DFW_SLURM_ACCOUNT \
        -J ${CW_DFW_SLURM_ACCOUNT}-nccl:test \
        -t 00:05:00 \
        -n 1 \
	-c 48 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}

function get_cuda_home() {
    echo "$CW_DFW_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$CW_DFW_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$CW_DFW_CUDA_HOME/lib64:$CW_DFW_OPENMPI_HOME/lib"
}

function get_mpi_params() {
    echo "$CW_DFW_MPI_PARAMS"
}

function get_extra_path() {
    echo "$CW_DFW_CUDA_HOME/bin:$CW_DFW_OPENMPI_HOME/bin"
}

function get_nccl_socket_ifname() {
    echo "$CW_DFW_NCCL_SOCKET_IFNAME"
}

function get_nccl_ib_sl() {
    echo "$CW_DFW_NCCL_IB_SL"
}

function get_planned_reserved() {
    echo "$CW_DFW_PLANNED_RESERVED"
}

function configure_test_env() {
    echo ""
}

function get_cluster_name() {
    echo "CW-DFW"
}

function get_gcperf_tools_path() {
    echo "/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/toolkits/gcperf-tools"
}
