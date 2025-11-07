# Config parameters for Draco-OCI cluster
# No host-specific code executes in this module to allow it to live elsewhere

# Targeting build to run on Draco-OCI
DO_GPU_ARCHS="80"

DO_OS_VERSION="20.04"
DO_CUDA_VERSION="12.8.0"
DO_BUILD_TOOLS_VERSION="2.0.0"
DO_BUILD_IMAGE_VERSION="${DO_BUILD_TOOLS_VERSION}-c${DO_CUDA_VERSION}-u${DO_OS_VERSION}"

DO_OPENMPI_VERSION="4.1.4"

# Parameters for executing tasks on Draco-OCI
DO_DOCKER_IMAGE_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/docker_sqsh"
DO_SLURM_ACCOUNT="coreai_libraries_nccl"

# Relevant paths for running test on bare metal
DO_OPENMPI_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/local/openmpi-${DO_OPENMPI_VERSION}"
DO_CUDA_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/local/cuda-${DO_CUDA_VERSION}"

# Running test in run container
DO_RUN_TOOLS_VERSION="1.0.1"
DO_RUN_TOOLS_IMAGE_VERSION="${DO_RUN_TOOLS_VERSION}-c${DO_CUDA_VERSION}-u${DO_OS_VERSION}"
DO_PLANNED_RESERVED="Planned"
DO_MPI_PARAMS="-mca btl tcp,self"

# Target configs
# Draco-OCI only has one type of GPU: A100
function get_gpu_archs() {
    echo "$DO_GPU_ARCHS"
}

function get_build_image_version() {
    echo "$DO_BUILD_IMAGE_VERSION"
}

# Build host configs
function get_build_tools_image() {
    build_image_version="$1"
    echo "$DO_DOCKER_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_run_tools_image() {
    run_image_version="$1"
    echo "$DO_DOCKER_IMAGE_DIR/nccl_run_tools-${run_image_version}.sqsh"
}

function get_slurm_account() {
    echo "$DO_SLURM_ACCOUNT"
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
        --account=$DO_SLURM_ACCOUNT \
        -J ${DO_SLURM_ACCOUNT}-nccl:test \
	-p batch_block1 \
        -t 00:05:00 \
        -n 1 \
	-c 48 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}

function get_cuda_home() {
    echo "$DO_CUDA_HOME"
}

function get_openmpi_home() {
    echo "$DO_OPENMPI_HOME"
}

function get_extra_ld_library_path() {
    echo "$DO_CUDA_HOME/lib64:$DO_OPENMPI_HOME/lib"
}

function configure_test_env() {
    export CUDA_HOME=$(get_cuda_home)
    export MPI_HOME=$(get_openmpi_home)

    # Slurm account to use to submit tests
    export SLURM_ACCOUNT=$(get_slurm_account)
    #export UCX_TLS=tcp

    # Set oci-iad specific environment variables for NCCL testing
    export HCOLL_ENABLE_MCAST_ALL=0
    export IB_RX_QUEUE_LEN=8192
    export NCCL_IB_TIMEOUT=18
    export NCCL_IB_SL=0
    export NCCL_IB_TC=41
    export NCCL_IGNORE_CPU_AFFINITY=0
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_CROSS_NIC=0
    export NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
    #export OMPI_MCA_pml="ucx"
    export OMPI_MCA_coll="^hcoll"
    export OMPI_MCA_coll_hcoll_enable=0
    export RX_QUEUE_LEN=8192
    export MPIRUN_SKIP_PPN=1
}

function get_planned_reserved() {
    echo "$DO_PLANNED_RESERVED"
}

function get_mpi_params() {
    echo "$DO_MPI_PARAMS"
}

function get_extra_path() {
    echo "$DO_CUDA_HOME/bin:$DO_OPENMPI_HOME/bin"
}
