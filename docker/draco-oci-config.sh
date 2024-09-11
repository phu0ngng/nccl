# config parameters for Draco-RNO cluster
# no host-specific code executes in this module to allow it to live elsewhere

# Relevant paths
DO_OPENMPI_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/local/openmpi-4.1.4"
DO_CUDA_HOME="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/local/cuda-12.0.1"
DO_DOCKER_IMAGE_DIR="/lustre/fsw/portfolios/coreai/projects/coreai_libraries_nccl/docker_sqsh"

OS_VERSION="20.04"
CUDA_VERSION="12.0.1"

DO_SLURM_ACCOUNT="coreai_libraries_nccl"

RUN_TOOLS_VERSION="1.0.5"
RUN_TOOLS_IMAGE_NAME="nccl_run_tools-${RUN_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}.sqsh"
DO_RUN_TOOLS_IMAGE="$DO_DOCKER_IMAGE_DIR/$RUN_TOOLS_IMAGE_NAME"

BUILD_TOOLS_VERSION="1.0.5"
BUILD_TOOLS_IMAGE_NAME="nccl_build_tools-${BUILD_TOOLS_VERSION}-c${CUDA_VERSION}-u${OS_VERSION}.sqsh"
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
    echo "$DO_CUDA_HOME/lib64:$DO_OPENMPI_HOME/lib"
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
        -t 00:05:00 \
        -n 1 \
	-c 48 \
        --container-image=$DO_BUILD_TOOLS_IMAGE \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}

function configure_test_env() {
    export CUDA_HOME=$(get_cuda_home)
    export MPI_HOME=$(get_openmpi_home)

    # Slurm account to use to submit tests
    export SLURM_ACCOUNT=$(get_slurm_account)

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
    export OMPI_MCA_pml="ucx"
    export OMPI_MCA_coll="^hcoll"
    export OMPI_MCA_coll_hcoll_enable=0
    export RX_QUEUE_LEN=8192
}
