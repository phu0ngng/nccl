# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for Lyris cluster
#
# Targeting build to run on Theia subsystem nodes in Lyris

## Lyris Cluster Slurm Config
THEIA_SLURM_ACCOUNT="coreai_libraries_nccl"

function get_slurm_account() {
    echo "$THEIA_SLURM_ACCOUNT"
}

# Use with sacct on to get Slurm wait times
THEIA_PLANNED_RESERVED="Planned"

function get_planned_reserved() {
    echo "$THEIA_PLANNED_RESERVED"
}

## CUDA Config
THEIA_CUDA_MAJOR_VERSION="12"
THEIA_CUDA_MINOR_VERSION="9"
THEIA_CUDA_VERSION="${THEIA_CUDA_MAJOR_VERSION}.${THEIA_CUDA_MINOR_VERSION}"
THEIA_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${THEIA_CUDA_VERSION}"

function get_cuda_home() {
    echo "$THEIA_CUDA_HOME"
}

function get_extra_ld_library_path() {
    echo "$THEIA_CUDA_HOME/lib64"
}

function get_extra_path() {
    echo "$THEIA_CUDA_HOME/bin"
}

## MPI config
THEIA_HPCX_TOOLKIT_VERSION="2.23"
THEIA_HPCX_TOOLKIT="/lustre/fsw/coreai_libraries_nccl/toolkits/hpcx-v${THEIA_HPCX_TOOLKIT_VERSION}-cuda${THEIA_CUDA_MAJOR_VERSION}"

# Call utility function to set MPI env config
source $THEIA_HPCX_TOOLKIT/hpcx-mt-init.sh && hpcx_load

function get_openmpi_home() {
    # $MPI_HOME is set when `hpcx_load` is called
    echo "$MPI_HOME"
}

function get_mpi_params() {
    # This is equivalent to a no-op
    # Expose MPI params as test env variables
    echo ""
}

# Build configs
# Theia GPU Type: GB300, Gencode: 103
# https://confluence.nvidia.com/display/DSE/Lyris+Hardware
THEIA_GPU_ARCHS="103"

function get_gpu_archs() {
    echo "$THEIA_GPU_ARCHS"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

# Baremetal build config
function get_build_command_bm() {
    echo "srun \
        --account=$THEIA_SLURM_ACCOUNT \
        -J $THEIA_SLURM_ACCOUNT-ci.nccl-build \
        -t 00:10:00 \
        -n 1 \
        -c 64 \
        -p gb300 \
        docker/build_nccl_bm.sh"
}

# Container build config
# Note: This doesn't work at the moment as the CTK in the image doesn't recognize gencode 103
THEIA_BUILD_TOOLS_VERSION="1.0.3"
THEIA_BUILD_TOOLS_CUDA_VERSION="12.8.0"
THEIA_BUILD_TOOLS_OS_VERSION="20.04"
THEIA_BUILD_TOOLS_IMAGE_VERSION="${THEIA_BUILD_TOOLS_VERSION}-c${THEIA_BUILD_TOOLS_CUDA_VERSION}-u${THEIA_BUILD_TOOLS_OS_VERSION}"
THEIA_BUILD_TOOLS_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"

function get_build_image_version() {
    echo ${THEIA_BUILD_TOOLS_IMAGE_VERSION}
}

function get_build_tools_image() {
    build_image_version="$1"
    echo "$THEIA_BUILD_TOOLS_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"
    build_tools_image="$(get_build_tools_image $build_image_version)"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$THEIA_SLURM_ACCOUNT \
        -J ${THEIA_SLURM_ACCOUNT}-ci.nccl-build \
        -p gb300 \
        -t 00:10:00 \
        -n 1 \
        -c 64 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh"
}

# Test configs
function configure_test_env() {
    export UCX_TLS="tcp,self"

    # NCCL things
    # E/W interfaces on Theia nodes
    export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3"
    export NCCL_IB_SL=1
    # Disable the usage of IB-Sharp (for CX8 support, HPCX-v2.23 doesn't support it)
    export NCCL_NET_PLUGIN=none

    # MPI params
    export OMPI_MCA_coll_hcoll_enable=0
}
