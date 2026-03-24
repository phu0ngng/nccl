# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for POLYPHE cluster
#
# Targeting build to run on POLYPHE

# POLYPHE GPU Type: GB200
POLYPHE_GPU_ARCHS="100"

## CUDA Config
POLYPHE_CUDA_MAJOR_VERSION="13"
POLYPHE_CUDA_MINOR_VERSION="1"
POLYPHE_CUDA_VERSION="${POLYPHE_CUDA_MAJOR_VERSION}.${POLYPHE_CUDA_MINOR_VERSION}"
POLYPHE_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${POLYPHE_CUDA_VERSION}"

# Container build config
# Note: This doesn't work at the moment as the CTK in the image doesn't recognize gencode 103
POLYPHE_BUILD_TOOLS_VERSION="2.0.0"
POLYPHE_BUILD_TOOLS_CUDA_VERSION="13.0.2"
POLYPHE_BUILD_TOOLS_OS_VERSION="24.04"
POLYPHE_BUILD_TOOLS_IMAGE_VERSION="${POLYPHE_BUILD_TOOLS_VERSION}-c${POLYPHE_BUILD_TOOLS_CUDA_VERSION}-u${POLYPHE_BUILD_TOOLS_OS_VERSION}"
POLYPHE_BUILD_TOOLS_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"

# Slurm account for executing jobs
POLYPHE_SLURM_ACCOUNT="coreai_libraries_nccl"

# Use with sacct on to get Slurm wait times
POLYPHE_PLANNED_RESERVED="Planned"

POLYPHE_NCCL_SOCKET_IFNAME="enP6p3s0f1np1"

# Target configs
function get_gpu_archs() {
    echo "$POLYPHE_GPU_ARCHS"
}

function get_slurm_account() {
    echo "$POLYPHE_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_image_version() {
    echo ${POLYPHE_BUILD_TOOLS_IMAGE_VERSION}
}

function get_build_tools_image() {
    build_image_version="$1"
    echo "$POLYPHE_BUILD_TOOLS_IMAGE_DIR/nccl_build_tools-${build_image_version}_arm.sqsh"
}

function get_build_command_bm() {
    echo "srun \
            --account=$POLYPHE_SLURM_ACCOUNT \
            -J $POLYPHE_SLURM_ACCOUNT-ci:nccl-build \
            -t 00:10:00 \
            -n 1 \
            -c 64 \
            -p batch \
            docker/build_nccl_bm.sh"
}

function get_build_command() {
    current_dir="$1"
    build_image_version="$2"
    build_tools_image="$(get_build_tools_image $build_image_version)"

    # ask for a lot of cores - otherwise we get 2
    echo "srun \
        --account=$POLYPHE_SLURM_ACCOUNT \
        -J ${POLYPHE_SLURM_ACCOUNT}-ci:nccl-build \
        -p batch \
        -t 00:10:00 \
        -n 1 \
        -c 64 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_cuda_home() {
    echo "$POLYPHE_CUDA_HOME"
}

function get_ccache_bin() {
    echo "/lustre/fsw/coreai_libraries_nccl/toolkits/ccache-4.12.1-linux-aarch64/ccache"
}

function get_extra_ld_library_path() {
    echo "$POLYPHE_CUDA_HOME/lib64"
}

function get_extra_path() {
    echo "$POLYPHE_CUDA_HOME/bin"
}


## MPI config
POLYPHE_HPCX_TOOLKIT_VERSION="2.24"
POLYPHE_HPCX_TOOLKIT="/lustre/fsw/coreai_libraries_nccl/toolkits/hpcx-v${POLYPHE_HPCX_TOOLKIT_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda${POLYPHE_CUDA_MAJOR_VERSION}-aarch64"

# Call utility function to set MPI env config
source $POLYPHE_HPCX_TOOLKIT/hpcx-mt-init.sh && hpcx_load

function get_openmpi_home() {
    # $MPI_HOME is set when `hpcx_load` is called
    echo "$MPI_HOME"
}

function get_mpi_params() {
    # Expose MPI params as test env variables
    echo ""
}


function get_planned_reserved() {
    echo "$POLYPHE_PLANNED_RESERVED"
}

function configure_test_env() {
    export UCX_TLS="tcp,self"

    # MPI params
    export OMPI_MCA_coll_hcoll_enable=0

    # NCCL params
    export NCCL_IB_TC=96
    export NCCL_NET_PLUGIN=none
    export NCCL_IB_ADAPTIVE_ROUTING=1
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5
    # WAR for failing GIN/DOCA on Polyphe (same as Theia)
    export NCCL_GIN_TYPE=2

    # export NCCL_GDRCOPY_ENABLE=1
    # export NCCL_GDRCOPY_SYNC_ENABLE=1
}

function get_cluster_name() {
    echo "Polyphe"
}

function get_gcperf_tools_path() {
    echo "/lustre/fsw/coreai_libraries_nccl/toolkits/gcperf-tools"
}
