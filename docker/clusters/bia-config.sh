# No host-specific code executes in this module to allow it to live elsewhere
#
# Config parameters for BIA cluster
#
# Targeting build to run on BIA

# BIA GPU Type: B300
BIA_GPU_ARCHS="103"

## CUDA Config
BIA_CUDA_MAJOR_VERSION="13"
BIA_CUDA_MINOR_VERSION="0"
BIA_CUDA_VERSION="${BIA_CUDA_MAJOR_VERSION}.${BIA_CUDA_MINOR_VERSION}"
BIA_CUDA_HOME="/lustre/fsw/coreai_libraries_nccl/toolkits/cuda-${BIA_CUDA_VERSION}"

# Container build config
# Note: This doesn't work at the moment as the CTK in the image doesn't recognize gencode 103
BIA_BUILD_TOOLS_VERSION="2.0.0"
BIA_BUILD_TOOLS_CUDA_VERSION="13.0.2"
BIA_BUILD_TOOLS_OS_VERSION="24.04"
BIA_BUILD_TOOLS_IMAGE_VERSION="${BIA_BUILD_TOOLS_VERSION}-c${BIA_BUILD_TOOLS_CUDA_VERSION}-u${BIA_BUILD_TOOLS_OS_VERSION}"
BIA_BUILD_TOOLS_IMAGE_DIR="/lustre/fsw/coreai_libraries_nccl/toolkits/docker_sqsh"

# Slurm account for executing jobs
BIA_SLURM_ACCOUNT="coreai_libraries_nccl"

# Use with sacct on to get Slurm wait times
BIA_PLANNED_RESERVED="Planned"

BIA_NCCL_SOCKET_IFNAME="enP6p3s0f1np1"

# Target configs
function get_gpu_archs() {
    echo "$BIA_GPU_ARCHS"
}

function get_slurm_account() {
    echo "$BIA_SLURM_ACCOUNT"
}

function get_num_build_procs() {
    # ignore the arg
    echo "nproc"
}

function get_build_image_version() {
    echo ${BIA_BUILD_TOOLS_IMAGE_VERSION}
}

function get_build_tools_image() {
    build_image_version="$1"
    echo "$BIA_BUILD_TOOLS_IMAGE_DIR/nccl_build_tools-${build_image_version}.sqsh"
}

function get_build_command_bm() {
    echo "srun \
            --account=$BIA_SLURM_ACCOUNT \
            -J $BIA_SLURM_ACCOUNT-ci:nccl-build \
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
        --account=$BIA_SLURM_ACCOUNT \
        -J ${BIA_SLURM_ACCOUNT}-ci:nccl-build \
        -p batch \
        -t 00:10:00 \
        -n 1 \
        -c 64 \
        --container-image=$build_tools_image \
        --container-mounts=${current_dir}:/nccl \
        /nccl/docker/build_nccl.sh --enable-ccache"
}

function get_cuda_home() {
    echo "$BIA_CUDA_HOME"
}

function get_ccache_bin() {
    echo "/lustre/fsw/coreai_libraries_nccl/toolkits/ccache-4.12.1-linux-x86_64/ccache"
}

function get_extra_ld_library_path() {
    echo "$BIA_CUDA_HOME/lib64"
}

function get_extra_path() {
    echo "$BIA_CUDA_HOME/bin"
}


## MPI config
BIA_HPCX_TOOLKIT_VERSION="2.24.1"
BIA_HPCX_TOOLKIT="/lustre/fsw/coreai_libraries_nccl/toolkits/hpcx-v${BIA_HPCX_TOOLKIT_VERSION}-gcc-doca_ofed-ubuntu24.04-cuda${BIA_CUDA_MAJOR_VERSION}-x86_64"

# Call utility function to set MPI env config
source $BIA_HPCX_TOOLKIT/hpcx-mt-init.sh && hpcx_load

function get_openmpi_home() {
    # $MPI_HOME is set when `hpcx_load` is called
    echo "$MPI_HOME"
}

function get_mpi_params() {
    # Expose MPI params as test env variables
    echo ""
}


function get_planned_reserved() {
    echo "$BIA_PLANNED_RESERVED"
}

function configure_test_env() {
    export UCX_TLS="tcp,self"

    # MPI params
    export OMPI_MCA_coll_hcoll_enable=0
    
    # NCCL params
    export NCCL_IB_TC=96
    export NCCL_NET_PLUGIN=none
    export NCCL_IB_ADAPTIVE_ROUTING=1
    export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_10,mlx5_11,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_20,mlx5_21,mlx5_22,mlx5_23
    
    # export NCCL_GDRCOPY_ENABLE=1
    # export NCCL_GDRCOPY_SYNC_ENABLE=1
}

function get_cluster_name() {
    echo "Bia"
}

function get_gcperf_tools_path() {
    echo "/lustre/fsw/coreai_libraries_nccl/toolkits/gcperf-tools"
}
