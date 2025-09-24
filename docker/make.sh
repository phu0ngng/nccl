#!/bin/bash -x

# Utility script to build NCCL on different clusters
# By default, uses the NCCL build tools containers but also supports baremetal builds.
# provide target cluster name (defaults to the build cluster name)
# identify build machine/cluster automatically
function usage() {
    echo "INFO:  Usage: $0 [ target_cluster_tag ] [ --clean ] [ --baremetal-build ] [ --ci-build ]"
    exit 1
}

if [ ! -d "docker" ]; then
  echo "Please launch from the top of NCCL source tree"
  exit 1
fi

source docker/cluster-utils.sh

function get_nvcc_gencodes() {
    # comma-separated list of numeric gpu_archs
    gpu_archs="$(echo $1 | tr ',' '\n' | sort -u)"

    gencode_string=""
    for gpu_arch in $gpu_archs; do
        gencode_string="$gencode_string -gencode=arch=compute_${gpu_arch},code=sm_${gpu_arch}"
    done
    echo "$gencode_string"
}

# identify build cluster first
build_cluster_tag="$(identify_cluster)"

# arg parsing
# defaults
target_cluster_arg="$build_cluster_tag"
make_clean=0
baremetal_build=0
ci_build=0
use_build_cluster_image=0

for arg in "$@"
do
    case $arg in
        --clean) make_clean=1
                 ;;
        --baremetal-build) baremetal_build=1
                           ;;
        --ci-build) ci_build=1
                 ;;
        --use-build-cluster-image) use_build_cluster_image=1
                                   ;;
        --help|-h) usage
                   ;;
        *) target_cluster_arg="$arg"
           ;;
    esac
done

export CI_BUILD=$ci_build

# process target cluster config
# comma separated list of cluster tags
target_cluster_tags="$(echo $target_cluster_arg | tr ',' ' ')"
gpu_arch_list=""
build_image_version=""

for target_cluster_tag in $target_cluster_tags; do
    source_cluster_config $target_cluster_tag
    if [ -z "$gpu_arch_list" ]; then
        gpu_arch_list="$(get_gpu_archs)"
    else
        gpu_arch_list="$(get_gpu_archs),$gpu_arch_list"
    fi

    # If --use-build-cluster-image is not set, then the build image is pulled from the target cluster(s)
    # If there are multiple targets, the versions must be the exact same to ensure the artifacts are produced correctly for all
    # If --use-build-cluster-image is set, then the build image is pulled from the build cluster, and the target cluster build images are ignored 
    # This should be used cautiously as it may result in artifacts being incompatible with the target clusters, but is useful if the build cluster has a newer image than the target clusters
    if [ "$baremetal_build" -eq 0 ] && [ "$use_build_cluster_image" -eq 0 ]; then
        if [ -z "$build_image_version" ]; then
            build_image_version="$(get_build_image_version)"
        else
            if [ "$build_image_version" != "$(get_build_image_version)" ]; then
                echo "ERROR: Build image version mismatch between: $target_cluster_tags"
                exit 1
            fi
        fi
    fi
done

export NVCC_GENCODE="$(get_nvcc_gencodes $gpu_arch_list)"

# reload the config with build cluster data
if [[ "$target_cluster_arg" != "$build_cluster_tag" ]]; then
    source_cluster_config $build_cluster_tag
fi

if [ "$use_build_cluster_image" -eq 1 ]; then
    build_image_version="$(get_build_image_version)"
fi

if [[ $make_clean == 1 && -d "build" ]]; then
    if [ "$baremetal_build" -eq 1 ]; then
        rm -rf build
    else
        # to be deleted by docker user later on
        mv build build.old
    fi
fi

# recreate as needed
mkdir -p build

export NUM_BUILD_PROCS="$(get_num_build_procs $(hostname))"

if [ "$baremetal_build" -eq 0 ]; then
    # assuming local drives or NFS mounted home dir
    current_dir=$(realpath .)

    # extract file ownership data
    export DOCKER_USER_ID=$(stat --format %u $0)
    export DOCKER_GROUP_ID=$(stat --format %g $0)

    # Disable GPU detection to allow enroot to launch this on CPU-only nodes
    # https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#gpu-enumeration
    export NVIDIA_VISIBLE_DEVICES=void
    
    eval "$(get_build_command $current_dir $build_image_version)"
else
    eval "$(get_build_command_bm)"
fi

make_result=$?
if [ "$make_result" -ne 0 ]; then
    exit $make_result
fi

# check symbols
if [ "$CHECK_SYMBOLS" -eq 1 ]; then
    test/binary/test_symbols.sh build/lib/libnccl.so
fi

exit_code=$?
exit $exit_code
