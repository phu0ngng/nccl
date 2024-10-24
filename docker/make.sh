#!/bin/bash -x
# use plain docker to build the binaries
# provide target cluster name (defaults to the build cluster name)
# identify build machine/cluster automatically
function usage() {
    echo "INFO:  Usage: $0 [ target_cluster_tag ] [ --clean ]"
    exit 1
}

function source_cluster_config() {
    config_file="docker/$1-config.sh"
    
    if [ ! -f "$config_file" ]; then
        echo "ERROR: Config file $config_file is not found."
        usage
    fi
  
    source $config_file
}

function get_nvcc_gencodes() {
    # comma-separated list of numeric gpu_archs
    gpu_archs="$(echo $1 | tr ',' '\n' | sort -u)"

    gencode_string=""
    for gpu_arch in $gpu_archs; do
        gencode_string="$gencode_string -gencode=arch=compute_${gpu_arch},code=sm_${gpu_arch}"
    done
    echo "$gencode_string"
}

function identify_build_cluster() {
    hostname="$(hostname)"
    
    if [[ "$hostname" =~ ^gc[01][0-9]$ ]]; then
        echo "gc"
	return
    fi

    if [[ "$hostname" =~ eos.clusters.nvidia.com$ ]]; then
        echo "eos"
	return
    fi
 
    if [[ "$hostname" =~ draco-rno-login- ]]; then
	echo "draco-rno"
	return
    fi

    if [[ "$hostname" =~ draco-oci-login- ]]; then
        echo "draco-oci"
	return
    fi

    echo "ERROR: Cluster unknown"
    exit 1	
}

# identify build cluster first
build_cluster_tag="$(identify_build_cluster)"

# arg parsing
# defaults
target_cluster_arg="$build_cluster_tag"
make_clean=0

for arg in "$@"
do
    case $arg in
        --clean) make_clean=1
                 ;;
        *) target_cluster_arg="$arg"
           ;;
    esac
done

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
    # make sure all build image versions are the same
    if [ -z "$build_image_version" ]; then
        build_image_version="$(get_build_image_version)"
    else
        if [ "$build_image_version" != "$(get_build_image_version)" ]; then
            echo "ERROR: Build image version mismatch between: $target_cluster_tags"
	    exit 1
	fi
    fi
done

export NVCC_GENCODE="$(get_nvcc_gencodes $gpu_arch_list)"

# reload the config with build cluster data
if [[ "$target_cluster_arg" != "$build_cluster_tag" ]]; then
    source_cluster_config $build_cluster_tag
fi
    
export DOCKER_JOB_COMMAND="$(get_docker_job_command $(hostname))"

if [ ! -d "docker" ]; then
  echo "Please launch from the top of NCCL source tree"
  exit 1
fi

if [[ $make_clean == 1 && -d "build" ]]; then
    # to be deleted by docker user later on
    mv build build.old
fi

# recreate as needed
mkdir -p build

# assuming local drives or NFS mounted home dir
current_dir=$(realpath .)

# extract file ownership data
export DOCKER_USER_ID=$(stat --format %u $0)
export DOCKER_GROUP_ID=$(stat --format %g $0)

eval "$(get_build_command $current_dir $build_image_version)"
