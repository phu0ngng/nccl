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
    gpu_archs=$(echo $1 | tr ',' '\n')
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
target_cluster_tag="$build_cluster_tag"
make_clean=0

for arg in "$@"
do
    case $arg in
        --clean) make_clean=1
                 ;;
        *) target_cluster_tag="$arg"
           ;;
    esac
done

# process target cluster config
source_cluster_config $target_cluster_tag

gpu_archs=$(get_gpu_archs)
export NVCC_GENCODE="$(get_nvcc_gencodes $gpu_archs)"

build_image_version="$(get_build_image_version)"

# reload the config with build cluster data
# special case gc-classic
if [[ "$target_cluster_tag" != "$build_cluster_tag" ]]; then
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
