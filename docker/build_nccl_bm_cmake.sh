#!/bin/bash -x
# Use to build NCCL baremetal using CMake

# TODO (kartiki) - Integrate CMake build into docker/make.sh and kill this script

if [ ! -d "docker" ]; then
  echo "Please launch from the top of NCCL source tree"
  exit 1
fi

nccl_src=$(pwd)
rm -rf $nccl_src/build

source docker/cluster-utils.sh

# TODO (kartiki) - Support CMake on all clusters; to be taken up as part of integration with docker/make.sh
target_cluster_tag=ipp6-slurm
# target_cluster_tag=$(identify_cluster)
source docker/clusters/$target_cluster_tag-config.sh

nccl_build_workspace=$(mktemp -d)/nccl
git clone --depth 1 file://$nccl_src $nccl_build_workspace

pushd $nccl_build_workspace

export CUDA_PATH=$(get_cuda_home)
export NCCL_HOME=$nccl_build_workspace

# TODO (kartiki) - Get arch list from cluster config; replace ',' with ';' for CMake
gpu_arch_list="80"

echo "Starting CMake Build..."

cmake -DCMAKE_CUDA_ARCHITECTURES=$gpu_arch_list -B build/ && cmake --build build/ --parallel

build_status=$?
if [ $build_status -eq 0 ]; then
    echo "INFO: CMake exited successfully"
else
    echo "ERROR: CMake exited with $build_status"
fi

rsync -a build $nccl_src/

# Cleanup temp directory
popd
rm -rf $nccl_build_workspace
