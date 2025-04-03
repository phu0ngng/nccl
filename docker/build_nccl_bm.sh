#!/bin/bash -x
# Use to build NCCL baremetal on a supported cluster, called as part of docker/make.sh

if [ ! -d "docker" ]; then
  echo "Please launch from the top of NCCL source tree"
  exit 1
fi

source docker/cluster-utils.sh

# Assumes that if you're building baremetal, then you're build and target cluster are the same
target_cluster_tag=$(identify_cluster)
source docker/$target_cluster_tag-config.sh

export CUDA_HOME=$(get_cuda_home)
export MPI_HOME=$(get_openmpi_home)
export LD_LIBRARY_PATH=$(get_extra_ld_library_path):${LD_LIBRARY_PATH}
export PATH=$CUDA_HOME/bin:${PATH}

export NCCL_HOME=$(pwd)/build

if [ $make_clean ]; then make clean; fi

# figure out a fair job number
jobs=$(eval "$NUM_BUILD_PROCS")

make -j$jobs test.build MPI=1 WERROR=1

build_status=$?

if [ $build_status -eq 0 ]; then
    echo "INFO: Make exited successfully"
else
    echo "ERROR: Make exited with $build_status"
fi

# propagate exit status
exit $build_status
