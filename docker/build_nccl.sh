#!/bin/bash -x
# use inside nccl build tools container
export CUDA_HOME=/usr/local/cuda
export MPI_HOME=/usr/local/openmpi
export LD_LIBRARY_PATH=${MPI_HOME}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH}

export NCCL_HOME=/nccl/build

cd /nccl

# clean up previous run if clean was requested
if [ -d "build.old" ]; then
  rm -rf build.old &
  make clean
fi

# figure out a fair job number
jobs=$(eval "$NUM_BUILD_PROCS")

make -j$jobs test.build MPI=1 WERROR=1 TRACE=$NCCL_BUILD_TRACE
# make -j$jobs pkg.build

build_status=$?

if [ $build_status -eq 0 ]; then
    echo "INFO: Make exited successfully"
else
    echo "ERROR: Make exited with $build_status"
fi

# propagate exit status
exit $build_status
