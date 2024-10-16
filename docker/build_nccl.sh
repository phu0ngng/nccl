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
jobs=$(eval "$DOCKER_JOB_COMMAND")

make -j$jobs test.build MPI=1 WERROR=1
# make -j$jobs pkg.build

# return file ownership to the user
chown -R ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} .

