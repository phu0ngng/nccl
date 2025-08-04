#!/bin/bash
cd /workspace/nccl/
# Needed for gitlab nccl make
apt-get update
apt-get install rsync -y
export MPI_HOME=/usr/local/mpi/
export NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export NCCL_HOME=/workspace/nccl/build
export PATH=/usr/local/cuda/bin:${PATH}
make -j`nproc` test.build MPI=1 BUILDDIR=build-dlfw
chmod -R a+rwx build-dlfw
