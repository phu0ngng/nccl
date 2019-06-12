#!/bin/bash
#PyTorch
gpumodel=$1
resdir="results_dlfw"
path=$resdir/$gpumodel
mkdir -p $path
result="$path/pytorch"

# Environment Variables
export INSTALL=/home/nightly/install
export PATH=$INSTALL/conda/envs/pytorch-py35/bin:$PATH

BENCH_DIR=$INSTALL/pytorch/examples/imagenet

# L1 perf test as in PyTorch container
CUR_DIR=$(pwd)
cd $INSTALL/pytorch/qa/L1_perftest
NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=INFO srun -p $gpumodel -t 40 --exclusive ./test.sh | tee $CUR_DIR/$result.out
cd $CUR_DIR
