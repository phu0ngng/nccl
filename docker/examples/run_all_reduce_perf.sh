#!/bin/bash
# run from the top of the source tree
build_dir="build"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${build_dir}/lib
export NCCL_DEBUG=INFO

echo NCCL_IB_HCA=$NCCL_IB_HCA

# nvidia-smi
# /usr/sbin/ibstat

binary="${build_dir}/test/perf/all_reduce_perf"

ldd $binary

$binary -b 8 -e 1G -f 2 -g 2

