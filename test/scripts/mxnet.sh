#!/bin/bash
#mxnet
gpumodel=$1
resdir="results_dlfw"
path=$resdir/$gpumodel
mkdir -p $path
result="$path/mxnet"

# Switch to Github version + nccl integration PR (#8294)
mxnet_path=/home/nightly/install/mxnet.github

# Option "nccl" now stands for reduce/broadcast (same as "nccl_reducebcast" used before)
NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
python $mxnet_path/example/image-classification/train_imagenet.py --gpu 0,1,2,3,4,5,6,7 --batch-size 1024 --num-epochs 1 --disp-batches 100 --network resnet-v1 --num-layers 50 --dtype float16 --benchmark 1 --kv-store nccl 2>&1 | \
#awk '/Speed/ {if (lines++) sum += $5} END {print sum/(lines-1)}' | \
tee $result.out
