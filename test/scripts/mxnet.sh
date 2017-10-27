#!/bin/bash
#mxnet
gpumodel=$1
mode=$2 #GROUP PARALLEL
resdir="results_dlfw"
path=$resdir/mxnet
mkdir -p $path
result="$path/$gpumodel.$mode"

mxnet_path=/home/nightly/install/mxnet

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN srun -p $gpumodel --exclusive \
python $mxnet_path/example/image-classification/train_imagenet.py --gpu 0,1,2,3,4,5,6,7 --batch-size 512 --num-epochs 1 --disp-batches 100 --network resnet-v1 --num-layers 50 --dtype float16 --benchmark 1 --kv-store nccl_reducebcast | \
tee /dev/stderr | \
#awk '/Speed/ {if (lines++) sum += $5} END {print sum/(lines-1)}' | \
tee -a $result.out
