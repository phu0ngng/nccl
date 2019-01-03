#!/bin/bash
#Caffe2
gpumodel=$1
resdir="results_dlfw"
path=$resdir/$gpumodel
mkdir -p $path
result="$path/caffe2"

# Environment Variables
export CAFFE2_ROOT=/home/nightly/install/caffe2
export PYTHONPATH=$PYTHONPATH:$CAFFE2_ROOT/build

if [ "$gpumodel" == "dgx1v" ]; then
  extra="--enable-tensor-core"
fi

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=INFO srun -p $gpumodel -t 40 --exclusive \
python $CAFFE2_ROOT/nvidia-examples/imagenet/train_resnet.py --train-lmdb /data/imagenet/train-lmdb-256x256 --skip-test --batch-size 1024 --num-iterations 1000 $extra --all-gpus --dtype float16 | \
#tee /dev/stderr | awk -F'=' '/Epoch/ {if (lines++) sum += $NF} END {print sum/(lines-1)}' | \
tee $result.out
