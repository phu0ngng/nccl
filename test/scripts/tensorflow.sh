#!/bin/bash
#Tensorflow
gpumodel=$1
resdir="results_dlfw"
path=$resdir/$gpumodel
mkdir -p $path
result="$path/tensorflow"

# Environment Variables
export INSTALL=/home/nightly/install
export TMP_PKG_DIR=$INSTALL/tmp_tensorflow_pkg

# TF needs installing python packages (we do that locally)
#python -m pip install $TMP_PKG_DIR/*whl --user
#python -m pip install --upgrade $TMP_PKG_DIR/*whl --user

BENCH_DIR=$INSTALL/tf_benchmarks/scripts/tf_cnn_benchmarks

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=INFO srun -p $gpumodel -t 40 --exclusive \
python $BENCH_DIR/tf_cnn_benchmarks.py --local_parameter_device=gpu --num_gpus=8 --batch_size=128 --model=resnet50 --variable_update=replicated --use_nccl=True | \
#awk '/total images/ {print $NF}' | \
tee $result.out
