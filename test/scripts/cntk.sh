#!/bin/bash
#CNTK
gpumodel=$1
mode=$2 #GROUP PARALLEL
resdir="results_dlfw"
path=$resdir/cntk
mkdir -p $path
result="$path/$gpumodel.$mode"

# Environment Variables
export CNTK_DIR=/home/nightly/install/cntk
export PATH=$CNTK_DIR/build/release/bin:$PATH
export LD_LIBRARY_PATH=$CNTK_DIR/build/release/lib:$LD_LIBRARY_PATH

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=WARN salloc -p $gpumodel -n 8 \
export OMP_NUM_THREADS=7 #nproc/ngpus
mpiexec --allow-run-as-root -np 8 cntk configFile=/data/networks/ResNet50/ResNet50.cntk meanFile=/data/images/ImageNet1K_mean.xml ConfigDir=/data/networks/ResNet50 trainMapFile=/data/map/fake_trainMap.txt OutputPath=/tmp/cntk-$(date +"%Y%m%d%I%M%S") epochSize=51200 batchSize=512 parallelTrain=true |\
tee /dev/stderr | awk '/^ Epoch/ {if (lines++ > 50) sum += $NF} END {print sum/(lines-1)}' | \
tee -a $result.out
