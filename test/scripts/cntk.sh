#!/bin/bash
#CNTK
gpumodel=$1
resdir="results_dlfw"
path=$resdir/$gpumodel
mkdir -p $path
result="$path/cntk"

# Environment Variables
export CNTK_DIR=/home/nightly/install/cntk
export PATH=$CNTK_DIR/build/release/bin:$PATH
export LD_LIBRARY_PATH=$CNTK_DIR/build/release/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=7 #nproc/ngpus (56 cores on t039)

NCCL_DISABLE_CHECKS=1 NCCL_DEBUG=INFO salloc -p $gpumodel -t 40 -n 8 \
mpiexec --allow-run-as-root -np 8 cntk configFile=/data/networks/ResNet50/ResNet50.cntk meanFile=/data/images/ImageNet1K_mean.xml ConfigDir=/data/networks/ResNet50 trainMapFile=/data/map/fake_trainMap.txt OutputPath=/tmp/cntk-$(date +"%Y%m%d%I%M%S") epochSize=512000 batchSize=512 parallelTrain=true 2>&1 | \
#tee /dev/stderr | awk '/^ Epoch/ {if (lines++ > 50) sum += $NF} END {print sum/(lines-1)}' | \
tee $result.out
