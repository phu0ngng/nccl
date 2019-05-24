#!/bin/bash

gpumodel=$1

maxgpu=$2

mode=$3

# get dir of test scripts
SHDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SHDIR/../../
NCCLROOT=$PWD
BLDDIR=$NCCLROOT/build
rm $BLDDIR/state

HOST=$(hostname)
export CUDA_HOME="${CUDA_HOME:-$HOME/nightly-$HOST/cuda}"
export MPI_HOME="${MPI_HOME:-$HOME/nightly-$HOST/openmpi}"
export OPAL_PREFIX=$MPI_HOME
export PATH=$MPI_HOME/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MPI_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# build
if [ "$DEBDIR" == "" ]; then
  make -j src.build 2>&1 | tee make_src.log
  DEBDIR=$BLDDIR
fi

# export library if not using the one installed on system
export LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH

# build tests
cd $NCCLROOT
make -j test.build MPI=1 NCCLDIR=${DEBDIR} 2>&1 | tee make_test_mpi.log

# SLURM setting
timeout=$((10 * $maxgpu))
if [ "$SLURM" == "1" ]; then
  srun_cmd="srun -p $gpumodel -t ${timeout} --exclusive "
  salloc_cmd="salloc -p $gpumodel -t ${timeout} --exclusive "
else
  srun_cmd="timeout ${timeout}m "
  salloc_cmd="timeout ${timeout}m "
fi

export NCCL_DEBUG=INFO

cd $BLDDIR
if [ "$mode" == "dlfw" ] && [ "$gpumodel" == "P100" ]; then
  $SHDIR/caffe2.sh $gpumodel
  $SHDIR/cntk.sh $gpumodel
  $SHDIR/tensorflow.sh $gpumodel
  $SHDIR/mxnet.sh $gpumodel
  $SHDIR/pytorch.sh $gpumodel
elif [[ "$mode" == *"multinode"* ]]; then
  # multinode test
  if [ "$gpumodel" == "dgx1" ] || [ "$gpumodel" == "dgx1v" ]; then
    $SHDIR/multinode_perf_graphs.sh $gpumodel 2 4 2 2
  elif [ "$gpumodel" == "mlperf" ]; then
    $SHDIR/multinode_perf_graphs.sh $gpumodel 2 16 1 1
    $SHDIR/multinode_perf_graphs.sh $gpumodel 4 32 1 1
    $SHDIR/multinode_perf_graphs.sh $gpumodel 8 64 1 1
    $SHDIR/multinode_perf_graphs.sh $gpumodel 16 128 1 1
    $SHDIR/multinode_perf_graphs.sh $gpumodel 32 256 1 1
  elif [ "$gpumodel" == "P100" ]; then
    $SHDIR/multinode_perf_graphs.sh all 2 4 2 2
    $SHDIR/multinode_env_test.sh all 2 8
  else
    echo "No multi-node test on $gpumodel"
  fi
elif [ "$mode" == "api" ]; then
  api_path="results_api/$gpumodel"
  mkdir -p $api_path
  $srun_cmd $BLDDIR/test/apitest/apitest 2>&1 | tee $api_path/apitest.out
else
  $salloc_cmd $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu $mode
fi
