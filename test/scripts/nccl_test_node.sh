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

# DGX specific setting
if [ "$gpumodel" == "dgx1" ] || [ "$gpumodel" == "dgx1v" ]; then
  source /etc/profile.d/modules.sh
  export PATH=/usr/local/bin:/usr/bin:$PATH
  source $HOME/cuda.sh
  MPI_HOME="${MPI_HOME:-$HOME/install/openmpi}"
  exclude="-x dgx1-prd-01 "
else
  source $SHDIR/cuda.sh
  MPI_HOME="${MPI_HOME:-/opt/mpi/openmpi}"
fi

# MPI Env
export MPI_HOME
export PATH=$MPI_HOME/bin:$PATH
if [ "$( which mpirun )" == "" ]; then
  echo "Cannot find MPI, please specify path using MPI_HOME=/path/to/MPI"
  exit 1
fi
export LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH

# build
if [ "$DEBDIR" == "" ] && [ "$INSTALL" != "1" ]; then
  make -j src.build 2>&1 | tee make_src.log
  DEBDIR=$BLDDIR
fi

# export library if not using the one installed on system
if [ "$INSTALL" != "1" ]; then
  export LD_LIBRARY_PATH=$DEBDIR/lib:$LD_LIBRARY_PATH
fi

# build tests
cd $NCCLROOT
make -j test.clean
if [ "$INSTALL" == "1" ]; then
  make -j test.build MPI=1 2>&1 | tee make_test_mpi.log
else
  make -j test.build MPI=1 NCCLDIR=${DEBDIR} 2>&1 | tee make_test_mpi.log
fi

# SLURM setting
timeout=2
if [ "$mode" == "all" ]; then
  timeout=`expr $timeout \* 26`
else
  timeout=`expr $timeout \* 15`
fi
if [ "$SLURM" == "1" ]; then
  srun_cmd="srun -p $gpumodel -t ${timeout} --exclusive $exclude "
  salloc_cmd="salloc -p $gpumodel -n $maxgpu -c 1 -t ${timeout} --exclusive $exclude "
else
  srun_cmd="timeout ${timeout}m "
  salloc_cmd="timeout ${timeout}m "
fi

cd $BLDDIR
if [ "$mode" == "dlfw" ] && [ "$gpumodel" == "P100" ]; then
  $SHDIR/caffe2.sh $gpumodel
  $SHDIR/cntk.sh $gpumodel
  $SHDIR/tensorflow.sh $gpumodel
  $SHDIR/mxnet.sh $gpumodel
  # warm-up run of pytorch
  $SHDIR/pytorch.sh $gpumodel
  $SHDIR/pytorch.sh $gpumodel
elif [[ "$mode" == *"mpi"* ]] || [[ "$mode" == *"multinode"* ]]; then
  # test (multi processes)
  if [[ "$mode" == *"mpi"* ]]; then
    echo "Testing $mode..."
    $salloc_cmd $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu $mode
  fi
  # multinode test
  if [[ "$mode" == *"multinode"* ]]; then
    if [ "$gpumodel" == "dgx1" ]; then
      $SHDIR/multinode_perf_graphs.sh dgx1 2 16 8 8
    elif [ "$gpumodel" == "P100" ]; then
      $SHDIR/multinode_perf_graphs.sh gpu-verbs 2 16 8 8
    else
      echo "No multi-node test on $gpumodel"
    fi
  fi
else
  # test (single process)
  if [ "$mode" == "api" ]; then
    api_path="results_api/$gpumodel"
    mkdir -p $api_path
    $srun_cmd $BLDDIR/test/apitest/apitest 2>&1 | tee $api_path/apitest.out
  else
    $srun_cmd $SHDIR/run_perf_graphs.sh $gpumodel $maxgpu $mode
  fi
fi

echo "NCCL_Complete" > state
