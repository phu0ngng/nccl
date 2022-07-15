#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

max=$1
if [ "$max" == "" ]; then max=1G; fi

shift
graph=$1
if [ "$graph" == "" ]; then graph=0; fi

opts="-n 5 -w 1 -G $graph"
range="-b 8 -e $max -f 2"

export LD_LIBRARY_PATH=$SHARP_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PLUGIN_PATH:$LD_LIBRARY_PATH
export NCCL_COLLNET_ENABLE=1
export NCCL_ALGO=COLLNET

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using SHARP_HOME=$SHARP_HOME"
echo "Using PLUGIN_PATH=$PLUGIN_PATH"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0

echo "=============================== all_reduce (CollNet) - $(date +\"%T\") ================================="
$SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/all_reduce_perf $range $opts
[ $? -ne 0 ] && let failure_count=$failure_count+1

exit $failure_count
