#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

max=$1
if [ "$max" == "" ]; then max=1G; fi

shift
graph=$1
if [ "$graph" == "" ]; then graph=0; fi

opts="-n 5 -w 1 -G $graph"
range="-b 8 -e $max -f 2"
enable_split_comm="-S 1 -P 1"
enable_local_register="-R 1"

export LD_LIBRARY_PATH=$SHARP_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HPCX_UCX_LIB:$PLUGIN_PATH:$LD_LIBRARY_PATH
export NCCL_COLLNET_ENABLE=1

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$NCCL_HOME"
echo "Using HPCX_UCX_LIB=$HPCX_UCX_LIB"
echo "Using SHARP_HOME=$SHARP_HOME"
echo "Using PLUGIN_PATH=$PLUGIN_PATH"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0

# Uncomment when sharp is re-enabled on GC
# export NCCL_ALGO=CollNetDirect,CollNetChain
echo "=============================== all_reduce (CollNet) - $(date +\"%T\") ================================="
$SALLOC $MPI_HOME/bin/mpirun $NCCL_HOME/test/perf/all_reduce_perf $range $opts
[ $? -ne 0 ] && let failure_count=$failure_count+1

echo "=============================== all_reduce (Split Share CollNet) - $(date +\"%T\") ====================="
$SALLOC $MPI_HOME/bin/mpirun $NCCL_HOME/test/perf/all_reduce_perf $range $opts $enable_split_comm
[ $? -ne 0 ] && let failure_count=$failure_count+1

echo "=============================== all_reduce (local registration CollNet) - $(date +\"%T\") ====================="
$SALLOC $MPI_HOME/bin/mpirun $NCCL_HOME/test/perf/all_reduce_perf $range $opts $enable_local_register
[ $? -ne 0 ] && let failure_count=$failure_count+1

exit $failure_count
