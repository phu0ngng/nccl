#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

TRACE_FILE=$1

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$NCCL_HOME"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using TRACE_FILE=$TRACE_FILE"

echo "=============================== Replaying $(basename $TRACE_FILE) - $(date +\"%T\") ================================="
$SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay $TRACE_FILE
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$TRACE_FILE")

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
