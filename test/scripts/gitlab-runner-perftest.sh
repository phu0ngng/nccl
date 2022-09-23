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
enable_ft="-B 0 -F 1"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv hypercube; do
  echo "=============================== $func (all sizes) - $(date +\"%T\") ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/${func}_perf $range $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (all sizes)")
done

rangetype="-b 16M -e 16M -o all -d all"
for func in all_reduce reduce reduce_scatter; do
  echo "=============================== $func (all ops/dtype) - $(date +\"%T\")  ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/all_reduce_perf $rangetype $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (all ops/dtype)")
done

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count