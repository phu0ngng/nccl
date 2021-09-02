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
for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv hypercube; do
  echo "=============================== $func (all sizes) ================================="
  $SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/${func}_perf $range $opts
done

rangetype="-b 16M -e 16M -o all -d all"
for func in all_reduce reduce reduce_scatter; do
  echo "=============================== $func (all ops/dtype)  ================================="
  $SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/all_reduce_perf $rangetype $opts
done
