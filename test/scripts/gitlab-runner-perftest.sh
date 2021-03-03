#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

max=$1
if [ "$max" == "" ]; then max=1G; fi

opts="-n 5 -w 1"

range="-b 8 -e $max -f 2"
for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv hypercube; do
  $SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/${func}_perf $range $opts
done

rangetype="-b 16M -e 16M"
for func in all_reduce reduce reduce_scatter; do
  $SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/all_reduce_perf $rangetype $opts
done
