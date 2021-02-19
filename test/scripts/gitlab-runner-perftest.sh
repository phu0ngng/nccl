#!/bin/bash

export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

$SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/all_reduce_perf -b 8 -e 1G -f 2
$SALLOC $MPI_HOME/bin/mpirun ./build/test/perf/alltoall_perf -b 8 -e 1G -f 2
