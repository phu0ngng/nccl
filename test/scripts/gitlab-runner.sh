#!/bin/bash

cmd=$1
run=0
if [ `hostname` == "gc00" ]; then
  makeprefix=''
  slurmp='P100 P40 M40 K40'
  ngpus=8
  MPI_HOME=/opt/mpi/openmpi-1.10.7
  run=1
elif [ `hostname` == "prom" ]; then
  makeprefix=''
  slurmp='backfill'
  ngpus=32
  MPI_HOME=/home/sjeaugey/install/ompi-v1.10.7-patches/
  run=1
elif [ `hostname` == "circe" ]; then
  makeprefix='srun -p interactive --pty'
  slurmp='backfill'
  ngpus=32
  MPI_HOME=/home/sjeaugey/install/ompi-v1.10.7-patches/
  run=1
fi
if [ "$run" == "0" ]; then
  echo "Unknown host, exiting."
fi

if [ "$cmd" == "build" ]; then
  $makeprefix make -j test.build MPI=1 MPI_HOME=$MPI_HOME
elif [ "$cmd" == "test" ]; then
  export OPAL_PREFIX=$MPI_HOME
  export PATH=$MPI_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$LD_LIBRARY_PATH
  for p in $slurmp; do
    srun -p$p -n 1 -c $ngpus ./build/test/apitest/apitest
    NCCL_P2P_DISABLE=1 srun -p$p -n 1 -c $ngpus ./build/test/apitest/apitest
    NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 srun -p$p -n 1 -c $ngpus ./build/test/apitest/apitest
    salloc -p$p -n $ngpus mpirun build/test/perf/all_reduce_perf -b 8 -e 1G -f 2
  done
fi

