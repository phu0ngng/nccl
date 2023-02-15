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
enable_split_share="-S 1"
split_range="-b 8 -e 1G -f 2"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf hypercube_perf; do
  echo "=============================== $func (all sizes) - $(date +\"%T\") ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $range $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func $range $opts")
done

rangetype="-b 16M -e 16M -o all -d all"
for func in all_reduce_perf reduce_perf reduce_scatter_perf; do
  echo "=============================== $func (all ops/dtype) - $(date +\"%T\")  ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $rangetype $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func $rangetype $opts")
done

export NCCL_TESTS_SPLIT_MASK="0x1"
for func in all_reduce alltoall reduce_scatter all_gather broadcast; do
  echo "=============================== $func (split share all sizes) - $(date +\"%T\") =========================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/${func}_perf $split_range $opts $enable_split_share
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (split share all sizes)")
done
unset NCCL_TESTS_SPLIT_MASK

export NCCL_DEBUG="" # disable WARN information
echo "=============================== all_reduce (FT tests) - $(date +\"%T\") ================================="
$SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/all_reduce_perf $range $opts $enable_ft
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("all_reduce_perf $range $opts $enable_ft")

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count