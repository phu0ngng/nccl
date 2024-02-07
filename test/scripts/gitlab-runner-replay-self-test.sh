#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using SALLOC_SELF_MULTI=$SALLOC_SELF_MULTI"
echo "Using SALLOC_SELF_SINGLE=$SALLOC_SELF_SINGLE"

echo "=============================== Generating all_reduce trace file - $(date +\"%T\") ================================="
generate_all_reduce_trace=$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_all_reduce.%h.%p ./build/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all
$generate_all_reduce_trace
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Generate all_reduce trace file:\n$generate_all_reduce_trace")
cat trace_all_reduce* > full_trace_all_reduce.txt
rm trace_all_reduce*

echo "=============================== Generating alltoall trace file - $(date +\"%T\") ================================="
generate_alltoall_trace=$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_alltoall.%h.%p ./build/test/perf/alltoall_perf -w5 -n5 -b1K -e10M -f2
$generate_alltoall_trace
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Generate alltoall trace file:\n$generate_alltoall_trace")
cat trace_alltoall* > full_trace_alltoall.txt
rm trace_alltoall*

echo "=============================== "Replay full_trace_all_reduce" - $(date +\"%T\") ================================="
replay_all_reduce=$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/replay/replay -b full_trace_all_reduce.txt
$replay_all_reduce
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay full_trace_all_reduce:\nGenerate alltoall trace file : $generate_all_reduce_trace\nReplay full_trace_all_reduce : $replay_all_reduce")
fi

echo "=============================== Replay full_trace_alltoall - $(date +\"%T\") ================================="
replay_alltoall=$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/replay/replay -b full_trace_alltoall.txt
$replay_alltoall
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay full_trace_alltoall\nGenerate alltoall trace file : $generate_alltoall_trace\nReplay full_trace_alltoall : $replay_alltoall")
fi

echo "=============================== Replay force-fit full_trace_all_reduce - $(date +\"%T\") ================================="
replay_ar_ff=$SALLOC_SELF_SINGLE $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/replay/replay -x -b full_trace_all_reduce.txt
$replay_ar_ff
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay force-fit full_trace_all_reduce:\nGenerate alltoall trace file : $generate_all_reduce_trace\nReplay force-fit full_trace_all_reduce : $replay_ar_ff")
fi

for str in "${failure_names[@]}"
do
  printf "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
