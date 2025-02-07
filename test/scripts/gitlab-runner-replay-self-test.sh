#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$NCCL_HOME"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using SALLOC_SELF_MULTI=$SALLOC_SELF_MULTI"
echo "Using SALLOC_SELF_SINGLE=$SALLOC_SELF_SINGLE"

echo "=============================== Generating all_reduce trace file - $(date +\"%T\") ================================="
$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_all_reduce.%h.%p $NCCL_HOME/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Generate all_reduce trace file: mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_all_reduce.%h.%p $NCCL_HOME/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all")
fi
cat trace_all_reduce* > full_trace_all_reduce.txt
rm trace_all_reduce*

echo "=============================== Generating alltoall trace file - $(date +\"%T\") ================================="
$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_alltoall.%h.%p $NCCL_HOME/test/perf/alltoall_perf -w5 -n5 -b1K -e10M -f2
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Generate alltoall trace file: mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_alltoall.%h.%p $NCCL_HOME/test/perf/alltoall_perf -w5 -n5 -b1K -e10M -f2")
fi
cat trace_alltoall* > full_trace_alltoall.txt
rm trace_alltoall*

echo "=============================== "Replay full_trace_all_reduce" - $(date +\"%T\") ================================="
$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay full_trace_all_reduce.txt
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay full_trace_all_reduce:")
  failure_names+=("Generate all_reduce trace file : mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_all_reduce.%h.%p $NCCL_HOME/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all && cat trace_all_reduce* >> full_trace_all_reduce.txt")
  failure_names+=("Replay full_trace_all_reduce : mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay -b full_trace_all_reduce.txt")
fi

echo "=============================== Replay full_trace_alltoall - $(date +\"%T\") ================================="
$SALLOC_SELF_MULTI $MPI_HOME/bin/mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay full_trace_alltoall.txt
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay full_trace_alltoall:")
  failure_names+=("Generate alltoall trace file : mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_alltoall.%h.%p $NCCL_HOME/test/perf/alltoall_perf -w5 -n5 -b1K -e10M -f2 && cat trace_alltoall* >> full_trace_alltoall.txt")
  failure_names+=("Replay full_trace_alltoall : mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay -b full_trace_alltoall.txt")
fi

echo "=============================== Replay force-fit full_trace_all_reduce - $(date +\"%T\") ================================="
$SALLOC_SELF_SINGLE $MPI_HOME/bin/mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay -x full_trace_all_reduce.txt
if [ $? -ne 0 ]; then
  let failure_count=$failure_count+1
  failure_names+=("Replay force-fit full_trace_all_reduce:")
  failure_names+=("Generate alltoall trace file : mpirun $MPI_PARAMS -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_FILE=trace_all_reduce.%h.%p $NCCL_HOME/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all && cat trace_all_reduce* >> full_trace_alltoall.txt")
  failure_names+=("Replay full_trace_all_reduce : mpirun $MPI_PARAMS $NCCL_HOME/test/replay/replay -x -b full_trace_all_reduce.txt")
fi

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
