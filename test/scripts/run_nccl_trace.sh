LOG_FILE_DIR=~/nccl_trace
export NCCL_DEBUG_FILE=$LOG_FILE_DIR/nccl_trace.replay.%h.%p
export TRACE_FILE=$1
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG=TRACE
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1
if ls ~/nccl_trace/nccl_trace.replay* 1> /dev/null 2>&1; then
    rm ~/nccl_trace/nccl_trace.replay*
fi

if test -f "~/nccl_trace/results_out.txt"; then
    rm ~/nccl_trace/results_out*
fi

# -output-filename ~/nccl_trace/results_out
mpirun -output-filename ~/nccl_trace/results_out -np 8 ~/nccl/build/test/replay/replay $TRACE_FILE -p
