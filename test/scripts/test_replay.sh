#!/bin/bash

NUM_GPUS=1
NUM_THREADS=1
MAX_GPUS=8
TIMEOUT=6000
NUM_RANKS=1

# How many ranks can we run?
MAX_RANKS=$((MAX_GPUS/(NUM_GPUS*NUM_THREADS)))

ARGS="-dint32 -t$NUM_THREADS -g$NUM_GPUS -b100k -e100M -f2 -n 44 -w 0"

SCRIPT_LOG_FILE="results_out.txt"
NCCL_TRACE_PREFIX="nccl_trace"

# Trace replay
NCCL_DEBUG=TRACE
NCCL_DEBUG_SUBSYS=ALL
LOG_FILE_DIR=~/nccl_trace
# export CUDA_DEVICE_WAITS_ON_EXCEPTION=1

mkdir $LOG_FILE_DIR

rm $LOG_FILE_DIR/$NCCL_TRACE_PREFIX*

# Supposedly will make this script stop when an error occurs
set -e

# Clean old trace
if [ -e $LOG_FILE_DIR/$SCRIPT_LOG_FILE ]; then
    rm $LOG_FILE_DIR/$SCRIPT_LOG_FILE
fi

echo "[test_replay] Running tests with up to $MAX_RANKS ranks" >> $LOG_FILE_DIR/$SCRIPT_LOG_FILE
for (( i=$NUM_RANKS; i<=$MAX_RANKS; i++ ))
do
    echo "[test_replay] Running tests with $i ranks" >> $LOG_FILE_DIR/$SCRIPT_LOG_FILE
    for perf_test in all_gather_perf all_gatherv_perf all_reduce_perf alltoall_perf broadcast_perf gather_perf reduce_perf reduce_perf reduce_scatter_perf reduce_scatterv_perf scatter_perf sendrecv_perf
    do
        # Clean old traces
        [ -e $LOG_FILE_DIR/nccl_trace_combined.txt ] && rm $LOG_FILE_DIR/$NCCL_TRACE_PREFIX*

        # Invoke perf test
        export NCCL_DEBUG_FILE=$LOG_FILE_DIR/$NCCL_TRACE_PREFIX.$perf_test.%h.%p

        echo "[test_trace] Running $perf_test with $i ranks using args $ARGS" >> $LOG_FILE_DIR/$SCRIPT_LOG_FILE
        mpirun -np $i $NCCL_HOME/test/perf/$perf_test $ARGS

        # Combine traces into one file
        cat $LOG_FILE_DIR/$NCCL_TRACE_PREFIX.* >> $LOG_FILE_DIR/nccl_trace_combined.txt

        # Set trace for the replay for debugging
        export NCCL_DEBUG_FILE=$LOG_FILE_DIR/$NCCL_TRACE_PREFIX.$perf_test.replay.%h.%p

        # Run command with timeout to detect hangs
        echo "[test_trace] ($perf_test:$i) Replaying nccl_trace_combined.txt on $i ranks with timeout $TIMEOUT seconds">> $LOG_FILE_DIR/$SCRIPT_LOG_FILE
        timeout $TIMEOUT mpirun -np $i -output-filename ~/nccl_trace/results_out $NCCL_HOME/test/replay/replay $LOG_FILE_DIR/nccl_trace_combined.txt -p
    done
done
