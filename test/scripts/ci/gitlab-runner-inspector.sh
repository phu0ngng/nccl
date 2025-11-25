#!/bin/bash
#
# Simple CI runner for NCCL Inspector plugin
#
set -e
source test/scripts/ci/ci-utils.sh

load_test_ci_variables
source $CLUSTER_CONFIG
load_cluster_ci_variables

PERF_TESTS=(all_reduce_perf reduce_scatter_perf all_gather_perf broadcast_perf alltoall_perf sendrecv_perf)
CHECK_LOG_TESTS=(all_reduce_perf reduce_scatter_perf all_gather_perf)

opts="-w 0 -n 5 -O0 -G 0"
range="-b 64M -e 64M"

# Build the inspector plugin
make CUDA_HOME=$CUDA_HOME -C $NCCL_HOME/../ext-profiler/inspector -j

for func in "${PERF_TESTS[@]}"; do
  LOG_DIR=$(pwd)/inspector_logs_${func}_${CI_PIPELINE_ID}_${CI_JOB_ID}
  mkdir -p "$LOG_DIR"
  export NCCL_INSPECTOR_DUMP_DIR="$LOG_DIR"
  echo "Running $func with Inspector, logs in $LOG_DIR"
  run_command "${func}_inspector" $RUN_MODE $NGPUS "" \
    "NCCL_PROFILER_PLUGIN=$NCCL_HOME/../ext-profiler/inspector/libnccl-profiler-inspector.so NCCL_INSPECTOR_ENABLE=1 NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500 NCCL_INSPECTOR_DUMP_DIR=$LOG_DIR " \
    "$NCCL_HOME/test/perf/$func" "$range $opts"

  # For selected tests, check that a non-zero log file was created
  if [[ " ${CHECK_LOG_TESTS[@]} " =~ " $func " ]]; then
    log_count=$(find "$LOG_DIR" -name "*.log" -size +0c | wc -l)
    if [ "$log_count" -eq 0 ]; then
      echo "ERROR: No non-zero log files found for $func in $LOG_DIR"
      exit 1
    else
      echo "Found $log_count non-zero log files for $func in $LOG_DIR"
    fi
  fi
done

ci_exit
