#!/bin/bash


threshold=$1
if [ "$threshold" == "" ]; then threshold=10; fi

shift

ref=$1
if [ "$ref" == "" ]; then ref="$REGRESSION_BASELINE"; fi
baseline_build_dir="build-${ref}"

failure_count=0
failure_names=()

# Check if the baseline build is cached
if [[ ! -e "$baseline_build_dir"/test/perf/all_reduce_perf ]]; then
  let failure_count=$failure_count+1 && failure_names+=("Can't find ref: $ref")
fi

# Only run comparison if the baseline exists and we are back in the dev branch
if [[ $failure_count -eq 0 ]]; then
  perf_files_dir="perf_files"
  rm -rf $perf_files_dir
  mkdir $perf_files_dir
  baseline_perf_files_dir="baseline_perf_files"
  rm -rf $baseline_perf_files_dir
  mkdir $baseline_perf_files_dir

  export OPAL_PREFIX=$MPI_HOME
  LD_LIBRARY_PATH_BASE=$MPI_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
  LD_LIBRARY_NEW=$PWD/build/lib:$LD_LIBRARY_PATH_BASE
  LD_LIBRARY_OLD=$baseline_build_dir/lib:$LD_LIBRARY_PATH_BASE

  # Environment setup
  echo "=============================== Environment Setup ================================="

  # Set Persistence mode:
  # sudo nvidia-persistenced

  # Set Persistence mode For POWER9:
  # systemctl status nvidia-persistenced
  # sudo systemctl enable nvidia-persistenced
  # sudo systemctl restart nvidia-persistenced.service

  # Query Max Clocks for each GPU:
  # nvidia-smi -i ${GPU_ID} -q -d clock
  # Check Section "Max Clocks"

  # Set Max clock for each GPU (x is the GPU index, 0-7 in 8GPUs server):
  # sudo /usr/bin/nvidia-smi -i ${GPU_ID} -ac 1410,1410

  # Query for each GPU:
  # nvidia-smi -i ${GPU_ID} -q -d clock

  # Disbale Auto Boost (Only for K80/M40)
  # sudo nvidia-smi -i ${GPU_ID} --auto-boost-default=0

  echo "HOSTNAME=$HOSTNAME"
  echo "Using CUDA_HOME=$CUDA_HOME"
  echo "Using MPI_HOME=$MPI_HOME"
  echo "Using NCCL_HOME=$PWD/build"
  echo "Using UCX_TLS: $UCX_TLS"
  echo "Using LD_LIBRARY_NEW=$LD_LIBRARY_NEW"
  echo "Using LD_LIBRARY_OLD=$LD_LIBRARY_OLD"
  echo "Using REGRESSION_BASELINE=$REGRESSION_BASELINE"
  echo "Using REGRESSION_THRESHOLD=$REGRESSION_THRESHOLD"

  # Bandwidth Tests
  opts="-b 512M -e 512M -d float -w 5 -n 5 -c1"
  for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv; do
    echo "=============================== $func BANDWIDTH ($opts) ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$perf_files_dir/${func}_perf_bandwidth $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_bandwidth failed to run") && echo "New ${func}_perf_bandwidth failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$baseline_perf_files_dir/${func}_perf_bandwidth $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_bandwidth failed to run") && echo "Baseline ${func}_perf_bandwidth failed to run"
  done

  # Collectives Latency Tests
  opts="-b 64 -e 64 -w 1 -n 200 -c0"
  for func in all_reduce broadcast reduce sendrecv; do
    echo "=============================== $func 64B COLLECTIVES LATENCY ($opts) ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$perf_files_dir/${func}_perf_64b_coll_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_64b_coll_latency failed to run") && echo "New ${func}_perf_64b_coll_latency failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$baseline_perf_files_dir/${func}_perf_64b_coll_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_64b_coll_latency failed to run") && echo "Baseline ${func}_perf_64b_coll_latency failed to run"
  done

  # P2P Latency Tests
  let LATENCY_SIZE_8B=$SLURM_NTASKS*8
  opts="-b $LATENCY_SIZE_8B -e $LATENCY_SIZE_8B -w 1 -n 200 -c0"
  for func in alltoall scatter gather; do
    echo "=============================== $func P2P 8B LATENCY ($opts) ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$perf_files_dir/${func}_perf_p2p_8b_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_p2p_8b_latency failed to run") && echo "New ${func}_perf_p2p_8b_latency failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$baseline_perf_files_dir/${func}_perf_p2p_8b_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_p2p_8b_latency failed to run") && echo "Baseline ${func}_perf_p2p_8b_latency failed to run"
  done

  # 16B Round Robin Latency Tests
  let LATENCY_SIZE_16B=$SLURM_NTASKS*16
  opts="-b $LATENCY_SIZE_16B -e $LATENCY_SIZE_16B -w 1 -n 200 -c0"
  for func in all_gather reduce_scatter; do
    echo "=============================== $func ROUND ROBIN 16B LATENCY ($opts) ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$perf_files_dir/${func}_perf_16b_round_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_16b_round_latency failed to run") && echo "New ${func}_perf_16b_round_latency failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$baseline_perf_files_dir/${func}_perf_16b_round_latency $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_16b_round_latency failed to run") && echo "Baseline ${func}_perf_16b_round_latency failed to run"
  done

  # Overhead Tests
  opts="-t 1 -g 1 -b 512M -e 512M -n 50 -w 1 -c0 -C 1"
  for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv; do
    echo "=============================== $func OVERHEAD ($opts) ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$perf_files_dir/${func}_perf_cpu_overhead $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_cpu_perf_overhead failed to run") && echo "New ${func}_perf_cpu_overhead failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $SALLOC $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts > "$baseline_perf_files_dir/${func}_perf_cpu_overhead $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_cpu_perf_overhead failed to run") && echo "Baseline ${func}_perf_cpu_overhead failed to run"
  done

  for f in $perf_files_dir/*.txt; do
    f2=$(basename "$f")
    baseline_f="$baseline_perf_files_dir/$f2"
    echo "Comparing $f and $baseline_f"
    if [ -e "$baseline_f" ]; then
      ./test/scripts/perf_regression.py old="$baseline_f" new="$f" threshold=$threshold
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$f")
    else
      echo "$baseline_f doesn't exist"
      let failure_count=$failure_count+1 && failure_names+=("$f")
    fi
  done

fi

for str in "${failure_names[@]}"
do
  echo "Failed check: $str"
done

echo "$failure_count regressions detected"
exit $failure_count