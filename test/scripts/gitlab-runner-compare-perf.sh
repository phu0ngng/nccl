#!/bin/bash


threshold=$1
if [ "$threshold" == "" ]; then threshold=10; fi

shift

iterations=$1
if [ "$iterations" == "" ]; then iterations=5; fi

shift

check_cpu_overhead=$1
if [ "$check_cpu_overhead" == "" ]; then check_cpu_overhead=0; fi

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
start=$(date +%s)

# Only run comparison if the baseline exists and we are back in the dev branch
if [[ $failure_count -eq 0 ]]; then
  perf_files_dir="perf_files"
  rm -rf $perf_files_dir
  mkdir $perf_files_dir

  cpu_perf_files_dir="cpu_perf_files"
  rm -rf $cpu_perf_files_dir
  mkdir $cpu_perf_files_dir

  baseline_perf_files_dir="baseline_perf_files"
  rm -rf $baseline_perf_files_dir
  mkdir $baseline_perf_files_dir

  cpu_baseline_perf_files_dir="cpu_baseline_perf_files"
  rm -rf $cpu_baseline_perf_files_dir
  mkdir $cpu_baseline_perf_files_dir

  dev_branch_info=$(git show --oneline -s)
  baseline_branch_info=$(git log --format=%B -n 1 $ref)

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
  if [ "$GPU_APPLICATION_CLOCK" != "" ]; then
    echo "Setting all GPU application clocks to $GPU_APPLICATION_CLOCK"
    sudo nvidia-smi -ac $GPU_APPLICATION_CLOCK,$GPU_APPLICATION_CLOCK
  fi

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
  echo "Using baseline ref=$ref (non-MR REGRESSION_BASELINE=$REGRESSION_BASELINE)"
  # Print out commit message
  echo "Baseline commit message=$baseline_branch_info"
  echo "Dev branch commit message=$dev_branch_info"
  echo "Using threshold=$threshold"
  let cpu_threshold=threshold*2
  echo "Using cpu_threshold=$cpu_threshold"
  echo "Using iterations=$iterations"
  echo "Using check_cpu_overhead=$check_cpu_overhead"

  # Bandwidth Tests - No iterations
  opts="-b 512M -e 512M -d float -w 5 -n 5 -c1 --tbytes 5G" # Send no more than 10G of data for large scale tests
  for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv; do
    echo "=============================== $func BANDWIDTH ($opts) - $(date +%T)  ================================="
    export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
    $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts >> "$perf_files_dir/${func}_perf_bandwidth $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_bandwidth failed to run") && echo "New ${func}_perf_bandwidth failed to run"

    export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
    $MPI_HOME/bin/mpirun --bind-to numa -q $baseline_build_dir/test/perf/${func}_perf $opts >> "$baseline_perf_files_dir/${func}_perf_bandwidth $opts.txt"
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_bandwidth failed to run") && echo "Baseline ${func}_perf_bandwidth failed to run"
  done

  # Outer loop, appending iterations to files
  for i in $(seq 1 $iterations)
  do
    echo "Iteration # $i, total iterations=$iterations"

    # Collectives Latency Tests
    opts="-b 64 -e 64 -w10 -n50 -c0 --tbytes 512M" # Send no more than 512M of data for large scale tests
    for func in all_reduce broadcast reduce sendrecv; do
      echo "=============================== $func 64B COLLECTIVES LATENCY ($opts) - $(date +%T)  ================================="
      export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
      $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts >> "$perf_files_dir/${func}_perf_64b_coll_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_64b_coll_latency failed to run") && echo "New ${func}_perf_64b_coll_latency failed to run"

      export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
      $MPI_HOME/bin/mpirun --bind-to numa -q $baseline_build_dir/test/perf/${func}_perf $opts >> "$baseline_perf_files_dir/${func}_perf_64b_coll_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_64b_coll_latency failed to run") && echo "Baseline ${func}_perf_64b_coll_latency failed to run"
    done

    # P2P Latency Tests
    let LATENCY_SIZE_8B=$SLURM_NTASKS*8
    opts="-b $LATENCY_SIZE_8B -e $LATENCY_SIZE_8B -w10 -n50 -c0 --tbytes 512M" # Send no more than 512M of data for large scale tests
    for func in alltoall scatter gather; do
      echo "=============================== $func P2P 8B LATENCY ($opts) - $(date +%T)  ================================="
      export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
      $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts >> "$perf_files_dir/${func}_perf_p2p_8b_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_p2p_8b_latency failed to run") && echo "New ${func}_perf_p2p_8b_latency failed to run"

      export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
      $MPI_HOME/bin/mpirun --bind-to numa -q $baseline_build_dir/test/perf/${func}_perf $opts >> "$baseline_perf_files_dir/${func}_perf_p2p_8b_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_p2p_8b_latency failed to run") && echo "Baseline ${func}_perf_p2p_8b_latency failed to run"
    done

    # 16B Round Robin Latency Tests
    let LATENCY_SIZE_16B=$SLURM_NTASKS*16
    opts="-b $LATENCY_SIZE_16B -e $LATENCY_SIZE_16B -w10 -n50 -c0 --tbytes 512M" # Send no more than 512M of data for large scale tests
    for func in all_gather reduce_scatter; do
      echo "=============================== $func ROUND ROBIN 16B LATENCY ($opts) - $(date +%T)  ================================="
      export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
      $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts >> "$perf_files_dir/${func}_perf_16b_round_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_perf_16b_round_latency failed to run") && echo "New ${func}_perf_16b_round_latency failed to run"

      export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
      $MPI_HOME/bin/mpirun --bind-to numa -q $baseline_build_dir/test/perf/${func}_perf $opts >> "$baseline_perf_files_dir/${func}_perf_16b_round_latency $opts.txt"
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_perf_16b_round_latency failed to run") && echo "Baseline ${func}_perf_16b_round_latency failed to run"
    done
  done

  # Outer loop, appending iterations to files
  # Double iterations for CPU overhead tests
  if [[ "$check_cpu_overhead" != "0" ]]; then
    echo "Running CPU overhead regression checks - $(date +%T)"
    let iterations=$iterations*2
    for i in $(seq 1 $iterations)
    do
      echo "Iteration # $i, total iterations=$iterations"
      # Overhead Tests
      opts="-t 1 -g 1 -b 512M -e 512M -w10 -n500 -c0 -C1 --tbytes 20G" # "Send" no more than 20G of data for large scale tests
      for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv; do
        echo "=============================== $func OVERHEAD ($opts) - $(date +%T)  ================================="
        export LD_LIBRARY_PATH=$LD_LIBRARY_NEW
        $MPI_HOME/bin/mpirun --bind-to numa -q ./build/test/perf/${func}_perf $opts >> "$cpu_perf_files_dir/${func}_perf_cpu_overhead $opts.txt"
        [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("New ${func}_cpu_perf_overhead failed to run") && echo "New ${func}_perf_cpu_overhead failed to run"

        export LD_LIBRARY_PATH=$LD_LIBRARY_OLD
        $MPI_HOME/bin/mpirun --bind-to numa -q $baseline_build_dir/test/perf/${func}_perf $opts >> "$cpu_baseline_perf_files_dir/${func}_perf_cpu_overhead $opts.txt"
        [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Baseline ${func}_cpu_perf_overhead failed to run") && echo "Baseline ${func}_perf_cpu_overhead failed to run"
      done
    done

    for f in $cpu_perf_files_dir/*.txt; do
      f2=$(basename "$f")
      baseline_f="$cpu_baseline_perf_files_dir/$f2"
      echo "Comparing $f and $baseline_f"
      if [ -e "$baseline_f" ]; then
        ./test/scripts/perf_regression.py old="$baseline_f" new="$f" threshold=$cpu_threshold rmad_threshold=0.2
        [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$f")
      else
        echo "$baseline_f doesn't exist"
        let failure_count=$failure_count+1 && failure_names+=("$f")
      fi
    done
  fi

  for f in $perf_files_dir/*.txt; do
    f2=$(basename "$f")
    baseline_f="$baseline_perf_files_dir/$f2"
    echo "Comparing $f and $baseline_f"
    if [ -e "$baseline_f" ]; then
      ./test/scripts/perf_regression.py old="$baseline_f" new="$f" threshold=$threshold rmad_threshold=0.2
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

echo "$failure_count regressions detected comparing dev branch $dev_branch_info ($CI_COMMIT_SHORT_SHA) and baseline $baseline_branch_info ($ref)"

end=$(date +%s)
let seconds=$end-$start
printf 'Test completed in %dh:%dm:%ds\n' $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
exit $failure_count