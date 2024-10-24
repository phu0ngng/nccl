#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN

max=$1
if [ "$max" == "" ]; then max=1G; fi

shift
graph=$1
if [ "$graph" == "" ]; then graph=0; fi

shift
nvls=$1
if [ "$nvls" == "" ]; then nvls=0; fi

if [ "$SALLOC" == "" ]; then
  export SACCT_FORMAT_STRING="JobID,JobName%100,User%10,Partition%15,NNodes,Timelimit,$PLANNED_RESERVED"
  ./test/scripts/slurm_job_summary.sh
fi

opts="-w 1 -G $graph -s 512M"
range="-b 8 -e $max -f 2"
enable_ft="-B 0 -F 1"
enable_split_test="-S 1 -P 1"
split_range="-b 8 -e 1G -f 2"
enable_local_register="-R 1"
enable_graph_register="-G 1"
enable_parallel_init="-p 1"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using $NGPUS GPUs per node"
echo "SKIP_MULTI_GPU=$SKIP_MULTI_GPU"
echo "SKIP_FT=$SKIP_FT"

for func in all_reduce_perf reduce_perf reduce_scatter_perf broadcast_perf all_gather_perf alltoall_perf gather_perf scatter_perf sendrecv_perf; do
  echo "=============================== $func (all sizes) - $(date +\"%T\") ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $range $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (all sizes): $func $range $opts")
done

if [ "$NGPUS" -ge "3" ];
then
  for func in all_reduce_perf alltoall_perf; do
    let np=$NNODES
    echo "=============================== $func Multi-thread 3-GPU per-node (all sizes) - $(date +\"%T\") ================================="
    $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node ./build/test/perf/$func $range $opts -t 3 -g 1 -n 1
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func Multi-thread 3-GPU per-node (all sizes): $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node $func $range $opts -t 3 -g 1 -n 1")

    if [ "$NGPUS" -ge "6" ];
    then
      echo "=============================== $func Multi-thread 6-GPU per-node (all sizes) - $(date +\"%T\") ================================="
      $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node ./build/test/perf/$func $range $opts -t 6 -g 1 -n 1
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func Multi-thread 6-GPU per-node (all sizes): $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node $func $range $opts -t 6 -g 1 -n 1")
    fi
  done
fi

if [ "$SKIP_MULTI_GPU" != "1" ];
then
  let np=$NNODES
  for func in all_reduce_perf alltoall_perf; do
    echo "=============================== $func All-GPU (all sizes) - $(date +\"%T\") ================================="
    $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node ./build/test/perf/$func $range $opts -t 1 -g $NGPUS -n 1
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func All-GPU (all sizes): $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node $func $range $opts -t 1 -g $NGPUS -n 1")

    let nthreads=$NGPUS/2
    if [ $nthreads -gt 0 ]
    then
      echo "=============================== $func 2-GPU (parallel init all sizes) - $(date +\"%T\") ==================="
      $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node ./build/test/perf/$func $range $opts -t $nthreads -g2 $enable_parallel_init -n 1
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func 2-GPU (parallel init all sizes): $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node $func $range $opts -t $nthreads -g2 $enable_parallel_init -n 1")
    fi

    let nthreads=$NGPUS/4
    if [ $nthreads -gt 0 ]
    then
      echo "=============================== $func 4-GPU (all sizes) - $(date +\"%T\") ================================="
      $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node ./build/test/perf/$func $range $opts -t $nthreads -g4 -n 1
      [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func 4-GPU (all sizes): $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -np $np --map-by ppr:1:node $func $range $opts -t $nthreads -g4 -n 1")
    fi
  done
fi

rangetype="-b 16M -e 16M -o all -d all -n 5"
for func in all_reduce_perf reduce_perf reduce_scatter_perf; do
  echo "=============================== $func (all ops/dtype) - $(date +\"%T\")  ================================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $rangetype $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (all ops/dtype): $func $rangetype $opts")
done

for func in all_reduce reduce reduce_scatter broadcast all_gather alltoall gather scatter sendrecv hypercube; do
  echo "=============================== $func (split share all sizes) - $(date +\"%T\") =========================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/${func}_perf $split_range $opts $enable_split_test -n 1
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (split share all sizes): ${func}_perf $split_range $opts $enable_split_test")
done

if [ "$nvls" == "1" ]; then
  export NCCL_ALGO=NVLS
  for func in all_reduce reduce_scatter all_gather; do
    echo "=============================== $func NVLS (local registration all sizes) - $(date +\"%T\") =========================="
    $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_ALGO ./build/test/perf/${func}_perf $range $opts $enable_local_register -n 1
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func NVLS (local registration all sizes): NCCL_ALGO=NVLS ${func}_perf $range $opts $enable_local_register -n 1")
  done
fi

export NCCL_ALGO=Tree
echo "=============================== all_reduce Tree (local registration all sizes) - $(date +\"%T\") =========================="
$SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_ALGO ./build/test/perf/all_reduce_perf $range $opts $enable_local_register -n 1
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("all_reduce Tree (local registration all sizes): NCCL_ALGO=Tree all_reduce_perf $range $opts $enable_local_register -n 1")

export NCCL_ALGO=Ring
for func in all_reduce all_gather broadcast; do
  echo "=============================== $func Ring (local registration all sizes) - $(date +\"%T\") =========================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_ALGO ./build/test/perf/${func}_perf $range $opts $enable_local_register -n 1
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func Ring (local registration all sizes): NCCL_ALGO=Ring ${func}_perf $range $opts $enable_local_register -n 1")
done

for func in all_reduce all_gather broadcast; do
  echo "=============================== $func Ring (graph registration all sizes) - $(date +\"%T\") =========================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_ALGO ./build/test/perf/${func}_perf -b 1G -e 1G -n 1 -w 1 $enable_graph_register
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func Ring (graph registration all sizes): ${func}_perf -b 1G -e 1G -n 1 -w 1 $enable_graph_register")
done

if [ "$IS_DRACO_OCI_IAD" != "1" ]; then
  export NCCL_ALGO=Ring
  for func in all_reduce all_gather broadcast; do
    echo "=============================== $func Ring 1RPN (local registration all sizes) - $(date +\"%T\") =========================="
    $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_ALGO -x NCCL_SHM_DISABLE=1 -x NCCL_P2P_DISABLE=1 ./build/test/perf/${func}_perf $range $opts $enable_local_register
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func Ring 1RPN (local registration all sizes): ${func}_perf $range $opts $enable_local_register")
  done

  for func in sendrecv alltoall; do
    echo "=============================== $func (local registration all sizes) - $(date +\"%T\") =========================="
    $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS -x NCCL_PXN_DISABLE=1 ./build/test/perf/${func}_perf $range $opts $enable_local_register
    [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func (local registration all sizes): ${func}_perf $range $opts $enable_local_register")
  done
fi
unset NCCL_ALGO

export NCCL_DEBUG=""
echo "=============================== all_reduce (Output File) - $(date +\"%T\") ================================="
$SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/all_reduce_perf -b8 -e8 -w0 -n1 -J test_out.json
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("all_reduce (Output File): all_reduce_perf -b8 -e8 -w0 -n1 -J test_out.json")

export NCCL_DEBUG="" # disable WARN information
echo "=============================== all_reduce (FT tests) - $(date +\"%T\") ================================="
if [ "$SKIP_FT" != "1" ]
then
  if [ "$SKIP_FT_INIT" == "1" ]
  then
    echo "Skipping init FT test..."
    enable_ft="$enable_ft -L allreduce,alltoall,finalize,split,abort"
  fi
  NCCL_SOCKET_RETRY_SLEEP_MSEC=1 $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/all_reduce_perf $range $opts $enable_ft
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("all_reduce (FT tests): NCCL_SOCKET_RETRY_SLEEP_MSEC=1 all_reduce_perf $range $opts $enable_ft")
else
  echo "Skipping FT tests..."
fi

export NCCL_NET_MERGE_LEVEL=PHB
for func in all_reduce alltoall; do
  echo "=============================== $func NIC Fusion (PHB) - $(date +\"%T\") =========================="
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/${func}_perf -b 8 -e 128M -f2 $opts -n 1
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func NIC Fusion (PHB): export NCCL_NET_MERGE_LEVEL=PHB; ${func}_perf $range $opts -n 1")

  # Multithreaded
  $SALLOC $MPI_HOME/bin/mpirun $MPI_PARAMS --map-by ppr:1:node ./build/test/perf/${func}_perf -b 8 -e 128M -f2 $opts -t $NGPUS -n 1
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func NIC Fusion (PHB): export NCCL_NET_MERGE_LEVEL=PHB; ${func}_perf $range $opts -t $NGPUS -n 1")
done

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
