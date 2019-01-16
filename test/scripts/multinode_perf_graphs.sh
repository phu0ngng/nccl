#!/bin/bash

generate_perf() {
gpumodel=$1
nnode=$2
nproc=$3
nthread=$4
ngpus=$5
op=$6

if [ "$NCCL_IB_DISABLE" == "1" ]; then
  resdir="results_socket"
else
  resdir="results_multinode"
fi

timeout=10

mkdir -p $resdir/$gpumodel/

result=$resdir/$gpumodel/$op.$nproc.$nthread.$ngpus

nperproc=$(expr $nthread \* $ngpus)

if [ "$SLURM" == "1" ]; then
  salloc_cmd="salloc -p $gpumodel $req_hosts -N $nnode -n $nproc -c $nperproc -t ${timeout} --exclusive "
else
  mpi_hosts="-host $gpumodel -oversubscribe "
  if [ "$MPI_HOME" == "" ]; then
    echo "Please specify MPI_HOME by: export MPI_HOME=/path/to/MPI"
    exit 1
  fi
  prefix="--prefix $MPI_HOME "
fi

npn=$(expr $nproc / $nnode)

$salloc_cmd mpirun $prefix $mpi_hosts --bind-to none -x NCCL_DEBUG -x NCCL_LAUNCH_MODE -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000 -e 1960000 -i 40000 -d all -w 5 -n 20 2>&1 | tee $result.out
$salloc_cmd mpirun $prefix $mpi_hosts --bind-to none -x NCCL_DEBUG -x NCCL_LAUNCH_MODE -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 2000000 -e 38000000 -i 2000000 -d all -w 2 -n 4 2>&1 | tee -a $result.out
$salloc_cmd mpirun $prefix $mpi_hosts --bind-to none -x NCCL_DEBUG -x NCCL_LAUNCH_MODE -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 40000000 -e 400000000 -i 40000000 -d all -w 1 -n 2 2>&1 | tee -a $result.out

# latency test
resdir+="_latency"
mkdir -p $resdir/$gpumodel/
result=$resdir/$gpumodel/$op.$nproc.$nthread.$ngpus

$salloc_cmd mpirun $prefix $mpi_hosts --bind-to none -x NCCL_DEBUG -x NCCL_LAUNCH_MODE -np $nproc -npernode $npn test/perf/${op}_perf -t $nthread -g $ngpus -b 64 -e 128K -f 2 -w 5 -n 20 2>&1 | tee $result.out
}

perf_ptg_loop() {
gpumodel=$1
nnode=$2
maxproc=$3
maxthread=$4
maxgpu=$5
op=$6

declare -i nproc=2
declare -i nthread=1
declare -i ngpus=1
while [[ $nproc -le $maxproc ]] ; do
  while [[ $nthread -le $maxthread ]] ; do
    while [[ $ngpus -le $maxgpu ]]; do
      echo "Running test/perf/${op}_perf on $nnode nodes, $nproc processes, each process having $nthread threads with $ngpus GPUs ..."
      generate_perf $gpumodel $nnode $nproc $nthread $ngpus $op
      ngpus+=$ngpus
    done
    nthread+=$nthread
    ngpus=$maxgpu
  done
  nproc+=$nproc
  nthread=$maxthread
  ngpus=$maxgpu
done
}

gpumodel=$1
nnode=$2
maxproc=$3
maxthread=$4
maxgpu=$5

if [ "$maxgpu" == "" ]; then
  echo "Usage : $0 <gpumodel> <nnode> <maxproc> <maxthread> <maxgpu>"
  exit 1
fi

export NCCL_DEBUG=INFO

perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu reduce
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu all_reduce
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu reduce_scatter
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu all_gather
perf_ptg_loop $gpumodel $nnode $maxproc $maxthread $maxgpu broadcast
