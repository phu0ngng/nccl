#!/bin/bash

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:$PWD/build/lib:$CUDA_HOME/lib64:$PWD/ext-net/mock:$LD_LIBRARY_PATH

# Needed for collnet / p2p tuning and nic fusion tests
export NCCL_COLLNET_ENABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_GRAPH_DUMP_FILE=graph.xml
export NCCL_NET=MockPlugin

compare_graph=$1
if [ "$compare_graph" == "" ]; then compare_graph=test/nic-fusion/mock-a40-mixed-graph.xml; fi

opts="-n 1 -w 0 -c 0"
range="-b 8 -e 4G -f 2"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using NCCL_COLLNET_ENABLE=$NCCL_COLLNET_ENABLE"
echo "Using NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "Using NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "Using NCCL_GRAPH_DUMP_FILE=$NCCL_GRAPH_DUMP_FILE"
echo "Using NCCL_NET=$NCCL_NET"
echo "Using compare_graph=$compare_graph"
echo "Using $NGPUS GPUs per node"

for func in all_reduce_perf; do
  echo "=============================== $func (all sizes) - $(date +\"%T\") ================================="
  $SALLOC -n $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $range $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func singlethreaded: $func $range $opts")

  diff $NCCL_GRAPH_DUMP_FILE $compare_graph
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("singlethreaded: diff $NCCL_GRAPH_DUMP_FILE $compare_graph")

  # Multithreaded
  $SALLOC -n 1 -c $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ./build/test/perf/$func $range $opts -t $NGPUS
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func multithreaded: $func $range $opts -t $NGPUS")

  diff $NCCL_GRAPH_DUMP_FILE $compare_graph
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("multithreaded: diff $NCCL_GRAPH_DUMP_FILE $compare_graph")
done

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
  echo $NCCL_GRAPH_DUMP_FILE
  echo $compare_graph
done

echo "$failure_count tests failed"
exit $failure_count
