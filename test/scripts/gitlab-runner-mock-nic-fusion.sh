#!/bin/bash

if [ -z "$NCCL_HOME" ]; then
  export NCCL_HOME="${PWD}/build"
fi

export OPAL_PREFIX=$MPI_HOME
export LD_LIBRARY_PATH=$MPI_HOME/lib:${NCCL_HOME}/lib:$CUDA_HOME/lib64:$PWD/ext-net/mock:$LD_LIBRARY_PATH

# Needed for collnet / p2p tuning and nic fusion tests
export NCCL_COLLNET_ENABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NET=MockPlugin
export NCCL_DEBUG=WARN

compare_graph1=$1
if [ "$compare_graph1" == "" ]; then compare_graph1=test/nic-fusion/mock-a40-mixed-graph.xml; fi

compare_graph2=$2
if [ "$compare_graph2" == "" ]; then compare_graph2=test/nic-fusion/mock-a40-mixed-graph-1rpn.xml; fi

opts="-n 1 -w 0 -c 0"
range="-b 8 -e 4G -f 2"

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using MPI_HOME=$MPI_HOME"
echo "Using NCCL_HOME=$NCCL_HOME"
echo "Using UCX_TLS: $UCX_TLS"
echo "Using NCCL_COLLNET_ENABLE=$NCCL_COLLNET_ENABLE"
echo "Using NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "Using NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
echo "Using NCCL_NET=$NCCL_NET"
echo "Using compare_graph=$compare_graph"
echo "Using $NGPUS GPUs per node"

for func in all_reduce_perf; do
  export NCCL_TOPO_DUMP_FILE="topo1.xml"
  export NCCL_GRAPH_DUMP_FILE=graph1.xml
  echo "=============================== $func (all sizes) - $(date +\"%T\") ================================="
  $SALLOC -n $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ${NCCL_HOME}/test/perf/$func $range $opts
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func singlethreaded: $func $range $opts")

  echo "=============================== $func (all sizes) compare graph - $(date +\"%T\") ================================="
  diff $NCCL_GRAPH_DUMP_FILE $compare_graph1
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("singlethreaded: diff $NCCL_GRAPH_DUMP_FILE $compare_graph")

  # Multithreaded
  export NCCL_TOPO_DUMP_FILE="topo2.xml"
  export NCCL_GRAPH_DUMP_FILE=graph2.xml
  echo "=============================== $func (all sizes multithreaded) - $(date +\"%T\") ================================="
  $SALLOC -n 1 -c $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ${NCCL_HOME}/test/perf/$func $range $opts -t $NGPUS
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func multithreaded: $func $range $opts -t $NGPUS")

  echo "=============================== $func (all sizes multithreaded) compare graph - $(date +\"%T\") ================================="
  diff $NCCL_GRAPH_DUMP_FILE $compare_graph2
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("multithreaded: diff $NCCL_GRAPH_DUMP_FILE $compare_graph")

  echo "=============================== $func FORCE_MERGE (all sizes) - $(date +\"%T\") ================================="
  # FORCE_MERGE
  NCCL_NET_FORCE_MERGE="mock_0,mock_1;mock_2;mock_3;" $SALLOC -n 1 -c $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ${NCCL_HOME}/test/perf/$func $range $opts -t $NGPUS
  [ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func force_merge: NCCL_NET_FORCE_MERGE=\"mock_0,mock_1;mock_2;mock_3;\" $func $range $opts -t $NGPUS")

  echo "=============================== $func FORCE_MERGE (all sizes) expect failure - $(date +\"%T\") ================================="
  # FORCE_MERGE expect failure
  NCCL_NET_FORCE_MERGE="mock_0,mock_1;mock_2,mock_3;" $SALLOC -n 1 -c $NGPUS $MPI_HOME/bin/mpirun $MPI_PARAMS ${NCCL_HOME}/test/perf/$func $range $opts -t $NGPUS
  [ $? -eq 0 ] && let failure_count=$failure_count+1 && failure_names+=("$func force_merge_expect_failure: NCCL_NET_FORCE_MERGE=\"mock_0,mock_1;mock_2,mock_3;\" $func $range $opts -t $NGPUS")
done

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
  echo $NCCL_GRAPH_DUMP_FILE
  echo $compare_graph1
done

echo "$failure_count tests failed"
exit $failure_count
