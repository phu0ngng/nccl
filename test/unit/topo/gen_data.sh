#!/bin/bash

# Typically launch with
# salloc -N {nnodes} -n {nnodes x ngpus} ./gen_data.sh {ngpus} {nnodes} {topo}
#
# Don't forget to run make test.build before you can see the data using the model test :
# make -j test.build
# (cd build/test/unit; ./model {topo})

ngpus=$1
nnodes=$2
shift 2

if [ "$nnodes" == "" ]; then
  echo "Usage : $0 <ngpus> <nnodes> [path/to/perf/tests] [topo] [path/to/topo]"
  exit 1
fi

perftest_path=build/test/perf
topo_path=.

while [ "$1" != "" ]; do
  if [ -f $1/all_reduce_perf ]; then
    perftest_path=$1
  elif [ -d test/unit/topo/$1 ]; then
    topo_path=test/unit/topo/$1
  elif [ -d $1 ]; then
    topo_path=$1
  else
    echo "Don't know what to do with extra arg $1"
  fi
  shift
done
topo_path=$topo_path/data/$ngpus/$nnodes

echo "##########################################"
echo "# Generating data to :                   #"
printf "# %38s #\n" $topo_path
echo "# Perf tests in :                        #"
printf "# %38s #\n" $perftest_path
echo "##########################################"
echo

REPS=20
nranks=`expr $ngpus \* $nnodes`

gen_data() {
  perftest=$1
  topo_path=$2
  coll=$3
  algos=$4
  protos=$5
  if [ "$algos" != "" -a "$protos" != "" ]; then
    for algo in $algos; do
      for proto in $protos; do
        dir="$topo_path/$coll/$algo/$proto"
        mkdir -p $dir
        echo "############ $coll/$algo/$proto -> $dir/time.txt #############"
        if [ "$proto" == "LL" ]; then
          maxsize="1G"
        else
          maxsize="16G"
        fi
        mpirun -np $nranks --bind-to numa -x NCCL_ALGO=$algo -x NCCL_PROTO=$proto -x NCCL_TESTS_DUMP_FILE=$dir/time.txt $perftest -w 1 -n $REPS -b 8 -e $maxsize -f 2 -c 0
      done
    done
  else
    # Default
    echo "############ $coll/Default -> $topo_path/$coll/time.txt #############"
    mpirun -np $nranks --bind-to numa -x NCCL_TESTS_DUMP_FILE=$topo_path/$coll/time.txt $perftest -w 1 -n $REPS -b 8 -e $maxsize -f 2 -c 0
  fi
}

gen_data $perftest_path/all_reduce_perf $topo_path AllReduce "Ring Tree" "LL LL128 Simple"
if [ "$nnodes" == 1 ]; then
  gen_data $perftest_path/all_reduce_perf $topo_path AllReduce "NVLS" "Simple"
else
  gen_data $perftest_path/all_reduce_perf $topo_path AllReduce "NVLSTree" "Simple"
fi
gen_data $perftest_path/all_reduce_perf $topo_path AllReduce
gen_data $perftest_path/all_gather_perf $topo_path AllGather "Ring" "LL LL128 Simple"
if [ "$ngpus" == 1 ]; then
  gen_data $perftest_path/all_gather_perf $topo_path AllGather "PAT" "Simple"
fi
gen_data $perftest_path/all_gather_perf $topo_path AllGather
gen_data $perftest_path/reduce_scatter_perf $topo_path ReduceScatter "Ring" "LL LL128 Simple"
if [ "$ngpus" == 1 ]; then
  gen_data $perftest_path/reduce_scatter_perf $topo_path ReduceScatter "PAT" "Simple"
fi
gen_data $perftest_path/reduce_scatter_perf $topo_path ReduceScatter
gen_data $perftest_path/broadcast_perf $topo_path Broadcast "Ring" "LL LL128 Simple"
gen_data $perftest_path/broadcast_perf $topo_path Broadcast
gen_data $perftest_path/reduce_perf $topo_path Reduce "Ring" "LL LL128 Simple"
gen_data $perftest_path/reduce_perf $topo_path Reduce
