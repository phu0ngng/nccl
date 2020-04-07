#!/bin/bash -e

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

gen_data() {
  perftest=$1
  topo_path=$2
  coll=$3
  algos="Ring"
  if [ "$coll" == "AllReduce" ]; then
    algos="$algos Tree"
  fi
  for algo in $algos; do
    for proto in LL LL128 Simple; do
      dir="$topo_path/$coll/$algo/$proto"
      mkdir -p $dir
      echo "############ $coll/$algo/$proto -> $dir/time.txt #############"
      mpirun -x NCCL_ALGO=$algo -x NCCL_PROTO=$proto -x NCCL_TESTS_DUMP_FILE=$dir/time.txt $perftest -w 1 -n $REPS -b 8 -e 4G -f 2 -o all -c 0
    done
  done
  # Default
  echo "############ $coll/Default -> $topo_path/$coll/time.txt #############"
  mpirun -x NCCL_TESTS_DUMP_FILE=$topo_path/$coll/time.txt $perftest -w 1 -n $REPS -b 8 -e 4G -f 2 -o all -c 0
}

gen_data $perftest_path/all_reduce_perf $topo_path AllReduce
gen_data $perftest_path/broadcast_perf $topo_path Broadcast
gen_data $perftest_path/reduce_perf $topo_path Reduce
gen_data $perftest_path/all_gather_perf $topo_path AllGather
gen_data $perftest_path/reduce_scatter_perf $topo_path ReduceScatter
