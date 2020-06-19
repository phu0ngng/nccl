#!/bin/bash

set -e

platform=$1
nodes=$2
mode=${3:-"time"}
coll=${4:-"AllReduce"}
gpus=$5

if [ "$nodes" == "" ]; then
  echo "Usage : $0"
  echo "        <platform>"
  echo "        <nodes>"
  echo "      [ <mode=time/algbw/busbw>"
  echo "      [ <coll=AllReduce/AllGather/Broadcast/Reduce/ReduceScatter>"
  echo "      [ <gpus> ]]]"
  exit 1
fi

#if [ "$coll" == "" ]; then coll=AllReduce; fi
#if [ "$mode" == "" ]; then mode=time; fi
if [ "$gpus" == "" ]; then 
  gpus=1
  for g in `ls $platform/data/`; do
    if [ "$g" -gt "$gpus" ]; then gpus=$g; fi
  done
fi

path=$platform/data/$gpus/$nodes/$coll

gen_bw() {
  factor=$1
  size=8
  while read line; do
    echo "$size / $line * $factor" | bc -l
    size=`echo "$size * 2" | bc -l`;
  done
}

if [ "$mode" != "time" ]; then
  factor=1
  if [ "$mode" == "busbw" ]; then
    nranks=`echo "$gpus * $nodes" | bc`
    factor=`echo "2 * ( $nranks - 1) / $nranks" | bc -l`
  fi
  # Generate alg/bus BW files
  for p in $path $path/Ring/LL $path/Ring/LL128 $path/Ring/Simple $path/Tree/LL $path/Tree/LL128 $path/Tree/Simple; do
    cat $p/time.txt | gen_bw $factor > $p/$mode.txt
  done
fi

OUTPUT="nccl-${coll}-${mode}-${platform}-${gpus}x${nodes}.png"
(
echo    'set term pngcairo size 1200, 800'
echo    'set output "'$OUTPUT'"'
if [ "$mode" == "time" ]; then echo    'set logscale y'; fi
echo    'set title "NCCL '$coll' '$mode', '$platform', GPUs('$gpus') x '$nodes'"'
echo -n 'set xtics (                 "8"   0, "16"   1, "32"   2, "64"   3, "128"   4, "256"   5, "512"   6, '
echo -n ' "1K"  7, "2K"  8, "4K"  9, "8K" 10, "16K" 11, "32K" 12, "64K" 13, "128K" 14, "256K" 15, "512K" 16, '
echo -n ' "1M" 17, "2M" 18, "4M" 19, "8M" 20, "16M" 21, "32M" 22, "64M" 23, "128M" 24, "256M" 25, "512M" 26, '
echo    ' "1G" 27, "2G" 28, "4G" 29)'
echo    'set xtics rotate by 60 right'
echo    'set grid xtics ytics mytics'
echo    'set key left'
echo -n "plot '$path/$mode.txt'             title 'Default'     with linespoint lt rgb "'"#333333"'" lw 2 dt 3,"
echo -n "     '$path/Ring/LL/$mode.txt'     title 'Ring/LL'     with linespoint lt rgb "'"#9999FF"'" lw 1,"
echo -n "     '$path/Ring/LL128/$mode.txt'  title 'Ring/LL128'  with linespoint lt rgb "'"#6666FF"'" lw 1,"
echo -n "     '$path/Ring/Simple/$mode.txt' title 'Ring/Simple' with linespoint lt rgb "'"#0000FF"'" lw 1,"
echo -n "     '$path/Tree/LL/$mode.txt'     title 'Tree/LL'     with linespoint lt rgb "'"#FF9999"'" lw 1,"
echo -n "     '$path/Tree/LL128/$mode.txt'  title 'Tree/LL128'  with linespoint lt rgb "'"#FF6666"'" lw 1,"
echo    "     '$path/Tree/Simple/$mode.txt' title 'Tree/Simple' with linespoint lt rgb "'"#FF0000"'" lw 1"
) | gnuplot

if [ "$mode" != "time" ]; then
  # Remove all generated alg/bus BW files
  for p in $path $path/Ring/LL $path/Ring/LL128 $path/Ring/Simple $path/Tree/LL $path/Tree/LL128 $path/Tree/Simple; do
    rm $p/$mode.txt
  done
fi

echo $OUTPUT

