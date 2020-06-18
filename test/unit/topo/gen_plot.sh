#!/bin/bash

platform=$1
nodes=$2
coll=$3
gpus=$4

if [ "$coll" == "" ]; then coll=AllReduce; fi
if [ "$gpus" == "" ]; then 
  gpus=1
  for g in `ls $platform/data/`; do
    if [ "$g" -gt "$gpus" ]; then gpus=$g; fi
  done
fi

path=$platform/data/$gpus/$nodes/$coll

(
echo    'set term pngcairo size 1200, 800'
echo    'set output "nccl-perf-'$platform'-'$gpus'x'$nodes'.png"'
echo    'set logscale y'
echo    'set title "NCCL '$coll' time, '$gpus' GPUs('$platform') x '$nodes'"'
echo -n 'set xtics (                 "8"   0, "16"   1, "32"   2, "64"   3, "128"   4, "256"   5, "512"   6, '
echo -n ' "1K"  7, "2K"  8, "4K"  9, "8K" 10, "16K" 11, "32K" 12, "64K" 13, "128K" 14, "256K" 15, "512K" 16, '
echo -n ' "1M" 17, "2M" 18, "4M" 19, "8M" 20, "16M" 21, "32M" 22, "64M" 23, "128M" 24, "256M" 25, "512M" 26, '
echo    ' "1G" 27, "2G" 28, "4G" 29)'
echo    'set xtics rotate by 60 right'
echo -n "plot '$path/time.txt'             title 'Default'     with lines lt rgb "'"#333333"'" lw 2 dt 3,"
echo -n "     '$path/Ring/LL/time.txt'     title 'Ring/LL'     with lines lt rgb "'"#9999FF"'" lw 1,"
echo -n "     '$path/Ring/LL128/time.txt'  title 'Ring/LL128'  with lines lt rgb "'"#6666FF"'" lw 1,"
echo -n "     '$path/Ring/Simple/time.txt' title 'Ring/Simple' with lines lt rgb "'"#0000FF"'" lw 1,"
echo -n "     '$path/Tree/LL/time.txt'     title 'Tree/LL'     with lines lt rgb "'"#FF9999"'" lw 1,"
echo -n "     '$path/Tree/LL128/time.txt'  title 'Tree/LL128'  with lines lt rgb "'"#FF6666"'" lw 1,"
echo    "     '$path/Tree/Simple/time.txt' title 'Tree/Simple' with lines lt rgb "'"#FF0000"'" lw 1"
) | gnuplot
