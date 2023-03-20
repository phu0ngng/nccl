#!/bin/bash

file=$HOME/Downloads/nccl_all_reduce_data.csv

for nodes in 4 8 16 32 64; do
  gpus=$(($nodes * 8))
  for algo in Tree Ring; do
    for proto in LL LL128 Simple; do
      dir=$nodes/AllReduce/$algo/$proto/
      mkdir -p $dir
      cat $file  | grep -i "$nodes,$gpus,1,$algo,$proto," | cut -d "," -f 7 | head -30 > $dir/time.txt
    done
  done
done
