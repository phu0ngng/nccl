#!/bin/bash

if [ "$1" == "" ]; then
  echo "Usage: $0 <ngpus>"
  exit 1
fi

ngpus=`expr $1 - 1`

get_random_type() {
  set -- ncclInt8 ncclInt32 ncclInt64 ncclFloat ncclHalf ncclDouble
  shift `expr $RANDOM % 6`
  echo $1
}

echo "ncclGroupStart"
for source in `seq 0 $ngpus`; do
  for destination in `seq $source $ngpus`; do
     size=`awk "BEGIN{print $RANDOM * 2 ^ ($RANDOM%12+1)}"`
     datatype=`get_random_type`
     echo "ncclSend $source $size $destination $datatype"
     echo "ncclRecv $destination $size $source $datatype"
  done
done
echo "ncclGroupEnd"

