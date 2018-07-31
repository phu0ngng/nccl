#!/bin/bash

mode=$1
shift 1

versions="$@"

base="$1"

vers="$1"
shift
while [ "$1" != "" ]; do
vers="$vers / $1"
shift
done
 
nnode=2
maxproc=16
maxthread=8
maxgpus=8
gpumodels=`ls $base/results`

mkdir -p multinode
html=multinode/index.html
cat > $html << EOF
<head><title>NCCL comparison - $vers</title></head>
<html>
<body>
<h1>NCCL comparison - $vers</h1>
EOF

for gpu in $gpumodels; do
  ./multinode_generate_plots.sh $mode $gpu $nnode $maxproc $maxthread $maxgpus $versions
#  ./generate_diffs.sh $gpu $nnode $maxproc $maxthread $maxgpus $versions
done

for op in all_reduce all_gather reduce_scatter reduce broadcast; do
  echo "<h2>$op</h2>" >> $html
  for gpu in $gpumodels; do
    echo "<h3>$gpu</h3>" >> $html
    echo "<table border=0><tr>" >> $html
    echo "<td> P2 x T1 x G1 </td>" >> $html
    echo "<td> P2 x T1 x G2 </td>" >> $html
    echo "<td> P2 x T2 x G2 </td>" >> $html
    echo "<td> P4 x T2 x G2 </td>" >> $html
    echo "</tr><tr>" >> $html
    echo "<td><a href=\"$gpu/$op-$nnode-2-1-1.png\"><img width=300 src=\"$gpu/$op-$nnode-2-1-1.png\"></a></td>" >> $html
    echo "<td><a href=\"$gpu/$op-$nnode-2-1-2.png\"><img width=300 src=\"$gpu/$op-$nnode-2-1-2.png\"></a></td>" >> $html
    echo "<td><a href=\"$gpu/$op-$nnode-2-2-2.png\"><img width=300 src=\"$gpu/$op-$nnode-2-2-2.png\"></a></td>" >> $html
    echo "<td><a href=\"$gpu/$op-$nnode-4-2-2.png\"><img width=300 src=\"$gpu/$op-$nnode-4-2-2.png\"></a></td>" >> $html
    echo "</tr></table>" >> $html
  done
done

echo "</body></html>" >> $html
