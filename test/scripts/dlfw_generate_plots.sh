#!/bin/bash

extract_value() {
gpumodel=$1
shift 1
path=comp/$gpumodel
i=0
for version in $@; do
  dpath=${version}_dlfw/results/$gpumodel
  value=$(cat $dpath/mxnet.out | awk '/Speed/ {if (lines++) sum += $5} END {print sum/(lines-1)}')
  echo "$i $version $value" >> $path/mxnet.values
  value=$(cat $dpath/caffe2.out | awk -F'=' '/Epoch/ {if (lines++) sum += $NF} END {print sum/(lines-1)}')
  echo "$i $version $value" >> $path/caffe2.values
  value=$(cat $dpath/cntk.out | awk '/^ Epoch/ {if (lines++ > 10) sum += $NF} END {print sum/(lines-11)}')
  echo "$i $version $value" >> $path/cntk.values
  value=$(cat $dpath/pytorch.out | awk '/512 \|/ {print $7}')
  echo "$i $version $value" >> $path/pytorch.values
  value=$(cat $dpath/tensorflow.out | awk '/total images/ {print $NF}')
  echo "$i $version $value" >> $path/tensorflow.values
  i=$((i+1))
done
}

generate_plot() {
gpumodel=$1
fw=$2
shift 2

graph=comp/$gpumodel/$fw
cat > $graph.plot << EOF
set term png
set terminal png size 1280,1024
set output "$graph.png"
set title "$fw x $gpumodel" noenhanced
set boxwidth 0.5
set style fill solid
set yrange [0:]
EOF

data=comp/$gpumodel/$fw
echo -n "plot \"$data.values\" using 1:3:xtic(2) with boxes notitle, '' using 1:(\$3+30):3 with labels notitle" >> $graph.plot
#echo "" >> $graph.plot
#echo "replot" >> $graph.plot
gnuplot $graph.plot
}

gpumodel=$1
if [ "$gpumodel" == "" ]; then
  echo "Usage $0 <gpumodel>"
  exit 1
fi

shift 1
mkdir -p comp/$gpumodel
echo -n "Generating images "
extract_value $gpumodel $@
generate_plot $gpumodel mxnet $@
generate_plot $gpumodel caffe2 $@
generate_plot $gpumodel cntk $@
generate_plot $gpumodel pytorch $@
generate_plot $gpumodel tensorflow $@
echo "done."
