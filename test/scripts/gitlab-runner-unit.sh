#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=============================== GRAPH TESTS Default ================================="
cd build/test/unit && ./graph_test && cd ../../../

echo -e "\n\n"

echo "=============================== FT TESTS Default ================================="
./build/test/unit/ft_test