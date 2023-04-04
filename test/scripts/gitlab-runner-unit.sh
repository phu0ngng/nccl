#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=============================== GRAPH TESTS Default ================================="
cd build/test/unit && ./graph_test && cd ../../../

echo -e "\n\n"

echo "=============================== FT TESTS Default - $(date +\"%T\") ================================="
./build/test/unit/ft_test
echo "=============================== FT TESTS Default DONE - $(date +\"%T\") ============================"
echo -e "\n\n"

echo "=============================== FT TESTS NO P2P - $(date +\"%T\") =================================="
NCCL_P2P_DISABLE=1 ./build/test/unit/ft_test
echo "=============================== FT TESTS NO P2P DONE - $(date +\"%T\") ============================="
echo -e "\n\n"

echo "=============================== FT TESTS Network - $(date +\"%T\") ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 ./build/test/unit/ft_test
echo "=============================== FT TESTS Network DONE - $(date +\"%T\") ============================"
