#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "NCCL_HOME: $NCCL_HOME"
echo "CUDA_HOME: $CUDA_HOME"
echo "pwd: $(pwd)"

echo "=============================== API TESTS Default ================================="
$SRUN ./build/test/apitest/apitest
if [ "$1" == "minimal" ]; then exit 0; fi

echo "=============================== API TESTS No P2P  ================================="
NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest

echo "=============================== API TESTS Network ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest
