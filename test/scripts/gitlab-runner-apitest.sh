#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
$SRUN ./build/test/apitest/apitest
NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest
