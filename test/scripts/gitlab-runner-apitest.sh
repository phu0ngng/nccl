#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count = 0

echo "=============================== API TESTS Default ================================="
$SRUN ./build/test/apitest/apitest
[ $? -neq 0 ] && failure_count=$($failure_count + 1)

if [ "$1" == "minimal" ]; then exit $failure_count; fi

echo "=============================== API TESTS No P2P  ================================="
NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest
[ $? -neq 0 ] && failure_count=$($failure_count + 1)

echo "=============================== API TESTS Network ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 $SRUN ./build/test/apitest/apitest
[ $? -neq 0 ] && failure_count=$($failure_count + 1)

exit $failure_count