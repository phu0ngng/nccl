#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

echo "=============================== GRAPH TESTS Default ================================="
cd build/test/unit
./graph_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("GRAPH TESTS Default")
cd ../../../
echo -e "\n\n"

echo "=============================== FT TESTS Default - $(date +\"%T\") ================================="
./build/test/unit/ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT TESTS Default")
echo "=============================== FT TESTS Default DONE - $(date +\"%T\") ============================"
echo -e "\n\n"

echo "=============================== FT TESTS NO P2P - $(date +\"%T\") =================================="
NCCL_P2P_DISABLE=1 ./build/test/unit/ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT test No P2P")
echo "=============================== FT TESTS NO P2P DONE - $(date +\"%T\") ============================="
echo -e "\n\n"

echo "=============================== FT TESTS Network - $(date +\"%T\") ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 ./build/test/unit/ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT test Network")
echo "=============================== FT TESTS Network DONE - $(date +\"%T\") ============================"

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
