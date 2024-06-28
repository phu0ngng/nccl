#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()
cd build/test/unit
TIMEFORMAT=%R

echo "=============================== ENQUEUE TESTS ARGS - $(date +\"%T\") ================================="
NCCL_WORK_FIFO_BYTES=0 NCCL_WORK_ARGS_BYTES=512 time $SRUN --export=ALL,LD_LIBRARY_PATH ./enqueue_test 2>&1
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("ENQUEUE TESTS ARGS")
echo -e "\n\n"

echo "=============================== ENQUEUE TESTS FIFO - $(date +\"%T\") ================================="
NCCL_WORK_FIFO_BYTES=1024 time $SRUN --export=ALL,LD_LIBRARY_PATH ./enqueue_test time 2>&1
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("ENQUEUE TESTS FIFO")
echo -e "\n\n"

echo "=============================== GRAPH TESTS Default - $(date +\"%T\") ================================="
time $SRUN --export=ALL,LD_LIBRARY_PATH ./graph_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("GRAPH TESTS Default")
echo -e "\n\n"

echo "=============================== Single-Process Mem Leak TESTS Default - $(date +\"%T\") ================================="
ASAN_OPTIONS=protect_shadow_gap=0 time $SRUN --export=ALL,LD_LIBRARY_PATH ./comm_leak_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Single-Process Mem Leak TESTS Default")
echo -e "\n\n"

echo "=============================== Single-Process Mem Leak TESTS NO P2P - $(date +\"%T\") ================================="
ASAN_OPTIONS=protect_shadow_gap=0 NCCL_P2P_DISABLE=1 time $SRUN --export=ALL,LD_LIBRARY_PATH ./comm_leak_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Single-Process Mem Leak TESTS NO P2P")
echo -e "\n\n"

echo "=============================== Single-Process Mem Leak TESTS Network - $(date +\"%T\") ================================="
ASAN_OPTIONS=protect_shadow_gap=0 NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 time $SRUN --export=ALL,LD_LIBRARY_PATH ./comm_leak_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("Single-Process Mem Leak TESTS Network")
echo -e "\n\n"

echo "=============================== FT TESTS Default - $(date +\"%T\") ================================="
time $SRUN --export=ALL,LD_LIBRARY_PATH ./ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT TESTS Default")
echo "=============================== FT TESTS Default DONE - $(date +\"%T\") ============================"
echo -e "\n\n"

echo "=============================== FT TESTS NO P2P - $(date +\"%T\") =================================="
NCCL_P2P_DISABLE=1 time $SRUN --export=ALL,LD_LIBRARY_PATH ./ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT TESTS NO P2P")
echo "=============================== FT TESTS NO P2P DONE - $(date +\"%T\") ============================="
echo -e "\n\n"

echo "=============================== FT TESTS Network - $(date +\"%T\") ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 time $SRUN --export=ALL,LD_LIBRARY_PATH ./ft_test
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("FT TESTS Network")
echo "=============================== FT TESTS Network DONE - $(date +\"%T\") ============================"

echo "=============================== PLUGIN TESTS Net/Tuner - $(date +\"%T\") ==========================="
time $SRUN --export=ALL,LD_LIBRARY_PATH make -C ../../../ext-mixed/example test
[ $? -ne 0 ] && let failure_count=$failer_count+1 && failure_names+=("PLUGIN TESTS Net/Tuner")
echo "=============================== PLUGIN TESTS Net/Tuner DONE - $(date +\"%T\") ======================"

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
