#!/bin/bash
export LD_LIBRARY_PATH=$PWD/build/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()
TIMEFORMAT=%R

if [ "$SRUN" == "" ]; then
  export SACCT_FORMAT_STRING="JobID,JobName%100,User%10,Partition%15,NNodes,Timelimit,$PLANNED_RESERVED"
  ./test/scripts/slurm_job_summary.sh
fi

echo "HOSTNAME=$HOSTNAME"
echo "Using CUDA_HOME=$CUDA_HOME"
echo "Using NCCL_HOME=$PWD/build"
echo "Using LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "=============================== API TESTS Default - $(date +\"%T\") ================================="
time $SRUN ./build/test/apitest/apitest
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("API TESTS Default")

if [ "$1" == "minimal" ]; then exit $failure_count; fi

echo "=============================== API TESTS No P2P - $(date +\"%T\")  ================================="
NCCL_P2P_DISABLE=1 time $SRUN ./build/test/apitest/apitest
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("API TESTS No P2P")

echo "=============================== API TESTS Network - $(date +\"%T\") ================================="
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 time $SRUN ./build/test/apitest/apitest
[ $? -ne 0 ] && let failure_count=$failure_count+1 && failure_names+=("API TESTS Network")

for str in "${failure_names[@]}"
do
  echo "Failed Step: $str"
done

echo "$failure_count tests failed"
exit $failure_count
