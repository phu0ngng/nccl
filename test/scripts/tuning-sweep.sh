#!/bin/bash

LL_SIZE_MAX_INTER="1G"
LL_SIZE_MAX_INTRA="4G"
STANDARD_SIZE_MAX="16G"
NCCL_PROTOS=("Simple" "LL" "LL128")
AR_INTER_ALGOS=("Ring" "Tree" "NVLSTree")
AR_INTRA_ALGOS=("Ring" "Tree" "NVLSTree" "NVLS")
USER_BUFFER_OPTIONS=("0" "1")
OUTER_ITERS=5

# LOG_all_gather_perf_N512n4096_SL-0_MASK-0x0_UB-1.txt

LOG_STRING="J${SLURM_JOB_NAME}_ID-${SLURM_JOB_ID}".txt

# We need to catch failures manually and then throw at the end to get gitlab to detect a failure
failure_count=0
failure_names=()

env | grep ^NCCL > nccl-envs-N$SLURM_JOB_NUM_NODES-$LOG_STRING

for iter in $(seq 1 $OUTER_ITERS); do
  if [[ "$SLURM_JOB_NUM_NODES" == "1" ]]; then
    echo "Single node cases"
    # Common options
    for nps in 2 4 8; do
      LOG_NPS=$(( $nps * $SLURM_JOB_NUM_NODES ))
      for UB in "${USER_BUFFER_OPTIONS[@]}"; do
        for NCCL_PROTO in "${NCCL_PROTOS[@]}"; do
          SIZE_MAX=$STANDARD_SIZE_MAX
          if [[ "$NCCL_PROTO" == "LL" ]]; then
            SIZE_MAX=$LL_SIZE_MAX_INTRA
          fi
          # TODO - Is PAT supported on intranode?
          for func in reduce_scatter_perf all_gather_perf; do
            for NCCL_ALGO in Ring; do
              LOG_NAME="LOG_${func}_N${SLURM_JOB_NUM_NODES}n${LOG_NPS}_iter-${iter}_NCCL_ALGO-${NCCL_ALGO}_NCCL_PROTO-${NCCL_PROTO}_UB-${UB}_$LOG_STRING"
              echo "$(date +\"%T\") $LOG_NAME"
              $MPI_HOME/bin/mpirun --map-by ppr:$nps:node $MPI_PARAMS ~/nccl/build/test/perf/$func -b8 -e $SIZE_MAX -R$UB -f2 > $LOG_NAME
            done
          done
          # Op-specific options
          for func in all_reduce_perf; do
            for NCCL_ALGO in "${AR_INTRA_ALGOS[@]}"; do
              LOG_NAME="LOG_${func}_N${SLURM_JOB_NUM_NODES}n${LOG_NPS}_iter-${iter}_NCCL_ALGO-${NCCL_ALGO}_NCCL_PROTO-${NCCL_PROTO}_UB-${UB}_$LOG_STRING"
              echo "$(date +\"%T\") $LOG_NAME"
              $MPI_HOME/bin/mpirun --map-by ppr:$nps:node $MPI_PARAMS ~/nccl/build/test/perf/$func -b8 -e $SIZE_MAX -R$UB -f2 > $LOG_NAME
            done
          done
        done
      done
    done
    exit 0
  else
  echo "Multi node cases"
   # Common options
    for nps in 1 2 4 8; do
      LOG_NPS=$(( $nps * $SLURM_JOB_NUM_NODES ))
      for UB in "${USER_BUFFER_OPTIONS[@]}"; do
        for NCCL_PROTO in "${NCCL_PROTOS[@]}"; do
          SIZE_MAX=$STANDARD_SIZE_MAX
          if [[ "$NCCL_PROTO" == "LL" ]]; then
            SIZE_MAX=$LL_SIZE_MAX_INTER
          fi
          # Op-specific options
          for func in all_reduce_perf; do
            for NCCL_ALGO in "${AR_INTER_ALGOS[@]}"; do
              LOG_NAME="LOG_${func}_N${SLURM_JOB_NUM_NODES}n${LOG_NPS}_iter-${iter}_NCCL_ALGO-${NCCL_ALGO}_NCCL_PROTO-${NCCL_PROTO}_UB-${UB}_$LOG_STRING"
              echo "$(date +\"%T\") $LOG_NAME"
              $MPI_HOME/bin/mpirun --map-by ppr:$nps:node $MPI_PARAMS ~/nccl/build/test/perf/$func -b8 -e $SIZE_MAX -R$UB -f2 > $LOG_NAME
            done
          done
          for func in reduce_scatter_perf all_gather_perf; do
            for NCCL_ALGO in Ring; do
              LOG_NAME="LOG_${func}_N${SLURM_JOB_NUM_NODES}n${LOG_NPS}_iter-${iter}_NCCL_ALGO-${NCCL_ALGO}_NCCL_PROTO-${NCCL_PROTO}_UB-${UB}_$LOG_STRING"
              echo "$(date +\"%T\") $LOG_NAME"
              $MPI_HOME/bin/mpirun --map-by ppr:$nps:node $MPI_PARAMS ~/nccl/build/test/perf/$func -b8 -e $SIZE_MAX -R$UB -f2 > $LOG_NAME
            done
    
            if [[ "$nps" == "1" ]]; then
              for NCCL_ALGO in PAT; do
                LOG_NAME="LOG_${func}_N${SLURM_JOB_NUM_NODES}n${LOG_NPS}_iter-${iter}_NCCL_ALGO-${NCCL_ALGO}_NCCL_PROTO-${NCCL_PROTO}_UB-${UB}_$LOG_STRING"
                echo "$(date +\"%T\") $LOG_NAME"
                $MPI_HOME/bin/mpirun --map-by ppr:$nps:node $MPI_PARAMS ~/nccl/build/test/perf/$func -b8 -e $SIZE_MAX -R$UB -f2 > $LOG_NAME
              done
            fi
          done
        done
      done
    done 
  fi
done
