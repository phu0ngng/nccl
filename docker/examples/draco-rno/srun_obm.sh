#!/bin/bash
# launch a simple test
date

source docker/draco-rno-config.sh

script_to_execute="$1"

if [ $# -eq 2 ]; then
    job_suffix="$2"
else
    job_suffix="test-manual"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(get_extra_ld_library_path)"
account=$(get_slurm_account)

srun \
    --account=${account} \
    -J ${account}-nccl:${job_suffix} \
    -p batch_dgx1_m2 \
    -t 00:03:00 \
    --nv-meta ml-model.dlss \
    --gpus-per-node 2 \
    --mpi=pmix \
    -n 1 \
    -c 1 \
    $script_to_execute

date

