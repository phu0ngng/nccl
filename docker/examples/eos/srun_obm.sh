#!/bin/bash
# launch a simple test On Bare Metal
date

source docker/eos-config.sh

script_to_execute="$1"

if [ $# -eq 2 ]; then
    job_suffix="$2"
else
    job_suffix="test-manual"
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(get_extra_ld_library_path)
account=$(get_slurm_account)

srun \
    --account=${account} \
    -J ${account}-nccl:${job_suffix} \
    --exclusive \
    -p batch \
    -t 00:03:00 \
    -n 1 \
    -c 1 \
    --mpi=pmix \
    $script_to_execute

date

