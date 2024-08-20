#!/bin/bash
# launch a simple test In Run Container
# mount home directory
# use absolute paths
# set env vars inside container as needed
date

# fixed path in the container
OPENMPI_LIBRARY_PATH="/usr/local/openmpi/lib"

source docker/draco-oci-config.sh

script_to_execute=$(realpath $1)
root_dir=$(dirname $(dirname $(dirname $script_to_execute)))

if [ $# -eq 2 ]; then
    job_suffix="$2"
else
    job_suffix="test-manual"
fi

account=$(get_slurm_account)
run_tools_image=$(get_run_tools_image)

srun \
    --account=${account} \
    -J ${account}-nccl:${job_suffix} \
    --exclusive \
    -p batch_block1 \
    -t 00:03:00 \
    --gpus-per-node 2 \
    -n 1 \
    -c 1 \
    --mpi=pmix \
    --container-image="$run_tools_image" \
    --container-mounts="$HOME:$HOME" \
    bash -c "\
        export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$OPENMPI_LIBRARY_PATH; \
        cd $root_dir; \
        $script_to_execute"

date

