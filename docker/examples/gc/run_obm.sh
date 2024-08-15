#!/bin/bash -x
# run the test on bare metal and without slurm
source docker/gc-config.sh

script_to_execute="$1"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(get_extra_ld_library_path)"

$script_to_execute

