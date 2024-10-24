#!/bin/bash
if [ "$PLANNED_RESERVED" != "Skip" ]; then
    sacct -X -j $SLURM_JOB_ID --format $SACCT_FORMAT_STRING
fi
