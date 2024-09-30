#!/bin/bash
# This script dumps fairshare and past job allocations on a slurm cluster
# This serves as as starting point for analysis of workload utilization
set -x
sshare -A coreai_libraries_nccl --format=Account,RawUsage,NormUsage,EffectvUsage,FairShare,LevelFS,RawShares
sacct -A coreai_libraries_nccl --format=JobId,JobName%80,User%20,Elapsed%10,ElapsedRaw,NNodes

