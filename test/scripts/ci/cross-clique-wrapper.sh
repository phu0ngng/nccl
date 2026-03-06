#!/bin/bash
# Wrapper script to emulate cross-clique and multi-NVLD P2P on NVL72 systems.
# Assigns NCCL_MNNVL_CLIQUE_ID and optionally NCCL_MNNVL_UUID based on SLURM_NODEID.
#
# Usage: CLIQUE_SIZE=2 [NVLDS=1] cross-clique-wrapper.sh <command> [args...]
#
# CLIQUE_SIZE controls how many nodes form a clique:
#   CLIQUE_SIZE=1: Each node is its own clique (max cross-clique traffic)
#   CLIQUE_SIZE=2: Pairs of nodes form cliques (2 cliques in a 4-node job)
#
# NVLDS controls how many NVL domains to emulate (default: 1 = single NVLD):
#   NVLDS=1: All nodes share same UUID (single NVLD, cross-clique only)
#   NVLDS=2: Nodes split into 2 NVLDs with different UUIDs
#            e.g., 4 nodes: nodes 0-1 = NVLD 0, nodes 2-3 = NVLD 1
#            Each NVLD has its own cross-clique topology

cliqueSize=${CLIQUE_SIZE:-1}
numNvlds=${NVLDS:-1}

if [ "$numNvlds" -gt 1 ]; then
  # Multi-NVLD: compute which NVLD this node belongs to
  nodesPerNvld=$(( $(( ${SLURM_NNODES:-4} )) / numNvlds ))
  nvldId=$(( ${SLURM_NODEID} / nodesPerNvld ))
  # Clique ID is relative within the NVLD
  localNodeId=$(( ${SLURM_NODEID} % nodesPerNvld ))
  clique=$(( localNodeId / cliqueSize ))
  # Use nvldId+1 as UUID (0 means "don't override")
  NCCL_MNNVL_UUID=$(( nvldId + 1 )) NCCL_MNNVL_CLIQUE_ID=$clique NCCL_MNNVL_CROSS_CLIQUE=1 "$@"
else
  # Single NVLD: just assign clique IDs
  clique=$(( ${SLURM_NODEID} / cliqueSize ))
  NCCL_MNNVL_CLIQUE_ID=$clique NCCL_MNNVL_CROSS_CLIQUE=1 "$@"
fi
