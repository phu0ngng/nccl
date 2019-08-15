/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "devcomm.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

enum ncclPathDist {
  PATH_PIX  = 0,
  PATH_PXB  = 1,
  PATH_PHB  = 2,
  PATH_NODE = 3,
  PATH_SYS  = 4,
  PATH_ARRAY_SIZE = 5
};

extern const char* pathDists[PATH_ARRAY_SIZE];

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem** system, int inter);

// For unit tests
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system);

// Query topology
ncclResult_t ncclTopoGetNvlink(struct ncclTopoSystem* system, int nvmlDev1, int nvmlDev2, int* nvlink);
ncclResult_t ncclTopoHasNvlink(struct ncclTopoSystem* system, int nvmlDev, int* nvlink);
ncclResult_t ncclTopoGpuDistance(struct ncclTopoSystem* system, int nvmlDev1, int nvmlDev2, int* distance);
ncclResult_t ncclTopoGetNet(struct ncclTopoSystem* system, int nvmlDev, int id, int* net);
ncclResult_t ncclTopoNetDistance(struct ncclTopoSystem* system, int nvmlDev, int netDev, int* distance);
ncclResult_t ncclTopoCpuCount(struct ncclTopoSystem* system, int* count);

#define NCCL_TOPO_MAX_NODES 256

#define NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP 1 // Split tree (send/recv from different ranks) always flowing in the same direction
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Split tree (send/recv from different ranks) flowing in both directions
#define NCCL_TOPO_PATTERN_TREE 3            // Simple tree (send/recv from same rank) flowing in both directions
#define NCCL_TOPO_PATTERN_RING 4            // Ring
struct ncclTopoGraph {
  int pattern;
  int crossNic;
  int speed;
  int nvlink;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int nChannels;
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* baseGraph);

struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeUpRecv[MAXCHANNELS];
  int treeUpSend[MAXCHANNELS];
  int treeDnRecv[MAXCHANNELS];
  int treeDnSend[MAXCHANNELS];
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm, int* firstRanks,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph,
    struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks,
    struct ncclTopoRanks** allTopoRanks, int* rings);

#endif
