/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

ncclResult_t getCudaPath(int cudaDev, char** path);

static int getNumaId(char *path) {
  char npath[PATH_MAX];
  snprintf(npath, PATH_MAX, "%s/numa_node", path);
  npath[PATH_MAX-1] = '\0';

  int numaId = -1;
  FILE *file = fopen(npath, "r");
  if (file == NULL) return -1;
  if (fscanf(file, "%d", &numaId) == EOF) { fclose(file); return -1; }
  fclose(file);

  return numaId;
}

enum ncclPathDist {
  PATH_PIX  = 0,
  PATH_PXB  = 1,
  PATH_PHB  = 2,
  PATH_NODE = 3,
  PATH_SYS  = 4,
  PATH_ARRAY_SIZE = 5
};

extern const char* pathDists[PATH_ARRAY_SIZE];

int pciDistance(char* path1, char* path2);

struct ncclTopoSystem;
ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem** system, int inter);

#define NCCL_TOPO_MAX_NODES 256

#define NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP 1 // Split tree (send/recv from different ranks) always flowing in the same direction
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Split tree (send/recv from different ranks) flowing in both directions
#define NCCL_TOPO_PATTERN_TREE 3            // Simple tree (send/recv from same rank) flowing in both directions
#define NCCL_TOPO_PATTERN_RING 4            // Ring
struct ncclTopoGraph {
  int pattern;
  int crossNic;
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
