/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"

static ncclResult_t ncclTopoFollowPath(struct ncclTopoLinkList* path, struct ncclTopoNode** node, int width) {
  for (int i=0; i<path->count; i++) {
    if (path->list[i]->width < width) {
      // Can't follow this path, rewind and exit
      for (int j=0; j<i; j++) path->list[j]->width += width;
      *node = NULL;
      return ncclSuccess;
    }
    path->list[i]->width -= width;
  }
  *node = path->list[path->count-1]->remNode;
  return ncclSuccess;
}

static int gpuPciWidth(struct ncclTopoNode* gpu) {
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* gpuLink = gpu->links+l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l=0; l<pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links+l;
      if (pciLink->remNode != gpu) continue;
      return std::min(gpuLink->width, pciLink->width);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraWidth;
  int interNhops;
  int interPciWidth;
  int interWidth;    // Most important
};

static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   if ((d = (s2->interWidth - s1->interWidth))) return d;
   if ((d = (s2->interPciWidth - s1->interPciWidth))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraWidth - s1->intraWidth))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraWidth = scores[0].intraWidth;
  int intraNhops = scores[0].intraNhops;
  for (int i=1; i<count; i++) {
    if (scores[i].intraWidth != intraWidth || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, const uint64_t flag, struct ncclTopoLinkList** netPaths) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].used & flag) {
      *netPaths=system->nodes[NET].nodes[n].paths[GPU];
      return ncclSuccess;
    }
  }
  return ncclInternalError;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, flag, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    scores[count].g = g;
    scores[count].startIndex = i;
    scores[count].intraNhops = paths[g].count;
    scores[count].intraWidth = paths[g].width;
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciWidth = gpuPciWidth(system->nodes[GPU].nodes+g);
      scores[count].interWidth = netPaths[g].width;
    }
    count++;
  }

  // Sort GPUs
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }
  *countPtr = count;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

#define NCCL_SEARCH_TIMEOUT (1ULL<<20) // This should get contain all search within a second or so.

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  if (*g == -1) return ncclInternalError;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoLinkList* paths, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int g, int speed) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
  if (paths) NCCLCHECK(ncclTopoFollowPath(paths+g, &gpu, speed));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    if (paths) NCCLCHECK(ncclTopoFollowPath(paths+g, &gpu, -speed));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    graph->nChannels++;
    if (graph->nChannels*graph->speed > saveGraph->nChannels*saveGraph->speed) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels*graph->speed == system->maxSpeed*graph->netFactor) *time = -1;
    }
    if (graph->nChannels < MAXCHANNELS) {
      //printf("New Channel %d\n", graph->nChannels);
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    return ncclSuccess;
  }
  const uint64_t flag = 1ULL<<(graph->nChannels);
  graph->intra[graph->nChannels*ngpus+step] = gpu->rank;
  if (step == backToNet) {
    // first get back to NIC 
    if (system->nodes[NET].count) {
      int maxWidth = 0;
      int speed = DIVUP(graph->speed, graph->netFactor);
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n=0; n<system->nodes[NET].count; n++) {
        if (graph->crossNic != 1 && (system->nodes[NET].nodes[n].used & flag) == 0) continue;
        maxWidth = std::max(paths[n].width, maxWidth);
      }
      for (int n=0; n<system->nodes[NET].count; n++) {
        if (graph->crossNic != 1 && (system->nodes[NET].nodes[n].used & flag) == 0) continue;
        if (paths[n].width == maxWidth) {
          struct ncclTopoNode* net;
          NCCLCHECK(ncclTopoFollowPath(paths+n, &net, speed));
          if (net) {
            graph->inter[graph->nChannels*2+1] = net->id;
            //printf("GPU/%d -> NET/%d\n", gpu->id, n);
            NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, -1, backToFirstRank, forcedOrder, time));
            NCCLCHECK(ncclTopoFollowPath(paths+n, &net, -speed));
          }
        }
      }
    }
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    struct ncclTopoLinkList* paths = gpu->paths[GPU];
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    for (int i=0; i<count; i++) {
      int g = next[i];
      int nvlink = graph->nvlink;
      graph->nvlink &= paths[g].nvlink;
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, paths, step+1, backToNet, backToFirstRank, forcedOrder, time, g, graph->speed));
      graph->nvlink = nvlink;
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int g;
    int rank = graph->intra[graph->nChannels*ngpus];
    for (g=0; g<ngpus; g++) {
      if (system->nodes[GPU].nodes[g].rank == rank) break;
    }
    if (g == ngpus) {
      WARN("Could not find GPU with rank %d\n", rank);
      return ncclInternalError;
    }
    struct ncclTopoLinkList* paths = gpu->paths[GPU];
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(paths+g, &firstGpu, graph->speed));
    if (firstGpu) {
      //printf("GPU/%d -> GPU/%d (%d)\n", gpu->id, g, step+1);
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(paths+g, &firstGpu, -graph->speed));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  const int speed = DIVUP(graph->speed, graph->netFactor);
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    struct ncclTopoNode* gpu;
    if (net->used == 0) {
      graph->inter[graph->nChannels*2] = net->id;
      net->used ^= flag;
      struct ncclTopoLinkList* paths = net->paths[GPU];

      // First try the PCI order to set a reference
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, paths, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, 0, speed));
      // Then try to replay the last channel
      if (graph->nChannels > 0) {
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, paths, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, g, speed));
      }

      // Then try the most local GPUs
      int maxWidth = 0, minHops = 0xfffffff;
      for (int g=0; g<system->nodes[GPU].count; g++) {
        if (paths[g].width > maxWidth) {
          maxWidth = paths[g].width;
          minHops = paths[g].count;
        } else if (paths[g].width == maxWidth && paths[g].count < minHops) {
          minHops = paths[g].count;
        }
      }
      if (maxWidth >= speed) {
        // In the first loop, avoid using GPUs in both directions between channels (one channel
        // sending from that GPU and one channel receiving to that GPU), since that usually leads
        // to lower BW.
        for (int tryGpuBidir=0; tryGpuBidir<2; tryGpuBidir++) {
          for (int g=0; g<system->nodes[GPU].count; g++) {
            if (paths[g].width == maxWidth && paths[g].count == minHops) {
              gpu = system->nodes[GPU].nodes+g;
              int gpuUsed = gpuPciWidth(gpu) > 0 ? 0 : 1;
              if (tryGpuBidir == gpuUsed) {
                NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, paths, 0, backToNet, backToFirstRank, 0, time, g, speed));
              }
            }
          }
        }
      }
      net->used ^= flag;
    }
  }
  return ncclSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->nodes[NET].count) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
    else if (pattern == NCCL_TOPO_PATTERN_TREE) *backToNet = 0;
    else *backToNet = 1;
    if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  } else {
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING || pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Start from GPU 0
    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, NULL, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, 0, graph->speed));
    if (graph->nChannels > 0) NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, NULL, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, 0, graph->speed));
    NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, NULL, 0, backToNet, backToFirstRank, 0, time, 0, graph->speed));
  }
  return ncclSuccess;
}

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|0 2 3 1|1 3 2 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0"
 * Rings with a non-matching number of ranks are ignored so we can provide
 * rings for multiple cases.
 */
#define MAX_ENV_RANKS 512
static ncclResult_t parseGraph(const char* str, int* nChannelsRet, int ngpus, int* channels) {
  int ranks[MAX_ENV_RANKS];
  int nChannels = 0;
  int rank = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      if (status == 0) {
        ranks[rank] = digit;
        status = 1;
      } else {
        ranks[rank] = ranks[rank]*10+digit;
      }
    } else {
      if (status == 1) {
        rank++;
        if (rank == MAX_ENV_RANKS) goto end;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == '\0') {
        // Ignore if ngpus doesn't match
        if (rank != ngpus) goto newchannel;

        for (int r=0; r<ngpus; r++) {
          int rank = ranks[r];
          // Ignore if ranks are out of bounds
          if (rank < 0 || rank >= ngpus) goto newchannel;
          // Ignore if ranks are duplicate
          for (int i=0; i<r; i++)
            if (ranks[i] == rank) goto newchannel;

          channels[nChannels*ngpus+r] = rank;
        }
        nChannels++;
newchannel:
        rank = 0;
      }
    }
  } while (str[offset++] != 0);
end:
  *nChannelsRet = nChannels;
  return ncclSuccess;
}

ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* baseGraph) {
  int ngpus = system->nodes[GPU].count;
  graph->speed = 0;
  graph->nvlink = 0;
  graph->nChannels = 0;

  char* str = getenv("NCCL_GRAPH");
  if (str) {
    NCCLCHECK(parseGraph(str, &graph->nChannels, ngpus, graph->intra));
    for (int i=0; i<graph->nChannels*ngpus; i++) {
      // Translate gpu numbers into ranks
      graph->intra[i] = system->nodes[GPU].nodes[graph->intra[i]].rank;
    }
    graph->speed = PCI_WIDTH+2;
    graph->nvlink = 0;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
      // Reverse the loop
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<=ngpus/2; i++) {
          int tmp = graph->intra[ngpus*c+i];
          graph->intra[ngpus*c+i] = graph->intra[ngpus*c+(ngpus-i)%ngpus];
          graph->intra[ngpus*c+ngpus-i] = tmp;
        }
      }
    }
    if (graph->nChannels) return ncclSuccess;
  }
  if (ngpus == 1) {
    graph->speed = PCI_WIDTH;
    graph->nvlink = 1;
    graph->nChannels = 1;
    graph->intra[0] = system->nodes[GPU].nodes[0].rank;
    if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;
    return ncclSuccess;
  }

  struct ncclTopoGraph tmpGraph;
  int crossNic = (system->nodes[NET].count > 1) && graph->crossNic ? 1 : 0;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
  int bestSpeed = 0;
  for (tmpGraph.speed = system->maxWidth*graph->netFactor; tmpGraph.speed >= 6; tmpGraph.speed -= 3) {
    for (tmpGraph.crossNic = 0; tmpGraph.crossNic <= crossNic; tmpGraph.crossNic++) {
      tmpGraph.pattern = graph->pattern;
      while (1) {
        int time = NCCL_SEARCH_TIMEOUT;
        tmpGraph.nvlink = 1;
        tmpGraph.nChannels = 0;
        NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
        printf("Pattern %d, crossNic %d, Speed %d, nChannels %d %s\n", tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.speed, graph->nChannels, time == 0 ? "TIMEOUT" : "");
        for (int c=0; c<graph->nChannels; c++) {
          printf("%2d : ", c);
          for (int g=0; g<ngpus; g++) {
            printf("%d ", graph->intra[c*ngpus+g]);
          }
          printf("\n");
        }
#endif
        if (time == -1) return ncclSuccess;
        // We already have a solution and we timed out so lower speed will just timeout as well
        if (time == 0 && graph->nChannels > 0) return ncclSuccess;
        if ((graph->nChannels > 0) && (bestSpeed == 0)) bestSpeed = graph->speed;
        if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) tmpGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
        else if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
        else break;
      }
    }
    if (tmpGraph.speed <= bestSpeed/2) break;
  }
  if (graph->nChannels == 0) {
    WARN("Could not find a path for pattern %d\n", graph->pattern);
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, speed %d, nvlink %d", graph->pattern, graph->crossNic, graph->nChannels, graph->speed, graph->nvlink);
  int ngpus = system->nodes[GPU].count;

  char line[1024];
  for (int c=0; c<graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[NET], graph->inter[2*c]);
      offset = strlen(line);
    }
    for (int i=0; i<ngpus; i++) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[GPU], graph->intra[ngpus*c+i]);
      offset = strlen(line);
    }
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%d", topoNodeTypeStr[NET], graph->inter[2*c+1]);
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNetDev(struct ncclTopoGraph* graph, int dir, int channelId, int* dev) {
  *dev = graph->inter[(channelId%graph->nChannels)*2+dir];
  return ncclSuccess;
}
