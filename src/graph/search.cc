/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int id, struct ncclTopoLinkList** path) {
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %d\n", t, id);
  return ncclInternalError;
}

static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  nextNodeList.count = 0;
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
  }
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->width = 0xfffffff;
  basePath->nvlink = 1;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        int width = std::min(path->width, link->width);
        if (remPath->width < width) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          // Copy the rest of the path
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->width = width;
          remPath->nvlink = path->nvlink & ((link->type == LINK_NVL) ? 1 : 0);

          // Add to the list for the next iteration if not already in the list
          // Disallow GPUs as intermediate steps for now
          if (remNode->type != GPU) {
            int i;
            for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
            if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
          }
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  char line[1024];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%X :", topoNodeTypeStr[node->type], node->id);
#else
  sprintf(line, "%s/%X :", topoNodeTypeStr[node->type], node->id);
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        sprintf(line+offset, "--%s->%s/%X", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type], remNode->id);
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%d)", line, node->paths[t][n].width);
#else
      sprintf(line+offset, "%s/%X (%d/%d%s) ", topoNodeTypeStr[t], n, node->paths[t][n].count, node->paths[t][n].width, node->paths[t][n].nvlink ? "/N" : "");
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

static ncclResult_t getGpuSpeed(struct ncclTopoNode* node, int* speed) {
  int nvlSpeed = 0;
  int nvlPeers = 0;
  for (int l=0; l<node->nlinks; l++) {
    if (node->links[l].type == LINK_NVL) nvlSpeed += node->links[l].width;
    if (node->links[l].remNode->type == GPU) nvlPeers++; else nvlPeers = 2;
  }
  *speed = std::min(*speed, std::max(nvlSpeed, PCI_WIDTH));
  return ncclSuccess;
}

static ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  if (system->searchInitDone) return ncclSuccess;
  system->maxSpeed = 0xfffffff;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
    NCCLCHECK(getGpuSpeed(system->nodes[GPU].nodes+g, &system->maxSpeed));
  }
  if (system->nodes[NET].count) {
    for (int n=0; n<system->nodes[NET].count; n++) {
      NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
    }
    // Try to assign one NIC per GPU
    int netMaxSpeed = 0;
    int netMaxSpeedCount = 0;
    for (int n=0; n<system->nodes[NET].count; n++) {
      int maxSpeed = 0;
      struct ncclTopoNode* net = system->nodes[NET].nodes+n;
      for (int g=0; g<system->nodes[GPU].count; g++) {
        maxSpeed = std::max(maxSpeed, net->paths[GPU][g].width);
      }
      if (maxSpeed > netMaxSpeed) {
        netMaxSpeed = maxSpeed;
        netMaxSpeedCount = 1;
      } else if (maxSpeed == netMaxSpeed) {
        netMaxSpeedCount++;
      }
    }
    system->maxSpeed = std::min(system->maxSpeed, netMaxSpeedCount*NET_WIDTH);
  }
  system->searchInitDone = 1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  return ncclSuccess;
}

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

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    graph->nChannels++;
    if (graph->nChannels*graph->speed > saveGraph->nChannels*saveGraph->speed) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels*graph->speed == system->maxSpeed) *time = -1;
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
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n=0; n<system->nodes[NET].count; n++) {
        if (graph->crossNic != 1 && (system->nodes[NET].nodes[n].used & flag) == 0) continue;
        maxWidth = std::max(paths[n].width, maxWidth);
      }
      for (int n=0; n<system->nodes[NET].count; n++) {
        if (graph->crossNic != 1 && (system->nodes[NET].nodes[n].used & flag) == 0) continue;
        if (paths[n].width == maxWidth) {
          struct ncclTopoNode* net;
          NCCLCHECK(ncclTopoFollowPath(paths+n, &net, graph->speed));
          if (net) {
            //printf("GPU/%d -> NET/%d\n", gpu->id, n);
            NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, -1, backToFirstRank, time));
            NCCLCHECK(ncclTopoFollowPath(paths+n, &net, -graph->speed));
          }
        }
      }
    }
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    struct ncclTopoLinkList* paths = gpu->paths[GPU];
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    for (int i=0; i<count; i++) {
      int g = next[i];
      struct ncclTopoNode* nextGpu;
      NCCLCHECK(ncclTopoFollowPath(paths+g, &nextGpu, graph->speed));
      if (nextGpu) {
        int nvlink = graph->nvlink;
        graph->nvlink = paths[g].nvlink;
        //printf("GPU/%d -> GPU/%d (%d/%d)\n", gpu->id, nextGpu->id, i, g);
        nextGpu->used ^= flag;
        NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, nextGpu, step+1, backToNet, backToFirstRank, time));
        nextGpu->used ^= flag;
        NCCLCHECK(ncclTopoFollowPath(paths+g, &nextGpu, -graph->speed));
        graph->nvlink = nvlink;
      }
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
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, time));
      NCCLCHECK(ncclTopoFollowPath(paths+g, &firstGpu, -graph->speed));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    if (net->used == 0) {
      net->used ^= flag;
      struct ncclTopoLinkList* paths = net->paths[GPU];
      int maxWidth = 0, minHops = 0xfffffff;
      for (int g=0; g<system->nodes[GPU].count; g++) {
        if (paths[g].width > maxWidth) {
          maxWidth = paths[g].width;
          minHops = paths[g].count;
        } else if (paths[g].width == maxWidth && paths[g].count < minHops) {
          minHops = paths[g].count;
        }
      }
      if (maxWidth >= graph->speed) {
        // In the first loop, avoid using GPUs in both directions between channels (one channel
        // sending from that GPU and one channel receiving to that GPU), since that usually leads
        // to lower BW.
        for (int tryGpuBidir=0; tryGpuBidir<2; tryGpuBidir++) {
          for (int g=0; g<system->nodes[GPU].count; g++) {
            if (paths[g].width == maxWidth && paths[g].count == minHops) {
              struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
              int gpuUsed = gpuPciWidth(gpu) > 0 ? 0 : 1;
              if (tryGpuBidir == gpuUsed) {
                NCCLCHECK(ncclTopoFollowPath(paths+g, &gpu, graph->speed));
                if (gpu) {
                  //printf("NET/%d -> GPU/%d (%d/%d)\n", n, g, maxWidth, minHops);
                  gpu->used ^= flag;
                  NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, 0, backToNet, backToFirstRank, time));
                  gpu->used ^= flag;
                  NCCLCHECK(ncclTopoFollowPath(paths+g, &gpu, -graph->speed));
                }
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
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+0;
    gpu->used ^= 1ULL<<graph->nChannels;
    ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, 0, backToNet, backToFirstRank, time);
    gpu->used ^= 1ULL<<graph->nChannels;
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

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* baseGraph) {
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

  NCCLCHECK(ncclTopoSearchInit(system));
  struct ncclTopoGraph tmpGraph;
  int crossNic = (system->nodes[NET].count > 1) && graph->crossNic ? 1 : 0;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));
  int bestSpeed = 0;
  for (tmpGraph.speed = system->maxWidth; tmpGraph.speed >= 6; tmpGraph.speed -= 3) {
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
