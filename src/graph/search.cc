/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"

// Initialize system->maxWidth. This is the per-channel (i.e. per-SM)
// max speed.
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  printf("Search Init\n");
  system->maxWidth = LOC_WIDTH;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    int gpuLinkType = LINK_PCI;
    for (int l=0; l<gpu->nlinks; l++) {
      if (gpu->links[l]->type == LINK_NVL) gpuLinkType = LINK_NVL;
    }
    int gpuMaxWidth = gpuLinkType == LINK_NVL ? (gpu->gpu.cudaCompCap > 60 ? VOLTA_NVLINK_WIDTH : PASCAL_NVLINK_WIDTH) : PCI_WIDTH;
    system->maxWidth = std::min(system->maxWidth, gpuMaxWidth);
    printf("GPU %d maxWidth %d type %d\n", g, gpuMaxWidth, gpuLinkType);
  }
  if (system->nodes[NET].count > 0) {
    system->maxWidth = PCI_WIDTH;
  }
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, int mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2;
  if (type1 == -1) return ncclSuccess;
  struct ncclTopoLinkList* pathFw = system->nodes[type1].nodes[index1].paths[type2]+index2;
  struct ncclTopoLinkList* pathBw = system->nodes[type2].nodes[index2].paths[type1]+index1;
  if (pathFw->count == 0 ) return ncclSuccess;

  // Now try to follow paths
  *node = NULL;
  int bidir = graph->pattern == NCCL_TOPO_PATTERN_TREE || graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE;
  int intra = type1 == GPU && type2 == GPU;
  int speed = intra ? graph->speedIntra : graph->speedInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  // Account for P2P inefficiency when going through Intel CPUs
  if (intra && pathFw->type == LINK_QPI &&
      system->nodes[CPU].nodes[0].cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
      system->nodes[CPU].nodes[0].cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL)
    speed = INTEL_P2P_OVERHEAD(speed);

  int speedFw = bidir ? DIVUP(speed, 2) : speed;
  int speedBw = bidir ? speed/2 : 0;

  // Pulling data from GPUs on PCI impacts performance in the other direction.
  int extraSpeedFw = (type2 == GPU && type1 != GPU && pathBw->type == LINK_PCI) ? DIVUP(speedBw,8) : 0;
  int extraSpeedBw = (type1 == GPU && type2 != GPU && pathFw->type == LINK_PCI) ? DIVUP(speedFw,8) : 0;

  speedFw *= mult; speedBw *= mult; extraSpeedFw *= mult; extraSpeedBw *= mult;

  int stepFw = 0, stepBw = 0;

  if (mult == 1 && pathFw->type > type) goto rewind;
  for (; stepFw<pathFw->count; stepFw++) {
    int speed = speedFw;
    if (stepFw == pathFw->count-1) speed += extraSpeedFw;
    if (pathFw->list[stepFw]->width < speed) goto rewind;
    pathFw->list[stepFw]->width -= speed;
  }

  if (mult == 1 && bidir && pathBw->type > type) goto rewind;
  for (; stepBw<pathBw->count; stepBw++) {
    int speed = speedBw;
    if (stepBw == pathBw->count-1) speed += extraSpeedBw;
    if (pathBw->list[stepBw]->width < speed) goto rewind;
    pathBw->list[stepBw]->width -= speed;
  }

  graph->nHops += mult*pathFw->count;
  *node = system->nodes[type2].nodes+index2;
  return ncclSuccess;

rewind:
  for (int j=0; j<stepFw; j++) {
    int speed = speedFw;
    if (j == pathFw->count-1) speed += extraSpeedFw;
    pathFw->list[j]->width += speed;
  }
  for (int j=0; j<stepBw; j++) {
    int speed = speedBw;
    if (j == pathBw->count-1) speed += extraSpeedBw;
    pathBw->list[j]->width += speed;
  }
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

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d\n", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx\n", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  int netId = graph->inter[graph->nChannels*2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;
    if (paths[g].count == 0) continue; // There is no path to that GPU
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

#define NCCL_SEARCH_TIMEOUT (1ULL<<18) // This should get contain all search within a second or so.
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1ULL<<10) // This should get contain all search within a second or so.

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  if (*g == -1) return ncclInternalError;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu;
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Constraint to get the same nChannels between Rings and Trees
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  // 2. Try to get better bandwidth
  if (graph->nChannels*graph->speedIntra < refGraph->nChannels*refGraph->speedIntra) return ncclSuccess;
  if (graph->nChannels*graph->speedIntra > refGraph->nChannels*refGraph->speedIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  // 3. Less hops (but not at the price of going cross NICs)
  if (graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;
    NCCLCHECK(ncclTopoCompareGraphs(graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    return ncclSuccess;
  }
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  int g = gpu - system->nodes[GPU].nodes;
  if (step == backToNet) {
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      for (int n=0; n<system->nodes[NET].count; n++) {
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->crossNic != 1 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, -1, backToFirstRank, forcedOrder, time));
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
        }
      }
    }
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
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
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int speed = graph->speedInter;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    struct ncclTopoNode* gpu;
    if (net->net.width >= speed) {
      graph->inter[graph->nChannels*2] = net->id;
      for (int i=0; i<system->nodes[NET].count; i++) {
        if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
            (system->nodes[NET].nodes[i].net.port == net->net.port)) {
          system->nodes[NET].nodes[i].net.width -= speed;
        }
      }

      // Then try to replay the last channel if sameChannels is 1
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
      } else {
        if (graph->nChannels == 0) {
          // Always try the PCI order first to set a reference
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, NET, n, 0));
        }

        // Then try the most local GPUs
        int maxWidth = 0, minHops = 0xfffffff;
        struct ncclTopoLinkList* paths = net->paths[GPU];
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
                  NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, g));
                }
              }
            }
          }
        }
      }

      for (int i=0; i<system->nodes[NET].count; i++) {
        if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
            (system->nodes[NET].nodes[i].net.port == net->net.port)) {
          system->nodes[NET].nodes[i].net.width += speed;
        }
      }
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
    if (graph->nChannels > 0 && graph->sameChannels == 1) {
       NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, 0));
    } else {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, 0));
    }
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

#define BW_COARSE_INC 30
#define BW_FINE_INC 10

ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int crossNic = (system->nodes[NET].count > 1) && graph->crossNic ? 1 : 0;
  graph->speedIntra = graph->speedInter = 0;
  if (graph->crossNic == 2) graph->crossNic = 0;
  graph->typeIntra = LINK_LOC;
  graph->typeInter = LINK_PCI;
  graph->nChannels = 0;
  graph->sameChannels = 1;

  char* str = getenv("NCCL_GRAPH");
  if (str) {
    NCCLCHECK(parseGraph(str, &graph->nChannels, ngpus, graph->intra));
    for (int i=0; i<graph->nChannels*ngpus; i++) {
      // Translate gpu numbers into ranks
      graph->intra[i] = system->nodes[GPU].nodes[graph->intra[i]].gpu.rank;
    }
    // TODO : let user specify NICs
    graph->inter[0] = graph->inter[1] = 0;
    graph->speedIntra = graph->speedInter = PCI_WIDTH+2;
    // TODO compute proper path
    graph->typeIntra = graph->typeInter = LINK_QPI;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
      // Reverse the loop
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<=ngpus/2; i++) {
          int tmp = graph->intra[ngpus*c+i];
          graph->intra[ngpus*c+i] = graph->intra[ngpus*c+(ngpus-i)%ngpus];
          graph->intra[ngpus*c+ngpus-i] = tmp;
        }
      }
    } else graph->pattern = NCCL_TOPO_PATTERN_TREE;
    if (graph->nChannels) return ncclSuccess;
  }

  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // First try crossnic, then decrease speed and finally increase speedIntra.
  tmpGraph.speedIntra = tmpGraph.speedInter = system->maxWidth;
  tmpGraph.pattern = graph->pattern;
  int pass = 1;

search:
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  printf("Pattern %d, crossNic %d, Speed %d/%d, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%d/%d %s\n", tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.speedInter, tmpGraph.speedIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->speedInter, graph->speedIntra, time == 0 ? "TIMEOUT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("\n");
  }
#endif
  if (time == -1) goto done;

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels
    if (tmpGraph.sameChannels == 1) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = 1;

    // We already have a solution and we timed out so lower speed will just timeout as well
    if (time == 0 && graph->nChannels > 0) goto done;

    // Try a simpler tree
    if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
      goto search;
    }
    if (tmpGraph.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    int maxTypeIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : LINK_QPI;
    if (tmpGraph.typeIntra < maxTypeIntra) {
      tmpGraph.typeIntra += 1;
      goto search;
    }
    tmpGraph.typeIntra = LINK_LOC;
    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < LINK_QPI) {
      tmpGraph.typeInter += 1;
      goto search;
    }
    tmpGraph.typeInter = LINK_PCI;

    if (crossNic && tmpGraph.crossNic == 0) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = crossNic;
      goto search;
    }
    tmpGraph.crossNic = graph->crossNic;

    // Decrease speed until we find a sotution
    if (graph->nChannels == 0) {
      if (tmpGraph.speedInter > BW_COARSE_INC) {
        tmpGraph.speedIntra = tmpGraph.speedInter -= BW_COARSE_INC;
        goto search;
      } else if (tmpGraph.speedInter > BW_FINE_INC) {
        tmpGraph.speedIntra = tmpGraph.speedInter -= BW_FINE_INC;
        goto search;
      }
    } else if (graph->nChannels == 1 && tmpGraph.speedIntra-BW_COARSE_INC > graph->speedIntra/2) {
      // It's OK to go from 1 path to 2 in order to gain some BW
      tmpGraph.speedInter = tmpGraph.speedIntra -= BW_COARSE_INC;
      goto search;
    }
  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    pass = 2;
  }

  // 2. try to fine tune the speedInter up a bit
  if (pass == 2) {
    if (tmpGraph.speedInter < system->maxWidth && tmpGraph.speedInter == graph->speedInter) {
      time = -1;
      tmpGraph.speedIntra = tmpGraph.speedInter += BW_FINE_INC;
      goto search;
    }
    // Fine tuning done. Moving to pass 3
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    pass = tmpGraph.pattern == NCCL_TOPO_PATTERN_RING ? 5 : 3;
  }


  // 3. See if we can increase speedIntra for trees (2 nodes case)
  if (pass == 3) {
    if (time != 0 && tmpGraph.speedIntra == graph->speedIntra && tmpGraph.speedIntra <= tmpGraph.speedInter*2-BW_COARSE_INC) {
      // Try to increase the intra speed only but keeping nChannels the same
      tmpGraph.speedIntra += BW_COARSE_INC;
      goto search;
    }
    // SpeedIntra done. Moving to pass 4
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    pass = 4;
  }

  // 4. try to fine tune the speedIntra up a bit
  if (pass == 4 && time != 0 && tmpGraph.speedIntra == graph->speedIntra && tmpGraph.speedIntra < tmpGraph.speedInter*2) {
    tmpGraph.speedIntra += BW_FINE_INC;
    goto search;
  }

  if (graph->nChannels == 0) {
    WARN("Could not find a path for pattern %d, falling back to simple order\n", graph->pattern);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->speedIntra = graph->speedInter = 1;
    graph->typeIntra = graph->typeInter = LINK_QPI;
    graph->nChannels = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, speed %d/%d, type %d/%d, sameChannels %d", graph->pattern, graph->crossNic, graph->nChannels, graph->speedIntra, graph->speedInter, graph->typeIntra, graph->typeInter, graph->sameChannels);
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
