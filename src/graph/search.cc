/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"

/******************************************************************/
/******************** Graph Search Functions **********************/
/******************************************************************/
#define NCCL_TOPO_SEARCH_MAX_REQS 32

struct ncclTopoNodeReqList {
  // We do not use an array here, so that :
  //  - we can pass a higher-level array
  //  - we can link requests together
  // Downside is, we need to malloc/free those lists.
  struct ncclTopoNode** list;
  //  Used to keep track of which nodes have been used already
  int state[NCCL_TOPO_MAX_NODES];
  int count;
};

static inline ncclResult_t ncclTopoNodeReqListInitFromSystem(struct ncclTopoNodeReqList* list, struct ncclTopoSystem* system, int type) {
  list->count = system->nodes[type].count;
  NCCLCHECK(ncclCalloc(&list->list, list->count));
  for (int i=0; i<list->count; i++) {
    struct ncclTopoNode* node = type != NET ? system->nodes[type].nodes+i :
      system->nodes[type].nodes[i].links[0].remNode; // Follow NET to find NIC
    list->list[i] = node;
    list->state[i] = 0;
  }
  return ncclSuccess;
}

static inline ncclResult_t ncclTopoNodeReqListInitSingle(struct ncclTopoNodeReqList* list, struct ncclTopoNode** node) {
  list->list = node;
  list->count = 1;
  list->state[0] = 0;
  return ncclSuccess;
}

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
};

struct ncclTopoSearchReq {
  struct ncclTopoNodeReqList *start;
  struct ncclTopoNodeReqList *end;
  struct ncclTopoNodeReqList *inter;
  int nhops;
};

struct ncclTopoSearchPath {
  struct ncclTopoNodeList nodes;
  struct ncclTopoLinkList links;
};

struct ncclTopoSearch {
  /* Search Request */
  int nReqs;
  struct ncclTopoSearchReq reqs[NCCL_TOPO_SEARCH_MAX_REQS];
  /* Current search */
  int req;
  int curWidth;
  int maxWidth;
  int minWidth;
  int minHops;
  struct ncclTopoSearchPath curPaths[NCCL_TOPO_SEARCH_MAX_REQS];
  /* Best solution */
  int nPaths;
  int width;
  struct ncclTopoSearchPath paths[NCCL_TOPO_SEARCH_MAX_REQS];
  int stop;
};

static inline int nodeInReqList(struct ncclTopoNodeReqList* l, struct ncclTopoNode* node) {
  for (int i=0; i<l->count; i++) if (node == l->list[i] && l->state[i] == 0) return i;
  return -1;
}

// Check that we can use the link to connect the previous node to the next node.
// In particular we don't want to go through PCI if there is NVLink connectivity
// between 2 GPUs.
static int linkCheck(struct ncclTopoNode* prevNode, struct ncclTopoLink* link) {
  struct ncclTopoNode* nextNode = link->remNode;
  if (prevNode->type != GPU || nextNode->type != GPU) return 1;
  if (link->type == LINK_NVL) return 1;
  for (int l=0; l<prevNode->nlinks; l++) {
    struct ncclTopoLink* link = prevNode->links+l;
    if (link->type != LINK_NVL) continue;
    struct ncclTopoNode* remNode = link->remNode;
    if (remNode == nextNode) return 0;
    if (remNode->type == NVS) {
      for (int l=0; l<nextNode->nlinks; l++) {
        if (nextNode->links[l].remNode == remNode) return 0;
      }
    }
  }
  return 1;
}

#define FOLLOW_LINK(linkList, l, curWidth, cmd) do { \
  l->width -= curWidth; \
   linkList->list[linkList->count++] = l; \
    cmd; \
   linkList->count--; \
  l->width += curWidth; \
} while (0)

#define FOLLOW_NODE(nodeList, n, reqList, index, cmd) do { \
  reqList->state[index] = 1; \
   nodeList->list[nodeList->count++] = n; \
    if (nodeList->count == req->nhops+1) search->req++; \
     cmd; \
    if (nodeList->count == req->nhops+1) search->req--; \
   nodeList->count--; \
  reqList->state[index] = 0; \
} while (0)

ncclResult_t ncclTopoSearchRec(struct ncclTopoSearch* search) {
  struct ncclTopoSearchPath* path  = search->curPaths+search->req;
  struct ncclTopoNodeList* nodeList = &path->nodes;
  struct ncclTopoLinkList* linkList = &path->links;
  struct ncclTopoSearchReq* req = search->reqs+search->req;

  if (search->req < search->nPaths && search->curPaths[search->req].links.count > search->paths[search->req].links.count) return ncclSuccess;

  if (nodeList->count == 0) {
    int pathHops = 0, curPathHops = 0, optimalHops = 0;
    if (search->req) {
      int copy = 0;

      // If we found a shorter path, overwrite unconditionally.
      for (int r=0; r<search->req; r++) pathHops += r<search->nPaths ? search->paths[r].links.count : 0xfffffff;
      for (int r=0; r<search->req; r++) curPathHops += search->curPaths[r].links.count;
      for (int r=0; r<search->req; r++) optimalHops += search->minHops;
      if (curPathHops < pathHops) copy = 1;

      // Also overwrite if we found more paths or a wider width.
      if (search->req > search->nPaths) copy = 1;
      else if (search->curWidth > search->width) copy = 1;

      if (copy) {
        memcpy(search->paths, search->curPaths, search->req*sizeof(struct ncclTopoSearchPath));
        search->nPaths = search->req;
        search->width = search->curWidth;
      }
    }
    if ((search->req*search->width == search->nReqs*search->maxWidth) && (curPathHops == optimalHops)) search->stop = 1;
    if (search->req < search->nReqs && curPathHops <= pathHops) {
      // Start a new path
      for (int i=0; i<req->start->count; i++) {
        if (req->start->state[i] == 0) {
          struct ncclTopoNode* node = req->start->list[i];
          FOLLOW_NODE(nodeList, node, req->start, i,
              NCCLCHECK(ncclTopoSearchRec(search)));
        }
      }
    }
  } else {
    struct ncclTopoNode* node = linkList->count == 0 ? nodeList->list[0] // First node
      : linkList->list[linkList->count-1]->remNode; // Intermediate node
    int prevWidth = search->curWidth;

    // Try links in descending BW order
    int curWidth = 0xfffffff;
    while (1) {
      // Find next best width
      int maxWidth = 0;
      for (int l=0; l<node->nlinks; l++) {
        int w = node->links[l].width;
        if (w < curWidth && w > maxWidth) maxWidth = w;
      }
      if (maxWidth == 0 || maxWidth < search->width) break;
      curWidth = maxWidth;
      search->curWidth = std::min(curWidth, search->curWidth);

      // Only rotate in the same direction on NVS
      int linkStart = 0;
      if (node->type == NVS) {
        // Try rotating, so start from last node.
        struct ncclTopoNode* lastNode = nodeList->list[nodeList->count-1];
        while (node->links[linkStart].remNode != lastNode && linkStart < node->nlinks) linkStart++;
        // We know this is the last node so no need to try it
        linkStart++;
      }

      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+((l+linkStart)%node->nlinks);

        if (link->width != curWidth) continue;
        // Do not go through the same link twice in the same direction.
        int oldLink = 0;
        for (int ol=0; ol<linkList->count; ol++) {
          if (linkList->list[ol] == link) oldLink = 1;
        }
        if (oldLink) continue;
        struct ncclTopoNode* remNode = link->remNode;
        int bridge = (remNode->type == CPU || remNode->type == PCI || remNode->type == NVS) ? 1 : 0;
        struct ncclTopoNodeReqList* reqList = req->nhops == nodeList->count ? req->end : req->inter;

        int found = nodeInReqList(reqList, remNode);
        if (found != -1) { // Found a node in our path
          if (linkCheck(nodeList->list[nodeList->count-1], link)) {
            FOLLOW_NODE(nodeList, remNode, reqList, found,
              FOLLOW_LINK(linkList, link, search->curWidth,
                NCCLCHECK(ncclTopoSearchRec(search))));
          }
        } else if (bridge) { // We can follow this as well
          FOLLOW_LINK(linkList, link, search->curWidth,
              NCCLCHECK(ncclTopoSearchRec(search)));
        }
        if (search->stop) break;

        // No need to try all combinations on NVSwitch, only rotate.
        if (node->type == NVS) break;
      }
    }
    search->curWidth = prevWidth;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchProgressive(struct ncclTopoSearch* search) {
  int maxReqs = search->nReqs;
  search->nReqs = 0;
  // Find one path after the other. It avoid going deep with unoptimal paths
  // on previous levels.
  while (search->nReqs < maxReqs) {
    search->nReqs++;
    search->stop = 0;
    search->curWidth = search->maxWidth;
    NCCLCHECK(ncclTopoSearchRec(search));
    if (search->nPaths < search->nReqs) break;
    search->maxWidth = search->width;
    search->minHops = search->paths[0].links.count;
  }
  search->nReqs = maxReqs;
  return ncclSuccess;
}

ncclResult_t printPaths(struct ncclTopoSearch* search) {
  char line[1024];
  for (int p=0; p<search->nPaths; p++) {
    sprintf(line, "Path %d :", p);
    int offset = strlen(line);
    struct ncclTopoNodeList* list = &search->paths[p].nodes;
    for (int i=0; i<list->count; i++) {
      struct ncclTopoNode* node = list->list[i];
      sprintf(line+offset, " %s/%X", topoNodeTypeStr[node->type], node->id);
      offset = strlen(line);
      if (node->type == GPU) {
        sprintf(line+offset, "(%d)", node->rank);
        offset = strlen(line);
      }
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclFollowPaths(struct ncclTopoSearchPath* paths, int nPaths, int width) {
  for (int p=0; p<nPaths; p++) {
    struct ncclTopoSearchPath* path = paths+p;
#if 0
    printf("Path %d : ", p);
    for (int n=0; n<path->nodes.count; n++) {
      struct ncclTopoNode* node = path->nodes.list[n];
      printf(" %s/%X", topoNodeTypeStr[node->type], node->id);
    }
    printf("\n");
#endif
    for (int l=0; l<path->links.count; l++) {
      struct ncclTopoLink* link = path->links.list[l];
      if (link->width < width) {
        WARN("Internal error : could not select path %d, link %d. Link width %d < %d\n", p, l, link->width, width);
        return ncclInternalError;
      }
      link->width -= width;
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

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* baseGraph) {
  int ngpus = system->nodes[GPU].count;

  char* str = getenv("NCCL_GRAPH");
  if (str) {
    if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) {
      NCCLCHECK(parseGraph(str, &graph->nChannels, ngpus, graph->intra));
      for (int i=0; i<graph->nChannels*ngpus; i++) {
        // Translate gpu numbers into ranks
        graph->intra[i] = system->nodes[GPU].nodes[graph->intra[i]].rank;
      }
      graph->speed = PCI_WIDTH+2;
      return ncclSuccess;
    }
  }

  struct ncclTopoSearch search;
  search.req = 0;
  search.nPaths = 0;
  search.width = 0;
  search.maxWidth = search.curWidth = system->maxWidth;
  search.minHops = 0;
  search.stop = 0;
  int maxChannels = search.nReqs = system->maxChannels;

  if (baseGraph && baseGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP && graph->pattern == NCCL_TOPO_PATTERN_RING) {
    // Do not recompute rings, just convert the SPLIT_TREE_LOOP into a RING, shifting ranks by one.
    INFO(NCCL_GRAPH, "Converting %d channels from split tree loop to ring", baseGraph->nChannels);
    graph->nChannels = baseGraph->nChannels;
    graph->speed = baseGraph->speed;
    for (int c=0; c<graph->nChannels; c++) {
      for (int i=0; i<ngpus; i++) {
        graph->intra[ngpus*c+i] = baseGraph->intra[ngpus*c+((ngpus-i)%ngpus)];
      }
    }
    return ncclSuccess;
  }

  if (system->nodes[NET].count) {
    maxChannels = search.nReqs = system->nodes[NET].count;
    if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) {
      // NIC start/end lists are common to use each NIC once
      struct ncclTopoNodeReqList nicStart;
      NCCLCHECK(ncclTopoNodeReqListInitFromSystem(&nicStart, system, NET));
      // Loop back to same NIC
      struct ncclTopoNodeReqList nicEnds[NCCL_TOPO_SEARCH_MAX_REQS];
      // .. or cross NIC
      struct ncclTopoNodeReqList nicEnd;
      NCCLCHECK(ncclTopoNodeReqListInitFromSystem(&nicEnd, system, NET));

      // GPU -> .. -> GPU lists are duplicated
      struct ncclTopoNodeReqList gpuStart[NCCL_TOPO_SEARCH_MAX_REQS];
      struct ncclTopoNodeReqList gpuEnd[NCCL_TOPO_SEARCH_MAX_REQS];
      struct ncclTopoNodeReqList gpuInter[NCCL_TOPO_SEARCH_MAX_REQS];

      // Find NIC->GPU->GPU-NIC paths first
      for (int n=0; n<maxChannels; n++) {
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter+n, system, GPU));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(nicEnds+n, search.curPaths[n].nodes.list+0));
        search.reqs[n].start = &nicStart;
        search.reqs[n].end = graph->crossNic == 1 ? &nicEnd : nicEnds+n;
        search.reqs[n].inter = gpuInter+n;
        search.reqs[n].nhops = 3;
      }
      NCCLCHECK(ncclTopoSearchRec(&search));
      free(nicStart.list);
      free(nicEnd.list);
      // Save the result of the NIC search
      int nPaths = search.nPaths;
      int width = search.width;
      struct ncclTopoSearchPath nicPaths[NCCL_TOPO_SEARCH_MAX_REQS];
      memcpy(nicPaths, search.paths, nPaths*sizeof(struct ncclTopoSearchPath));

      // Then find GPU loops that go with those paths.
      for (int n=0; n<nPaths; n++) {
        // 2nd GPU -> ... -> 1st GPU loop (linked with previous req solution)
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuStart+n, nicPaths[n].nodes.list+2));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuEnd+n, nicPaths[n].nodes.list+1));
        gpuInter[n].state[nicPaths[n].nodes.list[2]->id] = 1;
        gpuInter[n].state[nicPaths[n].nodes.list[1]->id] = 1;
        search.reqs[n].start = gpuStart+n;
        search.reqs[n].end = graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP ? gpuEnd+n : gpuInter+n;
        search.reqs[n].inter = gpuInter+n;
        search.reqs[n].nhops = ngpus-2;
        if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) search.reqs[n].nhops++;
      }

      search.nReqs = nPaths;
      while (search.nReqs) {
        NCCLCHECK(ncclFollowPaths(nicPaths, search.nReqs, width));
        search.req = 0;
        search.nPaths = 0;
        search.width = 0;
        search.maxWidth = search.curWidth = PCI_WIDTH+1;
	search.minHops = 0;
	search.stop = 0;
        NCCLCHECK(ncclTopoSearchProgressive(&search));
        NCCLCHECK(ncclFollowPaths(nicPaths, search.nReqs, -width));

        if (search.nPaths < search.nReqs) search.nReqs--;
        else break;
      }

      for (int n=0; n<nPaths; n++) free(gpuInter[n].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths;
      graph->speed = search.width;
      for (int c=0; c<graph->nChannels; c++) {
        graph->intra[ngpus*c] = nicPaths[c].nodes.list[1]->rank;
        graph->intra[ngpus*c+1] = nicPaths[c].nodes.list[2]->rank;
        for (int i=2; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.paths[c].nodes.list[i-1]->rank;
        }
      }
    } else if (graph->pattern == NCCL_TOPO_PATTERN_RING ||
               graph->pattern == NCCL_TOPO_PATTERN_TREE) {
      // NIC list is common : we use each NIC only once
      struct ncclTopoNodeReqList nicStart;
      NCCLCHECK(ncclTopoNodeReqListInitFromSystem(&nicStart, system, NET));
      // Loop back to same NIC
      struct ncclTopoNodeReqList nicEnds[NCCL_TOPO_SEARCH_MAX_REQS];
      // .. or cross NIC
      struct ncclTopoNodeReqList nicEnd;
      NCCLCHECK(ncclTopoNodeReqListInitFromSystem(&nicEnd, system, NET));

      // GPU lists are duplicated : we go through each GPUs for each chain
      struct ncclTopoNodeReqList gpuInter[NCCL_TOPO_SEARCH_MAX_REQS];

      for (int n=0; n<maxChannels; n++) {
        // NIC - > all GPUs
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter+n, system, GPU));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(nicEnds+n, search.curPaths[n].nodes.list+0));
        search.reqs[n].start = &nicStart;
        search.reqs[n].end =
          graph->pattern == NCCL_TOPO_PATTERN_TREE ? gpuInter+n : // Don't loop back to NIC
          graph->crossNic == 1 ?
          &nicEnd : // Loop back to any NIC
          nicEnds+n; // Loop back to same NIC
        search.reqs[n].inter = gpuInter+n;
        search.reqs[n].nhops = ngpus;
        if (graph->pattern == NCCL_TOPO_PATTERN_RING) search.reqs[n].nhops++;
      }
      NCCLCHECK(ncclTopoSearchProgressive(&search));
      free(nicStart.list);
      free(nicEnd.list);
      for (int n=0; n<system->nodes[NET].count; n++) free(gpuInter[n].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths;
      graph->speed = search.width;
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.paths[c].nodes.list[i+1]->rank;
        }
      }
    }
  } else {
    // Intra-node
    if (graph->pattern == NCCL_TOPO_PATTERN_RING ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE) {
      struct ncclTopoNodeReqList gpuStart[MAXCHANNELS];
      struct ncclTopoNodeReqList gpuEnd[MAXCHANNELS];
      struct ncclTopoNodeReqList gpuInter[MAXCHANNELS];
      struct ncclTopoNode* gpu0 = system->nodes[GPU].nodes;
      for (int c=0; c<maxChannels; c++) {
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuStart+c, &gpu0));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuEnd+c, &gpu0));
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter+c, system, GPU));
        gpuInter[c].state[0] = 1;
        search.reqs[c].start = gpuStart+c;
        search.reqs[c].end = graph->pattern == NCCL_TOPO_PATTERN_RING ? gpuEnd+c : gpuInter+c;
        search.reqs[c].inter = gpuInter+c;
        search.reqs[c].nhops = graph->pattern == NCCL_TOPO_PATTERN_RING ? ngpus : ngpus-1;
      }
      // Only explore 2+ rings with NVLink
      search.minWidth = system->maxWidth;
      NCCLCHECK(ncclTopoSearchProgressive(&search));
      for (int c=0; c<maxChannels; c++) free(gpuInter[c].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths;
      graph->speed = search.width;
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.paths[c].nodes.list[i]->rank;
        }
      }
    }
  }

  INFO(NCCL_GRAPH, "TopoCompute : pattern %d xNic %d : %d paths speed %d", graph->pattern, graph->crossNic, search.nPaths, search.width);
  NCCLCHECK(printPaths(&search));

  if (graph->nChannels < maxChannels) {
    // We might be suboptimal, see if another pattern would give more channels.
    struct ncclTopoGraph newGraph;
    memcpy(&newGraph, graph, sizeof(newGraph));
    if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) newGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
    else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) newGraph.pattern = NCCL_TOPO_PATTERN_TREE;
    else if (graph->crossNic == 2) newGraph.crossNic = 1;
    else return ncclSuccess;

    NCCLCHECK(ncclTopoCompute(system, &newGraph, baseGraph));
    if (newGraph.nChannels > graph->nChannels || (newGraph.nChannels == graph->nChannels && newGraph.speed > graph->speed)) {
      INFO(NCCL_GRAPH, "TopoCompute : Pattern/XNic %d/%d better than %d/%d (%d/%d channels vs %d/%d)", newGraph.pattern, newGraph.crossNic, graph->pattern, graph->crossNic, newGraph.nChannels, newGraph.speed, graph->nChannels, graph->speed);
      memcpy(graph, &newGraph, sizeof(newGraph));
    }
  }
  if (graph->crossNic == 2 && graph->nChannels == 0) {
    WARN("Could not find a path for GPUs.");
    return ncclInternalError;
  }
  return ncclSuccess;
}
