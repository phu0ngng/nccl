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
  int nReqs;
  int req;
  struct ncclTopoSearchReq reqs[NCCL_TOPO_SEARCH_MAX_REQS];
  struct ncclTopoSearchPath paths[NCCL_TOPO_SEARCH_MAX_REQS];
  int nPaths;
  struct ncclTopoSearchPath save[NCCL_TOPO_SEARCH_MAX_REQS];
};

static inline int nodeInReqList(struct ncclTopoNodeReqList* l, struct ncclTopoNode* node) {
  for (int i=0; i<l->count; i++) if (node == l->list[i] && l->state[i] == 0) return i;
  return -1;
}

ncclResult_t ncclTopoCopyPath(struct ncclTopoSearchPath* dst, struct ncclTopoSearchPath* src) {
  for (int i=0; i<src->nodes.count; i++) dst->nodes.list[i] = src->nodes.list[i];
  dst->nodes.count = src->nodes.count;
  for (int i=0; i<src->links.count; i++) dst->links.list[i] = src->links.list[i];
  dst->links.count = src->links.count;
  return ncclSuccess;
}
#define FOLLOW_LINK(linkList, l, cmd) do { \
  l->width--; \
   linkList->list[linkList->count++] = l; \
    cmd; \
   linkList->count--; \
  l->width++; \
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
  struct ncclTopoSearchPath* path  = search->paths+search->req;
  struct ncclTopoNodeList* nodeList = &path->nodes;
  struct ncclTopoLinkList* linkList = &path->links;
  struct ncclTopoSearchReq* req = search->reqs+search->req;

  if (nodeList->count == 0) {
    if (search->req >= search->nPaths) { // Save new solution
      int copy = 1;
      if (search->req == search->nPaths) {
        int saveHops = 0;
        for (int r=0; r<search->req; r++) saveHops += search->save[r].links.count;
        int pathHops = 0;
        for (int r=0; r<search->req; r++) pathHops += search->paths[r].links.count;
        if (pathHops >= saveHops) copy = 0;
      }
      if (copy) {
        for (int r=0; r<search->req; r++) {
          NCCLCHECK(ncclTopoCopyPath(search->save+r, search->paths+r));
        }
        search->nPaths = search->req;
      }
    }
    if (search->req < search->nReqs) { // Start a new req
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

    for (int l=0; l<NCCL_TOPO_MAX_LINKS && node->links[l].remNode; l++) {
      struct ncclTopoLink* link = node->links+l;
      if (link == NULL || link->width == 0) continue;
      struct ncclTopoNode* remNode = link->remNode;
      int bridge = (remNode->type == CPU || remNode->type == PCI || remNode->type == NVS) ? 1 : 0;
      struct ncclTopoNodeReqList* reqList = req->nhops == nodeList->count ? req->end : req->inter;
      int found = nodeInReqList(reqList, remNode);
      if (found != -1) { // Found a node in our path
        FOLLOW_NODE(nodeList, remNode, reqList, found,
            FOLLOW_LINK(linkList, link,
              NCCLCHECK(ncclTopoSearchRec(search))));
      } else if (bridge) { // We can follow this as well
        FOLLOW_LINK(linkList, link,
            NCCLCHECK(ncclTopoSearchRec(search)));
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoSearch search;
  search.req = 0;
  search.nPaths = 0;
  search.nReqs = 0;
  int maxChannels = 0;

  if (system->nodes[NET].count) {
    maxChannels = system->nodes[NET].count;
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

      for (int n=0; n<system->nodes[NET].count; n++) {
        // NIC -> 1st GPU -> 2nd GPU -> NIC
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter+n, system, GPU));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(nicEnds+n, search.paths[2*n].nodes.list+0));
        search.reqs[2*n].start = &nicStart;
        search.reqs[2*n].end = graph->crossNic == 1 ? &nicEnd : nicEnds+n;
        search.reqs[2*n].inter = gpuInter+n;
        search.reqs[2*n].nhops = 3;
        // 2nd GPU -> ... -> 1st GPU loop (linked with previous req solution)
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuStart+n, search.paths[2*n].nodes.list+2));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuEnd+n, search.paths[2*n].nodes.list+1));
        search.reqs[2*n+1].start = gpuStart+n;
        search.reqs[2*n+1].end = graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP ? gpuEnd+n : gpuInter+n;
        search.reqs[2*n+1].inter = gpuInter+n;
        search.reqs[2*n+1].nhops = ngpus-2;
        if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) search.reqs[2*n+1].nhops++;
      }
      search.nReqs = system->nodes[NET].count*2;
      NCCLCHECK(ncclTopoSearchRec(&search));
      free(nicStart.list);
      free(nicEnd.list);
      for (int n=0; n<system->nodes[NET].count; n++) free(gpuInter[n].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths/2;
      for (int c=0; c<graph->nChannels; c++) {
        graph->intra[ngpus*c] = search.save[2*c].nodes.list[1]->rank;
        graph->intra[ngpus*c+1] = search.save[2*c].nodes.list[2]->rank;
        for (int i=2; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.save[2*c+1].nodes.list[i-1]->rank;
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

      for (int n=0; n<system->nodes[NET].count; n++) {
        // NIC - > all GPUs
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter+n, system, GPU));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(nicEnds+n, search.paths[n].nodes.list+0));
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
      search.nReqs = system->nodes[NET].count;
      NCCLCHECK(ncclTopoSearchRec(&search));
      free(nicStart.list);
      free(nicEnd.list);
      for (int n=0; n<system->nodes[NET].count; n++) free(gpuInter[n].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths;
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.save[c].nodes.list[i+1]->rank;
        }
      }
    }
  } else {
    // FIXME : detect max channels depending on PCI/NVLink
    maxChannels = MAXCHANNELS;
    // Intra-node
    if (graph->pattern == NCCL_TOPO_PATTERN_RING ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE) {
      struct ncclTopoNodeReqList gpuStart[MAXCHANNELS];
      struct ncclTopoNodeReqList gpuEnd[MAXCHANNELS];
      struct ncclTopoNodeReqList gpuInter[MAXCHANNELS];
      struct ncclTopoNode* gpu0 = system->nodes[GPU].nodes;
      for (int c=0; c<MAXCHANNELS; c++) {
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuStart+c, &gpu0));
        NCCLCHECK(ncclTopoNodeReqListInitSingle(gpuEnd+c, &gpu0));
        NCCLCHECK(ncclTopoNodeReqListInitFromSystem(gpuInter, system, GPU));
        search.reqs[c].start = gpuStart+c;
        search.reqs[c].end = graph->pattern == NCCL_TOPO_PATTERN_RING ? gpuEnd+c : gpuInter+c;
        search.reqs[c].inter = gpuInter+c;
        search.reqs[c].nhops = ngpus;
      }
      search.nReqs = MAXCHANNELS;
      NCCLCHECK(ncclTopoSearchRec(&search));
      for (int c=0; c<MAXCHANNELS; c++) free(gpuInter[c].list);

      // Save result into graph -> inter/intra
      graph->nChannels = search.nPaths;
      for (int c=0; c<graph->nChannels; c++) {
        for (int i=0; i<ngpus; i++) {
          graph->intra[ngpus*c+i] = search.save[c].nodes.list[i]->rank;
        }
      }
    }
  }

  INFO(NCCL_GRAPH, "TopoCompute : pattern %d xNic %d : %d paths", graph->pattern, graph->crossNic, search.nPaths);
  char line[1024];
  for (int p=0; p<search.nPaths; p++) {
    sprintf(line, "Path %d :", p);
    int offset = strlen(line);
    struct ncclTopoNodeList* list = &search.save[p].nodes;
    for (int i=0; i<list->count; i++) {
      struct ncclTopoNode* node = list->list[i];
      sprintf(line+offset, " %s/%X", topoNodeTypeStr[node->type], node->id);
      offset = strlen(line);
      if (node->type == GPU) {
        sprintf(line+offset, "(%d)", node->rank);
        offset = strlen(line);
      }
    }
  }
  INFO(NCCL_GRAPH, "%s", line);

  if (graph->nChannels < maxChannels) {
    // We might be suboptimal, see if another pattern would give more channels.
    struct ncclTopoGraph newGraph;
    memcpy(&newGraph, graph, sizeof(newGraph));
    if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP) newGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
    else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) newGraph.pattern = NCCL_TOPO_PATTERN_TREE;
    else if (graph->crossNic == 2) newGraph.crossNic = 1;
    else return ncclSuccess;

    NCCLCHECK(ncclTopoCompute(system, &newGraph));
    if (newGraph.nChannels > graph->nChannels) {
      INFO(NCCL_GRAPH, "TopoCompute : Pattern/XNic %d/%d better than %d/%d (%d channels vs %d)", newGraph.pattern, newGraph.crossNic, graph->pattern, graph->crossNic, newGraph.nChannels, graph->nChannels);
      memcpy(graph, &newGraph, sizeof(newGraph));
    }
  }

  return ncclSuccess;
}
