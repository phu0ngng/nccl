/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"

#define PASCAL_NVLINK_WIDTH 18
#define VOLTA_NVLINK_WIDTH 21
#define PCI_WIDTH 12           // PCI Gen3 x16
#define INTEL_PCI_WIDTH 9
#define QPI_WIDTH 6
#define SKL_QPI_WIDTH 9
#define P9_WIDTH 30
#define NET_WIDTH 12           // 100Gbit

#define NCCL_TOPO_NODE_TYPES 6
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
static const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };

#define LINK_NVL 0
#define LINK_PCI 1
#define LINK_QPI 2
#define LINK_NET 3
static const char* topoLinkTypeStr[] = { "NVL", "PCI", "QPI", "NET" };

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  int width;
  struct ncclTopoNode* remNode;
};
#define NCCL_TOPO_MAX_LINKS 32
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)
#define SELECT_PATH 1
#define SELECT_LAST 2

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  int width;
  int nvlink;
};

struct ncclTopoNode {
  int type;
  int id;
  int rank;
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  uint64_t used;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  int maxSpeed;
  int maxChannels;
  int maxWidth;
  int searchInitDone;
};

static ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d\n", type);
    return ncclInternalError;
  }
  *node = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  (*node)->type = type;
  (*node)->id = id;
  return ncclSuccess;
}

static ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int width) {
  // Aggregate links into higher width for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->width += width;

  // Sort links in BW descending order
  struct ncclTopoLink linkSave;
  memcpy(&linkSave, link, sizeof(struct ncclTopoLink));
  while (link != node->links) {
    if ((link-1)->width >= linkSave.width) break;
    memcpy(link, link-1, sizeof(struct ncclTopoLink));
    link--;
  }
  memcpy(link, &linkSave, sizeof(struct ncclTopoLink));
  return ncclSuccess;
}

#endif
