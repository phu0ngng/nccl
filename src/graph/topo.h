/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "graph.h"
#include "core.h"
#include <sched.h>

#define LOC_WIDTH 50000
#define PASCAL_NVLINK_WIDTH 180
#define VOLTA_NVLINK_WIDTH 210
#define PCI_WIDTH 120           // PCI Gen3 x16
#define QPI_WIDTH 240
#define SKL_QPI_WIDTH 90
#define P9_WIDTH 320
#define AMD_WIDTH 50000         // Refine later
#define NET_WIDTH 120           // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, to GPU
// to GPU traffic consumed more PCI bandwidth.
#define INTEL_P2P(speed) (DIVUP(speed*9, 12))
#define INTEL_P2P_OVERHEAD(speed) (speed*12/9)

#define NCCL_TOPO_NODE_TYPES 7
#define GPU 0
#define PCI 1
#define NVS 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
#define NPT 6
extern const char* topoNodeTypeStr[];

#define LINK_LOC 0
#define LINK_NVL 1
#define LINK_PCI 2
#define LINK_QPI 3
#define LINK_NET 4
extern const char* topoLinkTypeStr[];

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  int width;
  struct ncclTopoNode* remNode;
};
#define NCCL_TOPO_MAX_LINKS 32
#define NCCL_TOPO_MAX_HOPS (NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES)

struct ncclTopoLinkList {
  struct ncclTopoLink* list[NCCL_TOPO_MAX_HOPS];
  int count;
  int width;
  int type;
};

#define NCCL_TOPO_CPU_INTEL_BDW 1
#define NCCL_TOPO_CPU_INTEL_SKL 2

#define NCCL_TOPO_UNDEF (-1)

struct ncclTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int cudaCompCap;
      int gdrSupport;
    }gpu;
    struct {
      uint64_t asic;
      int port;
      int width;
      int gdrSupport;
    }net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    }cpu;
  };
  int nlinks;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
  // Pre-computed paths to GPUs and NICs
  struct ncclTopoLinkList* paths[NCCL_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  int maxWidth;
};

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id);
ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int id);
ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int width);
ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system);
ncclResult_t ncclTopoLoadSystem(const char* xmlTopoFile, struct ncclTopoSystem* system);

ncclResult_t ncclTopoGetSystemFromXml(struct ncclXml* xml, struct ncclTopoSystem** topoSystem);
ncclResult_t ncclTopoGetGraphsFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph);
ncclResult_t ncclTopoGetXmlFromGraphs(struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* treeGraph, struct ncclTopoSystem* system, struct ncclXml *xml);

#endif
