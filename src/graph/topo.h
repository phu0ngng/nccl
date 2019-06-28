/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TOPO_H_
#define NCCL_TOPO_H_

#include "nvmlwrap.h"
#include "nvlink.h"
#include "net.h"

#define PASCAL_NVLINK_WIDTH 17
#define VOLTA_NVLINK_WIDTH 22
#define PCI_WIDTH 12           // PCI Gen3 x16
#define PCI_CPU_WIDTH 9
#define QPI_WIDTH 6
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
struct ncclTopoNode {
  int type;
  int id;
  int rank;
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
  int maxChannels;
  int maxWidth;
};

#endif
