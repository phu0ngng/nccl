/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "topo.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

ncclResult_t getCudaPath(int cudaDev, char** path) {
  char busId[BUSID_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, BUSID_SIZE, cudaDev));
  for (int i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

const char* pathDists[] = { "PIX", "PXB", "PHB", "NODE", "SYS" };

int pciDistance(char* path1, char* path2) {
  int score = 0;
  int depth = 0;
  int same = 1;
  for (int i=0; i<strlen(path1); i++) {
    if (path1[i] != path2[i]) same = 0;
    if (path1[i] == '/') {
      depth++;
      if (same == 1) score++;
    }
  }
  if (score <= 3) {
#ifdef __PPC__
    // NUMA distance detection and PATH_SYS not supported on IBM/Power nodes
    // nodes currently
    return PATH_NODE;
#else
    /* Split the former PATH_SOC distance into PATH_NODE and PATH_SYS based on numaId */
    int numaId1 = getNumaId(path1);
    int numaId2 = getNumaId(path2);
    TRACE(NCCL_INIT, "depth %d score %d path1 %s numaId %d path2 %s numaId %d", depth, score, path1, numaId1, path2, numaId2);
    return ((numaId1 == numaId2) ? PATH_NODE : PATH_SYS);
#endif
  }
  if (score == 4) return PATH_PHB;
  if (score == depth-1) return PATH_PIX;
  return PATH_PXB;
}

/* ============= new topo graph ================ */
#include "nvmlwrap.h"
#include "nvlink.h"
#include "net.h"

#define PASCAL_NVLINK_WIDTH 2 // 17 GB/s
#define VOLTA_NVLINK_WIDTH 2  // 22 GB/s
#define PCI_WIDTH 1
#define QPI_WIDTH 1           // PCI Gen3 x16, 12GB/s
#define NET_WIDTH 1           // 100Gbit, 12GB/s

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
};

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int width, struct ncclTopoSystem* system) {
  // Aggregate links into higher width for NVLink
  struct ncclTopoLink* link;
  for (int l=0; l<NCCL_TOPO_MAX_LINKS; l++) {
    link = node->links+l;
    if (link->remNode == NULL) break;
    if (link->remNode == remNode) {
      if (link->type == type) {
        link->width += width;
        return ncclSuccess;
      }
    }
  }
  link->type = type;
  link->width = width;
  link->remNode = remNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, int id) {
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

ncclResult_t ncclTopoConnectNVLink(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system) {
  struct ncclTopoNode* nvsNode = NULL;

  for (int r=0; r<system->nodes[GPU].count; r++) {
    int cudaMajor, cudaMinor;
    NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDevs[r], &cudaMajor, &cudaMinor));
    int maxNvLinks = cudaMajor > 6 ? 6 : 4;

    for (int l=0; l<maxNvLinks; ++l) {
      // Check whether we can use this NVLink for P2P
      unsigned canP2P;
      if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDevs[r], l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive;
      if ((wrapNvmlDeviceGetNvLinkState(nvmlDevs[r], l, &isActive) != ncclSuccess) || (isActive != NVML_FEATURE_ENABLED)) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevs[r], l, &remoteProc) != ncclSuccess) continue;

      // Make a lower case copy of the bus ID for calling ncclDeviceType
      // PCI system path is in lower case
      char* p = remoteProc.busId;
      char lowerId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      for (int c=0; c<NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE; c++) {
        if (p[c] == 0) break;
        lowerId[c] = tolower(p[c]);
      }

      enum ncclNvLinkDeviceType type;
      NCCLCHECK(ncclDeviceType(lowerId, &type));
      if (type == ncclNvLinkDeviceGpu) {
        for (int peer=0; peer<system->nodes[GPU].count; peer++) {
          nvmlPciInfo_t pci;
          NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDevs[peer], &pci));
          if (strncmp(pci.busId, remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
            NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+r, system->nodes[GPU].nodes+peer, LINK_NVL, cudaMajor == 6 ? PASCAL_NVLINK_WIDTH : VOLTA_NVLINK_WIDTH, system));
            break;
          }
        }
      } else if (type == ncclNvLinkDeviceBridge) {
        /* TODO : create topo for Power/NVLink */
      } else { // Nvswitch
        if (type == ncclNvLinkDeviceUnknown) {
          // The NVLink is up but we couldn't find the PCI device on the other
          // side. Assume it's an NVswitch outside a VM.
          INFO(NCCL_INIT, "Assuming NVLink is connected to NVswitch");
        }
        if (nvsNode == NULL) { // Create nvswitch
          NCCLCHECK(ncclTopoCreateNode(system, &nvsNode, NVS, 0));
        }
        NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+r, nvsNode, LINK_NVL, VOLTA_NVLINK_WIDTH, system));
        NCCLCHECK(ncclTopoConnectNodes(nvsNode, system->nodes[GPU].nodes+r, LINK_NVL, VOLTA_NVLINK_WIDTH, system));
      }
    }
  }
  return ncclSuccess;
}

// Walk backwards a PCI path to generate a PCI ID as an integer.
// For example, a path pointer pointing on the last "/" of
// /sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will give 02*4096+00*8+0 = 4096.
ncclResult_t pciHexToInt(char* path, int offset, int minOffset, int* id) {
  // Copy hex value without ':', '.' and stop when we reach '/'
  if (path[offset] == '/') offset--; // Ignore trailing '/'

  char* hexStr;
  NCCLCHECK(ncclCalloc(&hexStr, offset-minOffset+2));
  int hexOffset = offset-minOffset;
  for (; offset >= minOffset; offset--) {
    if (path[offset] == '.' || path[offset] == ':') continue;
    if (path[offset] == '/') {
      *id = strtol(hexStr+hexOffset+1, NULL, 16);
      free(hexStr);
      return ncclSuccess;
    }
    hexStr[hexOffset--] = path[offset];
  }
  WARN("Topo/PCI : could not find preceding '/'");
  free(hexStr);
  return ncclInternalError;
}

ncclResult_t ncclTopoCreatePciPath(struct ncclTopoSystem* system, struct ncclTopoNode* endNode, char* path) {
  struct ncclTopoNode* lastNode = endNode;
  // Find intermediate PCI switches
  int slashCount = 0;
  int offsetRC = 0;
  while (offsetRC < strlen(path)) {
    if (path[offsetRC] == '/') slashCount++;
    if (slashCount == 4) break;
    offsetRC++;
  }
  int offset = strlen(path);
  slashCount = 0;
  while (--offset > offsetRC) {
    if (path[offset] == '/') {
      slashCount++;
      // Find if already existing
      if ((slashCount%2) == 0) {
        int pciId;
        NCCLCHECK(pciHexToInt(path, offset, offsetRC, &pciId));
        for (int p=0; p<system->nodes[PCI].count; p++) {
          if (system->nodes[PCI].nodes[p].id == pciId) {
            // Found our PCI switch. Attach and stop since the rest should already
            // be connected
            NCCLCHECK(ncclTopoConnectNodes(system->nodes[PCI].nodes+p, lastNode, LINK_PCI, PCI_WIDTH, system));
            NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[PCI].nodes+p, LINK_PCI, PCI_WIDTH, system));
            return ncclSuccess;
          }
        }
        struct ncclTopoNode* pciNode;
        NCCLCHECK(ncclTopoCreateNode(system, &pciNode, PCI, pciId));
        NCCLCHECK(ncclTopoConnectNodes(pciNode, lastNode, LINK_PCI, PCI_WIDTH, system));
        NCCLCHECK(ncclTopoConnectNodes(lastNode, pciNode, LINK_PCI, PCI_WIDTH, system));
        lastNode = pciNode;
      }
    }
  }
  // Then attach to a CPU node
  int numaId = getNumaId(path);
  for (int n=0; n<system->nodes[CPU].count; n++) {
    if (system->nodes[CPU].nodes[n].id == numaId) {
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, lastNode, LINK_PCI, PCI_WIDTH, system));
      NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[CPU].nodes+n, LINK_PCI, PCI_WIDTH, system));
      return ncclSuccess;
    }
  }
  struct ncclTopoNode* numaNode;
  NCCLCHECK(ncclTopoCreateNode(system, &numaNode, CPU, numaId));
  NCCLCHECK(ncclTopoConnectNodes(numaNode, lastNode, LINK_PCI, PCI_WIDTH, system));
  NCCLCHECK(ncclTopoConnectNodes(lastNode, numaNode, LINK_PCI, PCI_WIDTH, system));
  return ncclSuccess;
}

ncclResult_t ncclTopoCudaPath(char* busId, char** path) {
  for (int i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectPCI(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system, int inter) {
  for (int r=0; r<system->nodes[GPU].count; r++) {
    char* path;
    nvmlPciInfo_t pci;
    NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDevs[r], &pci));
    NCCLCHECK(ncclTopoCudaPath(pci.busId, &path));
    NCCLCHECK(ncclTopoCreatePciPath(system, system->nodes[GPU].nodes+r, path));
    free(path);
  }

  if (inter) {
    // Connect the NICs
    int netDevCount;
    NCCLCHECK(ncclNetDevices(&netDevCount));
    for (int n=0; n<netDevCount; n++) {
      struct ncclTopoNode* netNode;
      NCCLCHECK(ncclTopoCreateNode(system, &netNode, NET, n));
      char* path;
      NCCLCHECK(ncclNetPciPath(n, &path));

      // Create NIC
      int pciId;
      // Use "strlen(path)-1" to remove trailing subdevice and merge multi-port cards into one
      NCCLCHECK(pciHexToInt(path, strlen(path)-2, 0, &pciId));
      for (int n=0; n<system->nodes[NIC].count; n++) {
        if (system->nodes[NIC].nodes[n].id == pciId) {
          // Found our NIC. Attach and stop since the rest should already be connected
          NCCLCHECK(ncclTopoConnectNodes(system->nodes[NIC].nodes+n, netNode, LINK_NET, NET_WIDTH, system));
          NCCLCHECK(ncclTopoConnectNodes(netNode, system->nodes[NIC].nodes+n, LINK_NET, NET_WIDTH, system));
          return ncclSuccess;
        }
      }
      struct ncclTopoNode* nicNode;
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, NIC, pciId));
      NCCLCHECK(ncclTopoConnectNodes(nicNode, netNode, LINK_NET, NET_WIDTH, system));
      NCCLCHECK(ncclTopoConnectNodes(netNode, nicNode, LINK_NET, NET_WIDTH, system));

      // Create the PCI path
      NCCLCHECK(ncclTopoCreatePciPath(system, nicNode, path));
      free(path);
    }
  }

  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_QPI, QPI_WIDTH, system));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem* system, int inter) {
  nvmlDevice_t* nvmlDevs;
  NCCLCHECK(ncclCalloc(&nvmlDevs, nranks));
  for (int r=0; r<nranks; r++) {
    struct ncclTopoNode* gpuNode;
    NCCLCHECK(ncclTopoCreateNode(system, &gpuNode, GPU, nvmlIndexes[r]));
    gpuNode->rank = rankIndexes[r];
    NCCLCHECK(wrapNvmlDeviceGetHandleByIndex(nvmlIndexes[r], nvmlDevs+r));
  }

  NCCLCHECK(ncclTopoConnectNVLink(nvmlDevs, system));
  NCCLCHECK(ncclTopoConnectPCI(nvmlDevs, system, inter));

  free(nvmlDevs);
  return ncclSuccess;
}

ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
  // Shift all links to have upLink as last link
  if (upNode) {
    int l=0;
    while (node->links[l].remNode != upNode) l++;
    struct ncclTopoLink upLink;
    memcpy(&upLink, node->links+l, sizeof(struct ncclTopoLink));
    while (node->links[l+1].remNode) {
      memcpy(node->links+l, node->links+l+1, sizeof(struct ncclTopoLink));
      l++;
    }
    memcpy(node->links+l, &upLink, sizeof(struct ncclTopoLink));
  }

  // Recursively sort the PCI tree
  for (int l=0; l<NCCL_TOPO_MAX_LINKS; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->remNode == NULL) break;
    if (link->type == LINK_PCI && link->remNode != upNode) NCCLCHECK(ncclTopoSort(link->remNode, node));
  }
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. QPI (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[CPU].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[CPU].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%X (%d)", topoNodeTypeStr[node->type], node->id, node->rank);
    INFO(NCCL_GRAPH, "%s", line);
  } else {
    sprintf(line+offset, "%s/%X", topoNodeTypeStr[node->type], node->id);
    INFO(NCCL_GRAPH, "%s", line);
  }
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<NCCL_TOPO_MAX_LINKS; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->remNode && link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2d] - ", topoLinkTypeStr[link->type], link->width);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrint(link->remNode, node, line, nextOffset));
      } else {
        sprintf(line+nextOffset, "%s/%X", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem** system, int inter) {
  struct ncclTopoSystem* s;
  NCCLCHECK(ncclCalloc(&s, 1));
  NCCLCHECK(ncclTopoGetSystem(nranks, nvmlIndexes, rankIndexes, s, inter));
  NCCLCHECK(ncclTopoSortSystem(s));
  INFO(NCCL_GRAPH, "================ Topology ================");
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrint(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  *system = s;
  return ncclSuccess;
}

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
