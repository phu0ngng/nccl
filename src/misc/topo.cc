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

#define PASCAL_NVLINK_WIDTH 2
#define VOLTA_NVLINK_WIDTH 2
#define PCI_WIDTH 1
#define QPI_WIDTH 1

#define NCCL_TOPO_NODE_TYPES 5
#define GPU 0
#define PCI 1
#define NVS 2
#define NUMA 3
#define NET 4
static const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "NUMA", "NET" };

#define LINK_NVL 0
#define LINK_PCI 1
#define LINK_QPI 2
static const char* topoLinkTypeStr[] = { "NVL", "PCI", "QPI" };

struct ncclTopoNode;
struct ncclTopoLink {
  int type;
  int width;
  struct ncclTopoNode* remNode;
};
#define NCCL_TOPO_MAX_LINKS 32
#define SELECT_PATH 1
#define SELECT_LAST 2
struct ncclTopoNode {
  int type;
  int id;
  int rank;
  int present; // Used by compute algorithms
  int select;  // Used by compute algorithms
  struct ncclTopoLink links[NCCL_TOPO_MAX_LINKS];
};

struct ncclTopoNodeSet {
  int count;
  struct ncclTopoNode nodes[NCCL_TOPO_MAX_NODES];
};

struct ncclTopoSystem {
  struct ncclTopoNodeSet nodes[NCCL_TOPO_NODE_TYPES];
};

struct ncclTopoPath {
  int hops;
  int width;
  struct ncclTopoLink* links[NCCL_TOPO_MAX_NODES*NCCL_TOPO_NODE_TYPES];
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
  (*node)->present = 1;
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
  int hexOffset = offset-minOffset+1;
  for (; offset > minOffset; offset--) {
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
  // Then attach to a NUMA node
  int numaId = getNumaId(path);
  for (int n=0; n<system->nodes[NUMA].count; n++) {
    if (system->nodes[NUMA].nodes[n].id == numaId) {
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[NUMA].nodes+n, lastNode, LINK_PCI, PCI_WIDTH, system));
      NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[NUMA].nodes+n, LINK_PCI, PCI_WIDTH, system));
      return ncclSuccess;
    }
  }
  struct ncclTopoNode* numaNode;
  NCCLCHECK(ncclTopoCreateNode(system, &numaNode, NUMA, numaId));
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
      NCCLCHECK(ncclTopoCreatePciPath(system, netNode, path));
      free(path);
    }
  }

  // And connect all NUMA nodes together
  for (int n=0; n<system->nodes[NUMA].count; n++) {
    for (int p=0; p<system->nodes[NUMA].count; p++) {
      if (n == p) continue;
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[NUMA].nodes+n, system->nodes[NUMA].nodes+p, LINK_QPI, QPI_WIDTH, system));
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

ncclResult_t ncclTopoSystemSelectReset(struct ncclTopoSystem* system) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++)
    for (int i=0; i<system->nodes[t].count; i++)
      system->nodes[t].nodes[i].select = 0;
  return ncclSuccess;
}

// We want the graph to be organized to ease/accelerate traversal :
// 1. NVLinks (already the case)
// 2. PCI down
// 3. PCI up
// 4. QPI (already the case)
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system) {
  for (int n=0; n<system->nodes[NUMA].count; n++) NCCLCHECK(ncclTopoSort(system->nodes[NUMA].nodes+n, NULL));
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoNode* node, int offset, struct ncclTopoNode* prevNode) {
  if (node->type == GPU) {
    printf("%s/%X (%d)\n", topoNodeTypeStr[node->type], node->id, node->rank);
  } else {
    printf("%s/%X\n", topoNodeTypeStr[node->type], node->id);
  }
  for (int l=0; l<NCCL_TOPO_MAX_LINKS; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->remNode && link->remNode != prevNode) {
      for (int i=0; i<offset; i++) printf(" ");
      printf("+ %s[%2d] - ", topoLinkTypeStr[link->type], link->width);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrint(link->remNode, offset + 11, node));
      } else {
        printf("%s/%X\n", topoNodeTypeStr[link->remNode->type], link->remNode->id);
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
  printf("================ Topology ================\n");
  for (int n=0; n<s->nodes[NUMA].count; n++) NCCLCHECK(ncclTopoPrint(s->nodes[NUMA].nodes+n, 0, NULL));
  printf("==========================================\n");
  *system = s;
  return ncclSuccess;
}

/******************************************************************/
/******************** Graph Search Functions **********************/
/******************************************************************/
ncclResult_t ncclTopoSystemResetPresent(struct ncclTopoSystem* system) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++)
    for (int n=0; n<system->nodes[t].count; n++)
      system->nodes[t].nodes[n].present = 1;
  return ncclSuccess;
}

struct ncclTopoMulti {
  int channel;
  int nChannels;
  struct ncclTopoNode** interList;
  struct ncclTopoPath* paths;
};

ncclResult_t ncclTopoFindPathMulti(struct ncclTopoSystem* system, struct ncclTopoMulti* multi);

//#define GRAPH_TRACE
ncclResult_t ncclTopoFindPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, struct ncclTopoPath* path, int nselect, struct ncclTopoMulti* multi) {
#ifdef GRAPH_TRACE
  if (multi) printf("[%d] ", multi->channel);
  for (int i=0; i<16-nselect; i++) printf(" ");
  printf("%p : %s/%X [%d/%d] rem %d\n", node, topoNodeTypeStr[node->type], node->id, node->present, node->select, nselect);
#endif
  int maxWidth = 0;
  int minHops = 9999;
  struct ncclTopoPath tmpPath;
  for (int l=0; l<NCCL_TOPO_MAX_LINKS && node->links[l].remNode; l++) {
    struct ncclTopoLink* link = node->links+l;
    struct ncclTopoNode* remNode = link->remNode;
#ifdef GRAPH_TRACE
    printf("+ %s[%2d] - ", topoLinkTypeStr[link->type], link->width);
    printf("%s/%X [%d/%d]\n", topoNodeTypeStr[remNode->type], remNode->id, remNode->present, remNode->select);
#endif
    if (link->width == 0) continue;
    if (remNode->present == 0) continue;
    if (remNode->select & SELECT_LAST) {
      if (nselect > 1) continue;
      else {
        path->links[path->hops] = link;
        minHops = 1;
        maxWidth = link->width;
#ifdef GRAPH_TRACE
        printf("Path complete\n");
#endif
        if (multi) {
          multi->channel++;
          NCCLCHECK(ncclTopoFindPathMulti(system, multi));
          multi->channel--;
        }
      }
    } else {
      tmpPath.width = link->width;
      link->width--;
      tmpPath.hops = 1;
      tmpPath.links[0] = link;
      int inPath = remNode->select & SELECT_PATH;
      remNode->present-=inPath;
      NCCLCHECK(ncclTopoFindPath(system, link->remNode, &tmpPath, nselect-inPath, multi));
      remNode->present+=inPath;
      if (tmpPath.width > maxWidth || (tmpPath.width == maxWidth && tmpPath.hops < minHops)) {
        maxWidth = tmpPath.width;
        minHops = tmpPath.hops;
        // Save current best solution in path
        for (int i=0; i<tmpPath.hops; i++) path->links[path->hops+i] = tmpPath.links[i];
      }
      link->width++;
    }
  }
  path->hops += minHops;
  path->width = std::min(path->width, maxWidth);
  return ncclSuccess;
}

ncclResult_t ncclTopoFindPathMulti(struct ncclTopoSystem* system, struct ncclTopoMulti* multi) {
  if (multi->channel == multi->nChannels) return ncclSuccess;
  NCCLCHECK(ncclTopoSystemSelectReset(system));
  for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_PATH;
  if (multi->interList) {
    multi->interList[2*multi->channel+1]->select = 0; // Start on second GPU
    multi->interList[2*multi->channel]->select = SELECT_LAST; // End on first GPU
    multi->paths[multi->channel].width = 99;
    NCCLCHECK(ncclTopoFindPath(system, multi->interList[2*multi->channel+1], multi->paths+multi->channel, system->nodes[GPU].count-1, multi));
  } else {
    struct ncclTopoNode* gpu0 = system->nodes[GPU].nodes+0;
    gpu0->select = SELECT_LAST;
    multi->paths[multi->channel].width = 99;
    NCCLCHECK(ncclTopoFindPath(system, gpu0, multi->paths+multi->channel, system->nodes[GPU].count, multi));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSelectPath(struct ncclTopoPath* path) {
  for (int i=0; i<path->hops; i++) path->links[i]->width -= path->width;
  return ncclSuccess;
}
ncclResult_t ncclTopoUnSelectPath(struct ncclTopoPath* path) {
  for (int i=0; i<path->hops; i++) path->links[i]->width += path->width;
  return ncclSuccess;
}

ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  struct ncclTopoPath* paths;
  NCCLCHECK(ncclCalloc(&paths, MAXCHANNELS));
  struct ncclTopoNode** interNodes = NULL;

  if (system->nodes[NET].count) {
    int nChannels = 0;
    NCCLCHECK(ncclCalloc(&interNodes, MAXCHANNELS*2));
    struct ncclTopoPath* netPaths;
    NCCLCHECK(ncclCalloc(&netPaths, system->nodes[NET].count));
    
    // Figure out how many NICs are local to two GPUs.
    int minHops = 9999;
    for (int net=0; net<system->nodes[NET].count; net++) {
      struct ncclTopoNode* netNode = system->nodes[NET].nodes+net;
      struct ncclTopoPath path;
      memset(&path, 0, sizeof(path));
      path.width = 99;
      NCCLCHECK(ncclTopoSystemSelectReset(system));
      for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_PATH;
      netNode->select = SELECT_LAST;
      NCCLCHECK(ncclTopoFindPath(system, netNode, &path, 3, NULL));
      if (path.width && path.hops < minHops) { minHops=path.hops; nChannels = 0; }
      if (path.width && path.hops == minHops) nChannels++;
    }
    
    if (nChannels) {
      // Try to mark the NIC->GPU->GPU->NIC path as used for each NIC.
      // If two NICs are using the same path only the first will be used.
      int offset = 0;
      for (int net=0; net<system->nodes[NET].count; net++) {
        struct ncclTopoNode* netNode = system->nodes[NET].nodes+net;
        struct ncclTopoPath *path = netPaths+net;
        path->hops = 0;
        path->width = 99;
        NCCLCHECK(ncclTopoSystemSelectReset(system));
        for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_PATH;
        netNode->select = SELECT_LAST;
        NCCLCHECK(ncclTopoFindPath(system, netNode, path, 3, NULL));
        if (path->width && path->hops == minHops) {
          NCCLCHECK(ncclTopoSelectPath(path));
          for (int i=0; i<path->hops; i++) {
            if (path->links[i]->remNode->type == GPU) interNodes[offset++] = path->links[i]->remNode;
          }
        }
      }
      // Then close the rings
      struct ncclTopoMulti multi = { 0, nChannels, interNodes, paths };
      NCCLCHECK(ncclTopoFindPathMulti(system, &multi));

      // Unselect netPaths
      for (int net=0; net<system->nodes[NET].count; net++) {
        struct ncclTopoPath *path = netPaths+net;
        if (path->width && path->hops == minHops) {
          NCCLCHECK(ncclTopoUnSelectPath(path));
        }
      }

      // Set nChannels in case we need the fallback
      if (paths[0].width == 0) nChannels = 0;
    }
    if (nChannels == 0) {
      printf("Fall back to one GPU\n");
      // Figure out which NICs are local to at least one GPU.
      int minHops = 9999;
      for (int net=0; net<system->nodes[NET].count; net++) {
        struct ncclTopoNode* netNode = system->nodes[NET].nodes+net;
        struct ncclTopoPath path;
        memset(&path, 0, sizeof(path));
        path.width = 99;
        NCCLCHECK(ncclTopoSystemSelectReset(system));
        for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_LAST;
        NCCLCHECK(ncclTopoFindPath(system, netNode, &path, 1, NULL));
        if (path.width && path.hops < minHops) { minHops=path.hops; nChannels = 0; }
        if (path.width && path.hops == minHops) nChannels++;
      }
      if (nChannels == 0) {
        WARN("Could not find path from NIC to GPU\n");
        return ncclInternalError;
      }
      printf("Found %d channels\n", nChannels);
      nChannels = 0;

      // Try to mark the NIC->GPU path as used for each NIC.
      int offset = 0;
      for (int net=0; net<system->nodes[NET].count; net++) {
        struct ncclTopoNode* netNode = system->nodes[NET].nodes+net;
        struct ncclTopoPath *path = netPaths+net;
        path->hops = 0;
        path->width = 99;
        NCCLCHECK(ncclTopoSystemSelectReset(system));
        // go to the first GPU
        for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_LAST;
        NCCLCHECK(ncclTopoFindPath(system, netNode, path, 1, NULL));
        if (path->width == 0 || path->hops > minHops) continue;

        NCCLCHECK(ncclTopoSelectPath(path));
        // then go through all GPUs and come back to NIC
        for (int n=0; n<system->nodes[GPU].count; n++) system->nodes[GPU].nodes[n].select = SELECT_PATH;
        struct ncclTopoNode* firstGpuNode = path->links[path->hops-1]->remNode;
        firstGpuNode->present = 0;
        netNode->select = SELECT_LAST;
        printf("Closing ring %d from %d/%d\n", net, firstGpuNode->id, firstGpuNode->rank);
        NCCLCHECK(ncclTopoFindPath(system, firstGpuNode, path, system->nodes[GPU].count, NULL));
        firstGpuNode->present = 1;
        if (path->width == 0) continue;
        NCCLCHECK(ncclTopoSelectPath(path));
        for (int i=0; i<path->hops; i++) {
          if (path->links[i]->remNode->type == GPU) interNodes[offset++] = path->links[i]->remNode;
        }
        memcpy(paths+nChannels++, path, sizeof(struct ncclTopoPath));
      }
    }
    free(netPaths);
  } else {
    // Intra-node only
    struct ncclTopoMulti multi = { 0, MAXCHANNELS, NULL, paths };
    NCCLCHECK(ncclTopoFindPathMulti(system, &multi));
  }

  int offset = 0;
  printf("Channels :\n");
  for (int c=0; c<MAXCHANNELS; c++) {
    if (paths[c].width) {
      if (interNodes) {
        graph->inter[c*2] = interNodes[c*2]->rank;
        graph->inter[c*2+1] = interNodes[c*2+1]->rank;
        printf("%d %d | ", graph->inter[c*2], graph->inter[c*2+1]);
        graph->intra[offset++] = interNodes[c*2+1]->rank;
        printf("%d ", interNodes[c*2+1]->rank);
      }
      for (int l=0; l<paths[c].hops; l++) {
        struct ncclTopoLink* link = paths[c].links[l];
        if (link->remNode->type == GPU) {
          graph->intra[offset++] = link->remNode->rank;
          printf("%d ", link->remNode->rank);
        }
      }
      printf("\n");
    } else {
      graph->nChannels = c;
      break;
    }
  }
  printf("End of channels.\n");
  free(interNodes);
  free(paths);
  return ncclSuccess;
}
