/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

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

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/

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

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int width, struct ncclTopoSystem* system) {
  // Aggregate links into higher width for NVLink
  struct ncclTopoLink* link;
  for (link = node->links; link->remNode; link++) {
    if (link->remNode == remNode && link->type == type) break;
  }
  if (link->remNode == NULL) node->nlinks++;
  link->type = type;
  link->remNode = remNode;
  link->width += width;
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

  int minNvlinks = 6, minWidth = VOLTA_NVLINK_WIDTH;
  for (int r=0; r<system->nodes[GPU].count; r++) {
    int cudaMajor, cudaMinor;
    NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDevs[r], &cudaMajor, &cudaMinor));
    int maxNvLinks, width;
    if (cudaMajor < 6) {
      maxNvLinks = 0;
      width = PCI_WIDTH;
    } else if (cudaMajor == 6) {
      maxNvLinks = 4;
      width = PASCAL_NVLINK_WIDTH;
    } else {
      maxNvLinks = 6;
      width = VOLTA_NVLINK_WIDTH;
    }

    int nvlinks = 0;
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
        lowerId[c] = tolower(p[c]);
        if (p[c] == 0) break;
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
          if (l == 0) INFO(NCCL_INIT, "%d/%d -> %s : Assuming NVLink is connected to NVswitch", r, l, lowerId);
        }
        if (nvsNode == NULL) { // Create nvswitch
          NCCLCHECK(ncclTopoCreateNode(system, &nvsNode, NVS, 0));
        }
        NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+r, nvsNode, LINK_NVL, VOLTA_NVLINK_WIDTH, system));
        NCCLCHECK(ncclTopoConnectNodes(nvsNode, system->nodes[GPU].nodes+r, LINK_NVL, VOLTA_NVLINK_WIDTH, system));
      }
      nvlinks++;
    }
    minNvlinks = std::min(minNvlinks, nvlinks);
    minWidth = std::min(minWidth, nvlinks ? width : PCI_WIDTH);
  }
  system->maxChannels = minNvlinks ? minNvlinks : 1;
  system->maxWidth = minWidth;
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
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, lastNode, LINK_PCI, PCI_CPU_WIDTH, system));
      NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[CPU].nodes+n, LINK_PCI, PCI_CPU_WIDTH, system));
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
      int found = 0;
      for (int n=0; n<system->nodes[NIC].count; n++) {
        if (system->nodes[NIC].nodes[n].id == pciId) {
          // Found our NIC. Attach to it.
          NCCLCHECK(ncclTopoConnectNodes(system->nodes[NIC].nodes+n, netNode, LINK_NET, NET_WIDTH, system));
          NCCLCHECK(ncclTopoConnectNodes(netNode, system->nodes[NIC].nodes+n, LINK_NET, NET_WIDTH, system));
         found = 1;
         break;
        }
      }
      if (!found) {
        struct ncclTopoNode* nicNode;
        NCCLCHECK(ncclTopoCreateNode(system, &nicNode, NIC, pciId));
        NCCLCHECK(ncclTopoConnectNodes(nicNode, netNode, LINK_NET, NET_WIDTH, system));
        NCCLCHECK(ncclTopoConnectNodes(netNode, nicNode, LINK_NET, NET_WIDTH, system));

        // Create the PCI path
        NCCLCHECK(ncclTopoCreatePciPath(system, nicNode, path));
      }
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
  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
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

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->remNode != prevNode) {
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
  INFO(NCCL_GRAPH, "=== System : maxChannels %1d maxWidth %2d ===", s->maxChannels, s->maxWidth);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrint(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  *system = s;
  return ncclSuccess;
}
