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

struct ncclTopoLink;
#define NCCL_TOPO_MAX_ARITY 32

enum ncclTopoNodeType { ncclTopoNodeGPU, ncclTopoNodePCI, ncclTopoNodeNVS, ncclTopoNodeNUMA, ncclTopoNodeNET };
enum ncclTopoLinkType { ncclTopoLinkNVL, ncclTopoLinkPCI, ncclTopoLinkQPI };

static const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "NUMA", "NET" };
static const char* topoLinkTypeStr[] = { "NVL", "PCI", "QPI" };

struct ncclTopoNode {
  enum ncclTopoNodeType type;
  int id;
  int rank;
  struct ncclTopoLink* links[NCCL_TOPO_MAX_ARITY];
};

struct ncclTopoLink {
  enum ncclTopoLinkType type;
  struct ncclTopoNode* nodes[2];
};

#define NCCL_MAX_GPU_NODES  2048
#define NCCL_MAX_PCI_NODES  2048
#define NCCL_MAX_NUMA_NODES 2048
#define NCCL_MAX_NET_NODES 2048
#define NCCL_MAX_NVS_NODES 1

struct ncclTopoSystem {
  struct ncclTopoNode* gpuNodes[NCCL_MAX_GPU_NODES];
  int gpuNodeCount;
  struct ncclTopoNode* pciNodes[NCCL_MAX_PCI_NODES];
  int pciNodeCount;
  struct ncclTopoNode* numaNodes[NCCL_MAX_NUMA_NODES];
  int numaNodeCount;
  struct ncclTopoNode* nvsNodes[NCCL_MAX_NVS_NODES];
  int nvsNodeCount;
  struct ncclTopoNode* netNodes[NCCL_MAX_NET_NODES];
  int netNodeCount;
};

ncclResult_t findFreeLink(struct ncclTopoNode* node, int* link) {
  for (int l=0; l<NCCL_TOPO_MAX_ARITY; l++) {
    if (node->links[l] == NULL) { *link = l; return ncclSuccess; }
  }
  WARN("Error : trying to build graph with too many links");
  return ncclInternalError;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node1, struct ncclTopoNode* node2, enum ncclTopoLinkType type) {
  struct ncclTopoLink* link;
  NCCLCHECK(ncclCalloc(&link, 1));
  link->type = type;
  link->nodes[0] = node1;
  link->nodes[1] = node2;
  int l;
  NCCLCHECK(findFreeLink(node1, &l));
  node1->links[l] = link;
  NCCLCHECK(findFreeLink(node2, &l));
  node2->links[l] = link;
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoNode** node, enum ncclTopoNodeType type, int id) {
  NCCLCHECK(ncclCalloc(node, 1));
  (*node)->type = type;
  (*node)->id = id;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNVLink(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system) {
  struct ncclTopoNode* nvsNode = NULL;

  for (int r=0; r<system->gpuNodeCount; r++) {
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
        for (int peer=0; peer<system->gpuNodeCount; peer++) {
          nvmlPciInfo_t pci;
          NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDevs[peer], &pci));
          if (strncmp(pci.busId, remoteProc.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE) == 0) {
            NCCLCHECK(ncclTopoConnectNodes(system->gpuNodes[r], system->gpuNodes[peer], ncclTopoLinkNVL));
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
          NCCLCHECK(ncclTopoCreateNode(&nvsNode, ncclTopoNodeNVS, 0));
        }
        NCCLCHECK(ncclTopoConnectNodes(system->gpuNodes[r], nvsNode, ncclTopoLinkNVL));
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
        for (int p=0; p<system->pciNodeCount; p++) {
          if (system->pciNodes[p]->id == pciId) {
            // Found our PCI switch. Attach and stop since the rest should already
            // be connected
            NCCLCHECK(ncclTopoConnectNodes(lastNode, system->pciNodes[p], ncclTopoLinkPCI));
            return ncclSuccess;
          }
        }
        struct ncclTopoNode* pciNode;
        NCCLCHECK(ncclTopoCreateNode(&pciNode, ncclTopoNodePCI, pciId));
        system->pciNodes[system->pciNodeCount++] = pciNode;
        NCCLCHECK(ncclTopoConnectNodes(lastNode, pciNode, ncclTopoLinkPCI));
        lastNode = pciNode;
      }
    }
  }
  // Then attach to a NUMA node
  int numaId = getNumaId(path);
  for (int n=0; n<system->numaNodeCount; n++) {
    if (system->numaNodes[n]->id == numaId) {
      NCCLCHECK(ncclTopoConnectNodes(lastNode, system->numaNodes[n], ncclTopoLinkPCI));
      return ncclSuccess;
    }
  }
  struct ncclTopoNode* numaNode;
  NCCLCHECK(ncclTopoCreateNode(&numaNode, ncclTopoNodeNUMA, numaId));
  system->numaNodes[system->numaNodeCount++] = numaNode;
  NCCLCHECK(ncclTopoConnectNodes(lastNode, numaNode, ncclTopoLinkPCI));
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

ncclResult_t ncclTopoConnectPCI(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system) {
  for (int r=0; r<system->gpuNodeCount; r++) {
    char* path;
    nvmlPciInfo_t pci;
    NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDevs[r], &pci));
    NCCLCHECK(ncclTopoCudaPath(pci.busId, &path));
    NCCLCHECK(ncclTopoCreatePciPath(system, system->gpuNodes[r], path));
    free(path);
  }

  // Connect the NICs
  int netDevCount;
  NCCLCHECK(ncclNetDevices(&netDevCount));
  for (int n=0; n<netDevCount; n++) {
    struct ncclTopoNode* netNode;
    NCCLCHECK(ncclTopoCreateNode(&netNode, ncclTopoNodeNET, n));
    char* path;
    NCCLCHECK(ncclNetPciPath(n, &path));
    NCCLCHECK(ncclTopoCreatePciPath(system, netNode, path));
    free(path);
  }

  // And connect all NUMA nodes together
  for (int n=0; n<system->numaNodeCount; n++) {
    for (int N=0; N<n; N++) {
      NCCLCHECK(ncclTopoConnectNodes(system->numaNodes[N], system->numaNodes[n], ncclTopoLinkQPI));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem* system) {
  nvmlDevice_t* nvmlDevs;
  NCCLCHECK(ncclCalloc(&nvmlDevs, nranks));
  system->gpuNodeCount = nranks;
  for (int r=0; r<nranks; r++) {
    struct ncclTopoNode* gpuNode;
    NCCLCHECK(ncclCalloc(&gpuNode, 1));
    system->gpuNodes[r] = gpuNode;
    gpuNode->rank = rankIndexes[r];
    gpuNode->id = nvmlIndexes[r];
    NCCLCHECK(wrapNvmlDeviceGetHandleByIndex(nvmlIndexes[r], nvmlDevs+r));
  }

  NCCLCHECK(ncclTopoConnectNVLink(nvmlDevs, system));
  NCCLCHECK(ncclTopoConnectPCI(nvmlDevs, system));

  return ncclSuccess;
}

ncclResult_t ncclTopoFreeNode(struct ncclTopoNode* node) {
  // Disconnect/free link
  for (int l=0; l<NCCL_TOPO_MAX_ARITY; l++) {
    struct ncclTopoLink* link = node->links[l];
    if (link == NULL) continue;
    struct ncclTopoNode** nodes = node->links[l]->nodes;
    // Disconnect from link
    if (nodes[0] == node) nodes[0] = NULL;
    if (nodes[1] == node) nodes[1] = NULL;
    // If link is no longer connected, free it
    if (nodes[0] == nodes[1]) free(link);
  }
  // Free node
  free(node);
  return ncclSuccess;
}

ncclResult_t ncclTopoFreeSystem(struct ncclTopoSystem* system) {
  for (int g=0; g<system->gpuNodeCount; g++) {
    NCCLCHECK(ncclTopoFreeNode(system->gpuNodes[g]));
  }
  for (int p=0; p<system->pciNodeCount; p++) {
    NCCLCHECK(ncclTopoFreeNode(system->pciNodes[p]));
  }
  for (int n=0; n<system->numaNodeCount; n++) {
    NCCLCHECK(ncclTopoFreeNode(system->numaNodes[n]));
  }
  for (int n=0; n<system->nvsNodeCount; n++) {
    NCCLCHECK(ncclTopoFreeNode(system->nvsNodes[n]));
  }
  for (int n=0; n<system->netNodeCount; n++) {
    NCCLCHECK(ncclTopoFreeNode(system->netNodes[n]));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoNode* node, int offset, struct ncclTopoLink* prevLink) {
  if (node->type == ncclTopoNodeGPU) {
    printf("%s/%X (%d)\n", topoNodeTypeStr[node->type], node->id, node->rank);
  } else {
    printf("%s/%X\n", topoNodeTypeStr[node->type], node->id);
  }
  for (int l=0; l<NCCL_TOPO_MAX_ARITY; l++) {
    struct ncclTopoLink* link = node->links[l];
    if (link && link != prevLink) {
      struct ncclTopoNode* remNode = link->nodes[0] == node ? link->nodes[1] : link->nodes[0];
      for (int i=0; i<offset; i++) printf(" ");
      printf("+ %s - ", topoLinkTypeStr[link->type]);
      if (link->type == ncclTopoLinkPCI) {
        NCCLCHECK(ncclTopoPrint(remNode, offset + 8, link));
      } else {
        printf("%s/%X\n", topoNodeTypeStr[remNode->type], remNode->id);
      }
    }
  } 
  return ncclSuccess;
}

ncclResult_t ncclTopoCompute(int nranks, int* nvmlIndexes, int* rankIndexes) {
  struct ncclTopoSystem* system;
  NCCLCHECK(ncclCalloc(&system, 1));
  NCCLCHECK(ncclTopoGetSystem(nranks, nvmlIndexes, rankIndexes, system));
  for (int n=0; n<system->numaNodeCount; n++) NCCLCHECK(ncclTopoPrint(system->numaNodes[n], 0, NULL));

  /* TODO : Compute trees and rings ! */

  NCCLCHECK(ncclTopoFreeSystem(system));
  free(system);
  return ncclSuccess;
}
