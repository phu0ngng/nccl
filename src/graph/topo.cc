/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "nvmlwrap.h"
#include "net.h"
#include <sys/stat.h>
#include <fcntl.h>


#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* pathDists[] = { "PIX", "PXB", "PHB", "NODE", "SYS" };

/******************************************************************/
/******************* Graph Creation Functions *********************/
/******************************************************************/
static int getNumaId(char *path) {
  char npath[PATH_MAX];
  snprintf(npath, PATH_MAX, "%s/numa_node", path);
  npath[PATH_MAX-1] = '\0';

  int numaId = -1;
  FILE *file = fopen(npath, "r");
  if (file == NULL) return -1;
  if (fscanf(file, "%d", &numaId) == EOF) { fclose(file); return -1; }
  fclose(file);

  return numaId;
}

ncclResult_t getPciPath(char* busId, char** path) {
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

ncclResult_t getNvmlPath(int nvmlDev, char** path) {
  nvmlPciInfo_t pci;
  NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDev, &pci));
  NCCLCHECK(getPciPath(pci.busId, path));
  return ncclSuccess;
}

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path) {
  char busId[BUSID_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, BUSID_SIZE, cudaDev));
  NCCLCHECK(getPciPath(busId, path));
  return ncclSuccess;
}


int interCpuWidth = 0;
int cpuPciWidth = 0;

static ncclResult_t getCpuWidths() {
  // Check if already detected
  if (interCpuWidth + cpuPciWidth) return ncclSuccess;

  // Defaults
  char cpu[256];
  sprintf(cpu, "Generic");
  cpuPciWidth = interCpuWidth = PCI_WIDTH;

#ifdef __PPC__
  sprintf(cpu, "ppc64");
  interCpuWidth = P9_WIDTH;
#endif
#ifdef __x86_64__
  sprintf(cpu, "x86_64");
  union {
    struct {
      // CPUID 0 String register order
      uint32_t ebx;
      uint32_t edx;
      uint32_t ecx;
    };
    char vendor[12];
  } cpuid0;

  asm volatile("cpuid" : "=b" (cpuid0.ebx), "=c" (cpuid0.ecx), "=d" (cpuid0.edx) : "a" (0));
  if (strncmp(cpuid0.vendor, "GenuineIntel", 12) == 0) sprintf(cpu, "Intel");

  if (strcmp(cpu, "Intel") == 0) {
    cpuPciWidth = INTEL_PCI_WIDTH;
    union {
      struct {
        int steppingId:4;
        int model:4;
        int familyId:4;
        int processorType:2;
        int resv0:2;
        int extModelId:4;
        int modelId:8;
        int resv1:4;
      };
      uint32_t val;
    } cpuid1;
    asm volatile("cpuid" : "=a" (cpuid1.val) : "a" (1));
    if (cpuid1.familyId == 6 && cpuid1.modelId >= 0x55) { // Skylake
      sprintf(cpu, "Intel/Skylake (or later)");
      interCpuWidth = SKL_QPI_WIDTH;
    } else {
      interCpuWidth = QPI_WIDTH;
    }
  }
#endif
  INFO(NCCL_GRAPH, "%s CPU (PCI %d, InterCpu %d)", cpu, cpuPciWidth, interCpuWidth);
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetInterCpuWidth(int* width) {
  NCCLCHECK(getCpuWidths());
  *width = interCpuWidth;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetCpuPciP2pWidth(int* width) {
  NCCLCHECK(getCpuWidths());
  *width = cpuPciWidth;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetPciWidth(int* width) {
  *width = PCI_WIDTH;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetNetWidth(int* width) {
  *width = NET_WIDTH;
  return ncclSuccess;
}

enum ncclNvLinkDeviceType {
  ncclNvLinkDeviceUnknown,
  ncclNvLinkDeviceGpu,
  ncclNvLinkDeviceSwitch,
  ncclNvLinkDeviceBridge, // IBM/Power NVLink bridge (Device 04ea)
};

static ncclResult_t ncclDeviceType(const char* busId, enum ncclNvLinkDeviceType* type) {
  char classPath[] =  "/sys/bus/pci/devices/0000:00:00.0/class";
  memcpy(classPath+sizeof("/sys/bus/pci/devices/")-1, busId, sizeof("0000:00:00.0")-1);
  char* rPath = realpath(classPath, NULL);
  int fd;
  if ((fd = open(rPath, O_RDONLY)) == -1) {
    // Could not find device. It might be because we're in a VM and
    // we don't see the whole machine. This is handled silently so
    // we don't want to print an INFO error.
    TRACE(NCCL_INIT, "Open of %s failed : %s\n", rPath, strerror(errno));
    return ncclSystemError;
  }
  free(rPath);
  char pciClass[9];
  strncpy(pciClass, "0x000000", 9);
  int len;
  SYSCHECKVAL(read(fd, pciClass, 8), "read", len);
  SYSCHECK(close(fd), "close");
  if (strcmp(pciClass, "0x068000") == 0) {
    // PCI device is of type "Bridge / Other Bridge Device" (NVswitch)
    *type = ncclNvLinkDeviceSwitch;
  } else if (strcmp(pciClass, "0x068001") == 0) {
    // PCI device is of type "Bridge: IBM Device 04ea"
    *type = ncclNvLinkDeviceBridge;
  } else if (strcmp(pciClass, "0x030200") == 0 // "3D Controller" (Tesla)
      || strcmp(pciClass, "0x030000") == 0) {  // "VGA Controller" (GeForce)
    *type = ncclNvLinkDeviceGpu;
  } else {
    *type = ncclNvLinkDeviceUnknown;
  }
  return ncclSuccess;
}


ncclResult_t ncclTopoConnectNVLink(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system) {
  struct ncclTopoNode* nvsNode = NULL;

  int minNvlinks = 6, minWidth = VOLTA_NVLINK_WIDTH;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    int cudaMajor, cudaMinor;
    NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDevs[g], &cudaMajor, &cudaMinor));
    int maxNvLinks, width;
    if (cudaMajor < 6) {
      maxNvLinks = 0;
      width = 0;
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
      if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDevs[g], l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

      // Make sure the Nvlink is up. The previous call should have trained the link.
      nvmlEnableState_t isActive;
      if ((wrapNvmlDeviceGetNvLinkState(nvmlDevs[g], l, &isActive) != ncclSuccess) || (isActive != NVML_FEATURE_ENABLED)) continue;

      // Try to figure out what's on the other side of the NVLink
      nvmlPciInfo_t remoteProc;
      if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevs[g], l, &remoteProc) != ncclSuccess) continue;

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
            NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+g, system->nodes[GPU].nodes+peer, LINK_NVL, width));
            break;
          }
        }
      } else if (type == ncclNvLinkDeviceBridge) {
        // Nvlink between GPU and CPU (PPC)
        // Since the remote bridge does not have a valid numa_node, assume we
        // are connected to the closest CPU.
        char* path;
        NCCLCHECK(getNvmlPath(nvmlDevs[g], &path));
        int numaId = getNumaId(path);
        free(path);
        struct ncclTopoNode* cpuNode = NULL;
        for (int c=0; c<system->nodes[CPU].count; c++) {
          if (system->nodes[CPU].nodes[c].id == numaId) cpuNode = system->nodes[CPU].nodes+c;
        }
        if (cpuNode == NULL) { // Create CPU
          NCCLCHECK(ncclTopoCreateNode(system, &cpuNode, CPU, numaId));
        }
        NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+g, cpuNode, LINK_NVL, width));
        NCCLCHECK(ncclTopoConnectNodes(cpuNode, system->nodes[GPU].nodes+g, LINK_NVL, width));
      } else { // Nvswitch
        if (type == ncclNvLinkDeviceUnknown) {
          // The NVLink is up but we couldn't find the PCI device on the other
          // side. Assume it's an NVswitch outside a VM.
          if (l == 0) INFO(NCCL_INIT, "%d/%d -> %s : Assuming NVLink is connected to NVswitch", g, l, lowerId);
        }
        if (nvsNode == NULL) { // Create nvswitch
          NCCLCHECK(ncclTopoCreateNode(system, &nvsNode, NVS, 0));
        }
        NCCLCHECK(ncclTopoConnectNodes(system->nodes[GPU].nodes+g, nvsNode, LINK_NVL, VOLTA_NVLINK_WIDTH));
        NCCLCHECK(ncclTopoConnectNodes(nvsNode, system->nodes[GPU].nodes+g, LINK_NVL, VOLTA_NVLINK_WIDTH));
      }
      nvlinks++;
    }
    minNvlinks = std::min(minNvlinks, nvlinks);
    minWidth = std::min(minWidth, width);
  }
  int pciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(&pciWidth));
  system->maxSpeed = minNvlinks ? minNvlinks*minWidth : pciWidth;
  system->maxChannels = minNvlinks ? minNvlinks : 1;
  system->maxWidth = minNvlinks ? minWidth : pciWidth;
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
  int pciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(&pciWidth));
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
            NCCLCHECK(ncclTopoConnectNodes(system->nodes[PCI].nodes+p, lastNode, LINK_PCI, pciWidth));
            NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[PCI].nodes+p, LINK_PCI, pciWidth));
            return ncclSuccess;
          }
        }
        struct ncclTopoNode* pciNode;
        NCCLCHECK(ncclTopoCreateNode(system, &pciNode, PCI, pciId));
        NCCLCHECK(ncclTopoConnectNodes(pciNode, lastNode, LINK_PCI, pciWidth));
        NCCLCHECK(ncclTopoConnectNodes(lastNode, pciNode, LINK_PCI, pciWidth));
        lastNode = pciNode;
      }
    }
  }
  // Then attach to a CPU node
  int numaId = getNumaId(path);
  for (int n=0; n<system->nodes[CPU].count; n++) {
    if (system->nodes[CPU].nodes[n].id == numaId) {
      int width;
      NCCLCHECK(ncclTopoGetCpuPciP2pWidth(&width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, lastNode, LINK_PCI, width));
      NCCLCHECK(ncclTopoConnectNodes(lastNode, system->nodes[CPU].nodes+n, LINK_PCI, width));
      return ncclSuccess;
    }
  }
  struct ncclTopoNode* numaNode;
  NCCLCHECK(ncclTopoCreateNode(system, &numaNode, CPU, numaId));
  NCCLCHECK(ncclTopoConnectNodes(numaNode, lastNode, LINK_PCI, pciWidth));
  NCCLCHECK(ncclTopoConnectNodes(lastNode, numaNode, LINK_PCI, pciWidth));
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectPCI(nvmlDevice_t* nvmlDevs, struct ncclTopoSystem* system, int inter) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    char* path;
    NCCLCHECK(getNvmlPath(nvmlDevs[g], &path));
    NCCLCHECK(ncclTopoCreatePciPath(system, system->nodes[GPU].nodes+g, path));
    free(path);
  }

  if (inter) {
    // Connect the NICs
    int netDevCount;
    NCCLCHECK(ncclNetDevices(&netDevCount));
    int netWidth;
    NCCLCHECK(ncclTopoGetNetWidth(&netWidth));

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
          NCCLCHECK(ncclTopoConnectNodes(system->nodes[NIC].nodes+n, netNode, LINK_NET, netWidth));
          NCCLCHECK(ncclTopoConnectNodes(netNode, system->nodes[NIC].nodes+n, LINK_NET, netWidth));
          found = 1;
          break;
        }
      }
      if (!found) {
        struct ncclTopoNode* nicNode;
        NCCLCHECK(ncclTopoCreateNode(system, &nicNode, NIC, pciId));
        NCCLCHECK(ncclTopoConnectNodes(nicNode, netNode, LINK_NET, netWidth));
        NCCLCHECK(ncclTopoConnectNodes(netNode, nicNode, LINK_NET, netWidth));

        // Create the PCI path
        NCCLCHECK(ncclTopoCreatePciPath(system, nicNode, path));
      }
      free(path);
    }
    system->maxSpeed = std::min(system->maxSpeed, netDevCount*netWidth);
    system->maxChannels = std::max(system->maxChannels, netDevCount);
    system->maxWidth = netWidth;
  }

  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      int width;
      NCCLCHECK(ncclTopoGetInterCpuWidth(&width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_QPI, width));
    }
  }
  return ncclSuccess;
}

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int id, struct ncclTopoLinkList** path) {
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %d\n", t, id);
  return ncclInternalError;
}

static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  nextNodeList.count = 0;
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
  }
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->width = 0xfffffff;
  basePath->nvlink = 1;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        int width = std::min(path->width, link->width);
        if (remPath->width < width) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          // Copy the rest of the path
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->width = width;
          remPath->nvlink = path->nvlink & ((link->type == LINK_NVL) ? 1 : 0);

          // Add to the list for the next iteration if not already in the list
          // Disallow GPUs as intermediate steps for now
          if (remNode->type != GPU) {
            int i;
            for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
            if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
          }
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  char line[1024];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%X :", topoNodeTypeStr[node->type], node->id);
#else
  sprintf(line, "%s/%X :", topoNodeTypeStr[node->type], node->id);
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        sprintf(line+offset, "--%s->%s/%X", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type], remNode->id);
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%d)", line, node->paths[t][n].width);
#else
      sprintf(line+offset, "%s/%X (%d/%d%s) ", topoNodeTypeStr[t], n, node->paths[t][n].count, node->paths[t][n].width, node->paths[t][n].nvlink ? "/N" : "");
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

static ncclResult_t getGpuSpeed(struct ncclTopoNode* node, int* speed) {
  int nvlSpeed = 0;
  int nvlPeers = 0;
  for (int l=0; l<node->nlinks; l++) {
    if (node->links[l].type == LINK_NVL) nvlSpeed += node->links[l].width;
    if (node->links[l].remNode->type == GPU) nvlPeers++; else nvlPeers = 2;
  }
  int pciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(&pciWidth));
  *speed = std::min(*speed, std::max(nvlSpeed, pciWidth));
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  if (system->searchInitDone) return ncclSuccess;
  system->maxSpeed = 0xfffffff;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));
    NCCLCHECK(getGpuSpeed(system->nodes[GPU].nodes+g, &system->maxSpeed));
  }
  if (system->nodes[NET].count) {
    for (int n=0; n<system->nodes[NET].count; n++) {
      NCCLCHECK(ncclTopoSetPaths(system->nodes[NET].nodes+n, system));
    }
    // Try to assign one NIC per GPU
    int netMaxSpeed = 0;
    int netMaxSpeedCount = 0;
    for (int n=0; n<system->nodes[NET].count; n++) {
      int maxSpeed = 0;
      struct ncclTopoNode* net = system->nodes[NET].nodes+n;
      for (int g=0; g<system->nodes[GPU].count; g++) {
        maxSpeed = std::max(maxSpeed, net->paths[GPU][g].width);
      }
      if (maxSpeed > netMaxSpeed) {
        netMaxSpeed = maxSpeed;
        netMaxSpeedCount = 1;
      } else if (maxSpeed == netMaxSpeed) {
        netMaxSpeedCount++;
      }
    }
    system->maxSpeed = std::min(system->maxSpeed, netMaxSpeedCount*NET_WIDTH);
  }
  system->searchInitDone = 1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
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
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        sprintf(line+nextOffset, "%s/%X", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxChannels %1d maxWidth %2d ===", s->maxChannels, s->maxWidth);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  return ncclSuccess;
}

static ncclResult_t ncclTopoSort(struct ncclTopoNode* node, struct ncclTopoNode* upNode) {
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

ncclResult_t ncclTopoGetSystem(int nranks, int* nvmlIndexes, int* rankIndexes, struct ncclTopoSystem** system, int inter) {
  struct ncclTopoSystem* s;
  NCCLCHECK(ncclCalloc(&s, 1));
  nvmlDevice_t* nvmlDevs;
  NCCLCHECK(ncclCalloc(&nvmlDevs, nranks));
  for (int r=0; r<nranks; r++) {
    struct ncclTopoNode* gpuNode;
    NCCLCHECK(ncclTopoCreateNode(s, &gpuNode, GPU, nvmlIndexes[r]));
    gpuNode->rank = rankIndexes[r];
    NCCLCHECK(wrapNvmlDeviceGetHandleByIndex(nvmlIndexes[r], nvmlDevs+r));
  }

  NCCLCHECK(ncclTopoConnectNVLink(nvmlDevs, s));
  NCCLCHECK(ncclTopoConnectPCI(nvmlDevs, s, inter));

  free(nvmlDevs);
  NCCLCHECK(ncclTopoSortSystem(s));
  NCCLCHECK(ncclTopoPrint(s));
  NCCLCHECK(ncclTopoSearchInit(s));
  *system = s;
  return ncclSuccess;
}

static ncclResult_t nvmlToIndex(struct ncclTopoSystem* system, int nvmlDev, int* index) {
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].id == nvmlDev) {
      *index = i;
      return ncclSuccess;
    }
  }
  WARN("nvmlToIndex : error finding NVML device %d", nvmlDev);
  return ncclInternalError;
}

ncclResult_t ncclTopoGetNvlink(struct ncclTopoSystem* system, int nvmlDev1, int nvmlDev2, int* nvlink) {
  int id1, id2;
  NCCLCHECK(nvmlToIndex(system, nvmlDev1, &id1));
  NCCLCHECK(nvmlToIndex(system, nvmlDev2, &id2));
  *nvlink = system->nodes[GPU].nodes[id1].paths[GPU][id2].nvlink;
  return ncclSuccess;
}

ncclResult_t ncclTopoHasNvlink(struct ncclTopoSystem* system, int nvmlDev, int* nvlink) {
  int id;
  NCCLCHECK(nvmlToIndex(system, nvmlDev, &id));
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (i == id) continue;
    if (system->nodes[GPU].nodes[id].paths[GPU][i].nvlink) {
      *nvlink = 1;
      return ncclSuccess;
    }
  }
  *nvlink = 0;
  return ncclSuccess;
}

// Return one of the closest nets and its distance ; round robin based on "rrid"
ncclResult_t ncclTopoGetNet(struct ncclTopoSystem* system, int nvmlDev, int rrid, int* net) {
  int id;
  NCCLCHECK(nvmlToIndex(system, nvmlDev, &id));
  int maxWidth = 0;
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoLinkList* links = system->nodes[GPU].nodes[id].paths[NET]+n;
    if (links->width > maxWidth) maxWidth = links->width;
  }
  if (maxWidth == 0) {
    WARN("Error : could not find network");
    return ncclInternalError;
  }
  int i = 0;
  while (1) {
    for (int n=0; n<system->nodes[NET].count; n++) {
      struct ncclTopoLinkList* links = system->nodes[GPU].nodes[id].paths[NET]+n;
      if (links->width == maxWidth) {
        if (i == rrid) {
          *net = n;
          return ncclSuccess;
        }
        i++;
      }
    }
  }
}

static int pathDistance(struct ncclTopoLinkList* links) {
  int distance = PATH_PIX;
  if (links->count > 2) distance = PATH_PXB;
  for (int l=0; l<links->count; l++) {
    if (links->list[l]->remNode->type == CPU) distance = PATH_PHB;
    if (links->list[l]->type == LINK_QPI) distance = PATH_SYS;
  }
  return distance;
}

ncclResult_t ncclTopoGpuDistance(struct ncclTopoSystem* system, int nvmlDev1, int nvmlDev2, int* distance) {
  int id1, id2;
  NCCLCHECK(nvmlToIndex(system, nvmlDev1, &id1));
  NCCLCHECK(nvmlToIndex(system, nvmlDev2, &id2));
  *distance = pathDistance(system->nodes[GPU].nodes[id1].paths[GPU]+id2);
  return ncclSuccess;
}

ncclResult_t ncclTopoNetDistance(struct ncclTopoSystem* system, int nvmlDev, int netDev, int* distance) {
  int id;
  NCCLCHECK(nvmlToIndex(system, nvmlDev, &id));
  *distance = pathDistance(system->nodes[GPU].nodes[id].paths[NET]+netDev);
  return ncclSuccess;
}

ncclResult_t ncclTopoCpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[CPU].count;
  return ncclSuccess;
}
