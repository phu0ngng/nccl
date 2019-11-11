/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "nvmlwrap.h"
#include "net.h"
#include <sys/stat.h>
#include <fcntl.h>

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

const char* pathDists[] = { "PIX", "PXB", "PHB", "NODE", "SYS" };

const char* topoNodeTypeStr[] = { "GPU", "PCI", "NVS", "CPU", "NIC", "NET" };
const char* topoLinkTypeStr[] = { "LOC", "NVL", "PCI", "QPI", "NET" };

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

static ncclResult_t getPciPath(char* busId, char** path) {
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

// Get an int64 from a PCI path. For example, sys/class/pci0000:00/0000:00:02.0/0000:02:00.0/ will return 0x000002000.
ncclResult_t pciPathToInt64(char* path, int offset, int minOffset, int64_t* id) {
  char* str = path+offset;
  // Remove trailing "/"
  if (*str == '/') str--;
  // Find next /
  while (*str != '/') str--;
  str++;
  int64_t numid;
  NCCLCHECK(busIdToInt64(str, &numid));
  // Ignore subdevice because those should use the same PCI link so we want to merge nodes.
  numid -= numid & 0xf;
  *id = numid;
  return ncclSuccess;
}

static ncclResult_t idToIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  *index = -1;
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (system->nodes[GPU].nodes[i].id == id) {
      *index = i;
    }
  }
  return ncclSuccess;
}


static ncclResult_t getPath(int64_t id, char** path) {
  char busId[] = "0000:00:00.0";
  NCCLCHECK(int64ToBusId(id, busId));
  NCCLCHECK(getPciPath(busId, path));
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

static ncclResult_t ncclTopoGetCpuInfo(struct ncclTopoNode* cpu) {
#if defined(__PPC__)
  cpu->cpu.type = NCCL_TOPO_CPU_POWER;
  INFO(NCCL_GRAPH, "CPU: PPC");
#elif defined(__aarch64__)
  cpu->cpu.type = NCCL_TOPO_CPU_ARM;
  INFO(NCCL_GRAPH, "CPU: Arm");
#elif defined(__x86_64__)
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
  if (strncmp(cpuid0.vendor, "GenuineIntel", 12) == 0) {
    cpu->cpu.type = NCCL_TOPO_CPU_INTEL;
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
      cpu->cpu.model = NCCL_TOPO_CPU_INTEL_SKL;
      INFO(NCCL_GRAPH, "CPU : Intel Skylake or later");
    } else {
      cpu->cpu.model = NCCL_TOPO_CPU_INTEL_BDW;
      INFO(NCCL_GRAPH, "CPU : Intel Broadwell or earlier");
    }
  }
  else if (strncmp(cpuid0.vendor, "AuthenticAMD", 12) == 0) {
    cpu->cpu.type = NCCL_TOPO_CPU_AMD;
    INFO(NCCL_GRAPH, "CPU : AMD");
  }
#endif
  return ncclSuccess;
}

static ncclResult_t ncclTopoGetInterCpuWidth(struct ncclTopoNode* cpu, int* width) {
  switch (cpu->cpu.type) {
    case NCCL_TOPO_CPU_INTEL: *width = cpu->cpu.model == NCCL_TOPO_CPU_INTEL_SKL ? SKL_QPI_WIDTH : QPI_WIDTH; break;
    case NCCL_TOPO_CPU_POWER: *width = P9_WIDTH; break;
    default: *width = LOC_WIDTH;
  }
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetNetWidth(int* width) {
  *width = NET_WIDTH;
  return ncclSuccess;
}
static ncclResult_t ncclTopoGetPciWidth(char* path, int offset, int* pciWidth) {
  char* filePath;
  NCCLCHECK(ncclCalloc(&filePath, offset + sizeof("/max_link_speed")));
  int fd;

  /* Get speed from max_link_speed */
  int speed = 0;
  memcpy(filePath, path, offset);
  sprintf(filePath+offset, "/max_link_speed");
  if ((fd = open(filePath, O_RDONLY)) != -1) {
    char str[8];
    int len;
    SYSCHECKVAL(read(fd, str, 8), "read", len);
    str[len-1] = '\0'; // Replace \n by \0
    close(fd);
    if (strcmp(str, "2.5 GT/s") == 0) speed=187; // Gen 1
    if (strcmp(str, "5 GT/s") == 0) speed=375;   // Gen 2
    if (strcmp(str, "8 GT/s") == 0) speed=750;   // Gen 3
    if (strcmp(str, "16 GT/s") == 0) speed=1500; // Gen 4
  }

  /* Get width from max_link_width */
  int width = 0;
  sprintf(filePath+offset, "/max_link_width");
  if ((fd = open(filePath, O_RDONLY)) != -1) {
    char str[8];
    int len;
    SYSCHECKVAL(read(fd, str, 8), "read", len);
    str[len-1] = '\0'; // Replace \n by \0
    close(fd);
    width = strtol(str, NULL, 0);
  }
  int GBps = speed*width/1000;
  memcpy(filePath, path, offset);
  filePath[offset] = '\0';
  *pciWidth = GBps ? 10*GBps : PCI_WIDTH;
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

ncclResult_t ncclTopoGetNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  for (int i=0; i<system->nodes[type].count; i++) {
    if (system->nodes[type].nodes[i].id == id) {
      *node = system->nodes[type].nodes+i;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCreateNode(struct ncclTopoSystem* system, struct ncclTopoNode** node, int type, uint64_t id) {
  if (system->nodes[type].count == NCCL_TOPO_MAX_NODES) {
    WARN("Error : tried to create too many nodes of type %d\n", type);
    return ncclInternalError;
  }
  struct ncclTopoNode* n = system->nodes[type].nodes+system->nodes[type].count;
  system->nodes[type].count++;
  n->type = type;
  n->id = id;
  if (type == GPU) {
    // Create link to itself (used in some corner cases)
    n->nlinks=1;
    n->links[0].type = LINK_LOC;
    n->links[0].remNode = n;
    n->links[0].width = LOC_WIDTH;
    n->gpu.dev = NCCL_TOPO_UNDEF;
    n->gpu.rank = NCCL_TOPO_UNDEF;
    n->gpu.cudaCompCap = NCCL_TOPO_UNDEF;
  } else if (type == CPU) {
    n->cpu.type = NCCL_TOPO_UNDEF;
    n->cpu.model = NCCL_TOPO_UNDEF;
  } else if (type == NET) {
    n->net.asic = 0ULL;
    n->net.port = NCCL_TOPO_UNDEF;
    n->net.width = NCCL_TOPO_UNDEF;
  }
  *node = n;
  return ncclSuccess;
}

ncclResult_t ncclTopoRemoveNode(struct ncclTopoSystem* system, int type, int index) {
  struct ncclTopoNode* delNode = system->nodes[type].nodes+index;
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      if (node == delNode) continue;
      for (int l=0; l<node->nlinks; l++) {
        while (node->links[l].remNode == delNode) {
          memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
          node->nlinks--;
        }
        if (node->links[l].remNode->type == type && node->links[l].remNode >= delNode) {
          node->links[l].remNode--;
        }
      }
    }
  }
  memmove(delNode, delNode+1, (system->nodes[type].count-index-1)*sizeof(struct ncclTopoNode));
  system->nodes[type].count--;
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNodes(struct ncclTopoNode* node, struct ncclTopoNode* remNode, int type, int width) {
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

ncclResult_t ncclTopoConnectCpu(struct ncclTopoSystem* system, int numaId, struct ncclTopoNode* node, int linkType, int linkWidth) {
  struct ncclTopoNode* cpuNode = NULL;
  NCCLCHECK(ncclTopoGetNode(system, &cpuNode, CPU, numaId));
  if (cpuNode == NULL) { // Create CPU
    NCCLCHECK(ncclTopoCreateNode(system, &cpuNode, CPU, numaId));
    NCCLCHECK(ncclTopoGetCpuInfo(cpuNode));
  }
  NCCLCHECK(ncclTopoConnectNodes(node, cpuNode, linkType, linkWidth));
  NCCLCHECK(ncclTopoConnectNodes(cpuNode, node, linkType, linkWidth));
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectNVLink(nvmlDevice_t nvmlDev, struct ncclTopoNode* gpu, struct ncclTopoSystem* system) {
  int maxNvLinks, width;
  if (gpu->gpu.cudaCompCap < 60) {
    maxNvLinks = 0;
    width = 0;
  } else if (gpu->gpu.cudaCompCap < 70) {
    maxNvLinks = 4;
    width = PASCAL_NVLINK_WIDTH;
  } else {
    maxNvLinks = 6;
    width = VOLTA_NVLINK_WIDTH;
  }

  int nvlinks = 0;

  for (int l=0; l<gpu->nlinks; l++) {
    if (gpu->links[l].type == LINK_NVL) {
      nvlinks += gpu->links[l].width / width;
      maxNvLinks = 0; // Disable auto detection if nvlink topology is already defined.
    }
  }

  if (nvmlDev == NULL && maxNvLinks > 0) {
    INFO(NCCL_GRAPH, "No NVML handle for gpu %d, not detecting NVLinks\n", gpu->gpu.dev);
    maxNvLinks = 0;
  }

  for (int l=0; l<maxNvLinks; ++l) {
    // Check whether we can use this NVLink for P2P
    unsigned canP2P;
    if ((wrapNvmlDeviceGetNvLinkCapability(nvmlDev, l, NVML_NVLINK_CAP_P2P_SUPPORTED, &canP2P) != ncclSuccess) || !canP2P) continue;

    // Make sure the Nvlink is up. The previous call should have trained the link.
    nvmlEnableState_t isActive;
    if ((wrapNvmlDeviceGetNvLinkState(nvmlDev, l, &isActive) != ncclSuccess) || (isActive != NVML_FEATURE_ENABLED)) continue;

    // Try to figure out what's on the other side of the NVLink
    nvmlPciInfo_t remoteProc;
    if (wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDev, l, &remoteProc) != ncclSuccess) continue;

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
      int64_t remoteId;
      NCCLCHECK(busIdToInt64(lowerId, &remoteId));
      int peer;
      NCCLCHECK(idToIndex(system, remoteId, &peer));
      if (peer != -1) {
        NCCLCHECK(ncclTopoConnectNodes(gpu, system->nodes[GPU].nodes+peer, LINK_NVL, width));
        nvlinks++;
      }
    } else if (type == ncclNvLinkDeviceBridge) {
      // Nvlink between GPU and CPU (PPC)
      // Since the remote bridge does not have a valid numa_node, assume we
      // are connected to the closest CPU.
      char* path;
      NCCLCHECK(getPath(gpu->id, &path));
      int numaId = getNumaId(path);
      free(path);
      NCCLCHECK(ncclTopoConnectCpu(system, numaId, gpu, LINK_NVL, width));
      nvlinks++;
    } else { // Nvswitch
      if (type == ncclNvLinkDeviceUnknown) {
        // The NVLink is up but we couldn't find the PCI device on the other
        // side. Assume it's an NVswitch outside a VM.
        if (l == 0) INFO(NCCL_INIT, "%d/%d -> %s : Assuming NVLink is connected to NVswitch", gpu->gpu.dev, l, lowerId);
      }
      struct ncclTopoNode* nvsNode = NULL;
      NCCLCHECK(ncclTopoGetNode(system, &nvsNode, NVS, 0));
      if (nvsNode == NULL) { // Create nvswitch
        NCCLCHECK(ncclTopoCreateNode(system, &nvsNode, NVS, 0));
      }
      NCCLCHECK(ncclTopoConnectNodes(gpu, nvsNode, LINK_NVL, width));
      NCCLCHECK(ncclTopoConnectNodes(nvsNode, gpu, LINK_NVL, width));
      nvlinks++;
    }
  }
  if (nvlinks > 0) system->maxWidth = std::min(system->maxWidth, width);
  else system->maxWidth = PCI_WIDTH;
  return ncclSuccess;
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

  // Retain device PCI max speed, then get the port max speed and
  // take the min.
  int devPciWidth, portPciWidth;
  NCCLCHECK(ncclTopoGetPciWidth(path, offset, &devPciWidth));

  slashCount = 0;
  while (--offset > offsetRC) {
    if (path[offset] == '/') {
      slashCount++;
      // Find if already existing
      if ((slashCount%2) == 1) {
        NCCLCHECK(ncclTopoGetPciWidth(path, offset, &portPciWidth));
      }
      if ((slashCount%2) == 0) {
        int64_t pciId;
        int width = std::min(portPciWidth, devPciWidth);
        NCCLCHECK(pciPathToInt64(path, offset, offsetRC, &pciId));
        struct ncclTopoNode* pciNode = NULL;
        int cont = 0;
        NCCLCHECK(ncclTopoGetNode(system, &pciNode, PCI, pciId));
        if (pciNode == NULL) {
          NCCLCHECK(ncclTopoCreateNode(system, &pciNode, PCI, pciId));
          cont = 1;
        }
        NCCLCHECK(ncclTopoConnectNodes(pciNode, lastNode, LINK_PCI, width));
        NCCLCHECK(ncclTopoConnectNodes(lastNode, pciNode, LINK_PCI, width));

        // We found an already existing PCI switch. No need to continue.
        if (cont == 0) return ncclSuccess;

        lastNode = pciNode;
        // Get this device pci width.
        NCCLCHECK(ncclTopoGetPciWidth(path, offset, &devPciWidth));
      }
    }
  }
  // Then attach to a CPU node
  int numaId = getNumaId(path);
  NCCLCHECK(ncclTopoGetPciWidth(path, offset, &portPciWidth));
  int width = std::min(portPciWidth, devPciWidth);
  NCCLCHECK(ncclTopoConnectCpu(system, numaId, lastNode, LINK_PCI, width));
  return ncclSuccess;
}

// Try to detect if IB cards are in fact the same physical NIC, hence sharing ports.
#include <glob.h>
#define IB_GUID_PATH "%s/infiniband/mlx5_*/sys_image_guid"
uint64_t getIbGuid(char* path) {
  uint64_t guid = 0ULL;
  char guidPath[PATH_MAX];
  snprintf(guidPath, PATH_MAX, IB_GUID_PATH, path);
  // PATH has a wildcard in it so use glob()
  glob_t globbuf;
  glob(guidPath, 0, NULL, &globbuf);
  if (globbuf.gl_pathc > 0)
    strncpy(guidPath, globbuf.gl_pathv[0], PATH_MAX);
  globfree(&globbuf);
  guidPath[PATH_MAX-1] = '\0';
  FILE *file = fopen(guidPath, "r");
  if (file != NULL) {
    uint64_t a, b, c, d;
    if (fscanf(file, "%04lx:%04lx:%04lx:%04lx", &a, &b, &c, &d) != EOF) {
      guid = (a << 48) + (b << 32) + (c<<16) + d;
      TRACE(NCCL_GRAPH, "Opened %s guid %lx", guidPath, guid);
    }
    fclose(file);
  }
  return guid;
}

#define IB_RATE_PATH "%s/infiniband/mlx5_*/ports/%d/rate"
int getIbWidth(char* path, int port) {
  char ratePath[PATH_MAX];
  snprintf(ratePath, PATH_MAX, IB_RATE_PATH, path, port);
  // PATH has a wildcard in it so use glob()
  glob_t globbuf;
  glob(ratePath, 0, NULL, &globbuf);
  if (globbuf.gl_pathc <= 0) return 0;
  strncpy(ratePath, globbuf.gl_pathv[0], PATH_MAX);
  globfree(&globbuf);
  ratePath[PATH_MAX-1] = '\0';
  FILE *file = fopen(ratePath, "r");
  if (file == NULL) return 0;
  int rate;
  if (fscanf(file, "%d Gb/sec", &rate) != EOF) {
    TRACE(NCCL_GRAPH, "Opened %s rate %d", ratePath, rate);
  } else {
    TRACE(NCCL_GRAPH, "Could not read rate from %s.", ratePath);
    rate = 0;
  }
  fclose(file);
  return 10*rate/8;
}

ncclResult_t ncclTopoAddNet(struct ncclTopoSystem* system) {
  // Connect the NICs
  int netDevCount;
  NCCLCHECK(ncclNetDevices(&netDevCount));

  for (int n=0; n<netDevCount; n++) {
    char* path = NULL;
    ncclResult_t res = ncclNetPciPath(n, &path);
    if (res != ncclSuccess) path = NULL;

    // Create NIC and attach it to the PCI tree
    int64_t id;
    NCCLCHECK(pciPathToInt64(path, strlen(path), 0, &id));
    struct ncclTopoNode* nicNode = NULL;
    NCCLCHECK(ncclTopoGetNode(system, &nicNode, NIC, id));
    if (nicNode == NULL) {
      NCCLCHECK(ncclTopoCreateNode(system, &nicNode, NIC, id));
      if (path) {
        // Create the PCI path
        NCCLCHECK(ncclTopoCreatePciPath(system, nicNode, path));
      } else {
        // This is probably a virtual NIC. Just attach it directly to CPU 0
        NCCLCHECK(ncclTopoConnectCpu(system, 0, nicNode, LINK_PCI, PCI_WIDTH));
      }
    }

    // Create the network side
    struct ncclTopoNode* netNode;
    NCCLCHECK(ncclTopoCreateNode(system, &netNode, NET, n));

    if (netNode->net.asic == NCCL_TOPO_UNDEF) {
      uint64_t ibGuid = getIbGuid(path);
      netNode->net.asic = (ibGuid == 0) ? n : ibGuid;
    }
    if (netNode->net.port == NCCL_TOPO_UNDEF) {
      netNode->net.port = 0;
      // Same PCI path -> different ports of the same NIC
      for (int i=0; i<n; i++) if (system->nodes[NET].nodes[i].id == netNode->id) netNode->net.port++;
    }
    if (netNode->net.width == NCCL_TOPO_UNDEF) {
      netNode->net.width = getIbWidth(path, netNode->net.port+1); // IB ports start at 1
      if (netNode->net.width == 0) netNode->net.width = NET_WIDTH;
    }
    TRACE(NCCL_GRAPH, "%s -> %x/%lx/%d/%d", path, netNode->id, netNode->net.asic, netNode->net.port, netNode->net.width);
    free(path);

    NCCLCHECK(ncclTopoConnectNodes(nicNode, netNode, LINK_NET, netNode->net.width));
    NCCLCHECK(ncclTopoConnectNodes(netNode, nicNode, LINK_NET, netNode->net.width));
  }

  for (int n=system->nodes[NET].count-1; n>=0; n--) {
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    if (net->net.asic == NCCL_TOPO_UNDEF || net->net.port == NCCL_TOPO_UNDEF || net->net.width == NCCL_TOPO_UNDEF)
      NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectCpus(struct ncclTopoSystem* system) {
  // And connect all CPU nodes together
  for (int n=0; n<system->nodes[CPU].count; n++) {
    for (int p=0; p<system->nodes[CPU].count; p++) {
      if (n == p) continue;
      int width;
      NCCLCHECK(ncclTopoGetInterCpuWidth(system->nodes[CPU].nodes+n, &width));
      NCCLCHECK(ncclTopoConnectNodes(system->nodes[CPU].nodes+n, system->nodes[CPU].nodes+p, LINK_QPI, width));
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclTopoPrintRec(struct ncclTopoNode* node, struct ncclTopoNode* prevNode, char* line, int offset) {
  if (node->type == GPU) {
    sprintf(line+offset, "%s/%lX (%d)", topoNodeTypeStr[node->type], node->id, node->gpu.rank);
  } else if (node->type == CPU) {
    sprintf(line+offset, "%s/%lX (%d/%d)", topoNodeTypeStr[node->type], node->id, node->cpu.type, node->cpu.model);
  } else {
    sprintf(line+offset, "%s/%lX", topoNodeTypeStr[node->type], node->id);
  }
  INFO(NCCL_GRAPH, "%s", line);
  for (int i=0; i<offset; i++) line[i] = ' ';

  for (int l=0; l<node->nlinks; l++) {
    struct ncclTopoLink* link = node->links+l;
    if (link->type == LINK_LOC) continue;
    if (link->remNode != prevNode) {
      sprintf(line+offset, "+ %s[%2d] - ", topoLinkTypeStr[link->type], link->width);
      int nextOffset = strlen(line);
      if (link->type == LINK_PCI) {
        NCCLCHECK(ncclTopoPrintRec(link->remNode, node, line, nextOffset));
      } else {
        if (link->remNode->type == NET) {
          sprintf(line+nextOffset, "%s/%lX (%lx/%d/%d)", topoNodeTypeStr[link->remNode->type], link->remNode->id, link->remNode->net.asic, link->remNode->net.port, link->remNode->net.width);
        } else {
          sprintf(line+nextOffset, "%s/%lX", topoNodeTypeStr[link->remNode->type], link->remNode->id);
        }
        INFO(NCCL_GRAPH, "%s", line);
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrint(struct ncclTopoSystem* s) {
  INFO(NCCL_GRAPH, "=== System : maxWidth %2d ===", s->maxWidth);
  char line[1024];
  for (int n=0; n<s->nodes[CPU].count; n++) NCCLCHECK(ncclTopoPrintRec(s->nodes[CPU].nodes+n, NULL, line, 0));
  INFO(NCCL_GRAPH, "==========================================");
  NCCLCHECK(ncclTopoPrintPaths(s));
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

ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system) {
  struct ncclTopoSystem* s;
  NCCLCHECK(ncclCalloc(&s, 1));

  char* xmlTopoFile = getenv("NCCL_TOPO_FILE");
  if (xmlTopoFile) {
    NCCLCHECK(ncclTopoLoadSystemFromXml(xmlTopoFile, s));
  }

  s->maxWidth = LOC_WIDTH;

  // Add/Set GPUs
  for (int r=0; r<comm->nRanks; r++) {
    nvmlDevice_t nvmlDev = NULL;
    if (comm->peerInfo[r].hostHash == comm->peerInfo[comm->rank].hostHash) {
      char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
      NCCLCHECK(int64ToBusId(comm->peerInfo[r].busId, busId));
      if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev) != ncclSuccess) nvmlDev = NULL;

      struct ncclTopoNode* gpu = NULL;
      NCCLCHECK(ncclTopoGetNode(s, &gpu, GPU, comm->peerInfo[r].busId));
      if (gpu == NULL) {
        NCCLCHECK(ncclTopoCreateNode(s, &gpu, GPU, comm->peerInfo[r].busId));
        char* path;
        NCCLCHECK(getPath(gpu->id, &path));
        NCCLCHECK(ncclTopoCreatePciPath(s, gpu, path));
        free(path);
      }
      if (gpu->gpu.dev == NCCL_TOPO_UNDEF && nvmlDev != NULL) {
        NCCLCHECK(wrapNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&gpu->gpu.dev));
      }
      if (gpu->gpu.cudaCompCap == NCCL_TOPO_UNDEF && nvmlDev != NULL) {
        int cudaMajor, cudaMinor;
        NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
        gpu->gpu.cudaCompCap = cudaMajor*10+cudaMinor;
      }
      if (gpu->gpu.cudaCompCap == NCCL_TOPO_UNDEF) continue; // GPU will be removed later

      NCCLCHECK(ncclTopoConnectNVLink(nvmlDev, gpu, s));
      gpu->gpu.rank = r;
    }
  }
  for (int g=s->nodes[GPU].count-1; g>=0; g--) {
    struct ncclTopoNode* gpu = s->nodes[GPU].nodes+g;
    if (gpu->gpu.rank == -1) NCCLCHECK(ncclTopoRemoveNode(s, GPU, g));
  }

  NCCLCHECK(ncclTopoAddNet(s));
  NCCLCHECK(ncclTopoConnectCpus(s));
  NCCLCHECK(ncclTopoSortSystem(s));
  *system = s;
  return ncclSuccess;
}

ncclResult_t ncclTopoGetNvlink(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* nvlink) {
  int g1, g2;
  NCCLCHECK(idToIndex(system, busId1, &g1));
  NCCLCHECK(idToIndex(system, busId2, &g2));
  *nvlink = g1 != -1 && g2 != -1 && system->nodes[GPU].nodes[g1].paths[GPU][g2].type == LINK_NVL;
  return ncclSuccess;
}

ncclResult_t ncclTopoHasNvlink(struct ncclTopoSystem* system, int64_t busId, int* nvlink) {
  int g;
  NCCLCHECK(idToIndex(system, busId, &g));
  for (int i=0; i<system->nodes[GPU].count; i++) {
    if (i == g) continue;
    if (system->nodes[GPU].nodes[g].paths[GPU][i].type == LINK_NVL) {
      *nvlink = 1;
      return ncclSuccess;
    }
  }
  *nvlink = 0;
  return ncclSuccess;
}

static int pathDistance(struct ncclTopoLinkList* links) {
  int distance = PATH_PIX;
  if (links->count > 2) distance = PATH_PXB;
  for (int l=0; l<links->count; l++) {
    // PHB if we go through 1 CPU, SYS if we go through 2 CPUs
    if (links->list[l]->remNode->type == CPU) distance = (distance == PATH_PHB) ? PATH_SYS : PATH_PHB;
  }
  return distance;
}

ncclResult_t ncclTopoGpuDistance(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* distance) {
  int g1, g2;
  NCCLCHECK(idToIndex(system, busId1, &g1));
  NCCLCHECK(idToIndex(system, busId2, &g2));
  *distance = pathDistance(system->nodes[GPU].nodes[g1].paths[GPU]+g2);
  return ncclSuccess;
}

ncclResult_t ncclTopoNetDistance(struct ncclTopoSystem* system, int64_t busId, int netDev, int* distance) {
  int g;
  NCCLCHECK(idToIndex(system, busId, &g));
  *distance = pathDistance(system->nodes[GPU].nodes[g].paths[NET]+netDev);
  return ncclSuccess;
}

ncclResult_t ncclTopoCpuCount(struct ncclTopoSystem* system, int* count) {
  *count = system->nodes[CPU].count;
  return ncclSuccess;
}
