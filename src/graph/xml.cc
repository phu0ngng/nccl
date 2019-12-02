/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include "core.h"
#include "nvmlwrap.h"
#include "xml.h"

/*******************/
/* XML File Parser */
/*******************/

ncclResult_t xmlGetChar(int fd, char* c) {
  if (read(fd, c, 1) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetValue(int fd, char* strValue, int* intValue, int* valueType, char* last) {
  char c;
  NCCLCHECK(xmlGetChar(fd, &c));
  if (c == '"') {
    *valueType = KEY_TYPE_STR;
    int o = 0;
    do {
      NCCLCHECK(xmlGetChar(fd, &c));
      strValue[o++] = c;
    } while (strValue[o-1] != '"');
    strValue[o-1] = '\0';
    NCCLCHECK(xmlGetChar(fd, last));
  } else {
    *valueType = KEY_TYPE_INT;
    int value = 0;
    while (1) {
      if (c == ' ' || c == '\n' || c == '\r' || c == '/' || c == '>') break;
      int digit = c-'0';
      if (digit < 0 || digit > 9) {
        WARN("XML Parse : invalid digit : %c\n", c);
        return ncclInternalError;
      }
      value = value*10 + digit;
      NCCLCHECK(xmlGetChar(fd, &c));
    }
    *intValue = value;
    *last = c;
  }
  return ncclSuccess;
}

ncclResult_t xmlGetToken(int fd, char* name, char* strValue, int* intValue, int* valueType, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  if (valueType) *valueType = KEY_TYPE_NONE;
  do {
    NCCLCHECK(xmlGetChar(fd, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (strValue == NULL || intValue == NULL || valueType == NULL) {
        WARN("XML Parse : Unexpected value with name %s\n", ptr);
        return ncclInternalError;
      }
      return xmlGetValue(fd, strValue, intValue, valueType, last);
    }
    ptr[o] = c;
    if (o == MAX_STR_LEN-1) {
      ptr[o] = '\0';
      WARN("Error : name %s too long (max %d)", ptr, MAX_STR_LEN);
      return ncclInternalError;
    }
    o++;
  } while (c != ' ' && c != '>' && c != '/' && c != '\n' && c != '\r');
  ptr[o-1] = '\0';
  *last = c;
  return ncclSuccess;
}

ncclResult_t xmlGetNode(int fd, struct xmlNode* node) {
  node->type = NODE_TYPE_NONE;
  char c = ' ';
  while (c == ' ' || c == '\n' || c == '\r') {
    if (read(fd, &c, 1) == 0) return ncclSuccess;
  }
  if (c != '<') {
    WARN("XML Parse error : expecting '<', got '%c'", c);
    return ncclInternalError;
  }
  // Read XML element name
  NCCLCHECK(xmlGetToken(fd, node->name, NULL, NULL, NULL, &c));

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(xmlGetToken(fd, node->name, NULL, NULL, NULL, &c));
    if (c != '>') {
      WARN("XML Parse error : unexpected trailing %c in closing tag %s\n", c, node->name);
      return ncclInternalError;
    }
    return ncclSuccess;
  }

  node->type = NODE_TYPE_OPEN;

  // Get Attributes
  int a = 0;
  while (c == ' ') {
    NCCLCHECK(xmlGetToken(fd, node->attrs[a].key, node->attrs[a].strValue, &node->attrs[a].intValue, &node->attrs[a].type, &c));
    if (node->attrs[a].type != KEY_TYPE_NONE) {
      if (a == MAX_ATTR_COUNT) {
        INFO(NCCL_GRAPH, "XML Parse : Ignoring extra attributes (max %d)\n", MAX_ATTR_COUNT);
        // Actually we need to still consume the extra attributes so we have an extra one.
      } else a++;
    }
  }
  node->nAttrs = a;
  if (c == '/') {
    node->type = NODE_TYPE_SINGLE;
    char str[MAX_STR_LEN];
    NCCLCHECK(xmlGetToken(fd, str, NULL, NULL, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*xmlHandlerFunc_t)(int, struct xmlSystem*, struct xmlNode*);

struct xmlHandler {
  const char * name;
  xmlHandlerFunc_t func;
};

ncclResult_t xmlLoadSub(int fd, struct xmlSystem* system, struct xmlNode* head, struct xmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    if (system->maxIndex == MAX_NODES) {
      WARN("Error : XML parser is limited to 1024 nodes\n");
      return ncclInternalError;
    }
    struct xmlNode* node = system->nodes+system->maxIndex;
    memset(node, 0, sizeof(struct xmlNode));
    NCCLCHECK(xmlGetNode(fd, node));
    if (node->type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node->type == NODE_TYPE_CLOSE) {
      if (strcmp(node->name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node->name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node->name, handlers[h].name) == 0) {
        if (head) head->subs[head->nSubs++] = node;
        node->parent = head;
        system->maxIndex++;
        NCCLCHECK(handlers[h].func(fd, system, node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node->name);
      NCCLCHECK(xmlLoadSub(fd, system, node, NULL, 0));
    }
  }
}

/**************/
/* XML Writer */
/**************/

ncclResult_t ncclTopoDumpXmlRec(int indent, int fd, struct xmlNode* node) {
  for (int i=0; i<indent; i++) dprintf(fd, " ");
  dprintf(fd, "<%s", node->name);

  for (int a=0; a<node->nAttrs; a++) {
    if (node->attrs[a].type == KEY_TYPE_INT) {
      dprintf(fd, " %s=%d", node->attrs[a].key, node->attrs[a].intValue);
    } else {
      dprintf(fd, " %s=\"%s\"", node->attrs[a].key, node->attrs[a].strValue);
    }
  }
  if (node->nSubs == 0) {
    dprintf(fd, "/>\n");
  } else {
    dprintf(fd, ">\n");
    for (int s=0; s<node->nSubs; s++) {
      NCCLCHECK(ncclTopoDumpXmlRec(indent+2, fd, node->subs[s]));
    }
    for (int i=0; i<indent; i++) dprintf(fd, " ");
    dprintf(fd, "</%s>\n", node->name);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpSystemToXml(const char* xmlTopoFile, struct xmlSystem* system) {
  int fd = open(xmlTopoFile, O_TRUNC|O_CREAT|O_WRONLY, 0644);
  if (fd == -1) {
    WARN("Unable to open %s, not dumping topology.", xmlTopoFile);
    return ncclSuccess;
  }
  printf("Opened %s\n", xmlTopoFile);
  NCCLCHECK(ncclTopoDumpXmlRec(0, fd, system->nodes));
  printf("Closed %s\n", xmlTopoFile);
  close(fd);
  return ncclSuccess;
}

/****************************************/
/* Parser rules for our specific format */
/****************************************/

ncclResult_t ncclTopoXmlLoadNvlink(int fd, struct xmlSystem* system, struct xmlNode* head) {
  NCCLCHECK(xmlLoadSub(fd, system, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(int fd, struct xmlSystem* system, struct xmlNode* head) {
  struct xmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink } };
  NCCLCHECK(xmlLoadSub(fd, system, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(int fd, struct xmlSystem* system, struct xmlNode* head) {
  NCCLCHECK(xmlLoadSub(fd, system, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(int fd, struct xmlSystem* system, struct xmlNode* head) {
  struct xmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(xmlLoadSub(fd, system, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(int fd, struct xmlSystem* system, struct xmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic} };
  NCCLCHECK(xmlLoadSub(fd, system, head, handlers, 3));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(int fd, struct xmlSystem* system, struct xmlNode* head) {
  struct xmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci } };
  NCCLCHECK(xmlLoadSub(fd, system, head, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvs(int fd, struct xmlSystem* system, struct xmlNode* head) {
  NCCLCHECK(xmlLoadSub(fd, system, head, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadSystem(int fd, struct xmlSystem* system, struct xmlNode* head) {
  int a;
  for (a=0; a<MAX_ATTR_COUNT; a++) {
    if (head->attrs[a].type == KEY_TYPE_STR && strcmp(head->attrs[a].key, "name") == 0) {
      INFO(NCCL_GRAPH, "Loading topology %s", head->attrs[a].strValue);
      break;
    }
  }
  if (a == MAX_ATTR_COUNT) INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct xmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu }, { "nvs", ncclTopoXmlLoadNvs } };
  NCCLCHECK(xmlLoadSub(fd, system, head, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromFile(const char* xmlTopoFile, struct xmlSystem* system) {
  int fd = open(xmlTopoFile, O_RDONLY);
  struct xmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  system->maxIndex = 0;
  NCCLCHECK(xmlLoadSub(fd, system, NULL, handlers, 1));
  close(fd);
  return ncclSuccess;
}

/**********************/
/* XML creation       */
/* from autodetection */
/**********************/

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))
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

ncclResult_t ncclTopoGetStrFromSys(const char* path, const char* fileName, char* strValue) {
  char* filePath;
  NCCLCHECK(ncclCalloc(&filePath, strlen(path) + strlen(fileName)));
  strcpy(filePath, path);
  sprintf(filePath, "%s/%s", path, fileName);
  int offset = 0;
  int fd;
  if ((fd = open(filePath, O_RDONLY)) != -1) {
    int len = 1;
    while (len != 0 && offset < MAX_STR_LEN) {
      SYSCHECKVAL(read(fd, strValue+offset, MAX_STR_LEN-offset), "read", len);
      offset += len;
    }
    close(fd);
  }
  if (offset == 0) {
    strValue[0] = '\0';
    INFO(NCCL_GRAPH, "Topology detection : could not read %s, ignoring", filePath);
  } else {
    strValue[offset-1] = '\0';
  }
  free(filePath);
  return ncclSuccess;
}

ncclResult_t ncclTopoSetAttrFromSys(struct xmlNode* pciNode, const char* path, const char* fileName, const char* attrName) {
  char strValue[MAX_STR_LEN];
  NCCLCHECK(ncclTopoGetStrFromSys(path, fileName, strValue));
  if (strValue[0] != '\0') { NCCLCHECK(xmlSetAttrStr(pciNode, attrName, strValue)); }
  TRACE(NCCL_GRAPH, "Read from sys %s/%s -> %s=%s\n", path, fileName, attrName, strValue);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromCpu(struct xmlNode* cpuNode, struct xmlSystem* system) {
  int index;
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "affinity", &index));
  if (index == -1) {
    int numaId;
    NCCLCHECK(xmlGetAttrInt(cpuNode, "numaid", &numaId));
    // Set affinity
    char cpumaskPath[] = "/sys/devices/system/node/node0000";
    sprintf(cpumaskPath, "/sys/devices/system/node/node%d", numaId);
    NCCLCHECK(ncclTopoSetAttrFromSys(cpuNode, cpumaskPath, "cpumap", "affinity"));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "arch", &index));
  if (index == -1) {
    // Fill CPU type / vendor / model
#if defined(__PPC__)
    NCCLCHECK(xmlSetAttrStr(cpuNode, "arch", "ppc64"));
#elif defined(__aarch64__)
    NCCLCHECK(xmlSetAttrStr(cpuNode, "arch", "arm64"));
#elif defined(__x86_64__)
    NCCLCHECK(xmlSetAttrStr(cpuNode, "arch", "x86_64"));
#endif
  }

#if defined(__x86_64__)
  NCCLCHECK(xmlGetAttrIndex(cpuNode, "vendor", &index));
  if (index == -1) {
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
    char vendor[13];
    strncpy(vendor, cpuid0.vendor, 12);
    vendor[12] = '\0';
    NCCLCHECK(xmlSetAttrStr(cpuNode, "vendor", vendor));
  }

  NCCLCHECK(xmlGetAttrIndex(cpuNode, "familyid", &index));
  if (index == -1) {
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
    NCCLCHECK(xmlSetAttrInt(cpuNode, "familyid", cpuid1.familyId));
    NCCLCHECK(xmlSetAttrInt(cpuNode, "modelid", cpuid1.modelId));
  }
#endif
  return ncclSuccess;
}

ncclResult_t ncclTopoGetPciNode(struct xmlSystem* system, const char* busId, struct xmlNode** pciNode) {
  *pciNode = NULL;
  for (int i=0; i<system->maxIndex; i++) {
    struct xmlNode* node = system->nodes+i;
    if (strcmp(node->name, "pci") == 0) {
      int index;
      NCCLCHECK(xmlGetAttrIndex(node, "busid", &index));
      if (index != -1 && node->attrs[index].type == KEY_TYPE_STR && strcmp(node->attrs[index].strValue, busId) == 0) {
        *pciNode = node;
        break;
      }
    }
  }
  if (*pciNode == NULL) {
    // GPU not in topo yet. Add it.
    *pciNode = system->nodes+system->maxIndex++;
    strcpy((*pciNode)->name, "pci");
  }
  NCCLCHECK(xmlSetAttrStr(*pciNode, "busid", busId));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromSys(struct xmlNode* pciNode, struct xmlSystem* system) {
  // Fill info, then parent
  char* busId;
  NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &busId));
  char* path = NULL;
  int index;
  NCCLCHECK(xmlGetAttrIndex(pciNode, "class", &index));
  if (index == -1) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));
    NCCLCHECK(ncclTopoSetAttrFromSys(pciNode, path, "class", "class"));
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_speed", &index));
  if (index == -1) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));
    char deviceSpeedStr[MAX_STR_LEN];
    float deviceSpeed;
    NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_speed", deviceSpeedStr));
    sscanf(deviceSpeedStr, "%f GT/s", &deviceSpeed);
    char portSpeedStr[MAX_STR_LEN];
    float portSpeed;
    NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_speed", portSpeedStr));
    sscanf(portSpeedStr, "%f GT/s", &portSpeed);
    NCCLCHECK(xmlSetAttrStr(pciNode, "link_speed", portSpeed < deviceSpeed ? portSpeedStr : deviceSpeedStr));
  }
  NCCLCHECK(xmlGetAttrIndex(pciNode, "link_width", &index));
  if (index == -1) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));
    char strValue[MAX_STR_LEN];
    NCCLCHECK(ncclTopoGetStrFromSys(path, "max_link_width", strValue));
    int deviceWidth = strtol(strValue, NULL, 0);
    NCCLCHECK(ncclTopoGetStrFromSys(path, "../max_link_width", strValue));
    int portWidth = strtol(strValue, NULL, 0);
    NCCLCHECK(xmlSetAttrInt(pciNode, "link_width", std::min(deviceWidth,portWidth)));
  }
  struct xmlNode* parent = pciNode->parent;
  if (parent == NULL) {
    if (path == NULL) NCCLCHECK(getPciPath(busId, &path));

    // Save that for later in case next step is a CPU
    char numaIdStr[MAX_STR_LEN];
    NCCLCHECK(ncclTopoGetStrFromSys(path, "numa_node", numaIdStr));

    // Go up one level : rewind two "/"
    int slashCount = 0;
    int parentOffset;
    for (parentOffset = strlen(path)-1; parentOffset>0; parentOffset--) {
      if (path[parentOffset] == '/') slashCount++;
      if (slashCount == 2) break;
    }
    path[parentOffset] = '\0';

    if (strlen(path) == strlen("/sys/devices/pci0000:00")) {
      // This a CPU root complex. Create a CPU tag
      int numaId = strtol(numaIdStr, NULL, 0);
      struct xmlNode* topNode;
      NCCLCHECK(xmlFindTag(system, "system", &topNode));
      NCCLCHECK(xmlGetSubKvInt(topNode, "cpu", &parent, "numaid", numaId));
      if (parent == NULL) {
        NCCLCHECK(xmlAddSub(system, topNode, "cpu", &parent));
        NCCLCHECK(xmlSetAttrInt(parent, "numaid", numaId));
      }
    } else {
      // Continue on the upper PCI switch
      for (int i = strlen(path)-1; i>0; i--) {
        if (path[i] == '/') {
          NCCLCHECK(xmlFindTagKvStr(system, "pci", &parent, "busid", path+i+1));
          if (parent == NULL) {
            parent = system->nodes+system->maxIndex++;
            strcpy(parent->name, "pci");
            NCCLCHECK(xmlSetAttrStr(parent, "busid", path+i+1));
          }
          break;
        }
      }
    }
    pciNode->parent = parent;
    parent->subs[parent->nSubs++] = pciNode;
  }
  if (strcmp(parent->name, "pci") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromSys(parent, system));
  } else if (strcmp(parent->name, "cpu") == 0) {
    NCCLCHECK(ncclTopoGetXmlFromCpu(parent, system));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromGpu(struct xmlNode* pciNode, nvmlDevice_t nvmlDev, struct xmlSystem* system, struct xmlNode** gpuNodeRet) {
  struct xmlNode* gpuNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "gpu", &gpuNode));
  if (gpuNode == NULL) NCCLCHECK(xmlAddSub(system, pciNode, "gpu", &gpuNode));

  int index = -1;

  int dev = -1;
  NCCLCHECK(xmlGetAttrIndex(gpuNode, "dev", &index));
  if (index == -1) {
    if (nvmlDev == NULL) {
      WARN("No NVML, trying to use CUDA instead");
      char* busId;
      NCCLCHECK(xmlGetAttrStr(pciNode, "busid", &busId));
      if (cudaDeviceGetByPCIBusId(&dev, busId) != cudaSuccess) dev = -1;
    } else {
      NCCLCHECK(wrapNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&dev));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "dev", dev));
  }
  NCCLCHECK(xmlGetAttrInt(gpuNode, "dev", &dev));
  if (dev == -1) return ncclSuccess;

  NCCLCHECK(xmlGetAttrIndex(gpuNode, "sm", &index));
  if (index == -1) {
    int cudaMajor, cudaMinor;
    if (nvmlDev == NULL) {
      cudaDeviceProp devProp;
      CUDACHECK(cudaGetDeviceProperties(&devProp, dev));
      cudaMajor = devProp.major; cudaMinor = devProp.minor;
    } else {
      NCCLCHECK(wrapNvmlDeviceGetCudaComputeCapability(nvmlDev, &cudaMajor, &cudaMinor));
    }
    NCCLCHECK(xmlSetAttrInt(gpuNode, "sm", cudaMajor*10+cudaMinor));
  }
  int sm;
  NCCLCHECK(xmlGetAttrInt(gpuNode, "sm", &sm));

  struct xmlNode* nvlNode = NULL;
  NCCLCHECK(xmlGetSub(pciNode, "nvlink", &nvlNode));
  if (nvlNode == NULL) {
    // NVML NVLink detection
    int maxNvLinks = (sm < 60) ? 0 : (sm < 70) ? 4 : 6;

    if (maxNvLinks > 0 && nvmlDev == NULL) {
      WARN("No NVML device handle. Skipping nvlink detection.\n");
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
      
      NCCLCHECK(xmlGetSubKvStr(gpuNode, "nvlink", &nvlNode, "target", lowerId));
      if (nvlNode == NULL) {
        NCCLCHECK(xmlAddSub(system, gpuNode, "nvlink", &nvlNode));
        NCCLCHECK(xmlSetAttrStr(nvlNode, "target", lowerId));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", 1));
      } else {
        int count;
        NCCLCHECK(xmlGetAttrInt(nvlNode, "count", &count));
        NCCLCHECK(xmlSetAttrInt(nvlNode, "count", count+1));
      }
    }
  }
  // Fill target classes
  for (int s=0; s<gpuNode->nSubs; s++) {
    struct xmlNode* sub = gpuNode->subs[s];
    if (strcmp(sub->name, "nvlink") != 0) continue;
    int index;
    NCCLCHECK(xmlGetAttrIndex(sub, "tclass", &index));
    if (index == -1) {
      char* busId;
      NCCLCHECK(xmlGetAttrStr(sub, "target", &busId));
      char* path;
      NCCLCHECK(getPciPath(busId, &path));
      NCCLCHECK(ncclTopoSetAttrFromSys(sub, path, "class", "tclass"));
    }
  }
  *gpuNodeRet = gpuNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillGpu(struct xmlSystem* system, const char* busId, struct xmlNode** gpuNode) {
  struct xmlNode* node;
  NCCLCHECK(ncclTopoGetPciNode(system, busId, &node));
  NCCLCHECK(ncclTopoGetXmlFromSys(node, system));
  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());
  nvmlDevice_t nvmlDev;
  if (wrapNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev) != ncclSuccess) nvmlDev = NULL;
  NCCLCHECK(ncclTopoGetXmlFromGpu(node, nvmlDev, system, gpuNode));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetXmlFromNet(struct xmlNode* nicNode, struct xmlSystem* system, const char* netSysPath, struct xmlNode** netNodeRet) {
  struct xmlNode* netNode;
  NCCLCHECK(xmlGetSub(nicNode, "net", &netNode));
  if (netNode == NULL) {
    NCCLCHECK(xmlAddSub(system, nicNode, "net", &netNode));
  }
  int index;
  NCCLCHECK(xmlGetAttrIndex(netNode, "name", &index));
  if (index == -1 && netSysPath) {
    int offset = strlen(netSysPath)-1;
    while (netSysPath[offset] != '/') offset--;
    NCCLCHECK(xmlSetAttrStr(netNode, "name", netSysPath+offset+1));
  }
  // IP interfaces
  NCCLCHECK(xmlGetAttrIndex(netNode, "speed", &index));
  if (index == -1 && netSysPath) {
    NCCLCHECK(ncclTopoSetAttrFromSys(netNode, netSysPath, "speed", "speed"));
  }
  // Infiniband interfaces
  NCCLCHECK(xmlGetAttrIndex(netNode, "sys_guid", &index));
  if (index == -1 && netSysPath) {
    NCCLCHECK(ncclTopoSetAttrFromSys(netNode, netSysPath, "sys_image_guid", "sys_guid"));
  }
  NCCLCHECK(xmlGetAttrIndex(netNode, "link_rate", &index));
  if (index == -1 && netSysPath) {
    NCCLCHECK(ncclTopoSetAttrFromSys(netNode, netSysPath, "ports/1/rate", "link_rate"));
  }
  *netNodeRet = netNode;
  return ncclSuccess;
}

ncclResult_t ncclTopoFillNic(struct xmlSystem* system, const char* sysPath, struct xmlNode** netNode) {
  // First detect whether it is the net sysPath (old behavior) or the pci sysPath (new behavior)
  int old = 1;
  char* netName = NULL;
  if (sysPath != NULL) {
    char* subsystemPath;
    NCCLCHECK(ncclCalloc(&subsystemPath, strlen(sysPath)+sizeof("/subsystem")));
    sprintf(subsystemPath, "%s/subsystem", sysPath);
    char* subsystemRealPath = realpath(subsystemPath, NULL);
    if (strcmp(subsystemRealPath, "/sys/bus/pci") != 0) {
      old = 0;
      int offset = strlen(subsystemRealPath)-1;
      while (subsystemRealPath[offset] != '/') offset--;
      NCCLCHECK(ncclCalloc(&netName, strlen(subsystemRealPath)));
      strcpy(netName, subsystemRealPath+offset+1);
    }
    free(subsystemRealPath);
  }

  char *pciSysPath = NULL, *netSysPath = NULL;
  if (old == 1) {
    pciSysPath = strdup(sysPath);
  } else {
    netSysPath = strdup(sysPath);
    char* deviceFilePath;
    NCCLCHECK(ncclCalloc(&deviceFilePath, strlen(sysPath)+sizeof("/device")));
    sprintf(deviceFilePath, "%s/device", sysPath);
    struct stat s;
    if (stat(deviceFilePath, &s) == 0) {
      pciSysPath = realpath(deviceFilePath, NULL);
    }
  }

  struct xmlNode* nicNode;
  if (pciSysPath != NULL) {
    struct xmlNode* pciNode;
    int offset;
    for (offset=strlen(pciSysPath)-1; sysPath[offset] != '/'; offset--);
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    strcpy(busId, pciSysPath+offset+1);
    NCCLCHECK(ncclTopoGetPciNode(system, busId, &pciNode));
    NCCLCHECK(ncclTopoGetXmlFromSys(pciNode, system));
    NCCLCHECK(xmlGetSub(pciNode, "nic", &nicNode));
    if (nicNode == NULL) {
      NCCLCHECK(xmlAddSub(system, pciNode, "nic", &nicNode));
    }
  } else {
    // Virtual NIC, no PCI device, attach to first CPU
    struct xmlNode* cpuNode;
    NCCLCHECK(xmlFindTag(system, "cpu", &cpuNode));
    NCCLCHECK(xmlAddSub(system, cpuNode, "nic", &nicNode));
  }
  free(pciSysPath);
  
  if (netName != NULL) {
    int index = -1;
    NCCLCHECK(xmlGetAttrIndex(nicNode, "type", &index));
    if (index == -1) {
      NCCLCHECK(xmlSetAttrStr(nicNode, "type", netName));
    }
    free(netName);
  }
  
  NCCLCHECK(ncclTopoGetXmlFromNet(nicNode, system, netSysPath, netNode));
  free(netSysPath);
  return ncclSuccess;
}
