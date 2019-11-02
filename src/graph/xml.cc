/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "topo.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

// XML Parser.

// A few constraints to make the implementation simpler
#define MAX_STR_LEN 16
#define MAX_ATTR_COUNT 8

#define KEY_TYPE_NONE 0
#define KEY_TYPE_INT 1
#define KEY_TYPE_STR 2

#define NODE_TYPE_NONE 0
#define NODE_TYPE_OPEN 1
#define NODE_TYPE_CLOSE 2
#define NODE_TYPE_SINGLE 3

struct xmlNode {
  char name[MAX_STR_LEN];
  struct {
    char key[MAX_STR_LEN];
    char strValue[MAX_STR_LEN];
    int intValue;
    int type;
  } attrs[MAX_ATTR_COUNT+1]; // Need an extra one to consume extra params
  int nAttrs;
  int type;
};

ncclResult_t ncclTopoGetChar(int fd, char* c) {
  if (read(fd, c, 1) == 0) {
    WARN("XML Parse : Unexpected EOF");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetValue(int fd, char* strValue, int* intValue, int* valueType, char* last) {
  char c;
  NCCLCHECK(ncclTopoGetChar(fd, &c));
  if (c == '"') {
    *valueType = KEY_TYPE_STR;
    int o = 0;
    do {
      NCCLCHECK(ncclTopoGetChar(fd, &c));
      strValue[o++] = c;
    } while (strValue[o-1] != '"');
    strValue[o-1] = '\0';
    NCCLCHECK(ncclTopoGetChar(fd, last));
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
      NCCLCHECK(ncclTopoGetChar(fd, &c));
    }
    *intValue = value;
    *last = c;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoGetToken(int fd, char* name, char* strValue, int* intValue, int* valueType, char* last) {
  char c;
  char* ptr = name;
  int o = 0;
  if (valueType) *valueType = KEY_TYPE_NONE;
  do {
    NCCLCHECK(ncclTopoGetChar(fd, &c));
    if (c == '=') {
      ptr[o] = '\0';
      if (strValue == NULL || intValue == NULL || valueType == NULL) {
        WARN("XML Parse : Unexpected value with name %s\n", ptr);
        return ncclInternalError;
      }
      return ncclTopoGetValue(fd, strValue, intValue, valueType, last);
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

ncclResult_t ncclTopoGetXmlNode(int fd, struct xmlNode* node) {
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
  NCCLCHECK(ncclTopoGetToken(fd, node->name, NULL, NULL, NULL, &c));

  // Check for closing tag
  if (node->name[0] == '\0' && c == '/') {
    node->type = NODE_TYPE_CLOSE;
    // Re-read the name, we got '/' in the first call
    NCCLCHECK(ncclTopoGetToken(fd, node->name, NULL, NULL, NULL, &c));
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
    NCCLCHECK(ncclTopoGetToken(fd, node->attrs[a].key, node->attrs[a].strValue, &node->attrs[a].intValue, &node->attrs[a].type, &c));
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
    NCCLCHECK(ncclTopoGetToken(fd, str, NULL, NULL, NULL, &c));
  }
  if (c != '>') {
    WARN("XML Parse : expected >, got '%c'", c);
    return ncclInternalError;
  }
  return ncclSuccess;
}

typedef ncclResult_t (*ncclTopoXmlHandlerFunc_t)(int, void*, struct xmlNode*);

struct ncclTopoXmlHandler {
  const char * name;
  ncclTopoXmlHandlerFunc_t func;
};

ncclResult_t ncclTopoLoadXmlSub(int fd, void* state, struct xmlNode* head, struct ncclTopoXmlHandler handlers[], int nHandlers) {
  if (head && head->type == NODE_TYPE_SINGLE) return ncclSuccess;
  while (1) {
    struct xmlNode node;
    memset(&node, 0, sizeof(struct xmlNode));
    NCCLCHECK(ncclTopoGetXmlNode(fd, &node));
    if (node.type == NODE_TYPE_NONE) {
      if (head) {
        WARN("XML Parse : unterminated %s", head->name);
        return ncclInternalError;
      } else {
        // All done
        return ncclSuccess;
      }
    }
    if (head && node.type == NODE_TYPE_CLOSE) {
      if (strcmp(node.name, head->name) != 0) {
        WARN("XML Mismatch : %s / %s", head->name, node.name);
        return ncclInternalError;
      }
      return ncclSuccess;
    }
    int found = 0;
    for (int h=0; h<nHandlers; h++) {
      if (strcmp(node.name, handlers[h].name) == 0) {
        NCCLCHECK(handlers[h].func(fd, state, &node));
        found = 1;
        break;
      }
    }
    if (!found) {
      if (nHandlers) INFO(NCCL_GRAPH, "Ignoring element %s", node.name);
      NCCLCHECK(ncclTopoLoadXmlSub(fd, state, &node, NULL, 0));
    }
  }
}

// NCCL Specific -- Build system from XML

#define MAX_NVLINKS 128
#define MAX_NVLINK_DEVICES 128

struct ncclXmlState {
  struct ncclTopoSystem* system;
  struct ncclTopoNode* parent;
  int pciSpeed;
  int nNvLinks;
  struct { 
    struct ncclTopoNode* src;
    int dst;
    int count;
  } nvlinks[MAX_NVLINKS];
  struct ncclTopoNode* nvlids[MAX_NVLINK_DEVICES];
};

struct ncclXmlKV {
  const char* k;
  int v;
};
struct ncclXmlKV ncclXmlKVStore[] = {
 // IB Speeds
 { "FDR", 56 }, { "EDR", 100 }, { "HDR", 200 },
 // PCI Speeds
 { "gen2", 375 }, { "gen3", 750 }, { "gen4", 1500 },
 // CPU models
 { "intel", NCCL_TOPO_CPU_INTEL }, { "broadwell", NCCL_TOPO_CPU_INTEL_BDW }, { "skylake", NCCL_TOPO_CPU_INTEL_SKL }, 
 { "amd", NCCL_TOPO_CPU_AMD }, { "arm", NCCL_TOPO_CPU_ARM }, { "power", NCCL_TOPO_CPU_POWER }
};

ncclResult_t ncclTopoStrConvert(char* str, int* value) {
  for (int s=0; s<sizeof(ncclXmlKVStore)/sizeof(struct ncclXmlKV); s++) {
    if (strcmp(str, ncclXmlKVStore[s].k) == 0) {
      *value = ncclXmlKVStore[s].v;
      return ncclSuccess;
    }
  }
  WARN("Ignoring unknown name %s", str);
  return ncclSuccess;
}

ncclResult_t ncclTopoGetAttribute(struct xmlNode* node, const char* key, void* value, int optional) {
  for (int a=0; a<node->nAttrs; a++) {
    if (strcmp(key, node->attrs[a].key) == 0) {
      if (node->attrs[a].type == KEY_TYPE_INT) {
        *(int*)value = node->attrs[a].intValue;
      } else if (node->attrs[a].type == KEY_TYPE_STR) {
        if (strcmp(key, "bus") == 0) {
          NCCLCHECK(busIdToInt64(node->attrs[a].strValue, (int64_t*)value));
        } else {
          NCCLCHECK(ncclTopoStrConvert(node->attrs[a].strValue, (int*)value));
        }
      }
      return ncclSuccess;
    }
  }
  if (optional) return ncclSuccess;
  WARN("Attribute %s for node %s not found.", key, node->name);
  return ncclInternalError;
}

ncclResult_t ncclTopoXmlLoadSub(int fd, void* state, struct xmlNode* head, struct ncclTopoNode* node, struct ncclTopoXmlHandler handlers[], int nHandlers) {
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  struct ncclTopoNode* parent = s->parent;
  s->parent = node;
  NCCLCHECK(ncclTopoLoadXmlSub(fd, state, head, handlers, nHandlers));
  s->parent = parent;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvlink(int fd, void* state, struct xmlNode* head) {
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  s->nvlinks[s->nNvLinks].src = s->parent;
  NCCLCHECK(ncclTopoGetAttribute(head, "target", &s->nvlinks[s->nNvLinks].dst, 0));
  NCCLCHECK(ncclTopoGetAttribute(head, "count", &s->nvlinks[s->nNvLinks].count, 0));
  s->nNvLinks++;
  
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, NULL, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlConnectPci(struct ncclXmlState* s, struct xmlNode* head, struct ncclTopoNode* node) {
  int pciSpeed = 0;
  int pci = 0;
  NCCLCHECK(ncclTopoGetAttribute(head, "speed", &pci, 0));
  if (pci) {
    int width = 0;
    NCCLCHECK(ncclTopoGetAttribute(head, "width", &width, 0));
    if (width) pciSpeed = pci*width/100;
  }
  if (pciSpeed == 0) pciSpeed = 120;
  if (s->parent) {
    NCCLCHECK(ncclTopoConnectNodes(s->parent, node, LINK_PCI, std::min(pciSpeed, s->pciSpeed)));
    NCCLCHECK(ncclTopoConnectNodes(node, s->parent, LINK_PCI, std::min(pciSpeed, s->pciSpeed)));
  }
  s->pciSpeed = pciSpeed;
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadGpu(int fd, void* state, struct xmlNode* head) {
  int64_t id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "bus", &id, 1));
  if (id == -1)
    NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 0));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, GPU, id));
  NCCLCHECK(ncclTopoGetAttribute(head, "dev", &node->gpu.dev, 1));
  NCCLCHECK(ncclTopoGetAttribute(head, "sm", &node->gpu.cudaCompCap, 1));
  int nvlid = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "nvlid", &nvlid, 1));
  if (nvlid != -1) s->nvlids[nvlid] = node;

  NCCLCHECK(ncclTopoXmlConnectPci(s, head, node));

  struct ncclTopoXmlHandler handlers[] = { { "nvlink", ncclTopoXmlLoadNvlink } };
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNet(int fd, void* state, struct xmlNode* head) {
  int id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "bus", &id, 1));
  if (id == -1)
    NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 0));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, NET, id));
  NCCLCHECK(ncclTopoGetAttribute(head, "asic", &node->net.asic, 1));
  NCCLCHECK(ncclTopoGetAttribute(head, "port", &node->net.port, 1));
  NCCLCHECK(ncclTopoGetAttribute(head, "speed", &node->net.width, 1));
  if (node->net.width != NCCL_TOPO_UNDEF) (node->net.width = node->net.width * 10 / 8); // Convert Gb/s to 100 MB/s

  NCCLCHECK(ncclTopoConnectNodes(s->parent, node, LINK_NET, node->net.width));
  NCCLCHECK(ncclTopoConnectNodes(node, s->parent, LINK_NET, node->net.width));

  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, NULL, 0));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNic(int fd, void* state, struct xmlNode* head) {
  int id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "bus", &id, 1));
  if (id == -1)
    NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 0));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, NIC, id));

  NCCLCHECK(ncclTopoXmlConnectPci(s, head, node));

  struct ncclTopoXmlHandler handlers[] = { { "net", ncclTopoXmlLoadNet } };
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, handlers, 1));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadPci(int fd, void* state, struct xmlNode* head) {
  int id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "bus", &id, 1));
  if (id == -1)
    NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 0));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, PCI, id));

  NCCLCHECK(ncclTopoXmlConnectPci(s, head, node));

  struct ncclTopoXmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic} };
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, handlers, 3));
  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadCpu(int fd, void* state, struct xmlNode* head) {
  int id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 1));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  // If unspecified, just add CPUs in order
  if (id == -1) id = s->system->nodes[CPU].count;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, CPU, id));
  NCCLCHECK(ncclTopoGetAttribute(head, "type", &node->cpu.type, 1));
  NCCLCHECK(ncclTopoGetAttribute(head, "model", &node->cpu.model, 1));

  NCCLCHECK(ncclTopoXmlConnectPci(s, head, node));

  struct ncclTopoXmlHandler handlers[] = { { "pci", ncclTopoXmlLoadPci }, { "gpu", ncclTopoXmlLoadGpu }, { "nic", ncclTopoXmlLoadNic} };
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, handlers, 3));

  return ncclSuccess;
}

ncclResult_t ncclTopoXmlLoadNvs(int fd, void* state, struct xmlNode* head) {
  int id = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "id", &id, 0));
  struct ncclTopoNode* node;
  struct ncclXmlState* s = (struct ncclXmlState*)state;
  // If unspecified, just add CPUs in order
  if (id == -1) id = s->system->nodes[NVS].count;
  NCCLCHECK(ncclTopoCreateNode(s->system, &node, NVS, id));
  int nvlid = -1;
  NCCLCHECK(ncclTopoGetAttribute(head, "nvlid", &nvlid, 1));
  if (nvlid != -1) s->nvlids[nvlid] = node;

  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, node, NULL, 0));
  return ncclSuccess;
}

// Load top "system" description
ncclResult_t ncclTopoXmlLoadSystem(int fd, void* state, struct xmlNode* head) {
  int a;
  for (a=0; a<MAX_ATTR_COUNT; a++) {
    if (head->attrs[a].type == KEY_TYPE_STR && strcmp(head->attrs[a].key, "name") == 0) {
      INFO(NCCL_GRAPH, "Loading topology %s", head->attrs[a].strValue);
      break;
    }
  }
  if (a == MAX_ATTR_COUNT) INFO(NCCL_GRAPH, "Loading unnamed topology");

  struct ncclTopoXmlHandler handlers[] = { { "cpu", ncclTopoXmlLoadCpu }, { "nvs", ncclTopoXmlLoadNvs } };
  NCCLCHECK(ncclTopoXmlLoadSub(fd, state, head, NULL, handlers, 2));
  return ncclSuccess;
}

ncclResult_t ncclTopoLoadSystemFromXml(const char* xmlTopoFile, struct ncclTopoSystem* system) {
  int fd = open(xmlTopoFile, O_RDONLY);
  struct ncclTopoXmlHandler handlers[] = { { "system", ncclTopoXmlLoadSystem } };
  struct ncclXmlState state;
  state.system = system;
  state.nNvLinks = 0;
  state.parent = NULL;
  NCCLCHECK(ncclTopoXmlLoadSub(fd, &state, NULL, NULL, handlers, 1));
  close(fd);
  for (int l=0; l<state.nNvLinks; l++) {
    struct ncclTopoNode* src = state.nvlinks[l].src, *dst = state.nvlids[state.nvlinks[l].dst];
    int cudaCompCap = src->type == GPU ? src->gpu.cudaCompCap : dst->gpu.cudaCompCap;
    int nvLinkSpeed = cudaCompCap > 60 ? VOLTA_NVLINK_WIDTH : PASCAL_NVLINK_WIDTH;
    NCCLCHECK(ncclTopoConnectNodes(src, dst, LINK_NVL, nvLinkSpeed));
  }
  // TODO : add NVLinks
  INFO(NCCL_GRAPH, "XML Import done. Topology is :");
  NCCLCHECK(ncclTopoPrint(system));
  return ncclSuccess;
}

