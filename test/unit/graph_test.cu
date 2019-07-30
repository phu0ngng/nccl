#include "topo.h"

const char* pci2R_topo[] = {
  "CPU/0-PCI/10-NIC/0-NET/0",
  "      PCI/10-GPU/0",
  "      PCI/10-GPU/1",
  "      PCI/10-GPU/2",
  "      PCI/10-GPU/3",
  "CPU/1-PCI/11-GPU/4",
  "      PCI/11-GPU/5",
  "      PCI/11-GPU/6",
  "      PCI/11-GPU/7",
  "CPU/0-CPU/1"
};

const char* pci1R_topo[] = {
  "CPU/0-PCI/10-NIC/0-NET/0",
  "      PCI/10-GPU/0",
  "      PCI/10-GPU/1",
  "      PCI/10-GPU/2",
  "      PCI/10-GPU/3",
  "CPU/0-PCI/20-GPU/4",
  "      PCI/20-GPU/5",
  "      PCI/20-GPU/6",
  "      PCI/20-GPU/7",
  "CPU/0-CPU/1"
};

const char* pciNV_topo[] = {
  "CPU/0-PCI/10-NIC/0-NET/0",
  "      PCI/10-GPU/0",
  "      PCI/10-GPU/1",
  "      PCI/10-GPU/2",
  "      PCI/10-GPU/3",
  "CPU/0-PCI/20-GPU/4",
  "      PCI/20-GPU/5",
  "      PCI/20-GPU/6",
  "      PCI/20-GPU/7",
  "GPU/0-GPU/1-GPU/0",
  "GPU/0-GPU/1-GPU/0",
  "CPU/0-CPU/1"
};

const char* dgx1p_topo[] = {
  "CPU/0-PCI/10-GPU/0",
  "      PCI/10-GPU/1",
  "      PCI/10-NIC/0-NET/0",
  "CPU/0-PCI/20-GPU/2",
  "      PCI/20-GPU/3",
  "      PCI/20-NIC/1-NET/1",
  "CPU/1-PCI/11-GPU/4",
  "      PCI/11-GPU/5",
  "      PCI/11-NIC/2-NET/2",
  "CPU/1-PCI/21-GPU/6",
  "      PCI/21-GPU/7",
  "      PCI/21-NIC/3-NET/3",
  "GPU/0-GPU/3-GPU/2-GPU/1-GPU/5-GPU/6-GPU/7-GPU/4-GPU/0",
  "GPU/0-GPU/2-GPU/6-GPU/4-GPU/5-GPU/7-GPU/3-GPU/1-GPU/0",
  "CPU/0-CPU/1"
};

const char* dgx1v_topo[] = {
  "CPU/0-PCI/10-GPU/0",
  "      PCI/10-GPU/1",
  "      PCI/10-NIC/0-NET/0",
  "CPU/0-PCI/20-GPU/2",
  "      PCI/20-GPU/3",
  "      PCI/20-NIC/1-NET/1",
  "CPU/1-PCI/11-GPU/4",
  "      PCI/11-GPU/5",
  "      PCI/11-NIC/2-NET/2",
  "CPU/1-PCI/21-GPU/6",
  "      PCI/21-GPU/7",
  "      PCI/21-NIC/3-NET/3",
  "GPU/0-GPU/3-GPU/2-GPU/1-GPU/5-GPU/6-GPU/7-GPU/4-GPU/0",
  "GPU/0-GPU/3-GPU/2-GPU/1-GPU/5-GPU/6-GPU/7-GPU/4-GPU/0",
  "GPU/0-GPU/2-GPU/6-GPU/4-GPU/5-GPU/7-GPU/3-GPU/1-GPU/0",
  "CPU/0-CPU/1"
};

const char* dgx2v_topo[] = {
  "CPU/0-PCI/10-PCI/110-GPU/0",
  "             PCI/110-GPU/1",
  "             PCI/110-NIC/0-NET/0",
  "      PCI/10-PCI/210-GPU/2",
  "             PCI/210-GPU/3",
  "             PCI/210-NIC/1-NET/1",
  "CPU/0-PCI/20-PCI/120-GPU/4",
  "             PCI/120-GPU/5",
  "             PCI/120-NIC/2-NET/2",
  "      PCI/20-PCI/220-GPU/6",
  "             PCI/220-GPU/7",
  "             PCI/220-NIC/3-NET/3",
  "CPU/1-PCI/11-PCI/111-GPU/8",
  "             PCI/111-GPU/9",
  "             PCI/111-NIC/4-NET/4",
  "      PCI/11-PCI/211-GPU/10",
  "             PCI/211-GPU/11",
  "             PCI/211-NIC/5-NET/5",
  "CPU/1-PCI/21-PCI/121-GPU/12",
  "             PCI/121-GPU/13",
  "             PCI/121-NIC/6-NET/6",
  "      PCI/21-PCI/221-GPU/14",
  "             PCI/221-GPU/15",
  "             PCI/221-NIC/7-NET/7",
  "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" ,
  "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" ,
  "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" ,
  "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" ,
  "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" ,
  "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" ,
  "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" ,
  "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" ,
  "NVS/0-GPU/8" , "NVS/0-GPU/8" , "NVS/0-GPU/8" , "NVS/0-GPU/8" , "NVS/0-GPU/8" , "NVS/0-GPU/8" ,
  "NVS/0-GPU/9" , "NVS/0-GPU/9" , "NVS/0-GPU/9" , "NVS/0-GPU/9" , "NVS/0-GPU/9" , "NVS/0-GPU/9" ,
  "NVS/0-GPU/10", "NVS/0-GPU/10", "NVS/0-GPU/10", "NVS/0-GPU/10", "NVS/0-GPU/10", "NVS/0-GPU/10",
  "NVS/0-GPU/11", "NVS/0-GPU/11", "NVS/0-GPU/11", "NVS/0-GPU/11", "NVS/0-GPU/11", "NVS/0-GPU/11",
  "NVS/0-GPU/12", "NVS/0-GPU/12", "NVS/0-GPU/12", "NVS/0-GPU/12", "NVS/0-GPU/12", "NVS/0-GPU/12",
  "NVS/0-GPU/13", "NVS/0-GPU/13", "NVS/0-GPU/13", "NVS/0-GPU/13", "NVS/0-GPU/13", "NVS/0-GPU/13",
  "NVS/0-GPU/14", "NVS/0-GPU/14", "NVS/0-GPU/14", "NVS/0-GPU/14", "NVS/0-GPU/14", "NVS/0-GPU/14",
  "NVS/0-GPU/15", "NVS/0-GPU/15", "NVS/0-GPU/15", "NVS/0-GPU/15", "NVS/0-GPU/15", "NVS/0-GPU/15",
  "CPU/0-CPU/1"
};

const char* dgx2a_topo[] = {
  "CPU/0-PCI/10-GPU/0",
  "      PCI/10-NIC/0-NET/0",
  "CPU/0-PCI/20-GPU/1",
  "      PCI/20-NIC/1-NET/1",
  "CPU/1-PCI/11-GPU/2",
  "      PCI/11-NIC/2-NET/2",
  "CPU/1-PCI/21-GPU/3",
  "      PCI/21-NIC/3-NET/3",
  "CPU/2-PCI/12-GPU/4",
  "      PCI/12-NIC/4-NET/4",
  "CPU/2-PCI/22-GPU/5",
  "      PCI/22-NIC/5-NET/5",
  "CPU/3-PCI/13-GPU/6",
  "      PCI/13-NIC/6-NET/6",
  "CPU/3-PCI/23-GPU/7",
  "      PCI/23-NIC/7-NET/7",
  "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" , "NVS/0-GPU/0" ,
  "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" , "NVS/0-GPU/1" ,
  "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" , "NVS/0-GPU/2" ,
  "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" , "NVS/0-GPU/3" ,
  "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" , "NVS/0-GPU/4" ,
  "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" , "NVS/0-GPU/5" ,
  "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" , "NVS/0-GPU/6" ,
  "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" , "NVS/0-GPU/7" ,
  "CPU/0-CPU/1-CPU/2-CPU/3-CPU/0", "CPU/1-CPU/3", "CPU/0-CPU/2"
};

const char* gcpnv_topo[] = {
  "CPU/0-PCI/10-NIC/0-NET/0",
  "CPU/0-PCI/20-GPU/0",
  "      PCI/20-GPU/1",
  "CPU/0-PCI/30-GPU/2",
  "      PCI/30-GPU/3",
  "CPU/1-PCI/21-GPU/4",
  "      PCI/21-GPU/5",
  "CPU/1-PCI/31-GPU/6",
  "      PCI/31-GPU/7",
  "GPU/0-GPU/1-GPU/3-GPU/2-GPU/7-GPU/6-GPU/4-GPU/5-GPU/0",
  "GPU/0-GPU/1-GPU/3-GPU/2-GPU/7-GPU/6-GPU/4-GPU/5-GPU/0",
  "GPU/0-GPU/1-GPU/3-GPU/2-GPU/7-GPU/6-GPU/4-GPU/5-GPU/0",
  "CPU/0-CPU/1"
};

const char* fbbug_topo[] = {
  "CPU/0-PCI/10-GPU/0",
  "      PCI/10-NIC/0-NET/0",
  "CPU/1-PCI/11-GPU/4",
  "      PCI/11-NIC/2-NET/2",
  "CPU/1-PCI/21-GPU/6",
  "      PCI/21-GPU/7",
  "      PCI/21-NIC/3-NET/3",
  "GPU/0-GPU/4-GPU/6-GPU/7-GPU/4",
  "CPU/0-CPU/1"
};

const char* p9_6v_topo[] = {
  "CPU/0-PCI/10-GPU/0",
  "CPU/0-PCI/20-GPU/1",
  "CPU/0-PCI/30-GPU/2",
  "CPU/0-PCI/40-NIC/0-NET/0",
  "             NIC/0-NET/1",
  "CPU/1-PCI/11-GPU/3",
  "CPU/1-PCI/21-GPU/4",
  "CPU/1-PCI/31-GPU/5",
  "CPU/1-PCI/41-NIC/1-NET/2",
  "             NIC/1-NET/3",
  // GPU-GPU NVLinks
  "GPU/0-GPU/1-GPU/2-GPU/0", "GPU/3-GPU/4-GPU/5-GPU/3",
  "GPU/0-GPU/1-GPU/2-GPU/0", "GPU/3-GPU/4-GPU/5-GPU/3",
  // GPU-P9 NVLinks
  "GPU/0-CPU/0", "GPU/1-CPU/0", "GPU/2-CPU/0", "GPU/3-CPU/1", "GPU/4-CPU/1", "GPU/5-CPU/1",
  "GPU/0-CPU/0", "GPU/1-CPU/0", "GPU/2-CPU/0", "GPU/3-CPU/1", "GPU/4-CPU/1", "GPU/5-CPU/1",
  // Model P9-P9 connection to permit at least 2 NVLinks flows = 44 GB/s
  "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1", "CPU/0-CPU/1"
};


#define ERROR(fmt, ...) do { \
  printf("%s:%d " fmt "\n", __FILE__, __LINE__, __VA_ARGS__); \
  exit(1); \
} while (0);

#define CHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    ERROR("NCCL Check failed : %d", res); \
    exit(1); \
  } \
} while (0);

void createSystem(struct ncclTopoSystem* system, const char* desc[], int descSize, int nvlinkWidth, int inter) {
  for (int i=0; i<descSize; i++) {
    const char* line = desc[i];
    struct ncclTopoNode* lastNode = NULL;
    while (*line == ' ') line++;

    while (1) {
      int type = -1;
      for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
        if (strncmp(line, topoNodeTypeStr[t], strlen(topoNodeTypeStr[t])) == 0) {
          type = t;
        line += strlen(topoNodeTypeStr[t]);
        if (*line != '/') ERROR("Missing '/' after %s.\n", topoNodeTypeStr[t]);
          line += 1;
          break;
        }
      }
      if (type == -1) ERROR("Unable to find type in %s", line);
      int id = -1;
      while (line[0] != '-' && line[0] != '\0') {
        int digit = line[0] - '0';
        if (digit < 0 || digit > 9) ERROR("Could not find id : %s", line);
        if (id == -1) id = digit;
        else id = id*10+digit;
        line++;
      }
      if (id == -1) ERROR("Unable to find id in %s", line);

      // Create new node if needed
      struct ncclTopoNode* node;
      CHECK(ncclTopoCreateNode(system, &node, type, id));

      // Don't add NICs in intra-node mode
      if (inter == 0 && node->type == NIC) break;

      // Connect
      if (lastNode) {
        int width;
        int type;
        int type1 = std::min(node->type, lastNode->type);
        int type2 = std::max(node->type, lastNode->type);
        if (type1 == GPU && type2 == GPU) {
          width = nvlinkWidth;
          type = LINK_NVL;
        } else if (type1 == GPU && type2 == NVS) {
          width = nvlinkWidth;
          type = LINK_NVL;
        } else if (type1 == GPU && type2 == CPU) {
          width = nvlinkWidth;
          type = LINK_NVL;
        } else if (type1 == CPU && type2 == CPU) {
          width = QPI_WIDTH;
          type = LINK_QPI;
        } else if (type1 == PCI && type2 == CPU) {
          width = PCI_CPU_WIDTH;
          type = LINK_PCI;
        } else if (type1 == NIC && type2 == NET) {
          width = NET_WIDTH;
          type = LINK_NET;
        } else {
          width = PCI_WIDTH;
          type = LINK_PCI;
        }
        CHECK(ncclTopoConnectNodes(node, lastNode, type, width));
        CHECK(ncclTopoConnectNodes(lastNode, node, type, width));
      }
      lastNode = node;

      if (line[0] == '\0') break;
      line++; // Skip the "-"
    }
  }

  // Compute maxChannels and maxWidth
  int minNvlinks = 0xfffffff;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* node = system->nodes[GPU].nodes+g;
    int nvlinks = 0;
    for (int l=0; l<node->nlinks; l++) {
      if (node->links[l].type == LINK_NVL) nvlinks += node->links[l].width / nvlinkWidth;
    }
    minNvlinks = std::min(minNvlinks, nvlinks);
  }
  system->maxChannels = minNvlinks ? minNvlinks : 1;
  system->maxWidth = minNvlinks ? nvlinkWidth : PCI_WIDTH;
  if (inter) {
    system->maxChannels = std::max(system->maxChannels, system->nodes[NET].count);
    system->maxWidth = NET_WIDTH;
  }

  // Sort system to accelerate search
  CHECK(ncclTopoSortSystem(system));
}

const char* treeMode[] = { "unknown", "split tree loop", "split tree", "tree" };

#include <sys/time.h>
uint64_t getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1000000+tv.tv_usec;
}

int checkTopo(const char* name, const char** topo, int topoSize, int nvlinkWidth, int inter, int expectedChannels, int expectedSpeed, int expectedTreePattern, int expectedCrossnic) {
  int errors = 0;
  struct ncclTopoSystem system;
  memset(&system, 0, sizeof(system));
  createSystem(&system, topo, topoSize, nvlinkWidth, inter);
  CHECK(ncclTopoPrint(&system));

  struct ncclTopoGraph treeGraph;
  memset(&treeGraph, 0, sizeof(treeGraph));
  treeGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP;
  treeGraph.crossNic = 2;

  struct ncclTopoGraph ringGraph;
  memset(&ringGraph, 0, sizeof(ringGraph));
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = 2;

  uint64_t computeTime = getTime();
  CHECK(ncclTopoCompute(&system, &treeGraph, NULL));
  CHECK(ncclTopoCompute(&system, &ringGraph, &treeGraph));
  computeTime = getTime() - computeTime;

  printf("%s (%s) : ", name, inter ? "inter" : "intra");
  int nChannels = std::min(ringGraph.nChannels, treeGraph.nChannels);
  int speed = std::min(ringGraph.speed, treeGraph.speed);
  printf("%2d x %2d    %15s    %9s   ", nChannels, speed, treeMode[treeGraph.pattern], ringGraph.crossNic == 1 ? "xNic ring" : "ring");
  if ((nChannels != expectedChannels) || (speed != expectedSpeed) || (treeGraph.pattern != expectedTreePattern) || (ringGraph.crossNic != expectedCrossnic)) {
    printf(" FAILED Expected %d x %d (%s/%s)\n", expectedChannels, expectedSpeed, treeMode[expectedTreePattern], expectedCrossnic == 1 ? "xNic ring" : "ring");
    errors++;
  } else if (computeTime > 1000000) {
    printf("   SLOW %ld ms\n", computeTime/1000);
    errors++;
  } else printf("     OK %ld ms\n", computeTime/1000);
  return errors;
}

int main() {
  setlinebuf(stdout);
  initDebug();
  int errors = 0;
  errors += checkTopo("PCI-1R", pci1R_topo, sizeof(pci1R_topo)/sizeof(const char*), 0, 0, 1, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("PCI-1R", pci1R_topo, sizeof(pci1R_topo)/sizeof(const char*), 0, 1, 1, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_TREE, 2);
  errors += checkTopo("PCI-2R", pci2R_topo, sizeof(pci2R_topo)/sizeof(const char*), 0, 0, 1, QPI_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("PCI-2R", pci2R_topo, sizeof(pci2R_topo)/sizeof(const char*), 0, 1, 1, QPI_WIDTH, NCCL_TOPO_PATTERN_TREE, 2);
  errors += checkTopo("PCI-NV", pciNV_topo, sizeof(pciNV_topo)/sizeof(const char*), PASCAL_NVLINK_WIDTH, 0, 1, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("PCI-NV", pciNV_topo, sizeof(pciNV_topo)/sizeof(const char*), PASCAL_NVLINK_WIDTH, 1, 1, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_TREE, 2);
  errors += checkTopo("DGX-1P", dgx1p_topo, sizeof(dgx1p_topo)/sizeof(const char*), PASCAL_NVLINK_WIDTH, 0, 4, PASCAL_NVLINK_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-1P", dgx1p_topo, sizeof(dgx1p_topo)/sizeof(const char*), PASCAL_NVLINK_WIDTH, 1, 4, NET_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-1V", dgx1v_topo, sizeof(dgx1v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 6, VOLTA_NVLINK_WIDTH , NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-1V", dgx1v_topo, sizeof(dgx1v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 4, NET_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-2V", dgx2v_topo, sizeof(dgx2v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 6, VOLTA_NVLINK_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-2V", dgx2v_topo, sizeof(dgx2v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 8, NET_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-2A", dgx2a_topo, sizeof(dgx2a_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 6, VOLTA_NVLINK_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("DGX-2A", dgx2a_topo, sizeof(dgx2a_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 8, NET_WIDTH, NCCL_TOPO_PATTERN_TREE, 1);
  errors += checkTopo("GCP-NV", gcpnv_topo, sizeof(gcpnv_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 6, VOLTA_NVLINK_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("GCP-NV", gcpnv_topo, sizeof(gcpnv_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 1, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  errors += checkTopo("FB-BUG", fbbug_topo, sizeof(fbbug_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 1, QPI_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE, 2);
  errors += checkTopo("FB-BUG", fbbug_topo, sizeof(fbbug_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 2, NET_WIDTH, NCCL_TOPO_PATTERN_TREE, 1);
  errors += checkTopo("P9-6V ", p9_6v_topo, sizeof(p9_6v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  0, 2, VOLTA_NVLINK_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE, 2);
  errors += checkTopo("P9-6V ", p9_6v_topo, sizeof(p9_6v_topo)/sizeof(const char*), VOLTA_NVLINK_WIDTH,  1, 2, PCI_CPU_WIDTH, NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP, 2);
  printf("%d errors (%s)\n", errors, errors ? "FAILED" : "PASSED");
  return errors;
}
