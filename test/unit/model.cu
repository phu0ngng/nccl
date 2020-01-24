#include "topo.h"
#include "comm.h"
#include "collectives.h"
#include "core.h"
#include "xml.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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

const char* protocolNames[] = { "LL", "LL128", "Simple" };
const char* algorithmNames[] = { "Tree", "Ring", "CollNet" };

void runTopo(const char* xmlTopoFile, const char* platform, int nnodes) {
  struct ncclXml* xmlSystem;
  INFO(NCCL_GRAPH, "Loading platform %s", platform);
  CHECK(ncclCalloc(&xmlSystem, 1));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystem));
  struct ncclTopoSystem* system;
  if (xmlSystem->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    return;
  }
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system));
  CHECK(ncclTopoComputePaths(system, NULL));
  if (nnodes == 1) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      CHECK(ncclTopoRemoveNode(system, NET, n));
  }
  CHECK(ncclTopoSearchInit(system));
  CHECK(ncclTopoPrint(system));

  struct ncclTopoGraph ringGraph;
  memset(&ringGraph, 0, sizeof(ringGraph));
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = 2;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = 16;

  struct ncclTopoGraph treeGraph;
  memset(&treeGraph, 0, sizeof(treeGraph));
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_SPLIT_TREE;
  treeGraph.crossNic = 2;
  treeGraph.collNet = 0;

  struct ncclTopoGraph cNetGraph;
  memset(&cNetGraph, 0, sizeof(cNetGraph));
  cNetGraph.id = 2;
  cNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  cNetGraph.crossNic = 2;
  cNetGraph.collNet = 1;

  /* Compute */
  CHECK(ncclTopoCompute(system, &ringGraph));
  CHECK(ncclTopoPrintGraph(system, &ringGraph));
  treeGraph.minChannels = 1;
  treeGraph.maxChannels = ringGraph.nChannels;
  CHECK(ncclTopoCompute(system, &treeGraph));
  CHECK(ncclTopoPrintGraph(system, &treeGraph));
  if (nnodes > 1) {
    cNetGraph.minChannels = 1;
    cNetGraph.maxChannels = ringGraph.nChannels;
    CHECK(ncclTopoCompute(system, &cNetGraph));
    CHECK(ncclTopoPrintGraph(system, &cNetGraph));
  }

  struct ncclComm comm;
  comm.topo = system;
  comm.nNodes = nnodes;
  comm.nRanks = system->nodes[GPU].count*nnodes;
  comm.nChannels = ringGraph.nChannels*2;
  comm.channels[0].treeUp.depth = system->nodes[GPU].count-1+log2i(nnodes);
  comm.channels[0].buffSize = 1 << 22;
  int compCap = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  CHECK(ncclTopoSetThresholds(&comm, compCap, compCap, &treeGraph, &ringGraph, &cNetGraph));
  struct ncclInfo info;
  info.comm = &comm;
  info.coll = ncclCollAllReduce;
  info.chunkSteps = ALLREDUCE_CHUNKSTEPS;
  info.sliceSteps = ALLREDUCE_SLICESTEPS;
  printf("#     Size ");
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      char fullName[15];
      sprintf(fullName, "%s/%s", algorithmNames[a], protocolNames[p]);
      printf("%14s ", fullName);
    }
  }
  printf("\n");
  
  for (ssize_t size=8; size<(2LL<<32); size<<=1) {
    float times[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
    info.nBytes = size;
    float minTime = -1;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        CHECK(ncclTopoGetAlgoTime(&info, a, p, &times[a][p]));
        if (minTime < 0 || (times[a][p] < minTime && times[a][p]>0)) minTime = times[a][p];
      }
    }
    
    printf("%10ld ", size);
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (times[a][p] == minTime) printf("%c[0;32m", 0x1b);
        printf("%14.1f ", times[a][p]);
        if (times[a][p] == minTime) printf("%c[00m", 0x1b);
      }
    }
    printf("\n");
  }
}

void runPlatform(const char* platform, int nnodes) {
  char xmlTopoFile[1024];
  sprintf(xmlTopoFile, "topo/%s/system.xml", platform);
  runTopo(xmlTopoFile, platform, nnodes);
}

#define RUN(...) runPlatform(__VA_ARGS__)

int main(int argc, const char* argv[]) {
  setlinebuf(stdout);
  if (argc > 2) {
    RUN(argv[1], atoi(argv[2]));
  } else {
    printf("Usage : %s <platform> <nnodes>\n", argv[0]);
    return 1;
  }
  return 0;
}
