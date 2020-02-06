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

// We don't support collnet yet
#undef NCCL_NUM_ALGORITHMS
#define NCCL_NUM_ALGORITHMS 2

const char* protocolNames[] = { "LL", "LL128", "Simple" };
const char* algorithmNames[] = { "Tree", "Ring", "CollNet" };

void runTopo(const char* xmlTopoFile, const char* platform, int nnodes) {
  int compareData = 0;
  char* str = getenv("COMPARE_DATA");
  if (str && atoi(str)) compareData=1;

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

  treeGraph.nChannels = ringGraph.nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);

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

  // Last column is used for min/best/default.
  int m = NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS;

  printf("----------+"); for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("---------------------+"); printf("\n");
  printf("     Size |");
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      printf(" %7s  / %7s  |", algorithmNames[a], protocolNames[p]);
    }
  }
  printf("%17s    |\n", "Default");
  if (compareData) {
    printf("          |");
    for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("[%9s] %9s|", "data", (i == m) ? "best" : "model");
    printf("\n");
  }
  printf("----------+"); for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("---------------------+"); printf("\n");

  int fd = 0;
  if (compareData) {
    char path[1024];
    sprintf(path, "topo/%s/data/%d.csv", platform, nnodes);
    fd = open(path, O_RDONLY);
    if (fd == -1) {
      printf("Could not open %s\n", path);
    }
  }

  for (ssize_t size=8; size<(2LL<<32); size<<=1) {
    float times[NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1];
    float data[NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1];
    info.nBytes = size;
    times[m] = -1.0; // Min time
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        int i = a*NCCL_NUM_PROTOCOLS+p;
        CHECK(ncclTopoGetAlgoTime(&info, a, p, times+i));
        if (times[m] < 0 || (times[i] < times[m] && times[i] > 0)) times[m] = times[i];
      }
    }

    if (compareData) {
      for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) data[i] = 0.0;
      int c, s = 0, o = 0, i = 0;
      char valueStr[128];
      while (fd != -1) {
        s = read(fd, &c, 1);
        if (s != 1) break;
        if (c == ',' || c == '\n') {
          valueStr[o] = '\0';
          data[i++] = atof(valueStr);
          o = 0;
        } else {
          valueStr[o++] = c;
        }
        if (c == '\n') break;
      }
      // Overwrite min model as the min of data instead of the min of model
      // This is more useful to see how good/bad the algo/proto choices are.
      times[m] = -1.0;
      for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS; i++) {
        if (times[m] < 0 || (data[i] < times[m] && data[i] > 0)) times[m] = data[i];
      }
      if (s != 1) { close(fd); fd = -1; }
    }

    printf("%10ld|", size);
    for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) {
      float delta;
      if (compareData && data[i] != 0.0) printf("[%9.1f] ", data[i]); else printf("%11s ", "");
      if (compareData) {
        if (data[i] != 0.0) {
          delta = (times[i]-data[i])/times[i]; delta *= delta;
          if (delta > .1) printf("%c[0;31m", 0x1b);
          else if (delta > .02) printf("%c[0;33m", 0x1b);
          else printf("%c[0;32m", 0x1b);
        }
      } else if (i != m && times[i] == times[m]) printf("%c[0;32m", 0x1b);
      printf("%9.1f|", times[i]);
      if ((compareData && data[i] != 0.0) || (compareData == 0 && i != m && times[i] == times[m])) printf("%c[00m", 0x1b);
    }
    printf("\n");
  }
  printf("----------+"); for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("---------------------+"); printf("\n");
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
