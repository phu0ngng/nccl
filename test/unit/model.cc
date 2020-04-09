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

int compactMode = 0;
int compareMode = 0;
float totalScore = 0.0;
int totalNpoints = 0;

void runTopo(const char* xmlTopoFile, const char* platform, int ngpus, int nnodes, ncclFunc_t coll) {
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
  // Only keep ngpus
  for (int g=system->nodes[GPU].count-1; g>=ngpus; g--) {
    CHECK(ncclTopoRemoveNode(system, GPU, g));
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
  comm.buffSizes[NCCL_PROTO_SIMPLE] = 1 << 22;
  int compCap = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  CHECK(ncclTopoTuneModel(&comm, compCap, compCap, &treeGraph, &ringGraph, &cNetGraph));
  struct ncclInfo info;
  info.comm = &comm;
  info.coll = coll;
  info.chunkSteps = ALLREDUCE_CHUNKSTEPS;
  info.sliceSteps = ALLREDUCE_SLICESTEPS;

  // Last column is used for min/best/default.
  const int m = NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS;

  int fds[m+1];
  char path[1024];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int i = a*NCCL_NUM_PROTOCOLS+p;
    sprintf(path, "topo/%s/data/%d/%d/%s/%s/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[coll], ncclAlgoStr[a], ncclProtoStr[p]);
    fds[i] = open(path, O_RDONLY);
  }
  sprintf(path, "topo/%s/data/%d/%d/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[coll]);
  fds[m] = open(path, O_RDONLY);
  float score = 0.0;
  int npoints = 0;

  if (!compactMode) {
    printf("----------+"); for (int i=0; i<m+1; i++) printf("---------------------+"); printf("\n");
    printf("     Size |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      printf(" %7s  / %7s  |", ncclAlgoStr[a], ncclProtoStr[p]);
    }
    printf("%19s  |\n", compareMode == 0 ? "Default (file)" : "Default (dry run)");
    printf("          |");
    for (int i=0; i<m+1; i++) printf("[%9s] %9s|", "data", (i == m) ? "best" : "model");
    printf("\n");
    printf("----------+"); for (int i=0; i<m+1; i++) printf("---------------------+"); printf("\n");
  } else {
    printf("%10s/%5d |", platform, nnodes);
  }

  for (ssize_t size=8; size<(2LL<<32); size<<=1) {
    float times[m+2];
    float data[m+2];
    info.nBytes = size;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      int i = a*NCCL_NUM_PROTOCOLS+p;
      CHECK(ncclTopoGetAlgoTime(&info, a, p, times+i));
    }

    for (int i=0; i<m+1; i++) {
      char valueStr[128];
      data[i] = -1.0;
      if (fds[i] != -1) {
        for (int o=0; fds[i] != -1 && o<128; o++) {
          int s = read(fds[i], valueStr+o, 1);
          if (s != 1) {
	    printf("Error while reading value for size %ld in file %d\n", size, i);
            close(fds[i]);
            fds[i] = -1;
          } else if (valueStr[o] == '\n') {
            valueStr[o] = '\0';
	    data[i] = atof(valueStr);
            break;
          }
        }
      }
    }
    // Compute best performance
    times[m] = -1.0;
    float bestModelTime = -1.0;
    int bestModel = -1;
    for (int i=0; i<m; i++) {
      if (data[i] < 0) continue;
      // find best data
      if (times[m] < 0 || times[m] > data[i]) times[m] = data[i];
      // find best model
      if (times[i] < 0) continue;
      if (bestModelTime < 0 || bestModelTime > times[i]) {
        bestModelTime = times[i];
        bestModel = i;
      }
    }
    // Add a column
    times[m+1] = times[m];
    // Store the data picked by the best model (as if we have done a dry default run)
    data[m+1] = -1;
    if (bestModel != -1) data[m+1] = data[bestModel];

    if (!compactMode) {
      printf("%10ld|", size);
      for (int i=0; i<m+1; i++) {
        float delta;
        if (i == m && compareMode != 0) i = m+1;
        if (data[i] != -1.0) {
          printf("[%9.1f] ", data[i]);
          delta = 1-(times[i]/data[i]);
          if (i < m) delta *= delta;
          float s = 1-delta;
          if (s < .8) printf("%c[0;31m", 0x1b);
          else if (s < .95) printf("%c[0;33m", 0x1b);
          else if (s > 1.1) printf("%c[0;34m", 0x1b);
          else printf("%c[0;32m", 0x1b);
        } else {
          printf("%11s ", "");
          if (i < m && times[i] == times[m]) printf("%c[0;32m", 0x1b);
        }
        printf("%9.1f", times[i]);
        if ((data[i] != -1.0) || (i < m && times[i] == times[m])) printf("%c[00m", 0x1b);
        printf("|");
      }
      printf("\n");
    } else {
      int n = (compareMode == 0) ? m : m+1; // which data to compare: m = data from file, m+1 = data chosen by model
      float s = times[n]/data[n];
      if (s < 0.8) printf("%c[0;31m#", 0x1b);
      else if (s < .95) printf("%c[0;33mX", 0x1b);
      else if (s > 1.1) printf("%c[0;34mO", 0x1b);
      else printf("%c[0;32mO", 0x1b);
      score += s;
      totalScore += s;
      npoints++;
      totalNpoints++;     
    }
  }
  if (!compactMode) {
    printf("----------+"); for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("---------------------+"); printf("\n");
  } else {
    printf("%c[00m| %.1f %%\n", 0x1b, 100.0*score/npoints);
  }
}

#define COMPACT_SEPARATOR printf("-----------------+------------------------------+-------\n")

void runPlatform(const char* platform, int ngpus, int nnodes, ncclFunc_t coll) {
  char xmlTopoFile[1024];
  sprintf(xmlTopoFile, "topo/%s/system.xml", platform);
  if (nnodes == 0) {
    for (int n=1; n<128; n<<=1) {
      runTopo(xmlTopoFile, platform, ngpus, n, coll);
    }
  } else {
    runTopo(xmlTopoFile, platform, ngpus, nnodes, coll);
  }
  if (compactMode) COMPACT_SEPARATOR;
}

#define RUN(...) runPlatform(__VA_ARGS__)

int main(int argc, const char* argv[]) {
  char* str = getenv("COMPARE_MODE");
  compareMode = str ? atoi(str) : 0;

  setlinebuf(stdout);
  if (argc > 4) {
    int coll = -1;
    for (int i=0; i<NCCL_NUM_FUNCTIONS; i++) {
      if (strcmp(argv[4], ncclFuncStr[i]) == 0) coll = i;
    }
    if (coll == -1) {
      printf("Error : unknown collective %s\n", argv[4]);
      return 1;
    }
    RUN(argv[1], atoi(argv[2]), atoi(argv[3]), (ncclFunc_t)coll);
  } else if (argc > 1) {
    printf("Usage : %s <platform> <ngpus> <nnodes> <collective>\n", argv[0]);
    printf("Set COMPARE_MODE to select which data to use as the Default: 0 - from file, 1 - dry run\n");
    return 1;
  } else {
    compactMode = 1;
    printf("%10s/%5s |    Delta at size 8 to 4G     | Score\n", "Platform", "Nodes");
    printf("-----------------+------------------------------+-------\n");
    RUN("DGX-1V", 8, 0, ncclCollAllReduce);
    RUN("DGX-2V", 16, 0, ncclCollAllReduce);
//  RUN("Luna", 8, 0, ncclCollallReduce);
    printf("           Total |                              | %.1f %%\n", 100.0*totalScore/totalNpoints);
  }
  return 0;
}
