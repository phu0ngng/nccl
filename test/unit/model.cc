#include "topo.h"
#include "comm.h"
#include "collectives.h"
#include "core.h"
#include "xml.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <getopt.h>

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

int nGpus = -1;
int nNodes = -1;
char* platform = NULL;
const char* platforms[] = { "DGX-1V", "DGX-2V", "Luna" };
ncclFunc_t function = ncclFuncAllReduce;

int compactMode = -1;
int compareMode = 0;

int strConvert(const char* option, const char* dict[], int nvalues, const char* str) {
  for (int i=0; i<nvalues; i++) {
    if (strcasecmp(dict[i], str) == 0) {
      return i;
    }
  }
  printf("Ignoring invalid value '%s' for option '%s'. Possible values are :\n", str, option);
  for (int i=0; i<nvalues; i++) {
    printf(" %s,", dict[i]);
  }
  printf("\n");
  return -1;
}

float totalScore = 0.0;
int totalNpoints = 0;
int totalX = 0;
int totalHash = 0;


void runTopo(const char* xmlTopoFile, const char* platform, int nnodes) {
  int ngpus = nGpus;
  struct ncclXml* xmlSystem;
  INFO(NCCL_GRAPH, "Loading platform %s", platform);
  CHECK(ncclCalloc(&xmlSystem, 1));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystem));
  struct ncclTopoSystem* system;
  if (xmlSystem->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    free(xmlSystem);
    return;
  }
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system));
  free(xmlSystem);

  CHECK(ncclTopoComputePaths(system, NULL));
  if (nnodes == 1) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      CHECK(ncclTopoRemoveNode(system, NET, n));
  }

  if (ngpus == -1 ) {
    ngpus = system->nodes[GPU].count;
  } else {
    // Only keep ngpus
    for (int g=system->nodes[GPU].count-1; g>=nGpus; g--) {
      CHECK(ncclTopoRemoveNode(system, GPU, g));
    }
  }

  // Last column is used for min/best/default.
  const int m = NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS;

  int fds[m+1];
  char path[1024];
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int i = a*NCCL_NUM_PROTOCOLS+p;
    sprintf(path, "topo/%s/data/%d/%d/%s/%s/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[function], ncclAlgoStr[a], ncclProtoStr[p]);
    fds[i] = open(path, O_RDONLY);
  }
  sprintf(path, "topo/%s/data/%d/%d/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[function]);
  fds[m] = open(path, O_RDONLY);
  float score = 0.0;
  int npoints = 0;
  if (compactMode) {
    int nfds = 0;
    for (int i=0; i<=m; i++) if (fds[i] != -1) nfds++;
    if (nfds == 0 || ngpus*nnodes == 1) {
      ncclTopoFree(system);
      return;
    }
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
  info.coll = function;
  info.chunkSteps = ALLREDUCE_CHUNKSTEPS;
  info.sliceSteps = ALLREDUCE_SLICESTEPS;

  if (!compactMode) {
    printf("%s/%dx%d, %s\n", platform, nnodes, ngpus, ncclFuncStr[function]);
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
    printf("%10s/%5dx%5d |", platform, nnodes, ngpus);
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
            close(fds[i]);
            fds[i] = -1;
          } else if (valueStr[o] == '\n') {
            valueStr[o] = '\0';
	    data[i] = atof(valueStr);
            if (data[i] == 0.0) data[i] = -1.0;
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
      if (data[n] == 0.0) {
        printf("%c[00m.", 0x1b);
      } else {
        float s = times[n]/data[n];
        if (s < 0.8) { printf("%c[0;31m#", 0x1b); totalHash++; }
        else if (s < .95) { printf("%c[0;33mX", 0x1b); totalX++; }
        else if (s > 1.1) printf("%c[0;34mO", 0x1b);
        else printf("%c[0;32mO", 0x1b);
        score += s;
        totalScore += s;
        npoints++;
        totalNpoints++;
      }
    }
  }
  if (!compactMode) {
    printf("----------+"); for (int i=0; i<NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS+1; i++) printf("---------------------+"); printf("\n");
  } else {
    printf("%c[00m| %.1f %%\n", 0x1b, 100.0*score/npoints);
  }
  ncclTopoFree(system);
}

#define COMPACT_SEPARATOR printf("-----------------------+------------------------------+--------\n")

void runPlatform(const char* platform) {
  char xmlTopoFile[1024];
  sprintf(xmlTopoFile, "topo/%s/system.xml", platform);
  if (nNodes == -1) {
    for (int n=1; n<128; n<<=1) {
      runTopo(xmlTopoFile, platform, n);
    }
  } else {
    runTopo(xmlTopoFile, platform, nNodes);
  }
  if (compactMode) COMPACT_SEPARATOR;
}

int main(int argc, char* argv[]) {
  setlinebuf(stdout);

  // Parse args
  int longindex;
  static struct option longopts[] = {
    {"nnodes", required_argument, 0, 'n'},
    {"ngpus", required_argument, 0, 'g'},
    {"platform", required_argument, 0, 'p'},
    {"function", required_argument, 0, 'f'},
    {"compact", required_argument, 0, 'c'},
    {"compare", required_argument, 0, 'C'},
    {"help", no_argument, 0, 'h'}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "n:g:p:f:c:C:h:", longopts, &longindex);

    if (c == -1)
      break;

    switch(c) {
      case 'n':
        nNodes = strtol(optarg, NULL, 0);
        break;
      case 'g':
        nGpus = strtol(optarg, NULL, 0);
        break;
      case 'p':
        platform = optarg;
        break;
      case 'f':
        function = (ncclFunc_t)strConvert("function", ncclFuncStr, NCCL_NUM_FUNCTIONS, optarg);
        break;
      case 'c':
        compactMode = strtol(optarg, NULL, 0);
        break;
      case 'C':
        compareMode = strtol(optarg, NULL, 0);
        break;
      case 'h':
      default:
        if (c != 'h') printf("invalid option '%c'\n", c);
        printf("USAGE: %s \n\t"
            "[-n,--nnodes <number of nodes (default : 1 to 128)>] \n\t"
            "[-g,--ngpus <gpus per node (default : All)>] \n\t"
            "[-p,--platform <platform (default : All)>] \n\t"
            "[-f,--function <function (default : AllReduce)>] \n\t"
            "[-c,--compact <compact mode : 0/1>\n\t"
            "[-C,--compare <compare mode : 0/1>\n\t"
	    "[-h,--help]\n",
            basename(argv[0]));
        return 0;
    }
  }

  if (compactMode == -1) compactMode = nGpus == -1 || nNodes == -1 || platform == NULL || function == -1 ? 1 : 0;

  if (compactMode) {
    COMPACT_SEPARATOR;
    printf("  Function             |    %15s           |\n", ncclFuncStr[function]);
    COMPACT_SEPARATOR;
    printf("%10s/%5sx%5s |    Delta at size 8 to 4G     | Score\n", "Platform", "Nodes", "Ngpus");
    COMPACT_SEPARATOR;
  }

  if (platform) {
    runPlatform(platform);
  } else {
    for (int p=0; p<sizeof(platforms)/sizeof(platform); p++) {
      runPlatform(platforms[p]);
    }
  }

  if (compactMode) {
    printf("                 Total |         %3d X, %3d #         | %.1f %%\n", totalX, totalHash, 100.0*totalScore/totalNpoints);
  }
  return 0;
}
