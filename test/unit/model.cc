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

int nGpus = -1;
int nNodes = -1;
char* platform = NULL;
const char* platforms[] = { "DGX-1V", "DGX-2V", "Luna", "Viking", "Umbriel" };
ncclFunc_t function = ncclFuncAllReduce;

int compactMode = -1;
int dispMode = 0;

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

enum colorNames            {     RED = 0, YELLOW = 1,   BLUE = 2,  GREEN = 3,  RESET = 4 };
const char* colorCodes[] = {    "[0;31m",   "[0;33m",   "[0;34m",   "[0;32m",     "[00m" };
const char  markers[]    = {         '#',        'X',        'O',        'O' };

int stats[RESET] = { 0, 0, 0, 0 };

#define GET_COLOR(s) (s < .80) ? RED : (s < .95) ? YELLOW : (s > 1.1) ? BLUE : GREEN;

#define PRINT_MODE(str, val) do { \
  float v = val; \
  if (dispMode > 0  && v != -1.0) v = size / v; \
  if (dispMode == 2 && v != -1.0) { \
    float nranks = nnodes*ngpus; \
    if (function == ncclFuncAllReduce) v *= 2*(nranks-1)/nranks; \
    if (function == ncclFuncReduceScatter) v *= (nranks-1)/nranks; \
    if (function == ncclFuncAllGather) v *= (nranks-1)/nranks; \
  } \
  printf(str, v); \
}while(0);

int algoProtoSupported(int a, int p, struct ncclTopoGraph** graphs) {
  if (graphs[a]->nChannels == 0) return 0;
  if (a >= NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) return 0;
  return 1;
}

void keepGpus(struct ncclXml* xmlSystem) {
  struct ncclXmlNode* node;
  CHECK(xmlFindTag(xmlSystem, "gpu", &node));
  while (node) {
    CHECK(xmlSetAttrInt(node, "keep", 1));
    CHECK(xmlFindNextTag(xmlSystem, "gpu", node, &node));
  }
}

#define MAX_MNNVL_NODES 64

void runTopo(const char* xmlTopoFile, const char* platform, int nnodes) {
  int ngpus = nGpus;
  struct ncclXml* xmlSystem;
  INFO(NCCL_GRAPH, "Loading platform %s", platform);
  CHECK(xmlAlloc(&xmlSystem, MAX_MNNVL_NODES*NCCL_TOPO_XML_MAX_NODES));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystem, 1));
  struct ncclTopoSystem* system;
  if (xmlSystem->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    free(xmlSystem);
    return;
  }
  keepGpus(xmlSystem);
  CHECK(ncclTopoTrimXml(xmlSystem));
  uint64_t hostHash = 0;
  {
    // Get the host_hash of the first CPU.
    struct ncclXmlNode* cpuNode;
    CHECK(xmlFindTag(xmlSystem, "cpu", &cpuNode));
    if (cpuNode) {
      const char* hostHashStr;
      CHECK(xmlGetAttr(cpuNode, "host_hash", &hostHashStr));
      if (hostHashStr)
        hostHash = strtoull(hostHashStr, NULL, 16);
    }
  }
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system, hostHash));
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
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.crossNic = 2;
  treeGraph.collNet = 0;

  struct ncclTopoGraph cNetGraph;
  memset(&cNetGraph, 0, sizeof(cNetGraph));
  cNetGraph.id = 2;
  cNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  cNetGraph.crossNic = 2;
  cNetGraph.collNet = 1;

  struct ncclTopoGraph nvlsGraph;
  memset(&nvlsGraph, 0, sizeof(nvlsGraph));
  nvlsGraph.id = 3;
  nvlsGraph.pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph.crossNic = 2;
  nvlsGraph.collNet = 0;
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;

  /* Compute */
  CHECK(ncclTopoCompute(system, &ringGraph));
  CHECK(ncclTopoPrintGraph(system, &ringGraph));
  treeGraph.minChannels = ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  CHECK(ncclTopoCompute(system, &treeGraph));
  CHECK(ncclTopoPrintGraph(system, &treeGraph));
  if (nnodes > 1) {
    cNetGraph.minChannels = 1;
    cNetGraph.maxChannels = ringGraph.nChannels;
    CHECK(ncclTopoCompute(system, &cNetGraph));
    CHECK(ncclTopoPrintGraph(system, &cNetGraph));
  }
  CHECK(ncclTopoCompute(system, &nvlsGraph));
  CHECK(ncclTopoPrintGraph(system, &nvlsGraph));

  treeGraph.nChannels = ringGraph.nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);

  int collNetSupport = (cNetGraph.nChannels == 0) ? 0 : 1;

  // Last column is used for min/best/default.
  const int M = NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS;

  int fds[M+1];
  char path[1024];
  for (int i=0; i<M+1; i++) fds[i] = -1;
  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    int i = a*NCCL_NUM_PROTOCOLS+p;
    sprintf(path, "topo/%s/data/%d/%d/%s/%s/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[function], ncclAlgoStr[a], ncclProtoStr[p]);
    fds[i] = open(path, O_RDONLY);
  }
  sprintf(path, "topo/%s/data/%d/%d/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[function]);
  fds[M] = open(path, O_RDONLY);
  float score = 0.0;
  int npoints = 0;
  if (compactMode) {
    int nfds = 0;
    for (int i=0; i<=M; i++) if (fds[i] != -1) nfds++;
    if (nfds == 0 || ngpus*nnodes == 1) {
      ncclTopoFree(system);
      return;
    }
  }

  struct ncclComm comm;
  comm.topo = system;
  comm.nNodes = nnodes;
  comm.nRanks = system->nodes[GPU].count*nnodes;
  comm.nChannels = ringGraph.nChannels*2;
  comm.channels[0].tree.depth = system->nodes[GPU].count-1+log2i(nnodes);
  comm.buffSizes[NCCL_PROTO_SIMPLE] = 1 << 22;
  comm.config.collnetEnable = collNetSupport;
  int compCap = system->nodes[GPU].nodes[0].gpu.cudaCompCap;
  comm.minCompCap = compCap;
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { &treeGraph, &ringGraph, &cNetGraph, &cNetGraph, &nvlsGraph, &nvlsGraph, &treeGraph };
  CHECK(ncclTopoInitTunerConstants(&comm));
  CHECK(ncclTopoTuneModel(&comm, compCap, compCap, graphs));

  if (!compactMode) {
    printf("%s/%dx%d, %s\n", platform, nnodes, ngpus, ncclFuncStr[function]);
    printf("-----------+");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      printf("---------------------+");
    }
    printf("-------------------------------+"); printf("\n");
    printf("     Size  |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      if (strlen(ncclAlgoStr[a]) <= 12)
        printf(" %12s/%-6s |", ncclAlgoStr[a], ncclProtoStr[p]);
      else
        printf(" %.12s/%-6s |", ncclAlgoStr[a], ncclProtoStr[p]);
    }
    printf("%19s            |\n", "Default");
    printf("           |");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      int i = a*NCCL_NUM_PROTOCOLS+p;
      printf("%9s  %9s |", "data", (i == M) ? "best" : "model");
    }
    printf("%9s  %9s %9s |", "best", "dryrun", "data");
    printf("\n");
    printf("-----------+");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      printf("---------------------+");
    }
    printf("-------------------------------+"); printf("\n");
  } else {
    printf("%10s/%5dx%5d |", platform, nnodes, ngpus);
  }

  for (ssize_t size=8; size<(2LL<<32); size<<=1) {
    float model[M];
    float data[M+1];
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      int i = a*NCCL_NUM_PROTOCOLS+p;
      CHECK(ncclTopoGetAlgoTime(&comm, function, a, p, size, 1, model+i));
    }

    for (int i=0; i<M+1; i++) {
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
    float dryrun = -1.0;
    float bestdata = -1.0;
    float bestmodel = -1.0;
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      int i = a*NCCL_NUM_PROTOCOLS+p;
      // Find best data
      if (data[i] > 0 && (bestdata < 0 || data[i] < bestdata)) bestdata = data[i];

      // Find best model and compute dryrun
      if (model[i] > 0 && (bestmodel < 0 || model[i] < bestmodel)) {
        bestmodel = model[i];
        dryrun = data[i] > 0 ? data[i] : model[i];
      }
    }

    // Display each protocol/algorithm
    if (!compactMode) {
      printf("%10ld |", size);
      for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
        if (algoProtoSupported(a, p, graphs) == 0) continue;
        int i = a*NCCL_NUM_PROTOCOLS+p;
        float ref = data[i];
        float value = model[i];
        int boldmodel = value == bestmodel ? 7 : 0;
        int boldref = ref == bestdata ? 7 : 0;
        if (ref == -1.0) {
          printf("%10s ", "");
          if (boldmodel) printf("%c[%d;32m", 0x1b, boldmodel);
        } else {
          if (boldref) printf("%c[%d;37m", 0x1b, boldref);
          PRINT_MODE("%9.1f  ", ref);
          if (boldref) printf("%c[00m", 0x1b);
          float s = 1-ref/value;
          s *= s;
          if (s > .15) printf("%c[%d;31m", 0x1b, boldmodel);
          else if (s > .08) printf("%c[%d;33m", 0x1b, boldmodel);
          else printf("%c[%d;32m", 0x1b, boldmodel);
        }
        PRINT_MODE("%9.1f", value);
        if ((ref != -1.0) || boldmodel) printf("%c[00m", 0x1b);
        printf(" |");
      }
    }

    // Last column : best model, best data, dryrun and how well we do.
    if (bestdata == -1.0) {
      if (compactMode) {
        printf("%c[00m.", 0x1b);
      } else {
        printf("%10s %9s", "", "");
        PRINT_MODE(" %9.1f |\n", bestmodel);
      }
    } else {
      float s = bestdata/dryrun;
      int c = GET_COLOR(s);
      if (compactMode) {
        printf("%c%s%c", 0x1b, colorCodes[c], markers[c]);
        stats[c]++;
        score += s;
        totalScore += s;
        npoints++;
        totalNpoints++;
      } else {
        PRINT_MODE("%9.1f ", bestdata);
        printf(" %c%s", 0x1b, colorCodes[c]);
        PRINT_MODE("%9.1f", dryrun);
        s = bestdata/data[M];
        c = GET_COLOR(s);
        printf(" %c%s", 0x1b, colorCodes[c]);
        PRINT_MODE("%9.1f", data[M]);
        printf("%c%s |\n", 0x1b, colorCodes[RESET]);
      }
    }
  }
  if (!compactMode) {
    printf("-----------+");
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (algoProtoSupported(a, p, graphs) == 0) continue;
      printf("---------------------+");
    }
    printf("-------------------------------+"); printf("\n");
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
    for (int n=1; n<=128; n<<=1) {
      runTopo(xmlTopoFile, platform, n);
    }
  } else {
    runTopo(xmlTopoFile, platform, nNodes);
  }
  if (compactMode) COMPACT_SEPARATOR;
}

int main(int argc, char* argv[]) {
  setlinebuf(stdout);
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 1);

  // Parse args
  int longindex;
  static struct option longopts[] = {
    {"nnodes", required_argument, 0, 'n'},
    {"ngpus", required_argument, 0, 'g'},
    {"platform", required_argument, 0, 'p'},
    {"function", required_argument, 0, 'f'},
    {"compact", required_argument, 0, 'c'},
    {"mode", required_argument, 0, 'm'},
    {"help", no_argument, 0, 'h'}
  };

  while(1) {
    int c;
    c = getopt_long(argc, argv, "n:g:p:f:c:m:h", longopts, &longindex);

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
      case 'm':
        dispMode = strtol(optarg, NULL, 0);
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
            "[-m,--mode <0:time, 1:algbw, 2:busbw>\n\t"
	    "[-h,--help]\n",
            basename(argv[0]));
        return 0;
    }
  }

  if (compactMode == -1) compactMode = nGpus == -1 || nNodes == -1 || platform == NULL || function == (ncclFunc_t)-1 ? 1 : 0;

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
    printf("                 Total |         %3d %c, %3d %c         | %.1f %%\n", stats[YELLOW], markers[YELLOW], stats[RED], markers[RED], 100.0*totalScore/totalNpoints);
  }
  return 0;
}
