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
struct testParam{
  int ngpus;
  int nnodes;
  char* platform;
  ncclFunc_t function;
  int compactMode;
  int dispMode;
  // NIC fusion
  int mergeLevel;
  const char* forceMerge;
};

#define TESTPARAM_INIT {\
  /*ngpus=*/-1,\
  /*nnodes=*/-1,\
  /*platform=*/NULL,\
  /*function=*/ncclFuncAllReduce,\
  /*compactMode=*/-1,\
  /*dispMode=*/0,\
  /*mergeLevel=*/PATH_LOC,\
  /*forceMerge=*/NULL\
}

struct testParam param;
int coll = 0;
const char* platforms[] = { "DGX-1V", "DGX-2V", "Luna", "Viking", "Umbriel" };

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
  if (param.dispMode > 0  && v != -1.0) v = size / v; \
  if (param.dispMode == 2 && v != -1.0) { \
    float nranks = param.nnodes*param.ngpus; \
    if (param.function == ncclFuncAllReduce) v *= 2*(nranks-1)/nranks; \
    if (param.function == ncclFuncReduceScatter) v *= (nranks-1)/nranks; \
    if (param.function == ncclFuncAllGather) v *= (nranks-1)/nranks; \
  } \
  printf(str, v); \
}while(0);

int algoProtoSupported(int a, int p, struct ncclTopoGraph** graphs) {
  if (graphs[a]->nChannels == 0) return 0;
  if (a >= NCCL_ALGO_COLLNET_DIRECT && p != NCCL_PROTO_SIMPLE) return 0;
  return 1;
}


#define MAX_MNNVL_NODES 64

#define NCCL_PLUGIN_MAX_RECVS 8
int max_requests = NCCL_NET_MAX_REQUESTS;
int nPhysDevs = 0; // number of physical devices only
int nVirtualDevs = 0; // number of virtual devices only
#define MAX_MOCK_DEVS 1024
#define MAX_MOCK_VDEVS MAX_MOCK_DEVS*8
#define MOCK_VDEV_NAME_LENGTH 256
struct mockVDev {
  char name[MOCK_VDEV_NAME_LENGTH];
  char pciPath[MOCK_VDEV_NAME_LENGTH];
  int speed;
  float latency;
  int ignore;
  int gdr;
  int maxConns;
  ncclNetVDeviceProps_t vProps;
  int used;
};
mockVDev mockVDevs[MAX_MOCK_VDEVS];
ncclNetProperties_t mockProps[MAX_MOCK_VDEVS];
void fakeNetPluginAddNetNode(struct ncclXmlNode* node) {
  int devIndex;
  CHECK(xmlGetAttrInt(node, "dev", &devIndex));

  // if devIndex is already in use, find the next suitable location
  mockVDev* dev = NULL;
  int count = -1;
  do {
    dev = mockVDevs + devIndex + (++count);
  } while (dev->used);
  devIndex += count;
  // if duplicates, overwrite the new devIndex and generate a unique name
  char nameSuffix[MAX_STR_LEN] = "";
  if (count > 0) {
    snprintf(nameSuffix, sizeof(nameSuffix), "-%d", count);
    CHECK(xmlSetAttrInt(node, "dev", devIndex));
  }
  dev->used = 1;
  dev->ignore = 0;

  dev->vProps.devs[0] = devIndex;
  ncclNetProperties_t* props = mockProps + devIndex;

  // get the original name and overwrite it if needed
  const char *name, *attr;
  CHECK(xmlGetAttrStr(node, "name", &name));
  char uniqueName[MAX_STR_LEN];
  snprintf(uniqueName, sizeof(uniqueName), "%s%s", name, nameSuffix);
  snprintf(dev->name, sizeof(dev->name), "%s", uniqueName);
  CHECK(xmlSetAttr(node, "name", uniqueName));
  props->name = dev->name;

  // get speed, port, guid, and optional latency
  CHECK(xmlGetAttrUint64Default(node, "guid", &props->guid, /*default=*/devIndex));
  CHECK(xmlGetAttrInt(node, "port", &props->port));
  CHECK(xmlGetAttrInt(node, "speed", &props->speed));
  dev->speed = props->speed;

  // missing the following argument is not an error.
  xmlGetAttr(node, "latency", &attr);
  if (attr) dev->latency = props->latency = strtof(attr, NULL);
  xmlGetAttr(node, "coll", &attr);
  if (attr) coll = strtol(attr, NULL, 0);
  xmlGetAttr(node, "gdr", &attr);
  if (attr) dev->gdr = strtol(attr, NULL, 0);
  xmlGetAttr(node, "maxconn", &attr);
  if (attr) dev->maxConns = strtol(attr, NULL, 0);
  int railId, planeId;
  CHECK(xmlGetAttrIntDefault(node, "rail", &railId, NCCL_NET_ID_UNDEF));
  CHECK(xmlGetAttrIntDefault(node, "plane", &planeId, NCCL_NET_ID_UNDEF));
  props->railId = railId;
  props->planeId = planeId;
  dev->ignore = 0;

  // get the busId of the first PCI parent
  const char* busId = NULL;
  struct ncclXmlNode* parent = node;
  while (busId == NULL && parent != NULL) {
    CHECK(xmlGetAttr(parent, "busid", &busId));
    parent = parent->parent;
  }
  if (busId) {
    // getPciPath will fail to resolve the path, so hardcode the imaginary path.
    // Getting the path right is not important, we need to have different path to avoid NIC fusion.
    snprintf(dev->pciPath, sizeof(dev->pciPath), "/sys/class/pci_bus/%.*s/%.*s", (int)strlen("0000:00"), busId, (int)strlen("0000:00:00.0"), busId);
    props->pciPath = dev->pciPath;
  }

  // It's assumed system.xml files won't have duplicate "dev" fields
  if (nPhysDevs < (devIndex + 1)) nPhysDevs = devIndex + 1;
}



void getTestParam(struct testParam* param) {
  // default parameters
  *param = TESTPARAM_INIT;

  const char* str = getenv("NCCL_MODEL_TEST_NGPUS");
  if (str) param->ngpus = atoi(str);

  str = getenv("NCCL_MODEL_TEST_NNodes");
  if (str) param->nnodes = atoi(str);

  str = getenv("NCCL_MODEL_TEST_Platform");
  if (str) param->platform = strdup(str);

  str = getenv("NCCL_MODEL_TEST_Function");
  if (str) param->function = (ncclFunc_t)strConvert("function", ncclFuncStr, NCCL_NUM_FUNCTIONS, str);

  str = getenv("NCCL_MODEL_TEST_CompactMode");
  if (str) param->compactMode = atoi(str);

  str = getenv("NCCL_MODEL_TEST_DispMode");
  if (str) param->dispMode = atoi(str);

  CHECK(ncclTopoGetFusionEnv(&param->mergeLevel, &param->forceMerge));
}

void fakeNetPluginInit(struct ncclXml* xmlSystem) {
  nPhysDevs = 0;
  nVirtualDevs = NCCL_UNDEF_DEV_COUNT;
  coll = 0;
  memset(mockVDevs, 0, sizeof(mockVDev)*MAX_MOCK_DEVS);
  memset(mockProps, 0, sizeof(ncclNetProperties_t)*MAX_MOCK_DEVS);

  struct ncclXmlNode* node;
  CHECK(xmlFindTag(xmlSystem, "net", &node));
  while (node) {
    // will increment the number of nPhysDevs
    fakeNetPluginAddNetNode(node);
    CHECK(xmlFindNextTag(xmlSystem, "net", node, &node));
  }

  // Add in fake devs if there are any skipped in system.xml
  for (int i = 0; i < nPhysDevs; i++) {
    mockVDev* dev = mockVDevs + i;
    ncclNetProperties_t* props = mockProps + i;
    if (dev->speed == 0) {
      snprintf(dev->name, sizeof(dev->name), "ignore_%d", i);
      dev->speed = 1;
      dev->ignore = 1;
      props->name = dev->name;
      props->speed = 1;
      props->railId = NCCL_NET_ID_UNDEF;
      props->planeId = NCCL_NET_ID_UNDEF;
    }
  }
}

ncclResult_t fakeNetPluginGetProperties(int dev, ncclNetProperties_t* props) {
  // if nVirtualDevs is NCCL_UNDEF_DEV_COUNT, NIC fusion hasn't happened yet > no virtual devices
  int totalDevs = nPhysDevs + (nVirtualDevs != NCCL_UNDEF_DEV_COUNT ? nVirtualDevs : 0);
  if (dev < totalDevs) {
    mockVDev* vDev = mockVDevs + dev;
    int pDevIndex = vDev->vProps.devs[0];
    memcpy(props, mockProps + pDevIndex, sizeof(ncclNetProperties_t));
    props->vProps = vDev->vProps;
    props->speed = vDev->speed;
    props->name  = vDev->name;
    props->latency  = vDev->latency;
    props->ptrSupport = NCCL_PTR_HOST;
    props->maxComms = vDev->maxConns;
    if (vDev->gdr) {
      props->ptrSupport |= NCCL_PTR_CUDA;
    }
    return ncclSuccess;
  } else {
    printf("[MOCK] trying to get the properties of dev %d, but only %d are known", dev, totalDevs);
    return ncclInvalidUsage;
  }
}

ncclResult_t fakeNetPluginDevices(int* ndev) {
  *ndev = nPhysDevs + (nVirtualDevs != NCCL_UNDEF_DEV_COUNT ? nVirtualDevs : 0);
  return ncclSuccess;
}

ncclResult_t fakeNetPluginMakeVDevice(int* d, ncclNetVDeviceProps_t* vProps) {
  if (nVirtualDevs == NCCL_UNDEF_DEV_COUNT) nVirtualDevs = 0;
  int totalDevs = nPhysDevs + nVirtualDevs;
  if (totalDevs < MAX_MOCK_VDEVS) {
    if (vProps->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) return ncclInvalidArgument;
    for (int i = 0; i < vProps->ndevs; i++) {
      int pDev = vProps->devs[i];
      if (mockVDevs[pDev].ignore) return ncclInvalidArgument;
    }
    int deviceIndex = totalDevs;
    mockVDev* mDev = mockVDevs + deviceIndex;
    memset(mDev, 0, sizeof(mockVDev));
    memcpy(&mDev->vProps, vProps, sizeof(ncclNetVDeviceProps_t));
    mDev->gdr = 1;
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int pDev = mDev->vProps.devs[i];
      mDev->speed    += mockProps[pDev].speed;
      mDev->latency  += mockProps[pDev].latency;
      mDev->gdr      &= mockVDevs[pDev].gdr;
      mDev->maxConns += mockVDevs[pDev].maxConns;
      if (i > 0) {
        snprintf(mDev->name + strlen(mDev->name), sizeof(mDev->name) - strlen(mDev->name), "+%s", mockProps[pDev].name);
      } else {
        strncpy(mDev->name, mockProps[pDev].name, 128);
      }
    }

    INFO(NCCL_GRAPH, "Fake/Plugin : Made vDevice %s speed=%d", mDev->name, mDev->speed);

    *d = totalDevs;
    nVirtualDevs++;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

static ncclResult_t fakeNetPluginGetDevCount(int pluginIndex, int* nPhys, int* nVirt){
  // if no new virtual device has been created, we return undefined to trigger the usage of makeVDevices
  *nPhys = nPhysDevs;
  *nVirt = nVirtualDevs;
  return ncclSuccess;
}
static ncclResult_t fakeNetPluginSetDevCount(int pluginIndex, int nVirt) {
  // if no new virtual device has been created, we return undefined to trigger the usage of makeVDevices
  if (nVirt != nVirtualDevs && nVirtualDevs >= 0) {
    printf("NCCL core is trying to set an inconsistent number of physical devices. I have tracked %d virtual devs in the plugin, NCCL core thinks it's %d.\n", nVirtualDevs, nVirt);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

void keepGpus(struct ncclXml* xmlSystem) {
  struct ncclXmlNode* node;
  CHECK(xmlFindTag(xmlSystem, "gpu", &node));
  while (node) {
    CHECK(xmlSetAttrInt(node, "keep", 1));
    CHECK(xmlFindNextTag(xmlSystem, "gpu", node, &node));
  }
}



void runTopo(const char* xmlTopoFile, const char* platform, int nnodes) {


  int ngpus = param.ngpus;
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
  // Inititalize a netState object for NIC fusion
  fakeNetPluginInit(xmlSystem);
  struct ncclTopoNetInfo netInfo{};
  netInfo.net = 1;
  netInfo.coll = coll > 0;
  netInfo.netPluginIndex = 0;
  netInfo.maxDevsPerNic = NCCL_NET_MAX_DEVS_PER_NIC;
  netInfo.dmaBufSupport = true;
  netInfo.mergeLevel = param.mergeLevel;
  netInfo.forceMerge = param.forceMerge;
  netInfo.getDevCount = fakeNetPluginGetDevCount;
  netInfo.setVirtDevCount = fakeNetPluginSetDevCount;
  netInfo.name = "Fake";
  netInfo.getProperties = fakeNetPluginGetProperties;
  netInfo.makeVDevice = fakeNetPluginMakeVDevice;
  netInfo.devices = fakeNetPluginDevices;
  CHECK(ncclTopoProcessNet(xmlSystem, NULL, &netInfo));
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
  system->inter = nnodes == 1 ? 0 : 1;

  if (ngpus == -1 ) {
    ngpus = system->nodes[GPU].count;
  } else {
    // Only keep ngpus
    for (int g=system->nodes[GPU].count-1; g>=ngpus; g--) {
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
    sprintf(path, "topo/%s/data/%d/%d/%s/%s/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[param.function], ncclAlgoStr[a], ncclProtoStr[p]);
    fds[i] = open(path, O_RDONLY);
  }
  sprintf(path, "topo/%s/data/%d/%d/%s/time.txt", platform, ngpus, nnodes, ncclFuncStr[param.function]);
  fds[M] = open(path, O_RDONLY);
  float score = 0.0;
  int npoints = 0;
  if (param.compactMode) {
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

  if (!param.compactMode) {
    printf("%s/%dx%d, %s\n", platform, nnodes, ngpus, ncclFuncStr[param.function]);
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
      CHECK(ncclTopoGetAlgoTime(&comm, param.function, a, p, size, 1, model+i));
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
    if (!param.compactMode) {
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
      if (param.compactMode) {
        printf("%c[00m.", 0x1b);
      } else {
        printf("%10s %9s", "", "");
        PRINT_MODE(" %9.1f |\n", bestmodel);
      }
    } else {
      float s = bestdata/dryrun;
      int c = GET_COLOR(s);
      if (param.compactMode) {
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
  if (!param.compactMode) {
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
  if (param.nnodes == -1) {
    for (int n=1; n<=128; n<<=1) {
      runTopo(xmlTopoFile, platform, n);
    }
  } else {
    runTopo(xmlTopoFile, platform, param.nnodes);
  }
  if (param.compactMode) COMPACT_SEPARATOR;
}

int main(int argc, char* argv[]) {
  setlinebuf(stdout);
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 1);
  getTestParam(&param);

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
        param.nnodes = strtol(optarg, NULL, 0);
        break;
      case 'g':
        param.ngpus = strtol(optarg, NULL, 0);
        break;
      case 'p':
        param.platform = optarg;
        break;
      case 'f':
        param.function = (ncclFunc_t)strConvert("function", ncclFuncStr, NCCL_NUM_FUNCTIONS, optarg);
        break;
      case 'c':
        param.compactMode = strtol(optarg, NULL, 0);
        break;
      case 'm':
        param.dispMode = strtol(optarg, NULL, 0);
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

  if (param.compactMode == -1) param.compactMode = param.ngpus == -1 || param.nnodes == -1 || param.platform == NULL || param.function == (ncclFunc_t)-1 ? 1 : 0;

  if (param.compactMode) {
    COMPACT_SEPARATOR;
    printf("  Function             |    %15s           |\n", ncclFuncStr[param.function]);
    COMPACT_SEPARATOR;
    printf("%10s/%5sx%5s |    Delta at size 8 to 4G     | Score\n", "Platform", "Nodes", "Ngpus");
    COMPACT_SEPARATOR;
  }

  if (param.platform) {
    runPlatform(param.platform);
  } else {
    for (int p=0; p<sizeof(platforms)/sizeof(platforms[0]); p++) {
      runPlatform(platforms[p]);
    }
  }

  if (param.compactMode) {
    printf("                 Total |         %3d %c, %3d %c         | %.1f %%\n", stats[YELLOW], markers[YELLOW], stats[RED], markers[RED], 100.0*totalScore/totalNpoints);
  }
  return 0;
}
