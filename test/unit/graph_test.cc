#include "topo.h"
#include "xml.h"
#include "nccl_net.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define ERROR(fmt, ...) do { \
  printf("%s:%d " fmt "\n", __FILE__, __LINE__, __VA_ARGS__); \
  exit(1); \
} while (0)

#define CHECK(call) do { \
  ncclResult_t res = call; \
  if (res != ncclSuccess) { \
    ERROR("NCCL Check failed : %d", res); \
    exit(1); \
  } \
} while (0);

#include <sys/time.h>
uint64_t getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1000000+tv.tv_usec;
}

const char* graphNames[] = { "Ring", "Tree", "CollNet", "NVLS" };

int coll = 0;

enum testType{
  TEST_INTRA,
  TEST_INTER
};

void compareGraphs(struct ncclTopoGraph* ref, struct ncclTopoGraph* out, int ngpus, enum testType type, bool dumpDiff, int* errors, int* warnings) {
  if (ref->nChannels == 0 && out->nChannels == 0) return;
  if (memcmp(ref, out, sizeof(struct ncclTopoGraph)) != 0) {
    if (ref->nChannels*ref->bwInter > out->nChannels*out->bwInter ||
        ref->nChannels*ref->bwIntra > out->nChannels*out->bwIntra ||
        ref->crossNic < out->crossNic ||
        ref->typeIntra < out->typeIntra ||
        ref->typeInter < out->typeInter) (*errors)++;
    else (*warnings)++;

    if (dumpDiff) {
      char line[1024];
      int margin = 37;
      int width = std::max(3*ngpus+10, 40);

      line[0] = '\0';
      while (strlen(line) < margin) sprintf(line+strlen(line), " ");
      sprintf(line+strlen(line), "    --- Reference --- ");
      while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
      sprintf(line+strlen(line), "    --- Computed  --- ");
      printf("%s\n", line);

      line[0] = '\0';
      sprintf(line+strlen(line), "                        Properties : ");
      while (strlen(line) < margin) sprintf(line+strlen(line), " ");
      sprintf(line+strlen(line), "%7s %2dx%4.1f/%4.1f %3s/%3s P%1d C%1d S%1d", graphNames[ref->id], ref->nChannels, ref->bwIntra, ref->bwInter, topoPathTypeStr[ref->typeIntra], topoPathTypeStr[ref->typeInter], ref->pattern, ref->crossNic, ref->sameChannels);
      while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
      sprintf(line+strlen(line), "%7s %2dx%4.1f/%4.1f %3s/%3s P%1d C%1d S%1d", graphNames[out->id], out->nChannels, out->bwIntra, out->bwInter, topoPathTypeStr[out->typeIntra], topoPathTypeStr[out->typeInter], out->pattern, out->crossNic, out->sameChannels);
      printf("%s\n", line);

      line[0] = '\0';
      for (int i=0; i<std::max(ref->nChannels, out->nChannels); i++) {
        sprintf(line, "                        Channel %2d : ", i);
        while (strlen(line) < margin) sprintf(line+strlen(line), " ");
        if (i < ref->nChannels) {
          if (type == TEST_INTER) sprintf(line+strlen(line), "[%2lx-%2lx %2lx-%2lx] ",
              NCCL_TOPO_ID_SYSTEM_ID(ref->inter[i*2]), NCCL_TOPO_ID_LOCAL_ID(ref->inter[i*2]),
              NCCL_TOPO_ID_SYSTEM_ID(ref->inter[i*2+1]), NCCL_TOPO_ID_LOCAL_ID(ref->inter[i*2+1]));
          for (int g=0; g<ngpus; g++) sprintf(line+strlen(line), "%2d ", ref->intra[i*ngpus+g]);
        }
        while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
        if (i < out->nChannels) {
          if (type == TEST_INTER) sprintf(line+strlen(line), "[%2lx-%2lx %2lx-%2lx] ",
              NCCL_TOPO_ID_SYSTEM_ID(out->inter[i*2]), NCCL_TOPO_ID_LOCAL_ID(out->inter[i*2]),
              NCCL_TOPO_ID_SYSTEM_ID(out->inter[i*2+1]), NCCL_TOPO_ID_LOCAL_ID(out->inter[i*2+1]));
          for (int g=0; g<ngpus; g++) sprintf(line+strlen(line), "%2d ", out->intra[i*ngpus+g]);
        }
        printf("%s\n", line);
      }
    }
  }
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

struct testParam{
  int ngpus;
  bool inter;
  bool intra;
  bool dumpDiff;
  bool dumpProcessedXml;
  // split Mask
  int splitMask;
  int color;
  // NIC fusion
  int mergeLevel;
  const char* forceMerge;
};

#define TESTPARAM_INIT {\
  /*ngpus=*/-1,\
  /*inter=*/true,\
  /*intra=*/true,\
  /*dumpDiff=*/1,\
  /*dumpProcessedXml=*/0,\
  /*splitMask=*/ -1,\
  /*color=*/0,\
  /*mergeLevel=*/PATH_LOC,\
  /*forceMerge=*/NULL\
}

void getTestParam(struct testParam* param) {
  // default parameters
  *param = TESTPARAM_INIT;

  // get params from the env
  const char* str = getenv("NCCL_GRAPH_TEST_DUMP");
  if (str) param->dumpDiff = atoi(str);

  str = getenv("NCCL_GRAPH_TEST_DUMP_SYSTEM_XML");
  if (str) param->dumpProcessedXml = atoi(str);

  str = getenv("NCCL_GRAPH_TEST_NGPUS");
  if (str) param->ngpus = atoi(str);

  str = getenv("NCCL_TESTS_SPLIT_MASK");
  if (str) param->splitMask = strtoul(str, NULL, 0);

  str = getenv("NCCL_GRAPH_TEST_COLOR");
  if (str) param->color = strtoul(str, NULL, 0);

  CHECK(ncclTopoGetFusionEnv(&param->mergeLevel, &param->forceMerge));

  str = getenv("NCCL_GRAPH_TEST_INTER");
  if (str) param->inter = atoi(str) > 0;

  str = getenv("NCCL_GRAPH_TEST_INTRA");
  if (str) param->intra = atoi(str) > 0;
}

#define TIME_RING 0
#define TIME_TREE 1
#define TIME_CNET 2
#define TIME_NVLS 3
#define TIME_TOTL 4
#define TIME_SIZE 5

void checkTopo(const char* xmlTopoFile, const char* xmlGraphFile, const char* platform, enum testType type, const struct testParam* param, int* errors, int* warnings) {
  struct ncclXml* xmlSystem;
  char dumpFile[PATH_MAX];
  INFO(NCCL_GRAPH, "Loading platform %s", platform);
  CHECK(xmlAlloc(&xmlSystem, MAX_MNNVL_NODES*NCCL_TOPO_XML_MAX_NODES));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystem, 1));
  struct ncclTopoSystem* system;
  if (xmlSystem->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    (*errors)++;
    return;
  }

  // Inititalize a netState object for NIC fusion
  fakeNetPluginInit(xmlSystem);
  struct ncclTopoNetInfo netInfo{};
  netInfo.coll = coll > 0;
  netInfo.netPluginIndex = 0;
  netInfo.dmaBufSupport = true;
  netInfo.mergeLevel = param->mergeLevel;
  netInfo.forceMerge = param->forceMerge;
  netInfo.getDevCount = fakeNetPluginGetDevCount;
  netInfo.setVirtDevCount = fakeNetPluginSetDevCount;
  netInfo.name = "Fake";
  netInfo.getProperties = fakeNetPluginGetProperties;
  netInfo.makeVDevice = fakeNetPluginMakeVDevice;
  netInfo.devices = fakeNetPluginDevices;
  CHECK(ncclTopoProcessNet(xmlSystem, NULL, &netInfo));
  // We need to force all GPUs as keep="1" here to avoid trimming them
  keepGpus(xmlSystem);
  if (param->dumpProcessedXml) {
    snprintf(dumpFile, sizeof(dumpFile), "%s.processed", xmlTopoFile);
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystem));
  }
  CHECK(ncclTopoTrimXml(xmlSystem));
  if (param->dumpProcessedXml) {
    snprintf(dumpFile, sizeof(dumpFile), "%s.processed_trimmed", xmlTopoFile);
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystem));
  }
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
  if (type == TEST_INTRA) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      CHECK(ncclTopoRemoveNode(system, NET, n));
  }
  // prune GPUs depending on the number of GPUs and splitMask
  if (param->ngpus != -1) {
    for (int g = system->nodes[GPU].count - 1; g >= param->ngpus; g--) CHECK(ncclTopoRemoveNode(system, GPU, g));
  }
  if (param->splitMask != -1) {
    for (int g = system->nodes[GPU].count - 1; g >= 0; g--) {
      if ((g & param->splitMask) != param->color) CHECK(ncclTopoRemoveNode(system, GPU, g));
    }
  }
  int ngpus = system->nodes[GPU].count;
  CHECK(ncclTopoComputePaths(system, NULL));
  CHECK(ncclTopoSearchInit(system));
  CHECK(ncclTopoPrint(system));

  char* str = getenv("NCCL_CROSS_NIC");
  int crossNic = str ? atoi(str) : 2;

  struct ncclTopoGraph ringGraph;
  memset(&ringGraph, 0, sizeof(ringGraph));
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.crossNic = crossNic;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = 16;

  struct ncclTopoGraph treeGraph;
  memset(&treeGraph, 0, sizeof(treeGraph));
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.crossNic = crossNic;
  treeGraph.collNet = 0;

  struct ncclTopoGraph cNetGraph;
  memset(&cNetGraph, 0, sizeof(cNetGraph));
  cNetGraph.id = 2;
  cNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  cNetGraph.crossNic = crossNic;
  cNetGraph.crossNic = 2;
  cNetGraph.collNet = 1;

  struct ncclTopoGraph nvlsGraph;
  memset(&nvlsGraph, 0, sizeof(nvlsGraph));
  nvlsGraph.id = 3;
  nvlsGraph.pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph.crossNic = crossNic;
  nvlsGraph.collNet = 0;

  /* Compute */
  uint64_t computeTime[TIME_SIZE];
  computeTime[TIME_RING] = getTime();
  CHECK(ncclTopoCompute(system, &ringGraph));
  computeTime[TIME_RING] = getTime() - computeTime[TIME_RING];
  CHECK(ncclTopoPrintGraph(system, &ringGraph));
  treeGraph.minChannels = ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  computeTime[TIME_TREE] = getTime();
  CHECK(ncclTopoCompute(system, &treeGraph));
  computeTime[TIME_TREE] = getTime() - computeTime[TIME_TREE];
  CHECK(ncclTopoPrintGraph(system, &treeGraph));
  cNetGraph.minChannels = cNetGraph.maxChannels = ringGraph.nChannels;
  computeTime[TIME_CNET] = getTime();
  CHECK(ncclTopoCompute(system, &cNetGraph));
  computeTime[TIME_CNET] = getTime() - computeTime[TIME_CNET];
  CHECK(ncclTopoPrintGraph(system, &cNetGraph));
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;
  computeTime[TIME_NVLS] = getTime();
  CHECK(ncclTopoCompute(system, &nvlsGraph));
  computeTime[TIME_NVLS] = getTime() - computeTime[TIME_NVLS];
  CHECK(ncclTopoPrintGraph(system, &nvlsGraph));
  computeTime[TIME_TOTL] = 0;
  for(int i=0; i<TIME_TOTL; ++i) computeTime[TIME_TOTL] += computeTime[i];

  int err = 0, warn = 0, incompleteRef = 0;

  /* Get reference graphs from XML */
  struct ncclXml* xmlGraph;
  CHECK(xmlAlloc(&xmlGraph, NCCL_GRAPH_XML_MAX_NODES));
  if (ncclTopoGetXmlGraphFromFile(xmlGraphFile, xmlGraph) != ncclSuccess) {
    warn = 1; incompleteRef = 1;
  } else {
    struct ncclTopoGraph refRingGraph, refTreeGraph, refCNetGraph, refNvlsGraph;
    memcpy(&refRingGraph, &ringGraph, sizeof(ringGraph));
    memcpy(&refTreeGraph, &treeGraph, sizeof(treeGraph));
    memcpy(&refCNetGraph, &cNetGraph, sizeof(cNetGraph));
    memcpy(&refNvlsGraph, &nvlsGraph, sizeof(nvlsGraph));
    // Get graphs from XML. We select the right graph based on the id.
    int refNChannels[4] = { 0, 0, 0, 0 };
    CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refRingGraph, refNChannels));
    CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refTreeGraph, refNChannels+1));
    CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refCNetGraph, refNChannels+2));
    CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refNvlsGraph, refNChannels+3));
    if (ringGraph.nChannels != refNChannels[0] || treeGraph.nChannels != refNChannels[1] || cNetGraph.nChannels != refNChannels[2] || nvlsGraph.nChannels != refNChannels[3]) {
      warn = 1;
      incompleteRef = 1;
    }
    /* Compare */
    compareGraphs(&refRingGraph, &ringGraph, ngpus, type, param->dumpDiff, &err, &warn);
    compareGraphs(&refTreeGraph, &treeGraph, ngpus, type, param->dumpDiff, &err, &warn);
    if (type == TEST_INTER) compareGraphs(&refCNetGraph, &cNetGraph, ngpus, type, param->dumpDiff, &err, &warn);
    compareGraphs(&refNvlsGraph, &nvlsGraph, ngpus, type, param->dumpDiff, &err, &warn);
  }
  free(xmlGraph);

  char suffix[1024] = "";
  if (param->splitMask != -1) snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "/0x%x-0x%x", param->splitMask, param->color);
  if (param->forceMerge) {
    snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "/%s", param->forceMerge);
  } else if (param->mergeLevel > PATH_PORT) {
    int i = 0;
    while (nicPathKvList[i].value != param->mergeLevel && nicPathKvList[i].str != NULL) i++;
    snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), "/%s", nicPathKvList[i].str ? nicPathKvList[i].str : "ERR");
  }
  while (strlen(suffix) < 12) snprintf(suffix + strlen(suffix), sizeof(suffix) - strlen(suffix), " ");
  printf(" %20s/%2d/%s%s  %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f", platform, ngpus, (type == TEST_INTER) ? "Inter":"Intra", suffix,
      ringGraph.nChannels, ringGraph.bwIntra, ringGraph.bwInter,
      treeGraph.nChannels, treeGraph.bwIntra, treeGraph.bwInter,
      cNetGraph.nChannels, cNetGraph.bwIntra, cNetGraph.bwInter,
      nvlsGraph.nChannels, nvlsGraph.bwIntra, nvlsGraph.bwInter);

  if (err || warn || incompleteRef) {
    snprintf(dumpFile, sizeof(dumpFile), "%s.dump", xmlGraphFile);
    struct ncclXml* xml;
    CHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    struct ncclTopoGraph* graphs[4] = { &ringGraph, &treeGraph, &cNetGraph, &nvlsGraph };
    CHECK(ncclTopoGetXmlFromGraphs(4, graphs, system, xml));
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xml));
    printf("\ngraph_test : Dumping XML to %s\n", dumpFile);
    free(xml);
    printf(" %s %5ld ms\n", (err || warn) ? "FAILED" : "  WARN", computeTime[TIME_TOTL] / 1000);
    printf("Dumping computed graph to %s\n", dumpFile);
  } else if (computeTime[TIME_TOTL] > 1e6) {
    bool tooSlow = computeTime[TIME_TOTL] > 5e6;
    printf("   %sSLOW %5ld ms (ring: %ld ms + tree: %ld ms + collNet: %ld ms + nvls %ld ms)\n", tooSlow ? "TOO " : "",\
           computeTime[TIME_TOTL] / 1000, \
           computeTime[TIME_RING] / 1000, computeTime[TIME_TREE] / 1000, computeTime[TIME_CNET] / 1000, computeTime[TIME_NVLS] / 1000);
    if (tooSlow) warn++;
  } else
    printf("     OK %5ld ms\n", computeTime[TIME_TOTL] / 1000);
  ncclTopoFree(system);
  *errors += err;
  *warnings += warn;
}

// Helper function to build graph filename based on parameters
static void buildGraphFilename(char* filename, size_t size, const char* topoDir, const char* platform, enum testType type, const struct testParam* param) {
  // Start with base path and graph type
  char modifiers[256] = "";

  // First handle splitMask which affects communicator structure
  if (param->splitMask != -1) {
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-sm%d-c%d", param->splitMask, param->color);
  }
  // Then add any merge/force modifiers
  if (param->forceMerge != NULL) {
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-fm%s", param->forceMerge);
  } else if (PATH_PORT < param->mergeLevel) {
    int i = 0;
    while (nicPathKvList[i].value != param->mergeLevel && nicPathKvList[i].str != NULL) i++;
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-ml%s", nicPathKvList[i].str ? nicPathKvList[i].str : "ERR");
  }
  // Finally add ngpus if specified
  if (param->ngpus != -1) { // Don't add ngpus if we have splitMask
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-%d", param->ngpus);
  }

  // Combine everything into final filename
  snprintf(filename, size, "%stopo/%s/%s-graph%s.xml", topoDir, platform, (type == TEST_INTER) ? "inter" : "intra", modifiers);
}

void checkPlatform(const char* platform, struct testParam* param, int* errors, int* warnings) {
  char xmlTopoFile[PATH_MAX];
  char xmlGraphFile[PATH_MAX];
  char topoDir[1024];
  const char* envTopoDir = getenv("TOPO_DIR");
  if (envTopoDir) {
    snprintf(topoDir, 1024, "%s", envTopoDir);
  } else {
    topoDir[0] = '\0';
  }

  // check for parameter correctness
  if (param->forceMerge && param->mergeLevel > PATH_PORT) {
    printf("NCCL graph_test does not support setting both NIC fusion NCCL_NET_MERGE_LEVEL and NCCL_NET_FORCE_MERGE. Discarding the value of NCCL_NET_FORCE_MERGE.\n");
    param->forceMerge = NULL;
  }

  // Build system topology file path
  snprintf(xmlTopoFile, PATH_MAX, "%stopo/%s/system.xml", topoDir, platform);

  // Check intra topology
  if (param->intra) {
    buildGraphFilename(xmlGraphFile, PATH_MAX, topoDir, platform, TEST_INTRA, param);
    checkTopo(xmlTopoFile, xmlGraphFile, platform, TEST_INTRA, param, errors, warnings);
  }
  if (param->inter) {
    buildGraphFilename(xmlGraphFile, PATH_MAX, topoDir, platform, TEST_INTER, param);
    checkTopo(xmlTopoFile, xmlGraphFile, platform, TEST_INTER, param, errors, warnings);
  }
}

#define RUN_INTRA(platform)                          \
  do {                                               \
    struct testParam p = param;                      \
    p.inter = false;                                 \
    checkPlatform(platform, &p, &errors, &warnings); \
  } while (0)

#define RUN(platform) checkPlatform(platform, &param, &errors, &warnings)

#define RUN_FUSION(platform, level)                  \
  do {                                               \
    struct testParam p = param;                      \
    p.mergeLevel = level;                            \
    p.intra = 0;                                   \
    checkPlatform(platform, &p, &errors, &warnings); \
  } while (0)

#define RUN_MULTI4(platform)                         \
  do {                                               \
    struct testParam p = param;                      \
    p.ngpus = 4;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
    p.ngpus = 2;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
    p.ngpus = 1;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
  } while (0)

#define RUN_MULTI8(platform)                         \
  do {                                               \
    struct testParam p = param;                      \
    p.ngpus = 8;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
    p.ngpus = 6;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
    RUN_MULTI4(platform);                            \
  } while (0)

#define RUN_SPLITMASK(platform, /*countGpu=*/N)          \
  do {                                                   \
    struct testParam p = param;                          \
    for (int s = 1; s < 32; ++s) {                       \
      int nComms = 1 << s;                               \
      if (nComms > N) break;                             \
      for (int c = 0; c < nComms; c++) {                 \
        p.splitMask = (1 << s) - 1;                      \
        p.color = c;                                     \
        checkPlatform(platform, &p, &errors, &warnings); \
      }                                                  \
    }                                                    \
  } while (0)

void printHelpMessage() {
  printf("This tool directly invokes NCCL topology and graph search code, and operates on a database of system.xml files. You can directly modify topo.cc or any other relevant files and test their behavior here.\n");
  printf("Usage : graph_test [platform] [ngpus]\n");
  printf("  platform : platform name (e.g. LOC-1G)\n");
  printf("  ngpus    : number of GPUs per node (default -1, all)\n");
  printf("  -h       : print this help message\n");
  printf("Set NCCL_GRAPH_TEST_NGPUS=N to change the number of GPUs per node.\n");
  printf("Set NCCL_TOPO_DIR to override the default topo directory. This is necessary to invoke graph_test from an outside directory.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP=0 to disable dumping of graph diffs.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP_SYSTEM_XML=1 to dump the processed system XML from NIC Fusion and then the fully trimmed system XML.\n");
  printf("Set NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH to get standard NCCL logs of your scenario.\n");
  printf("Set NCCL_TESTS_SPLIT_MASK=0xA to only get the graph for the communicator with the color 0 (unless changed, see NCCL_GRAPH_TEST_COLOR) in the case of NCCL_TESTS_SPLIT_MASK=0xA.\n");
  printf("Set NCCL_GRAPH_TEST_COLOR=0xA to change the color that is considered when using NCCL_TESTS_SPLIT_MASK.\n");
  printf("Set NCCL_NET_FORCE_MERGE to force the merge between devices, see NCCL documentation.\n");
  printf("Set NCCL_NET_MERGE_LEVEL to change the NIC fusion merge level, see NCCL documentation.\n");
  printf("Set NCCL_GRAPH_TEST_INTER=0/1 to enable the INTER test.\n");
  printf("Set NCCL_GRAPH_TEST_INTRA=0/1 to enable the INTRA test.\n");
}

int main(int argc, const char* argv[]) {
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 0); // Disable hardware health checks (NVML)
  setlinebuf(stdout);

  struct testParam param;
  getTestParam(&param);

  int errors = 0, warnings = 0;
  if (argc > 1) {
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
      printHelpMessage();
      return 0;
    }
    if (argc > 2) param.ngpus = atoi(argv[2]);
    checkPlatform(argv[1], &param, &errors, &warnings);
  } else {
    RUN("LOC-1G");
    RUN("PCI-1R");
    RUN("PCI-2R");
    RUN("PCI-No-Numa");
    RUN("PCI-NV");
    RUN("SKL-V100");
    RUN("MS-1G-2N");
    RUN("T4");
    RUN("A10-PCI");
#ifdef __x86_64__
    RUN("DGX-1P");
    RUN("DGX-1P-4G");
    RUN_MULTI8("DGX-1V");
    RUN("DGX-2V");
    RUN_MULTI8("DGX-2V");
    RUN("XMAN-3");
    RUN_MULTI8("Luna");
    RUN_MULTI8("DGX-A800");
    RUN_MULTI8("Luna-SHARP");
    RUN("DGX-2-Delta");
    RUN_MULTI4("Redstone");
    RUN("GCP-NV");
    RUN("AWS-NV");
    RUN("AWS-NV-EFA");
    RUN("Azure");
    RUN("FB-BUG");
    RUN("GCP-Shared-NVS");
    RUN("Dual-Delta-VM");
    RUN("ZionEX");
    RUN("FB-V100");
    RUN_MULTI8("DGX-H800");
    RUN_MULTI8("DGX-H800-4NIC");
    RUN_MULTI8("Viking");
    RUN_MULTI8("Viking-SHARP");
    RUN_MULTI4("Scout");
    RUN("PCI-H100-NV");
    RUN("OCI-HGX-A100");
    RUN("OCI-HGX-A100-flat");
    RUN("DualPort-CPU");
    RUN("Dell_R760xa");
    RUN("SMC521GE");
    RUN("Perlmutter");
    RUN_MULTI8("Umbriel");
#endif
    RUN("CG4");
    RUN("P9-6V");
    RUN("P9-4V");
    RUN("HP-ARM-V100");
    RUN("GB200");
    { // GB200-Ariel-NVL8
      RUN("GB200-Ariel-NVL8");
      RUN_FUSION("GB200-Ariel-NVL8", /*PXB=*/5);
    }
    RUN("GB200-AWS-NVL8");
    {// GB200-NVL36
      RUN("GB200-NVL36");
      RUN_SPLITMASK("GB200-NVL36", 8);
    }
    {// GB200-NVL72
      RUN("GB200-NVL72");
      RUN_SPLITMASK("GB200-NVL72", 8);
    }
    RUN("GB200-CX8-NVL4");
    RUN("GB200-CX8-NVL32");
    RUN("GB300-CX8-NVL4");
    RUN("GB300-CX8-NVL32");
    RUN("DGX-Spark");
    RUN("DGX-Spark-flat");
  }
  printf("%d errors, %d warnings (%s)\n", errors, warnings, (errors || warnings) ? "FAILED" : "PASSED");
  return (errors || warnings) ? 1 : 0;
}
