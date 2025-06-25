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

int dumpDiff = 1;
int coll = 0;
int dumpProcessedXml = 0;

void compareGraphs(struct ncclTopoGraph* ref, struct ncclTopoGraph* out, int ngpus, int inter, int* errors, int* warnings) {
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
          if (inter) sprintf(line+strlen(line), "[%2lx-%2lx %2lx-%2lx] ",
              NCCL_TOPO_ID_SYSTEM_ID(ref->inter[i*2]), NCCL_TOPO_ID_LOCAL_ID(ref->inter[i*2]),
              NCCL_TOPO_ID_SYSTEM_ID(ref->inter[i*2+1]), NCCL_TOPO_ID_LOCAL_ID(ref->inter[i*2+1]));
          for (int g=0; g<ngpus; g++) sprintf(line+strlen(line), "%2d ", ref->intra[i*ngpus+g]);
        }
        while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
        if (i < out->nChannels) {
          if (inter) sprintf(line+strlen(line), "[%2lx-%2lx %2lx-%2lx] ",
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
int nPhysDevs = 0;
int nVirtualDevs = 0;
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
mockVDev* mockVDevs = nullptr;
ncclNetProperties_t* mockProps = nullptr;

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
  CHECK(xmlGetAttrUint64(node, "guid", &props->guid));
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
  nVirtualDevs = nPhysDevs;
}

void fakeNetPluginInit(struct ncclXml* xmlSystem) {
  if (mockVDevs == NULL) {
    mockVDevs = (mockVDev*) malloc(sizeof(mockVDev)*MAX_MOCK_DEVS);
    mockProps = (ncclNetProperties_t*) malloc(sizeof(ncclNetProperties_t)*MAX_MOCK_DEVS);
  }

  nPhysDevs = 0;
  nVirtualDevs = 0;
  coll = 0;
  memset(mockVDevs, 0, sizeof(mockVDev)*MAX_MOCK_DEVS);
  memset(mockProps, 0, sizeof(ncclNetProperties_t)*MAX_MOCK_DEVS);

  struct ncclXmlNode* node;
  CHECK(xmlFindTag(xmlSystem, "net", &node));
  while (node) {
    fakeNetPluginAddNetNode(node);
    CHECK(xmlFindNextTag(xmlSystem, "net", node, &node));
  }

  // Add in fake devs if there are any skipped in system.xml
  for (int i = 0; i < nPhysDevs; i++) {
    mockVDev* dev = mockVDevs + i;
    ncclNetProperties_t* props = mockProps + i;
    if (dev->speed == 0) {
      dev->speed = 1;
      dev->ignore = 1;
      snprintf(dev->name, sizeof(dev->name), "ignore_%d", i);
      int size = strlen(dev->name) + 1;
      props->name = (char*) malloc(size);
      snprintf(props->name, size, "ignore_%d", i);
      props->speed = 1;
    }
  }
}

ncclResult_t fakeNetPluginGetProperties(int dev, ncclNetProperties_t* props) {
  if (dev < nVirtualDevs) {
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
    return ncclInvalidUsage;
  }
}

ncclResult_t fakeNetPluginDevices(int* ndev) {
  *ndev = nVirtualDevs;
  return ncclSuccess;
}

ncclResult_t fakeNetPluginMakeVDevice(int* d, ncclNetVDeviceProps_t* vProps) {
  if (nVirtualDevs < MAX_MOCK_VDEVS) {
    if (vProps->ndevs > NCCL_NET_MAX_DEVS_PER_NIC) return ncclInvalidArgument;
    for (int i = 0; i < vProps->ndevs; i++) {
      int pDev = vProps->devs[i];
      if (mockVDevs[pDev].ignore) return ncclInvalidArgument;
    }
    int deviceIndex = nVirtualDevs;
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

    INFO(NCCL_GRAPH, "Fake/Plugin : Made vDevice %s speed=%d\n", mDev->name, mDev->speed);

    *d = nVirtualDevs;
    nVirtualDevs++;
    return ncclSuccess;
  } else {
    return ncclInvalidUsage;
  }
}

void keepGpus(struct ncclXml* xmlSystem) {
  struct ncclXmlNode* node;
  CHECK(xmlFindTag(xmlSystem, "gpu", &node));
  while (node) {
    CHECK(xmlSetAttrInt(node, "keep", 1));
    CHECK(xmlFindNextTag(xmlSystem, "gpu", node, &node));
  }
}

#define TIME_RING 0
#define TIME_TREE 1
#define TIME_CNET 2
#define TIME_NVLS 3
#define TIME_TOTL 4
#define TIME_SIZE 5

void checkTopo(const char* xmlTopoFile, const char* xmlGraphFile, const char* platform, int inter, int ngpus, int* errors, int* warnings) {
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
  ncclTopoNetState netState = {-1,-1};
  fakeNetPluginInit(xmlSystem);
  CHECK(ncclTopoProcessNet(xmlSystem, coll, NULL, &netState,
    fakeNetPluginGetProperties, fakeNetPluginMakeVDevice, fakeNetPluginDevices, "Fake", true));
  // We need to force all GPUs as keep="1" here to avoid trimming them
  keepGpus(xmlSystem);
  if (dumpProcessedXml) {
    snprintf(dumpFile, sizeof(dumpFile), "%s.processed", xmlTopoFile);
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystem));
  }
  CHECK(ncclTopoTrimXml(xmlSystem));
  if (dumpProcessedXml) {
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
  if (inter == 0) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      CHECK(ncclTopoRemoveNode(system, NET, n));
  }
  if (ngpus != -1) {
    for (int g=system->nodes[GPU].count-1; g>=ngpus; g--)
      CHECK(ncclTopoRemoveNode(system, GPU, g));
  }
  ngpus = system->nodes[GPU].count;
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
    compareGraphs(&refRingGraph, &ringGraph, ngpus, inter, &err, &warn);
    compareGraphs(&refTreeGraph, &treeGraph, ngpus, inter, &err, &warn);
    if (inter) compareGraphs(&refCNetGraph, &cNetGraph, ngpus, inter, &err, &warn);
    compareGraphs(&refNvlsGraph, &nvlsGraph, ngpus, inter, &err, &warn);
  }
  free(xmlGraph);

  printf(" %15s/%2d/%s  %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f | %2dx%4.1f/%4.1f", platform, ngpus, inter ? "Inter":"Intra",
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

void checkPlatform(const char* platform, int ngpus, int intraOnly, int* errors, int* warnings) {
  char xmlTopoFile[PATH_MAX];
  char xmlGraphFile[PATH_MAX];
  char topoDir[1024];
  const char* envTopoDir = getenv("TOPO_DIR");
  if (envTopoDir) {
    snprintf(topoDir, 1024, "%s", envTopoDir);
  } else {
    topoDir[0] = '\0';
  }

  sprintf(xmlTopoFile, "%stopo/%s/system.xml", topoDir, platform);
  if (ngpus == -1) sprintf(xmlGraphFile, "%stopo/%s/intra-graph.xml", topoDir, platform);
  else sprintf(xmlGraphFile, "%stopo/%s/intra-graph-%d.xml", topoDir, platform, ngpus);
  checkTopo(xmlTopoFile, xmlGraphFile, platform, 0, ngpus, errors, warnings);
  if (ngpus == -1) sprintf(xmlGraphFile, "%stopo/%s/inter-graph.xml", topoDir, platform);
  else sprintf(xmlGraphFile, "%stopo/%s/inter-graph-%d.xml", topoDir, platform, ngpus);
  if (!intraOnly) checkTopo(xmlTopoFile, xmlGraphFile, platform, 1, ngpus, errors, warnings);
}

#define RUN_INTRA(platform) checkPlatform(platform, -1, 1, &errors, &warnings)

#define RUN(platform) checkPlatform(platform, -1, 0, &errors, &warnings)

#define RUN_MULTI4(platform) do { \
  checkPlatform(platform, 4, 0, &errors, &warnings); \
  checkPlatform(platform, 2, 0, &errors, &warnings); \
  checkPlatform(platform, 1, 0, &errors, &warnings); \
} while(0)

#define RUN_MULTI8(platform) do { \
  checkPlatform(platform, 8, 0, &errors, &warnings); \
  checkPlatform(platform, 6, 0, &errors, &warnings); \
  checkPlatform(platform, 4, 0, &errors, &warnings); \
  checkPlatform(platform, 2, 0, &errors, &warnings); \
  checkPlatform(platform, 1, 0, &errors, &warnings); \
} while(0)

void printHelpMessage() {
  printf("This tool directly invokes NCCL topology and graph search code, and operates on a database of system.xml files. You can directly modify topo.cc or any other relevant files and test their behavior here.\n");
  printf("Usage : graph_test [platform] [ngpus]\n");
  printf("  platform : platform name (e.g. LOC-1G)\n");
  printf("  ngpus    : number of GPUs per node (default -1, all)\n");
  printf("  -h       : print this help message\n");
  printf("Set NCCL_TOPO_DIR to override the default topo directory. This is necessary to invoke graph_test from an outside directory.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP=0 to disable dumping of graph diffs.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP_SYSTEM_XML=1 to dump the processed system XML from NIC Fusion and then the fully trimmed system XML.\n");
  printf("Set NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH to get standard NCCL logs of your scenario.\n");
}

int main(int argc, const char* argv[]) {
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 0); // Disable hardware health checks (NVML)
  setlinebuf(stdout);
  char* str = getenv("NCCL_GRAPH_TEST_DUMP");
  if (str) dumpDiff = atoi(str);
  int errors = 0, warnings = 0;
  str = getenv("NCCL_GRAPH_TEST_DUMP_SYSTEM_XML");
  if (str) dumpProcessedXml = atoi(str);
  if (argc > 1) {
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
      printHelpMessage();
      return 0;
    }
    if (argc > 2) checkPlatform(argv[1], atoi(argv[2]), 0, &errors, &warnings);
    else checkPlatform(argv[1], -1, 0, &errors, &warnings);
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
    RUN("GB200-NVL36");
    RUN("GB200-NVL72");
    RUN("GB200-CX8");
    RUN("DGX-Spark");
    RUN("DGX-Spark-flat");
  }
  printf("%d errors, %d warnings (%s)\n", errors, warnings, (errors || warnings) ? "FAILED" : "PASSED");
  return (errors || warnings) ? 1 : 0;
}
