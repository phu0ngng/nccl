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
  int speed;
  float latency;
  int ignore;
  ncclNetVDeviceProps_t vProps;
};
mockVDev* mockVDevs = nullptr;
ncclNetProperties_t* mockProps = nullptr;

void fakeNetPluginAddNetNode(struct ncclXmlNode* node) {
  int devIndex;
  CHECK(xmlGetAttrInt(node, "dev", &devIndex));
  mockVDev* dev = mockVDevs + devIndex;
  dev->ignore = 0;
  CHECK(xmlGetAttrInt(node, "speed", &dev->speed));
  xmlGetAttrFloat(node, "latency", &dev->latency);
  const char* name;
  CHECK(xmlGetAttrStr(node, "name", &name));
  snprintf(dev->name, sizeof(dev->name), "%s", name);
  ncclNetProperties_t* props = mockProps + devIndex;
  // Perhaps todo - Compute PCI Path
  CHECK(xmlGetAttrUint64(node, "guid", &props->guid));
  int size = strlen(name) + 1;
  props->name = (char*) malloc(size);
  snprintf(props->name, size, "%s", name);
  CHECK(xmlGetAttrInt(node, "speed", &props->speed));
  xmlGetAttrFloat(node, "latency", &props->latency);
  CHECK(xmlGetAttrInt(node, "port", &props->port));
  // We are assuming that all NICs are coll=1 on collnet platforms
  xmlGetAttrInt(node, "coll", &coll);

  // It's assumed system.xml files won't have duplicate "dev" fields
  if (nPhysDevs < (devIndex + 1)) nPhysDevs = devIndex + 1;
  nVirtualDevs = nPhysDevs;
}

void fakeNetPluginInit(struct ncclXml* xmlSystem) {
  if (mockVDevs == NULL) {
    mockVDevs = (mockVDev*) malloc(sizeof(mockVDev)*MAX_MOCK_DEVS);
    mockProps = (ncclNetProperties_t*) malloc(sizeof(ncclNetProperties_t)*MAX_MOCK_DEVS);
  }

  for (int i = 0; i < nPhysDevs; i++) {
    free(mockProps[i].name);
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
    for (int i = 0; i < mDev->vProps.ndevs; i++) {
      int pDev = mDev->vProps.devs[i];
      mDev->speed   += mockProps[pDev].speed;
      mDev->latency += mockProps[pDev].latency;
      if (i > 0) {
        snprintf(mDev->name + strlen(mDev->name), sizeof(mDev->name) - strlen(mDev->name), "+%s", mockProps[pDev].name);
      } else {
        strncpy(mDev->name, mockProps[pDev].name, 128);
      }
    }

    printf("Fake/Plugin : Made vDevice %s speed=%d\n", mDev->name, mDev->speed);

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

void checkTopo(const char* xmlTopoFile, const char* xmlGraphFile, const char* platform, int inter, int ngpus, int* errors, int* warnings) {
  struct ncclXml* xmlSystem;
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
  CHECK(ncclTopoTrimXml(xmlSystem));
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system, 0));
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
  uint64_t computeTime[5];
  computeTime[0] = getTime();
  CHECK(ncclTopoCompute(system, &ringGraph));
  computeTime[0] = getTime() - computeTime[0];
  CHECK(ncclTopoPrintGraph(system, &ringGraph));
  treeGraph.minChannels = ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  computeTime[1] = getTime();
  CHECK(ncclTopoCompute(system, &treeGraph));
  computeTime[1] = getTime() - computeTime[1];
  CHECK(ncclTopoPrintGraph(system, &treeGraph));
  cNetGraph.minChannels = cNetGraph.maxChannels = ringGraph.nChannels;
  computeTime[2] = getTime();
  CHECK(ncclTopoCompute(system, &cNetGraph));
  computeTime[2] = getTime() - computeTime[2];
  CHECK(ncclTopoPrintGraph(system, &cNetGraph));
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;
  computeTime[3] = getTime();
  CHECK(ncclTopoCompute(system, &nvlsGraph));
  computeTime[3] = getTime() - computeTime[3];
  CHECK(ncclTopoPrintGraph(system, &nvlsGraph));
  computeTime[4] = computeTime[0]+computeTime[1]+computeTime[2]+computeTime[3];

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
    char dumpFile[PATH_MAX];
    sprintf(dumpFile, "%s.dump", xmlGraphFile);
    struct ncclXml* xml;
    CHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    struct ncclTopoGraph* graphs[4] = { &ringGraph, &treeGraph, &cNetGraph, &nvlsGraph };
    CHECK(ncclTopoGetXmlFromGraphs(4, graphs, system, xml));
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xml));
    printf("\ngraph_test : Dumping XML to %s\n", dumpFile);
    free(xml);
    printf(" %s %5ld ms\n", err ? "FAILED" : "  WARN", computeTime[4]/1000);
    printf("Dumping computed graph to %s\n", dumpFile);
  } else if (computeTime[4] > 1000000) {
    printf("   SLOW %5ld ms [%ld+%ld+%ld+%ld]\n", computeTime[4]/1000,
        computeTime[0]/1000, computeTime[1]/1000, computeTime[2]/1000, computeTime[3]/1000);
    warn++;
  } else printf("     OK %5ld ms\n", computeTime[4]/1000);
  ncclTopoFree(system);
  *errors += err;
  *warnings += warn;
}

void checkPlatform(const char* platform, int ngpus, int* errors, int* warnings) {
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
  checkTopo(xmlTopoFile, xmlGraphFile, platform, 1, ngpus, errors, warnings);
}

#define RUN(platform) checkPlatform(platform, -1, &errors, &warnings)

#define RUN_MULTI4(platform) do { \
  checkPlatform(platform, 4, &errors, &warnings); \
  checkPlatform(platform, 2, &errors, &warnings); \
  checkPlatform(platform, 1, &errors, &warnings); \
} while(0)

#define RUN_MULTI8(platform) do { \
  checkPlatform(platform, 8, &errors, &warnings); \
  checkPlatform(platform, 6, &errors, &warnings); \
  checkPlatform(platform, 4, &errors, &warnings); \
  checkPlatform(platform, 2, &errors, &warnings); \
  checkPlatform(platform, 1, &errors, &warnings); \
} while(0)

int main(int argc, const char* argv[]) {
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 0); // Disable hardware health checks (NVML)
  setlinebuf(stdout);
  char* str = getenv("NCCL_GRAPH_TEST_DUMP");
  if (str) dumpDiff = atoi(str);
  int errors = 0, warnings = 0;
  if (argc > 1) {
    if (argc > 2) checkPlatform(argv[1], atoi(argv[2]), &errors, &warnings);
    else checkPlatform(argv[1], -1, &errors, &warnings);
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
    RUN("DualPort-CPU");
    RUN("Dell_R760xa");
    RUN("SMC521GE");
    RUN("Perlmutter");
#endif
    RUN("CG4");
    RUN("P9-6V");
    RUN("P9-4V");
    RUN("HP-ARM-V100");
    RUN("GB200-NVL36");
    RUN("GB200-NVL72");
  }
  printf("%d errors, %d warnings (%s)\n", errors, warnings, errors ? "FAILED" : "PASSED");
  return errors ? 1 : 0;
}
