#include "topo.h"
#include "xml.h"
#include "nccl_net.h"
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
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
  // multi-port systems
  int portRatio;
  // multinode NVLDs
  int nvldSize;
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
  /*forceMerge=*/NULL,\
  /*portRatio=*/1,\
  /*nvldSize=*/1,\
}

static void setStackSize(rlim_t size) {
  int res = 0;
  size_t max = 0;
  struct rlimit rl;
  if ((res = getrlimit(RLIMIT_STACK, &rl)) != 0) goto fail;
  if ((max = rl.rlim_max) < size) goto fail;
  if (rl.rlim_cur < size) {
    rl.rlim_cur = size;
    printf("setting stack size to %ld/%ld\n", size, rl.rlim_max);
    if ((res = setrlimit(RLIMIT_STACK, &rl)) != 0) goto fail;
  }
  return;
fail:
  printf("Failed to set the stack size to %ld kiB (max size = %ld kiB). Please run 'ulimit -s %ld' and try again.\n", size / 1024, max / 1024, size / 1024);
  exit(1);
}

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
      // Line: margin + 2 columns, ~8 chars per GPU total. Min 256 for headers.
      char* line = (char*)malloc(std::max(ngpus * 8, 256));
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
      free(line);
    }
  }
}
#define MAX_MNNVL_NODES 72 /*max rank count per NVLD*/

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
mockVDev* mockVDevs;
ncclNetProperties_t* mockProps;

static void allocateMock() {
  mockVDevs = (struct mockVDev*)calloc(MAX_MOCK_VDEVS, sizeof(mockVDev));
  mockProps = (ncclNetProperties_t*)calloc(MAX_MOCK_VDEVS, sizeof(ncclNetProperties_t));
}
static void freeMock() {
  free(mockVDevs);
  free(mockProps);
}

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
  INFO(NCCL_GRAPH | NCCL_NET, "%s: device %s:%d added with speed=%d, guid=0x%lx, pciPath=%s", __func__, props->name, props->port, props->speed, props->guid, props->pciPath);

  // It's assumed system.xml files won't have duplicate "dev" fields
  if (nPhysDevs < (devIndex + 1)) nPhysDevs = devIndex + 1;
}

void fakeNetPluginInit(struct ncclXml* xmlSystem, const struct testParam* param) {
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

  str = getenv("NCCL_GRAPH_TEST_PORT_RATIO");
  if (str) param->portRatio = atoi(str);
}

#define TIME_RING 0
#define TIME_TREE 1
#define TIME_CNET 2
#define TIME_NVLS 3
#define TIME_TOTL 4
#define TIME_SIZE 5

#define MAX_TOPO_NODES 128

static ncclResult_t xmlSplitNics(struct ncclXml* xmlSystem, int ratio){
  if (ratio == 1) return ncclSuccess;

  int listCount = 0;
  struct ncclXmlNode** nodeList = (struct ncclXmlNode**)calloc(MAX_TOPO_NODES,sizeof(struct ncclXmlNode*));
  {
    // first list all the nets in the system to avoid counting new nets
    struct ncclXmlNode* node;
    CHECK(xmlFindTag(xmlSystem, "net", &node));
    while (node) {
      if (listCount >= MAX_TOPO_NODES) {
        WARN("ERROR: too many networks devices in the topology for port duplication.");
        return ncclInvalidArgument;
      }
      nodeList[listCount++] = node;
      CHECK(xmlFindNextTag(xmlSystem, "net", node, &node));
    }
  }
  for(int n=0; n<listCount; ++n){
    const char *nameAttr;
    int speed = -1, port = -1;
    struct ncclXmlNode* node = nodeList[n];
    CHECK(xmlGetAttrInt(node, "speed", &speed));
    CHECK(xmlGetAttrIntDefault(node, "port", &port, 0));
    CHECK(xmlGetAttr(node, "name", &nameAttr));
    // copy the name to avoid recursive name modification
    char* name = strdup(nameAttr);

    // we need to get rid of the guid attr to avoid conflicts.
    // Add NIC is supported without guid so it will not be an issue later.
    int shift = 0;
    for (int i = 0; i < node->nAttrs; ++i) {
      if (strncmp(node->attrs[i].key, "guid", MAX_STR_LEN) == 0) {
        shift = 1;
      } else if (shift == 1) {
        node->attrs[i - 1] = node->attrs[i];
      }
    }
    node->nAttrs -= shift;

    // add other nets
    for (int i = 0; i < ratio; ++i) {
      char subName[MAX_STR_LEN];
      snprintf(subName, sizeof(subName), "%s-p%d", name, port + i);

      struct ncclXmlNode* sub = node;
      if (i > 0) CHECK(xmlAddNode(xmlSystem, node->parent, subName, &sub));
      memcpy(sub, node, sizeof(struct ncclXmlNode));

      // change the name, the speed, and the port, the rest stays the same
      CHECK(xmlSetAttr(sub,"name",subName));
      CHECK(xmlSetAttrInt(sub, "dev", n * ratio + i));
      CHECK(xmlSetAttrInt(sub, "port", port + i));
      CHECK(xmlSetAttrInt(sub, "speed", speed / ratio));
    }
    free(name);
  }
  free(nodeList);
  return ncclSuccess;
}

void checkTopo(const char* xmlTopoFile, const char* xmlGraphFile, const char* platform, enum testType type, const struct testParam* param, int* errors, int* warnings) {
  char* dumpFile = (char*)malloc(PATH_MAX);
  INFO(NCCL_GRAPH, "Loading platform %s - NVLD=%d", platform,param->nvldSize);
  // load the local XML (might contain the entire NVLD for backward compatibility reasons with GB200/300 topologies)
  struct ncclXml* xmlSystemLocal = NULL;
  CHECK(xmlAlloc(&xmlSystemLocal, MAX_MNNVL_NODES * NCCL_TOPO_XML_MAX_NODES));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystemLocal, 1));
  if (xmlSystemLocal->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    (*errors)++;
    free(dumpFile);
    return;
  }
  // Initiate the fake plugin with all the devices.
  // NIC duplication has to happen before the init.
  CHECK(xmlSplitNics(xmlSystemLocal, param->portRatio));
  fakeNetPluginInit(xmlSystemLocal, param);

  // process the local node
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
  CHECK(ncclTopoProcessNet(xmlSystemLocal, NULL, &netInfo));
  // We need to force all GPUs as keep="1" here to avoid trimming them
  keepGpus(xmlSystemLocal);
  if (param->dumpProcessedXml) {
    snprintf(dumpFile, PATH_MAX, "%s.processed", xmlTopoFile);
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystemLocal));
  }
  CHECK(ncclTopoTrimXml(xmlSystemLocal));
  if (param->dumpProcessedXml) {
    snprintf(dumpFile, PATH_MAX, "%s.processed_trimmed", xmlTopoFile);
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystemLocal));
  }

  // do the topo Fusion for MNNVL
  uint64_t hostHash = getHostHash();
  int rank = 0;
  struct ncclXml* xmlSystem = NULL;
  CHECK(xmlAlloc(&xmlSystem, MAX_MNNVL_NODES * NCCL_TOPO_XML_MAX_NODES));
  for (int i = 0; i < param->nvldSize; i++) {
    // Replace the hostHash on all cpuNode with a unique one for each host in the NVLD.
    struct ncclXmlNode* node = NULL;
    CHECK(xmlFindTag(xmlSystemLocal, "cpu", &node));
    while (node) {
      CHECK(xmlGetAttrUint64Default(node, "host_hash", &hostHash, 0x0));
      uint64_t hacc[2] = {1, 1};
      eatHash(hacc, &hostHash);
      eatHash(hacc, &i);
      hostHash = digestHash(hacc);
      CHECK(xmlSetAttrLong(node, "host_hash", hostHash));
      CHECK(xmlFindNextTag(xmlSystemLocal, "cpu", node, &node));
      INFO(NCCL_GRAPH,"host %d - cpu host_hash = 0x%lx",i,hostHash);
    }
    // update the rank for each GPU
    node = NULL;
    CHECK(xmlFindTag(xmlSystemLocal, "gpu", &node));
    while (node) {
      CHECK(xmlSetAttrInt(node, "rank", rank++));
      CHECK(xmlFindNextTag(xmlSystemLocal, "gpu", node, &node));
    }
    // Replace the guid for the net.
    node = NULL;
    CHECK(xmlFindTag(xmlSystemLocal, "net", &node));
    while (node) {
      uint64_t guid;
      CHECK(xmlGetAttrUint64Default(node, "guid", &guid, 0x0));
      uint64_t hacc[2] = {1, 1};
      eatHash(hacc, &guid);
      eatHash(hacc, &i);
      guid = digestHash(hacc);
      CHECK(xmlSetAttrLong(node, "guid", guid));
      CHECK(xmlFindNextTag(xmlSystemLocal, "net", node, &node));
    }
    // add the new host to the topology
    CHECK(ncclTopoFuseXml(xmlSystem, xmlSystemLocal));
  }
  snprintf(dumpFile, PATH_MAX, "%s.topo_global", xmlTopoFile);
  CHECK(ncclTopoDumpXmlToFile(dumpFile, xmlSystem));

  // we assume to be the on the last host, so we can reuse the hostHash that was set last
  struct ncclTopoSystem* system;
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system, /*last known hash*/hostHash));
  free(xmlSystem);
  free(xmlSystemLocal);

  system->inter = (type == TEST_INTRA) ? 0 : 1;
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
  ringGraph.maxChannels = MAXCHANNELS/2;

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
    snprintf(dumpFile, PATH_MAX, "%s.dump", xmlGraphFile);
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
  free(dumpFile);
  *errors += err;
  *warnings += warn;
}

// Helper function to build graph filename based on parameters
static void buildGraphFilename(char* filename, size_t size, const char* topoDir, const char* platform, enum testType type, const struct testParam* param) {
  // Start with base path and graph type
  char modifiers[256] = "";

  if(param->nvldSize >1){
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-NVLD%d", param->nvldSize);
  }
  // First handle splitMask which affects communicator structure
  if (param->splitMask != -1) {
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-sm%d-c%d", param->splitMask, param->color);
  }
  if(param->portRatio >1){
    snprintf(modifiers + strlen(modifiers), sizeof(modifiers) - strlen(modifiers), "-pr%d", param->portRatio);
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
  // recover the NVLD size and the system name from the platform name
  int count = 0;
  char * name = strdup(platform);
  char *token = strtok(name, "-");
  param->nvldSize = 1;
  while (token != NULL) {
    if (strncmp(token, "NVLD", strlen("NVLD")) == 0) {
      param->nvldSize = atoi(token + strlen("NVLD"));
      break;
    } else {
      if (count) count += 1;
      count += strlen(token);
    }
    token = strtok(NULL, "-");
  }
  strncpy(name,platform,count);

  // Build system topology file path
  char xmlTopoFile[PATH_MAX];
  char xmlGraphFile[PATH_MAX];
  char topoDir[1024];
  const char* envTopoDir = getenv("TOPO_DIR");
  if (envTopoDir) {
    snprintf(topoDir, 1024, "%s", envTopoDir);
  } else {
    topoDir[0] = '\0';
  }
  snprintf(xmlTopoFile, PATH_MAX, "%stopo/%s/system.xml", topoDir, name);

  // check for parameter correctness
  if (param->forceMerge && param->mergeLevel > PATH_PORT) {
    printf("NCCL graph_test does not support setting both NIC fusion NCCL_NET_MERGE_LEVEL and NCCL_NET_FORCE_MERGE. Discarding the value of NCCL_NET_FORCE_MERGE.\n");
    param->forceMerge = NULL;
  }

  // Check topologies
  if (param->intra) {
    buildGraphFilename(xmlGraphFile, PATH_MAX, topoDir, name, TEST_INTRA, param);
    checkTopo(xmlTopoFile, xmlGraphFile, name, TEST_INTRA, param, errors, warnings);
  }
  if (param->inter) {
    buildGraphFilename(xmlGraphFile, PATH_MAX, topoDir, name, TEST_INTER, param);
    checkTopo(xmlTopoFile, xmlGraphFile, name, TEST_INTER, param, errors, warnings);
  }
  free(name);
}

#define RUN_INTRA(platform)                          \
  do {                                               \
    struct testParam p = param;                      \
    p.inter = false;                                 \
    checkPlatform(platform, &p, &errors, &warnings); \
  } while (0)

#define RUN(platform) checkPlatform(platform, &param, &errors, &warnings)

#define RUN_PORT_RATIO(platform, r)                  \
  do {                                               \
    struct testParam p = param;                      \
    p.portRatio = r;                                 \
    p.intra = 0;                                     \
    checkPlatform(platform, &p, &errors, &warnings); \
  } while (0)

#define RUN_FUSION(platform, level)                  \
  do {                                               \
    struct testParam p = param;                      \
    p.mergeLevel = level;                            \
    p.intra = 0;                                     \
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
  printf("  -v       : enable the verbose mode, equivalent to NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH\n");
  printf("\n");
  printf("The name of the platform is given as <NAME>[-NVLD<X>], which indicates an NVLink domain of size <X>, where each host is <NAME>.\n");
  printf("For example:\n");
  printf(" - ./graph_test GB200 will consider an NVLink domain with a single host of GB200,\n");
  printf(" - ./graph_test GB200-NVLD8 will consider an NVLink domain with 8 hosts of GB200 (equivalent to 32 GPUs)\n");
  printf("\n");
  printf("Set NCCL_GRAPH_TEST_NGPUS=N to change the number of GPUs per node.\n");
  printf("Set NCCL_TOPO_DIR to override the default topo directory. This is necessary to invoke graph_test from an outside directory.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP=0 to disable dumping of graph diffs.\n");
  printf("Set NCCL_GRAPH_TEST_DUMP_SYSTEM_XML=1 to dump the processed system XML from NIC Fusion and then the fully trimmed system XML.\n");
  printf("Set NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GRAPH to get standard NCCL logs of your scenario. This is equivalent to using the -v flag.\n");
  printf("Set NCCL_TESTS_SPLIT_MASK=0xA to only get the graph for the communicator with the color 0 (unless changed, see NCCL_GRAPH_TEST_COLOR) in the case of NCCL_TESTS_SPLIT_MASK=0xA.\n");
  printf("Set NCCL_GRAPH_TEST_COLOR=0xA to change the color that is considered when using NCCL_TESTS_SPLIT_MASK.\n");
  printf("Set NCCL_NET_FORCE_MERGE to force the merge between devices, see NCCL documentation.\n");
  printf("Set NCCL_NET_MERGE_LEVEL to change the NIC fusion merge level, see NCCL documentation.\n");
  printf("Set NCCL_GRAPH_TEST_INTER=0/1 to enable the INTER test.\n");
  printf("Set NCCL_GRAPH_TEST_INTRA=0/1 to enable the INTRA test.\n");
}

static void parseArgs(int argc, const char* argv[], const char** platform, const char** ngpusArg) {
  bool verboseMode = false;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-v") == 0) {
      verboseMode = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      printHelpMessage();
      exit(0);
    } else if (*platform == NULL) {
      *platform = argv[i];
    } else if (*ngpusArg == NULL) {
      *ngpusArg = argv[i];
    }
  }
  // Set verbose mode environment variables
  if (verboseMode) {
    setenv("NCCL_DEBUG", "INFO", 1);
    // Append GRAPH to existing NCCL_DEBUG_SUBSYS or set it
    const char* existingSubsys = getenv("NCCL_DEBUG_SUBSYS");
    if (existingSubsys && strlen(existingSubsys) > 0) {
      char newSubsys[1024];
      snprintf(newSubsys, sizeof(newSubsys), "%s,GRAPH", existingSubsys);
      setenv("NCCL_DEBUG_SUBSYS", newSubsys, 1);
    } else {
      setenv("NCCL_DEBUG_SUBSYS", "GRAPH", 1);
    }
  }
}

int main(int argc, const char* argv[]) {
  setenv("NCCL_IGNORE_DISABLED_P2P", "2", 0); // Disable hardware health checks (NVML)
  setlinebuf(stdout);
  const char* platform = NULL;
  const char* ngpusArg = NULL;
  parseArgs(argc, argv, &platform, &ngpusArg); // Parse command line arguments
  setStackSize(16 * 1024 * 1024);              // set stack size to 16MiB to avoid stack overflow with large NVLDs

  struct testParam param;
  getTestParam(&param);
  allocateMock();

  int errors = 0, warnings = 0;
  if (platform != NULL) {
    if (ngpusArg != NULL) param.ngpus = atoi(ngpusArg);
    checkPlatform(platform, &param, &errors, &warnings);
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
    RUN_MULTI4("Cineca");
    RUN("GCP-NV");
    RUN("AWS-P5-H100");
    RUN("AWS-P5en-H100");
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
    {// GB300-NVL4
      RUN("GB300-CX8-NVL4");               // IB
      RUN_PORT_RATIO("GB300-CX8-NVL4", 2); // RoCE 2 ports
    }
    RUN("GB300-CX8-NVL32");
    RUN("GB300WS");
    RUN("DGX-Spark");
    RUN("DGX-Spark-flat");
    RUN("DGX-B300-RoCE");
  }
  printf("%d errors, %d warnings (%s)\n", errors, warnings, (errors || warnings) ? "FAILED" : "PASSED");
  freeMock();
  return (errors || warnings) ? 1 : 0;
}
