#include "topo.h"
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

#include <sys/time.h>
uint64_t getTime() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1000000+tv.tv_usec;
}

const char* graphNames[] = { "Ring", "Tree", "CollNet" };

int compareGraphs(struct ncclTopoGraph* ref, struct ncclTopoGraph* out, int ngpus, int inter) {
  int errors = 0;
  if (memcmp(ref, out, sizeof(struct ncclTopoGraph)) != 0) {
    errors++;
    char line[256];
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
    sprintf(line+strlen(line), "%7s %2dx%4d/%4d %3s/%3s P%1d C%1d S%1d", graphNames[ref->id], ref->nChannels, ref->speedIntra, ref->speedInter, topoLinkTypeStr[ref->typeIntra], topoLinkTypeStr[ref->typeInter], ref->pattern, ref->crossNic, ref->sameChannels);
    while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
    sprintf(line+strlen(line), "%7s %2dx%4d/%4d %3s/%3s P%1d C%1d S%1d", graphNames[out->id], out->nChannels, out->speedIntra, out->speedInter, topoLinkTypeStr[out->typeIntra], topoLinkTypeStr[out->typeInter], out->pattern, out->crossNic, out->sameChannels);
    printf("%s\n", line);

    line[0] = '\0';
    for (int i=0; i<std::max(ref->nChannels, out->nChannels); i++) {
      sprintf(line, "                        Channel %2d : ", i);
      while (strlen(line) < margin) sprintf(line+strlen(line), " ");
      if (i < ref->nChannels) {
        if (inter) sprintf(line+strlen(line), "[%2d %2d] ", ref->inter[i*2], ref->inter[i*2+1]);
        for (int g=0; g<ngpus; g++) sprintf(line+strlen(line), "%2d ", ref->intra[i*ngpus+g]);
      }
      while (strlen(line) < margin+width) sprintf(line+strlen(line), " ");
      if (i < out->nChannels) {
        if (inter) sprintf(line+strlen(line), "[%2d %2d] ", ref->inter[i*2], ref->inter[i*2+1]);
        for (int g=0; g<ngpus; g++) sprintf(line+strlen(line), "%2d ", ref->intra[i*ngpus+g]);
      }
      printf("%s\n", line);
    }
  }
  return errors;
}

int checkTopo(const char* xmlTopoFile, const char* xmlGraphFile, const char* platform, int inter) {
  struct ncclXml* xmlSystem;
  INFO(NCCL_GRAPH, "Loading platform %s", platform);
  CHECK(ncclCalloc(&xmlSystem, 1));
  CHECK(ncclTopoGetXmlFromFile(xmlTopoFile, xmlSystem));
  struct ncclTopoSystem* system;
  if (xmlSystem->maxIndex == 0) {
    printf("Error : no system in %s\n", xmlTopoFile);
    return 1;
  }
  CHECK(ncclTopoGetSystemFromXml(xmlSystem, &system));
  CHECK(ncclTopoComputePaths(system, NULL));
  if (inter == 0) {
    for (int n=system->nodes[NET].count-1; n>=0; n--)
      NCCLCHECK(ncclTopoRemoveNode(system, NET, n));
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
  uint64_t computeTime = getTime();
  CHECK(ncclTopoCompute(system, &ringGraph));
  treeGraph.minChannels = treeGraph.maxChannels = ringGraph.nChannels;
  CHECK(ncclTopoCompute(system, &treeGraph));
  cNetGraph.minChannels = cNetGraph.maxChannels = ringGraph.nChannels;
  CHECK(ncclTopoCompute(system, &cNetGraph));
  computeTime = getTime() - computeTime;

  /* Get reference graphs from XML */
  struct ncclXml* xmlGraph;
  CHECK(ncclCalloc(&xmlGraph, 1));
  if (ncclTopoGetXmlGraphFromFile(xmlGraphFile, xmlGraph) != ncclSuccess) {
    printf(" %10s/%s  Error : no graph in %s\n", platform, inter ? "Inter":"Intra", xmlGraphFile);
    return 1;
  }
  struct ncclTopoGraph refRingGraph, refTreeGraph, refCNetGraph;
  memcpy(&refRingGraph, &ringGraph, sizeof(ringGraph));
  memcpy(&refTreeGraph, &treeGraph, sizeof(treeGraph));
  memcpy(&refCNetGraph, &cNetGraph, sizeof(cNetGraph));
  // Get graphs from XML. We select the right graph based on the id.
  CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refRingGraph));
  CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refTreeGraph));
  CHECK(ncclTopoGetGraphFromXml(xmlGraph->nodes, system, &refCNetGraph));

  /* Compare */
  int errors = 0;
  errors += compareGraphs(&refRingGraph, &ringGraph, system->nodes[GPU].count, inter);
  errors += compareGraphs(&refTreeGraph, &treeGraph, system->nodes[GPU].count, inter);
  errors += compareGraphs(&refCNetGraph, &cNetGraph, system->nodes[GPU].count, inter);

  printf(" %15s/%s  %2dx%3d/%3d | %2dx%3d/%3d | %2dx%3d/%3d", platform, inter ? "Inter":"Intra",
      ringGraph.nChannels, ringGraph.speedIntra, ringGraph.speedInter,
      treeGraph.nChannels, treeGraph.speedIntra, treeGraph.speedInter,
      cNetGraph.nChannels, cNetGraph.speedIntra, cNetGraph.speedInter);
  if (errors > 0) {
    char dumpFile[PATH_MAX];
    sprintf(dumpFile, "%s.dump", xmlGraphFile);
    struct ncclXml* xml;
    CHECK(ncclCalloc(&xml, 1));
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &cNetGraph };
    CHECK(ncclTopoGetXmlFromGraphs(3, graphs, system, xml));
    CHECK(ncclTopoDumpXmlToFile(dumpFile, xml));
    free(xml);
    printf(" FAILED %5ld ms\n", computeTime/1000);
  } else if (computeTime > 1000000) {
    printf("   SLOW %5ld ms\n", computeTime/1000);
    errors++;
  } else printf("     OK %5ld ms\n", computeTime/1000);
  return errors;
}

int checkPlatform(const char* platform) {
  int errors = 0;
  char xmlTopoFile[1024];
  char xmlGraphFile[1024];
  sprintf(xmlTopoFile, "topo/%s/system.xml", platform);
  sprintf(xmlGraphFile, "topo/%s/intra-graph.xml", platform);
  errors += checkTopo(xmlTopoFile, xmlGraphFile, platform, 0);
  sprintf(xmlGraphFile, "topo/%s/inter-graph.xml", platform);
  errors += checkTopo(xmlTopoFile, xmlGraphFile, platform, 1);
  return errors;
}

#define RUN(...) errors += checkPlatform(__VA_ARGS__)

int main(int argc, const char* argv[]) {
  setlinebuf(stdout);
  int errors = 0;
  if (argc > 1) {
    for (int a=1; a<argc; a++) {
      RUN(argv[a]);
    }
  } else {
    RUN("LOC-1G");
    RUN("PCI-1R");
    RUN("PCI-2R");
    RUN("PCI-NV");
    RUN("T4");
#ifdef __x86_64__
    RUN("DGX-1P");
    RUN("DGX-1P-4G");
    RUN("DGX-1V");
    RUN("DGX-1V-4G");
    RUN("DGX-1V-SHARP");
    RUN("DGX-2V");
    RUN("XMAN-3");
    RUN("Luna");
    RUN("GCP-NV");
    RUN("AWS-NV");
    RUN("Azure");
    RUN("FB-BUG");
    RUN("DGX-1V-1G");
#endif
    RUN("P9-6V");
    RUN("P9-4V");
    RUN("HP-ARM-V100");
  }
  printf("%d errors (%s)\n", errors, errors ? "FAILED" : "PASSED");
  return errors;
}
