#include "xml.h"
#include "topo.h"

int main(int argc, char* argv[]) {
  struct ncclXml* dst;
  struct ncclXml* src;
  int nSrcs;
  if (argc < 4 || strcmp(argv[argc-2], "-o") != 0) {
    printf("Usage %s <XML source files...> -o <XML destination file>\n", argv[0]);
    exit(1);
  }
  nSrcs = argc - 3;
  NCCLCHECK(xmlAlloc(&dst, nSrcs*NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECK(xmlAlloc(&src, NCCL_TOPO_XML_MAX_NODES));
  for (int i=0; i<nSrcs; i++) {
    printf("Loading %s\n", argv[i+1]);
    NCCLCHECK(ncclTopoGetXmlFromFile(argv[i+1], src, 1));
    printf("Fusing %s\n", argv[i+1]);
    NCCLCHECK(ncclTopoFuseXml(dst, src));
  }
  NCCLCHECK(ncclTopoDumpXmlToFile(argv[argc-1], dst));
  printf("All done, result in %s\n", argv[argc-1]);
  return 0;
}
