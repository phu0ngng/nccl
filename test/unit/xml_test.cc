#include "nccl.h"
#include "xml.h"
#include "topo.h"

#define MAX_MNNVL_NODES 64

int main(int argc, char* argv[]) {
  struct ncclXml *xml;
  if (argc < 2) {
    printf("Usage %s <XML file>\n", argv[0]);
    exit(1);
  }
  NCCLCHECK(xmlAlloc(&xml, MAX_MNNVL_NODES*NCCL_TOPO_XML_MAX_NODES));
  NCCLCHECK(ncclTopoGetXmlFromFile(argv[1], xml, 1));
  return 0;
}
