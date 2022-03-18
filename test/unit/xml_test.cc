#include "nccl.h"
#include "xml.h"

int main(int argc, char* argv[]) {
  struct ncclXml xml;
  if (argc < 2) {
    printf("Usage %s <XML file>\n", argv[0]);
    exit(1);
  }
  NCCLCHECK(ncclTopoGetXmlFromFile(argv[1], &xml, 1));
  return 0;
}
