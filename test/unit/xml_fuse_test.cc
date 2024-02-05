#include "xml.h"

int main(int argc, char* argv[]) {
  struct ncclXml* dst;
  struct ncclXml* srcs;
  int nSrcs;
  if (argc < 4 || strcmp(argv[argc-2], "-o") != 0) {
    printf("Usage %s <XML source files...> -o <XML destination file>\n", argv[0]);
    exit(1);
  }
  nSrcs = argc - 3;
  dst = (struct ncclXml*)malloc(sizeof(*dst));
  srcs = (struct ncclXml*)malloc(nSrcs*sizeof(*srcs));
  for (int i=0; i<nSrcs; i++)
    NCCLCHECK(ncclTopoGetXmlFromFile(argv[i+1], srcs+i, 1));
  NCCLCHECK(ncclTopoFuseXmls(dst, srcs, nSrcs));
  NCCLCHECK(ncclTopoDumpXmlToFile(argv[argc-1], dst));
  return 0;
}
