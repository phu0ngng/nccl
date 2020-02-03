#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define ARRAY_SIZE 30 // Covers 8B to 4GB

float min(float a, float b) { return a < b ? a : b; }

static int log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}

void getData(float* data, const char* file) {
  int fd = open(file, O_RDONLY);
  if (fd == -1) { printf("Could not open %s\n", file); exit(1); }
  while (1) {
    char line[1024];
    for (int o=0; o<1024; o++) {
      if (read(fd, line+o, 1) != 1) goto eof;
      if (line[o] == '\n') {
        for (int i=o+1; i<1024; i++) line[i] = '\0';
        break;
      }
    }
    if (strncmp(line+29, "float", 5) != 0) continue;
    long size = atol(line);
    int s = log2i(size) - 3;
    if (s < 0 || s >= ARRAY_SIZE) continue;
    data[s] = min(data[s], min(atof(line+51), atof(line+83)));
  }
eof:
  close(fd);
}

int main(int argc, const char* argv[]) {
  if (argc < 2) {
    printf("Usage %s <files>\n", argv[0]);
    printf("Files should be in algo-then-proto order (tree-ll, tree-ll128, tree-simple, ring-ll, ring-ll128, ring-simple, ...), then default\n");
    exit(1);
  }
  float* data = (float*)malloc((argc-1)*ARRAY_SIZE*sizeof(float));
  for (int i=0; i<(argc-1)*ARRAY_SIZE; i++) data[i] = 1000000000.0; // Times should definitely be less than 1000 seconds.
  for (int i=0; i<argc-1; i++) getData(data+i*ARRAY_SIZE, argv[i+1]);
  for (int s=0; s<ARRAY_SIZE; s++) {
    for (int i=0; i<argc-1; i++) {
      printf("%f", data[s+i*ARRAY_SIZE]);
      if (i<argc-2) printf(",");
    }
    printf("\n");
  }
  return 0;
}
