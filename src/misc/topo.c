/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "topo.h"

#define BUSID_SIZE (sizeof("0000:00:00.0"))
#define BUSID_REDUCED_SIZE (sizeof("0000:00"))

ncclResult_t getCudaPath(int cudaDev, char** path) {
  char busId[BUSID_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, BUSID_SIZE, cudaDev));
  for (int i=0; i<BUSID_SIZE; i++) busId[i] = tolower(busId[i]);
  char busPath[] = "/sys/class/pci_bus/0000:00/../../0000:00:00.0";
  memcpy(busPath+sizeof("/sys/class/pci_bus/")-1, busId, BUSID_REDUCED_SIZE-1);
  memcpy(busPath+sizeof("/sys/class/pci_bus/0000:00/../../")-1, busId, BUSID_SIZE-1);
  *path = realpath(busPath, NULL);
  if (*path == NULL) {
    WARN("Could not find real path of %s", busPath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

const char* pathDists[PATH_ARRAY_SIZE] = { "PIX", "PXB", "PHB", "SOC" };

int pciDistance(char* path1, char* path2) {
  int score = 0;
  int depth = 0;
  int same = 1;
  for (int i=0; i<strlen(path1); i++) {
    if (path1[i] != path2[i]) same = 0;
    if (path1[i] == '/') {
      depth++;
      if (same == 1) score++;
    }
  }
  if (score <= 3) return PATH_SOC;
  if (score == 4) return PATH_PHB;
  if (score == depth-1) return PATH_PIX;
  return PATH_PXB;
}
