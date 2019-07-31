/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_common.h"
#include "param.h"
#include "graph.h"

uint64_t getScores(short* distances, int ndev) {
  uint64_t scores = 0ULL;
  for (int d=0; d<ndev; d++) {
    uint64_t score = 1 + PATH_SYS - distances[d];
    // Keep 3 bits of score info per dev
    scores |= ((score & NET_BITS_PER_IF_MASK)<<(NET_BITS_PER_IF*d));
  }
  return scores;
}

int getScore(uint64_t scores, int dev) {
  return (scores >> (dev*NET_BITS_PER_IF)) & NET_BITS_PER_IF_MASK;
}

ncclResult_t netDistance(int cudaDev, int dev, short* distance, netInfoFuncs* netInfo) {
  char* cudaPath = NULL;
  char* nicPath = NULL;
  ncclResult_t err;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));
  err = netInfo->pciPath(dev, &nicPath);
  *distance = (err != ncclSuccess || nicPath == NULL || cudaPath == NULL) ? PATH_SYS : pciDistance(nicPath, cudaPath);
  if (nicPath) free(nicPath);
  if (cudaPath) free(cudaPath);
  return ncclSuccess;
}

ncclResult_t netDevices(int* ndev, short** distances, netInfoFuncs* netInfo) {
  NCCLCHECK(netInfo->devices(ndev));
  if (*ndev == 0) {
    WARN("Error : Network returned 0 device");
    return ncclSystemError;
  }
  if (*ndev > NET_MAX_IFS) *ndev = NET_MAX_IFS;

  *distances = (short*)malloc(*ndev*sizeof(short));
  if (*distances == NULL) return ncclSystemError;

  // Find distance with current GPU
  int cudaDev, nvmlDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(getNvmlDevice(cudaDev, &nvmlDev))
  char line[1024];
  sprintf(line, "CUDA Dev %d[%d], %s NIC distance : ", cudaDev, nvmlDev, netInfo->name());
  for (int d=0; d<*ndev; d++) {
    NCCLCHECK(netDistance(cudaDev, d, *distances+d, netInfo));
    sprintf(line+strlen(line), " %s", pathDists[(*distances)[d]]);
  }
  INFO(NCCL_INIT|NCCL_NET, "%s", line);
  return ncclSuccess;
}

// In: cudaDev, ringId, netInfo
// In/out: netScores, netNDev
int getDev(int cudaDev, int ringId, uint64_t* netScores, int* netNDev, netInfoFuncs* netInfo) {
  uint64_t scores = netScores[cudaDev];
  if (scores == NET_SCORES_UNSET) {
    if (cudaDev >= NET_MAX_GPUS) {
      WARN("CUDA device %d >= MAX %d\n", cudaDev, NET_MAX_GPUS);
      return ncclInternalError;
    }
    int nDev;
    short* distances;
    NCCLCHECK(netDevices(&nDev, &distances, netInfo));
    netScores[cudaDev] = scores = getScores(distances, nDev);
    *netNDev = nDev;
    free(distances);
  }

  int dev = 0;
  int maxScore = 0;
  for (int d=0; d<*netNDev; d++) if (getScore(scores, d) > maxScore) maxScore = getScore(scores, d);
  int skip = ringId+1;
  while (skip) {
    for (int d=0; d<*netNDev; d++) {
      if (getScore(scores, d) == maxScore) {
        skip--;
        if (skip == 0) { dev = d; goto end; }
      }
    }
  }
end:
  return dev;
}

NCCL_PARAM(NetGdrRead, "NET_GDR_READ", -2);
NCCL_PARAM(NetGdrLevel, "NET_GDR_LEVEL", PATH_PHB);

ncclResult_t netGetGdrSupport(int dev, int read, int* useGdr, netInfoFuncs* netInfo) {
  *useGdr = 0;

  int cudaDev, nvmlDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(getNvmlDevice(cudaDev, &nvmlDev));

  if (read) { // For reads (sends) only enable under certain conditions
    int gdrReadParam = ncclParamNetGdrRead();
    if (gdrReadParam == 0) return ncclSuccess;
    if (gdrReadParam < 0) {
       int nvlink;
       NCCLCHECK(ncclNvlinkGpu(&nvlink));
       if (!nvlink) return ncclSuccess;
    }
  }

  // Check if we are close enough that it makes sense to enable GDR
  int netGdrLevel = ncclParamNetGdrLevel();
  short distance;
  NCCLCHECK(netDistance(cudaDev, dev, &distance, netInfo));
  if (distance >= netGdrLevel) {
    INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Disabled for GPU %d[%d] / HCA %d (distance %d >= %d)", netInfo->name(), cudaDev, nvmlDev, dev, distance, netGdrLevel);
    return ncclSuccess;
  }

  // Finally, check if the NIC supports it
  int flags;
  NCCLCHECK(netInfo->ptrSupport(dev, &flags));
  if ((flags & NCCL_PTR_CUDA) == 0) return ncclSuccess;
  *useGdr = 1;
  INFO(NCCL_NET,"NET/%s : GPU Direct RDMA Enabled for GPU %d[%d] / HCA %d (distance %d < %d), read %d", netInfo->name(), cudaDev, nvmlDev, dev, distance, netGdrLevel, read);
  return ncclSuccess;
}
