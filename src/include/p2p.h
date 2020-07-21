/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

struct ncclP2Pinfo {
  void* buff;
  ssize_t nbytes;
  struct ncclP2Pinfo* next;
};

struct ncclP2PConnect {
  int nrecv[MAXCHANNELS];
  int nsend[MAXCHANNELS];
  int* recv;
  int* send;
};

struct ncclP2Plist {
  struct ncclP2Pinfo **peerlist;
  struct ncclP2Pinfo **peerlistTail;
  int count;
};

static ncclResult_t enqueueP2pInfo(ncclP2Plist* p2p, int peer, void* buff, ssize_t nBytes) {
  struct ncclP2Pinfo* & head = p2p->peerlist[peer];
  struct ncclP2Pinfo* & tail = p2p->peerlistTail[peer];
  struct ncclP2Pinfo* next;
  NCCLCHECK(ncclCalloc(&next, 1));
  next->buff = buff;
  next->nbytes = nBytes;
  if (tail != NULL) tail->next = next;
  tail = next;
  if (head == NULL) head = next;
  return ncclSuccess;
}

static ncclResult_t dequeueP2pInfo(ncclP2Plist* p2p, int peer) {
  struct ncclP2Pinfo* & head = p2p->peerlist[peer];
  struct ncclP2Pinfo* & tail = p2p->peerlistTail[peer];
  struct ncclP2Pinfo* temp = head;
  head = head->next;
  if (tail == temp) tail = NULL;
  free(temp);
  return ncclSuccess;
}
#endif
