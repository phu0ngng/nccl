#ifndef NCCL_REGISTER_H_
#define NCCL_REGISTER_H_

struct ncclRegCache {
  struct ncclReg **slots;
  int capacity, population;
};

ncclResult_t ncclRegCleanup(struct ncclComm* comm);
ncclResult_t ncclRegFind(struct ncclComm* comm, void* data, size_t size, int* found);

#endif